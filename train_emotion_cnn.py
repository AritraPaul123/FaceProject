import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import cv2

# Check for GPU availability
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Download and extract the FER-2013 dataset
print("Downloading FER-2013 dataset...")
path = kagglehub.dataset_download("msambare/fer2013")
print("Dataset downloaded to:", path)

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to load images from directory structure
def load_dataset_from_folders(dataset_path):
    X = []
    y = []
    
    # Process train and test directories
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"Processing {split} data...")
        
        # For each emotion folder
        for emotion_idx, emotion in enumerate(emotion_labels):
            emotion_path = os.path.join(split_path, emotion)
            if not os.path.exists(emotion_path):
                continue
                
            print(f"  Loading {emotion} images...")
            image_files = os.listdir(emotion_path)
            
            for image_file in image_files:
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(emotion_path, image_file)
                    try:
                        # Read image in grayscale
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize to 48x48 (standard for FER)
                            img = cv2.resize(img, (48, 48))
                            X.append(img)
                            y.append(emotion_idx)
                    except Exception as e:
                        print(f"    Error loading {image_path}: {e}")
    
    return np.array(X), np.array(y)

# Load the dataset
X_data, y = load_dataset_from_folders(path)
print("Dataset loaded. Shape:", X_data.shape)

# Process pixel data
# Convert to numpy array and normalize
X_data = X_data.astype('float32') / 255.0

# Reshape for CNN input (samples, height, width, channels)
X_data = X_data.reshape(X_data.shape[0], 48, 48, 1)

# Process emotion labels
y = to_categorical(y, num_classes=7)

# Split dataset into training and validation sets (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# Build custom CNN model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Fully connected layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

# Train the model
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

# Save the model
model.save('emotion_model.h5')
print("Model saved as emotion_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Training history plots saved as training_history.png")