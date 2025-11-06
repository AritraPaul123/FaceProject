import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class EmotionRecognizer:
    def __init__(self, model_path='emotion_model.h5'):
        """
        Initialize the EmotionRecognizer with a trained model.
        
        Args:
            model_path (str): Path to the trained emotion model file
        """
        self.model_path = model_path
        self.model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Check for GPU availability
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        print(f"GPU Available: {self.gpu_available}")
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """
        Load the trained emotion recognition model.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file {self.model_path} not found. Please train the model first.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_face(self, face_image):
        """
        Preprocess a face image for emotion prediction.
        
        Args:
            face_image (numpy.ndarray): Face image from OpenCV webcam feed
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        try:
            # Convert to grayscale if it's a color image
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_image
            
            # Resize to 48x48
            resized_face = cv2.resize(gray_face, (48, 48))
            
            # Normalize to 0-1 range
            normalized_face = resized_face.astype('float32') / 255.0
            
            # Reshape to match model input (1, 48, 48, 1)
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            
            return reshaped_face
        except Exception as e:
            print(f"Error preprocessing face image: {e}")
            return None
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image.
        
        Args:
            face_image (numpy.ndarray): Face image from OpenCV webcam feed
            
        Returns:
            tuple: (emotion_label, confidence_score) or (None, None) if prediction fails
        """
        # Check if model is loaded
        if self.model is None:
            print("Model not loaded. Cannot predict emotion.")
            return None, None
        
        try:
            # Preprocess the face image
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None, None
            
            # Predict emotion
            predictions = self.model.predict(processed_face, verbose=0)
            
            # Get the predicted class index and confidence score
            predicted_class = np.argmax(predictions[0])
            confidence_score = predictions[0][predicted_class]
            
            # Get the emotion label
            emotion_label = self.emotion_labels[predicted_class]
            
            return emotion_label, confidence_score
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return None, None

# Example usage function
def main():
    """
    Example usage of the EmotionRecognizer class.
    """
    # Initialize the emotion recognizer
    recognizer = EmotionRecognizer()
    
    # Check if model was loaded successfully
    if recognizer.model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Example: Load an image and predict emotion
    # Note: You would replace this with actual webcam feed in a real application
    print("Emotion Recognizer initialized successfully.")
    print("Ready to predict emotions from face images.")

if __name__ == "__main__":
    main()