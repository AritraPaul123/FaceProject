import cv2
import numpy as np
# Remove deepface import
import customtkinter as ctk
from PIL import Image, ImageTk
from typing import Optional
import threading
import time
import os
from spotify_auth import SpotifyAuthManager

class EmotionMusicPlayer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize Spotify
        self.spotify = SpotifyAuthManager()
        
        # Configure window
        self.title("Emotion Music Player")
        self.geometry("800x600")
        
        # Initialize variables
        self.camera = cv2.VideoCapture(0)
        self.current_emotion: Optional[str] = None
        self.last_emotion_change = time.time()
        self.emotion_cooldown = 10  # Seconds between emotion changes
        self.running = True
        
        # Get the path to the Haar cascade file
        self.face_cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        
        # Create GUI elements
        self.create_gui()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
        self.camera_thread.start()
    
    def create_gui(self):
        """Create the GUI elements"""
        # Camera frame
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(pady=10)
        
        # Emotion display
        self.emotion_frame = ctk.CTkFrame(self)
        self.emotion_frame.pack(pady=20, padx=20, fill="x")
        
        self.emotion_label = ctk.CTkLabel(
            self.emotion_frame, 
            text="Detected Emotion: None",
            font=("Helvetica", 16)
        )
        self.emotion_label.pack(pady=10)
        
        # Control buttons
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=20, padx=20, fill="x")
        
        self.toggle_btn = ctk.CTkButton(
            self.control_frame,
            text="Stop",
            command=self.toggle_camera
        )
        self.toggle_btn.pack(side="left", padx=10)
        
        self.quit_btn = ctk.CTkButton(
            self.control_frame,
            text="Quit",
            command=self.quit_app
        )
        self.quit_btn.pack(side="right", padx=10)
    
    def update_camera(self):
        """Update camera feed and detect emotions"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                # Simple emotion detection based on facial features
                emotion = self.detect_simple_emotion(frame)
                
                # Update emotion if changed and cooldown passed
                current_time = time.time()
                if (emotion != self.current_emotion and 
                    current_time - self.last_emotion_change > self.emotion_cooldown):
                    self.current_emotion = emotion
                    self.last_emotion_change = current_time
                    # Update emotion label
                    self.emotion_label.configure(text=f"Detected Emotion: {emotion}")
                    # Play matching music
                    threading.Thread(
                        target=self.spotify.play_tracks_by_mood,
                        args=(emotion,),
                        daemon=True
                    ).start()
                
                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=img_tk)
                # Store reference to avoid garbage collection using setattr
                setattr(self.camera_label, '_image', img_tk)
    
    def detect_simple_emotion(self, frame):
        """Simple emotion detection based on facial features"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load Haar cascade for face detection using the correct path
        face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        
        # Check if the cascade file was loaded correctly
        if face_cascade.empty():
            print("Error: Could not load Haar cascade file")
            return 'neutral'
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # For simplicity, we'll just return a random emotion
            # In a real implementation, you would analyze facial features here
            emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise']
            return np.random.choice(emotions)
        
        return 'neutral'
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.toggle_btn.cget("text") == "Stop":
            self.toggle_btn.configure(text="Start")
            self.running = False
        else:
            self.toggle_btn.configure(text="Stop")
            self.running = True
            self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
            self.camera_thread.start()
    
    def quit_app(self):
        """Clean up and quit application"""
        self.running = False
        if self.camera.isOpened():
            self.camera.release()
        self.quit()

if __name__ == "__main__":
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run app
    app = EmotionMusicPlayer()
    app.mainloop()