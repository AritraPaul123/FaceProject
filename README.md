# Emotion-Based Spotify Music Player 🎵

A real-time emotion recognition app that detects facial expressions and plays mood-matching songs via Spotify.

## Features 🚀

- Real-time facial detection using OpenCV Haar Cascades
- Dynamic emotion-to-music mapping
- Spotify-controlled music playback
- Modern GUI built with CustomTkinter
- Automatic emotion updates as camera feed changes

## Requirements ⚙️

- Python 3.10+
- Webcam
- Spotify Premium Account
- Spotify Desktop App

## Installation 📥

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Spotify Setup 🎵

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. Note down your Client ID and Client Secret
4. Add `http://127.0.0.1:8888` to your application's Redirect URIs in the settings

## Configuration ⚙️

Set up your Spotify credentials using either method:

### Method 1: Config File (Recommended)
1. Copy `config.template.py` to `config.py`
2. Fill in your Spotify credentials:
```python
SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"
```

### Method 2: Environment Variables
Set the following environment variables:
```bash
# Windows
set SPOTIFY_CLIENT_ID=your_client_id_here
set SPOTIFY_CLIENT_SECRET=your_client_secret_here

# Unix/MacOS
export SPOTIFY_CLIENT_ID=your_client_id_here
export SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## Usage 🎮

1. Make sure your Spotify desktop app is running
2. Play any song once to activate your device (you can pause it after)
3. Run the application:
```bash
python main.py
```
4. On first run, you'll need to authenticate with Spotify via your browser

## Controls 🎛️

- **Start/Stop**: Toggle emotion detection
- **Quit**: Close the application

## How it Works 🤔

1. The app captures your facial expression through the webcam
2. OpenCV Haar Cascades detect faces in the video stream
3. A simple emotion detection algorithm assigns emotions (currently randomized for demonstration)
4. The detected emotion is mapped to a suitable music genre/mood
5. Spotify automatically plays matching songs

## Training Custom Emotion Model 🧠

To train a custom CNN model for emotion recognition:

1. Run the training script:
```bash
python train_emotion_cnn.py
```

2. The script will:
   - Automatically download the FER-2013 dataset
   - Preprocess the data (48×48 grayscale images)
   - Train a custom CNN model
   - Save the trained model as `emotion_model.h5`
   - Display training accuracy and loss curves

The model targets at least 70% validation accuracy and supports 7 emotion classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Troubleshooting 🔧

- **No active device found**: Open Spotify desktop app and play any song once
- **Authentication failed**: Check your credentials and redirect URI
- **Webcam not detected**: Ensure your webcam is properly connected and not in use by another application
- **Haar Cascade file not found**: The app should automatically locate the Haar Cascade file from your OpenCV installation