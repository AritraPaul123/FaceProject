import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Optional, Dict, Any

class SpotifyAuthManager:
    def __init__(self):
        # Read Spotify API credentials from environment variables or config file
        try:
            # First try to read from config.py
            from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
            self.client_id = 'ff81c60506d04ff990a5512706b1edb0'
            self.client_secret = 'da5b1f8264c1474699fa13d3b0b6688c'
        except ImportError:
            # If config.py doesn't exist, check environment variables
            import os
            self.client_id = 'ff81c60506d04ff990a5512706b1edb0'
            self.client_secret = 'da5b1f8264c1474699fa13d3b0b6688c'
            
            if not self.client_id or not self.client_secret:
                print("\nPlease set up your Spotify credentials by either:")
                print("1. Creating a config.py file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
                print("   OR")
                print("2. Setting environment variables SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET\n")
                raise ValueError("Missing Spotify credentials")
        
        # Validate that credentials are not None or empty
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials are empty or None")
            
        self.redirect_uri = "http://127.0.0.1:8888/"
        self.scope = "user-modify-playback-state user-read-playback-state"
        
        self.sp: Optional[spotipy.Spotify] = None
    
    def authenticate(self) -> spotipy.Spotify:
        """Authenticate with Spotify and return the client."""
        try:
            # Safe way to show first 10 characters of client ID
            client_id_display = self.client_id[:10] if self.client_id else "UNKNOWN"
            print(f"Attempting to authenticate with Client ID: {client_id_display}...")
            print(f"Using redirect URI: {self.redirect_uri}")
            
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                # Add cache_path to store tokens
                cache_path=".cache",
                show_dialog=True  # Force showing the auth dialog
            )
            
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test the authentication by making a simple API call
            try:
                user_info = self.sp.current_user()
                print("Successfully authenticated with Spotify")
                return self.sp
            except Exception as e:
                print(f"Authentication test failed: {str(e)}")
                raise Exception(f"Failed to authenticate with Spotify: {str(e)}")
                
        except Exception as e:
            raise Exception(f"Failed to authenticate with Spotify: {str(e)}")
    
    def get_client(self) -> Optional[spotipy.Spotify]:
        """Get the authenticated Spotify client."""
        return self.sp if self.sp else self.authenticate()

    def play_tracks_by_mood(self, mood: str) -> None:
        """Play tracks based on the detected mood."""
        # Ensure we have a valid Spotify client
        if not self.sp:
            try:
                self.authenticate()
            except Exception as e:
                print(f"Authentication failed: {str(e)}")
                return
                
        # Check again if authentication was successful
        if not self.sp:
            print("Failed to initialize Spotify client")
            return
            
        # Assign sp to a local variable for clearer type checking
        sp_client = self.sp
        if not sp_client:
            print("Spotify client is not available")
            return
            
        # Mood to music mapping
        mood_queries = {
            "happy": "genre:pop mood:happy",
            "sad": "genre:indie mood:sad",
            "angry": "genre:rock mood:intense",
            "neutral": "genre:ambient mood:peaceful",
            "fear": "genre:classical mood:calm",
            "surprise": "genre:electronic mood:energetic",
            "disgust": "genre:jazz mood:sophisticated"
        }
        
        try:
            # Get available devices
            devices_response = sp_client.devices()
            
            # Check if devices response is valid
            if not devices_response or not isinstance(devices_response, dict):
                print("Failed to retrieve Spotify devices")
                return
                
            devices = devices_response.get('devices', [])
            
            # Find an active device or the first available device
            device_id = None
            for device in devices:
                if device.get('is_active', False):
                    device_id = device.get('id')
                    break
            
            # If no active device found, use the first available device
            if not device_id and devices:
                device_id = devices[0].get('id')
                # Transfer playback to this device
                if device_id:
                    sp_client.transfer_playback(device_id=device_id)
            
            if not device_id:
                print("No Spotify devices found. Please open Spotify on your computer or mobile device.")
                return
                
            # Search for tracks matching the mood
            query = mood_queries.get(mood.lower(), "genre:pop")
            results = sp_client.search(q=query, limit=10, type="track")
            
            if results and results.get("tracks", {}).get("items"):
                # Get track URIs
                track_uris = [track["uri"] for track in results["tracks"]["items"]]
                # Start playback on the selected device
                sp_client.start_playback(device_id=device_id, uris=track_uris)
            else:
                print(f"No tracks found for mood: {mood}")
        except Exception as e:
            error_msg = str(e)
            if "invalid_client" in error_msg:
                print("Spotify authentication error: Invalid client credentials.")
                print("Please verify your SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in config.py")
                print("Make sure you've created an app at https://developer.spotify.com/dashboard/")
            else:
                print(f"Error playing tracks: {error_msg}")