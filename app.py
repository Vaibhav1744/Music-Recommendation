import numpy as np
import streamlit as st
import cv2
import sys
import random
sys.path.append('./libs')
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spotify API Setup
SPOTIPY_CLIENT_ID = 'fa1beec2349f4ea4b54bae7c91d76800'
SPOTIPY_CLIENT_SECRET = '3246a346647b4d75a3337209924f7ca1'

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))
except Exception as e:
    logger.error(f"Spotify API initialization failed: {e}")
    st.error("Failed to connect to Spotify. Please check your credentials.")

# Emotion Dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Song Recommendation Function
def get_spotify_recommendations(emotion, song_type, language, limit=10):
    """Fetch song recommendations from Spotify."""
    query = f"{emotion.lower()} {song_type.lower()} {language.lower()}"
    try:
        results = sp.search(q=query, type='track', limit=50)
        songs = [
            {'name': track['name'], 'artist': track['artists'][0]['name'], 'link': track['external_urls']['spotify']}
            for track in results['tracks']['items']
        ]
        random.shuffle(songs)
        return songs[:limit]
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        return []

# Emotion Preprocessing
def get_dominant_emotion(emotion_list):
    """Return the most frequent emotion."""
    if not emotion_list:
        return None
    counter = Counter(emotion_list)
    return counter.most_common(1)[0][0]

# Model Definition
def create_emotion_model():
    """Create and load the CNN model for emotion detection."""
    try:
        if not os.path.exists('model.h5'):
            st.error("model.h5 file not found in project directory!")
            return None
        
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        model.load_weights('model.h5')
        logger.info("Model weights loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model.h5: {e}")
        st.error("Failed to load model.h5. Ensure it’s in the project directory and compatible.")
        return None

# Initialize Model and Camera
cv2.ocl.setUseOpenCL(False)
model = create_emotion_model()
if model is None:
    st.error("Emotion model failed to initialize!")
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Initial camera access failed. Trying fallback...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("No camera available. Please check your webcam connection.")
        sys.exit(1)

# Load Haar Cascade
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    st.error(f"Failed to load {face_cascade_path}. Ensure OpenCV is installed correctly.")
    sys.exit(1)

# Streamlit UI
st.markdown("<h2 style='text-align: center; color: white;'><b>Emotion-Based Music Recommendation</b></h2>", unsafe_allow_html=True)

# User Preferences
with st.form(key='preferences'):
    song_type = st.selectbox("Song Type", ["Devotional", "Romantic", "Indie", "Pop", "Rock"])
    language = st.selectbox("Language", ["Hindi", "English", "Others"])
    st.form_submit_button(label="Set Preferences")

# Emotion Detection
st.subheader("Scan Your Emotion")
emotion_list = st.session_state.get('emotion_list', [])
snapshot_container = st.empty()  # For single window (real-time feed)
multi_window_container = st.columns(3)  # For multiple windows (snapshots)

if st.button('SCAN EMOTION'):
    with st.spinner("Scanning emotions..."):
        emotion_list = []
        snapshots = []
        frame_count = 0
        
        # Ensure camera is opened
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access camera. Please check your device permissions.")
                sys.exit(1)

        while frame_count < 28:  # Capture 28 frames
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera feed interrupted.")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Adjust face detection parameters for better detection
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            frame_count += 1

            if len(faces) == 0:
                cv2.putText(frame, "No Face Detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                
                # Prepare image for prediction
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    # Resize and preprocess the image
                    cropped_img = cv2.resize(roi_gray, (48, 48))
                    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
                    cropped_img = cropped_img.astype('float32') / 255.0  # Normalize
                    
                    # Make prediction
                    prediction = model.predict(cropped_img, verbose=0)
                    max_index = int(np.argmax(prediction))
                    emotion = emotion_dict[max_index]
                    emotion_list.append(emotion)
                    
                    # Display emotion on frame
                    cv2.putText(frame, emotion, (x+20, y-60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Capture snapshot
                    if len(snapshots) < 3 and random.random() < 0.3:
                        snapshot = frame.copy()
                        snapshots.append((snapshot, emotion))
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    continue

            # Display real-time feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            snapshot_container.image(frame_rgb, caption="Real-Time Emotion Detection", 
                                  use_container_width=True)
            time.sleep(0.05)

        # Display snapshots
        for idx, (snapshot, emotion) in enumerate(snapshots):
            with multi_window_container[idx]:
                st.image(cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB), 
                        caption=f"Snapshot: {emotion}", use_container_width=True)

        cap.release()
        # Removed cv2.destroyAllWindows() since we're not creating OpenCV windows
        
        if emotion_list:
            st.session_state.emotion_list = emotion_list
            st.success(f"Emotion scan complete! Detected {len(emotion_list)} emotions.")
        else:
            st.warning("No emotions detected. Please ensure your face is visible and well-lit.")

# Display Recommendations
if 'emotion_list' in st.session_state and st.session_state.emotion_list:
    dominant_emotion = get_dominant_emotion(st.session_state.emotion_list)
    st.markdown(f"<h4 style='text-align: center;'>Detected Emotion: {dominant_emotion}</h4>", unsafe_allow_html=True)
    
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs</b></h5>", unsafe_allow_html=True)
    songs = get_spotify_recommendations(dominant_emotion, song_type, language)
    
    if songs:
        for song in songs:
            st.markdown(f"<h4 style='text-align: center;'><a href='{song['link']}'>{song['name']} - {song['artist']}</a></h4>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; color: grey;'><i>{song['artist']}</i></h5>", unsafe_allow_html=True)
            st.write("---------------------------------------------------------------------------------------------------")
    else:
        st.warning("No songs found. Try adjusting preferences or rescanning.")
else:
    st.info("Please scan your emotion to get song recommendations.")

# Cleanup
if st.button("Reset"):
    st.session_state.clear()
    cap = cv2.VideoCapture(0)