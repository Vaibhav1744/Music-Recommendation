import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Creating the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the pre-trained weights
model.load_weights('model.h5')

# Emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Streamlit Title
st.title("😊 Real-time Emotion Detection")
st.write("**Press 'Stop' to exit the webcam feed.**")

# Start Webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

# Button to stop the webcam feed
stop_button = st.button("Stop", key="stop_button")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or stop_button:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y-50), (x + w, y + h + 10), (255, 0, 0), 2)
        
        # Crop face
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        
        # Predict emotion
        prediction = model.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        emotion = emotion_dict[max_index]
        
        # Display emotion text
        cv2.putText(frame, emotion, (x + 20, y- 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    
cap.release()
st.write("Webcam closed successfully.")
