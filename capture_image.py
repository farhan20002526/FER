import streamlit as st
import cv2
import numpy as np
from fer import FER
import time
import os
import tempfile

# Load the pre-trained age detection model
AGE_MODEL = "age_deploy.prototxt"
AGE_PROTO = "age_net.caffemodel"
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

# Define the list of age ranges based on the model
AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-30)', '(38-43)', '(48-53)', '(60-100)']

# Initialize the FER detector
detector = FER()

# Custom title and description
st.markdown(
    """
    <h1 style='color: white;'>Real-time Emotion and Age Detection</h1>
    <p style='color: yellow;'>Use your webcam to detect emotions and estimate age in real time.</p>
    """,
    unsafe_allow_html=True
)

# Placeholders for image display and information
stframe = st.empty()
emotion_label_placeholder = st.empty()
age_label_placeholder = st.empty()

# Capture webcam image from the user
camera_image = st.camera_input("Capture an image from your webcam")

# If an image is captured, process it
if camera_image is not None:
    # Convert the captured image to OpenCV format
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Detect faces and emotions
    results = detector.detect_emotions(frame)

    # Initialize dominant emotion and age
    dominant_emotion = None
    dominant_emotion_score = 0
    age = "N/A"

    for result in results:
        bounding_box = result['box']
        emotions = result['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        dominant_emotion_score = emotions[dominant_emotion] * 100  # Scale to percentage

        # Draw bounding box and labels
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), 
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                      (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (bounding_box[0], bounding_box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Age detection
        face = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], 
                     bounding_box[0]:bounding_box[0] + bounding_box[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                       (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_RANGES[age_preds[0].argmax()]  # Get the predicted age range

        # Display age
        cv2.putText(frame, f"Age: {age}", (bounding_box[0], bounding_box[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw the progress bar on the frame
    progress_bar_length = 200
    progress_bar_height = 20
    cv2.rectangle(frame, (bounding_box[0], bounding_box[1] + bounding_box[3] + 5), 
                      (bounding_box[0] + progress_bar_length, bounding_box[1] + bounding_box[3] + 5 + progress_bar_height), 
                      (255, 255, 255), -1)  # Background

    filled_length = int(progress_bar_length * (dominant_emotion_score / 100))  # Calculate filled length
    cv2.rectangle(frame, (bounding_box[0], bounding_box[1] + bounding_box[3] + 5), 
                      (bounding_box[0] + filled_length, bounding_box[1] + bounding_box[3] + 5 + progress_bar_height), 
                      (0, 255, 0), -1)  # Fill with color

    # Add percentage text on the progress bar
    percentage_text = f"{dominant_emotion_score:.2f}%"
    text_x = bounding_box[0] + (progress_bar_length // 2) - (cv2.getTextSize(percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] // 2)
    text_y = bounding_box[1] + bounding_box[3] + 5 + (progress_bar_height // 2) + 6
    cv2.putText(frame, percentage_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the frame in Streamlit
    stframe.image(frame, channels='BGR')

    # Update emotion and age labels
    if dominant_emotion:
        emotion_label_placeholder.write(f"Dominant Emotion: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
        age_label_placeholder.write(f"Estimated Age: {age}")

        # Save the frame to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, frame)
            st.success(f"Frame captured and saved temporarily at {tmp_file.name}")

        # Provide a download link for the captured image
        with open(tmp_file.name, "rb") as f:
            st.download_button(
                label="Download Captured Image",
                data=f,
                file_name=f"{dominant_emotion}_{time.strftime('%Y%m%d-%H%M%S')}.jpg",
                mime="image/jpeg"
            )
