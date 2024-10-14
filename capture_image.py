import streamlit as st
import cv2
import numpy as np
from fer import FER
import time
import base64
import os

# Function to get the base64 string of the image
def get_base64_of_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load your logo image as base64
logo_base64 = get_base64_of_image("logo.jpg")  # Replace with your logo image file path

logo_html = f"""
    <div style="text-align: left; padding: 10px;">
        <img src="data:image/jpeg;base64,{logo_base64}" style="border-radius: 50%; width: 100px; height: 100px;" alt="Logo"/>
    </div>
"""

st.markdown(logo_html, unsafe_allow_html=True)

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

# Initialize session state for webcam capture and detection
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'detection_started' not in st.session_state:
    st.session_state.detection_started = False
if 'captured_frames' not in st.session_state:
    st.session_state.captured_frames = []  # To store captured frames
if 'previous_emotion' not in st.session_state:
    st.session_state.previous_emotion = None  # To track the previous detected emotion

# Directory to save the captured images
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Placeholders for image display and information
stframe = st.empty()
progress_placeholder = st.empty()
emotion_label_placeholder = st.empty()
age_label_placeholder = st.empty()

# Function to get the color for the progress bar based on emotion score
def get_bar_color(score):
    return (0, 0, 0)  # Black for the bar

# Function to release the webcam
def release_webcam():
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release()
        st.session_state.video_capture = None
        cv2.destroyAllWindows()

# Start/Stop Emotion and Age Detection button
if st.button("Start Emotion and Age Detection" if not st.session_state.detection_started else "Stop Emotion and Age Detection"):
    st.session_state.detection_started = not st.session_state.detection_started
    if st.session_state.detection_started:
        st.session_state.video_capture = cv2.VideoCapture(0)  # Initialize webcam
        if not st.session_state.video_capture.isOpened():
            st.error("Could not open webcam. Please check your camera or permissions.")
            st.session_state.detection_started = False

# Process the video feed if detection is started
if st.session_state.detection_started:
    while st.session_state.detection_started:
        ret, frame = st.session_state.video_capture.read()

        if not ret:
            st.error("Failed to capture image")
            release_webcam()
            break

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
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Draw progress bar based on emotion score
            bar_x_start, bar_y_start = bounding_box[0], bounding_box[1] + bounding_box[3] + 10
            bar_x_end = bar_x_start + bounding_box[2]
            bar_y_end = bar_y_start + 20
            cv2.rectangle(frame, (bar_x_start, bar_y_start), (bar_x_end, bar_y_end), (50, 50, 50), -1)
            bar_filled_x_end = int(bar_x_start + (bounding_box[2] * (dominant_emotion_score / 100)))
            bar_color = get_bar_color(dominant_emotion_score)
            cv2.rectangle(frame, (bar_x_start, bar_y_start), (bar_filled_x_end, bar_y_end), bar_color, -1)
            cv2.putText(frame, f"{dominant_emotion_score:.2f}%", (bar_x_start, bar_y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Age detection
            face = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_RANGES[age_preds[0].argmax()]

            # Display age
            cv2.putText(frame, f"Age: {age}", (bounding_box[0], bounding_box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Automatically capture the frame if emotion changes and save it
            if st.session_state.previous_emotion != dominant_emotion:
                st.session_state.captured_frames.append(frame.copy())
                st.session_state.previous_emotion = dominant_emotion

                # Save the frame to disk
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_path = os.path.join(output_dir, f"{dominant_emotion}_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)

                st.success(f"Frame captured and saved as {image_path} due to emotion change to {dominant_emotion}")

        # Display the frame in Streamlit
        stframe.image(frame, channels='BGR')

        # Update emotion and age labels
        if dominant_emotion:
            progress_placeholder.progress(int(dominant_emotion_score))
            emotion_label_placeholder.write(f"Dominant Emotion: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
            age_label_placeholder.write(f"Estimated Age: {age}")

        time.sleep(0.1)  # Control the frame rate

# Release the webcam when the session ends
if not st.session_state.detection_started:
    release_webcam()
