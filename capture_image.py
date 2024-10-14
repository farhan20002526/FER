import streamlit as st
import cv2
import numpy as np
from fer import FER
import time
import os

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

# Directory to save the captured images
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Placeholders for image display and information
stframe = st.empty()
progress_placeholder = st.empty()
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

    # Display the frame in Streamlit
    stframe.image(frame, channels='BGR')

    # Update emotion and age labels
    if dominant_emotion:
        progress_placeholder.progress(int(dominant_emotion_score))
        emotion_label_placeholder.write(f"Dominant Emotion: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
        age_label_placeholder.write(f"Estimated Age: {age}")

        # Save the frame to disk if a dominant emotion is detected
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(output_dir, f"{dominant_emotion}_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        st.success(f"Frame captured and saved as {image_path}")
