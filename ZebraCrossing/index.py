import streamlit as st
import cv2
import os
import tempfile
import io
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import joblib
import pandas as pd

# Load environment variables
load_dotenv()
ENDPOINT = os.getenv("ENDPOINT")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize the prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, credentials)

# Load the trained model for predicting crossing time
time_prediction_model = joblib.load(os.path.join('models', 'zebra_crossing_model.pkl'))

def detect_objects(prediction_client, project_id, model_name, frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w, ch = np.array(image).shape

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Detect objects in the image
    results = prediction_client.detect_image(project_id, model_name, image_bytes)

    # Display the image with boxes around each detected object
    draw = ImageDraw.Draw(image)

    linewidth = 6
    color = 'red'

    # Initialize human counter
    human_count = 0

    for prediction in results.predictions:
        # Only show objects with a > 90% probability and labeled as "human"
        if (prediction.probability * 100) > 90 and prediction.tag_name.lower() == "human":
            human_count += 1
            left = prediction.bounding_box.left * w
            top = prediction.bounding_box.top * h
            height = prediction.bounding_box.height * h
            width = prediction.bounding_box.width * w
            # Draw the box
            points = [(left, top), (left + width, top), (left + width, top + height), (left, top + height), (left, top)]
            draw.line(points, fill=color, width=linewidth)

    return image, human_count

def video_to_frames(video_path, frame_skip, resize_width, resize_height):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            if resize_width and resize_height:
                frame = cv2.resize(frame, (resize_width, resize_height))
            frames.append((frame_count, frame))

        frame_count += 1

    cap.release()
    return frames

def process_video(video_path, prediction_client, project_id, model_name, max_frames=5, resize_width=None, resize_height=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)
    frame_count = 0
    people_counts = []

    frames = video_to_frames(video_path, frame_interval, resize_width, resize_height)

    # Ensure exactly 5 frames are extracted
    frames = frames[:max_frames]

    for frame_index, frame in frames:
        image, human_count = detect_objects(prediction_client, project_id, model_name, frame)
        people_counts.append(human_count)
        st.image(image, caption=f"Frame {frame_index} - Detected Humans: {human_count}")
        print(f"Frame {frame_index} - Detected Humans: {human_count}")

    cap.release()
    return people_counts

st.title("Video Upload for Human Detection")

# Input for resizing frames
resize_width = st.number_input("Resize Width", min_value=1, value=320)
resize_height = st.number_input("Resize Height", min_value=1, value=240)

# Input for lane width
lane_width = st.number_input("Lane Width (meters)", min_value=1.0, value=3.5)

uploaded_files = st.file_uploader("Upload Videos", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    all_people_counts = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        people_counts = process_video(uploaded_file.name, predictor, PROJECT_ID, MODEL_NAME, max_frames=5, resize_width=resize_width, resize_height=resize_height)
        all_people_counts.append(people_counts)
        os.remove(uploaded_file.name)

    # Calculate the maximum number of humans detected in any frame across all videos
    max_humans = max(max(counts) for counts in all_people_counts)
    avg_humans_per_video = [sum(counts) / len(counts) for counts in all_people_counts]

    # Create a DataFrame for the input data with the appropriate feature names
    input_data = pd.DataFrame([[max_humans, lane_width]], columns=['number_of_persons', 'lane_width'])

    # Predict the time required for them to cross the zebra crossing using the trained model
    predicted_time = time_prediction_model.predict(input_data)

    st.write(f"Maximum number of humans detected: {max_humans}")
    st.write(f"Predicted time required for them to cross the zebra crossing: {predicted_time[0]:.2f} seconds")
    for i, avg in enumerate(avg_humans_per_video):
        st.write(f"Average number of humans detected in video {i+1}: {avg:.2f}")