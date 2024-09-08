import streamlit as st
import tempfile
import cv2
import numpy as np
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import io
from dotenv import load_dotenv
import os
import time
import threading
import queue
from video_to_img import video_to_frames  # Import the video_to_frames function
from concurrent.futures import ThreadPoolExecutor
import joblib  # Import joblib to load the trained model
import pandas as pd  # Import pandas to handle DataFrame
import requests  # Import requests to fetch weather data
from datetime import datetime
import tensorflow as tf  # Import TensorFlow to load the trained model
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client  # Import Twilio client

def send_twilio_message(message_body, location):
    load_dotenv()
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    messaging_service_sid = os.getenv('TWILIO_MESSAGING_SERVICE_SID')
    to_phone_number = os.getenv('TO_PHONE_NUMBER')

    # Initialize the Twilio client
    client = Client(account_sid, auth_token)

    try:
        # Send the message
        message = client.messages.create(
            messaging_service_sid=messaging_service_sid,
            body=f'There is an emergency in {location}: {message_body}',
            to=to_phone_number
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")

def detect_objects(prediction_client, project_id, model_name, frame, location):
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

    # Initialize object counter and large vehicle counter
    object_count = 0
    large_vehicle_count = 0

    for prediction in results.predictions:
        # Only show objects with a > 50% probability
        if (prediction.probability * 100) > 50:
            object_count += 1
            if prediction.tag_name.lower() in ["ambulance", "truck"]:
                large_vehicle_count += 1
            left = prediction.bounding_box.left * w
            top = prediction.bounding_box.top * h
            height = prediction.bounding_box.height * h
            width = prediction.bounding_box.width * w
            # Draw the box
            points = [(left, top), (left + width, top), (left + width, top + height), (left, top + height), (left, top)]
            draw.line(points, fill=color, width=linewidth)

            # Check for "blur" or "blank" tags and send a Twilio message
            if prediction.tag_name.lower() in ["blur", "blank"]:
                message_body = f"Detected {prediction.tag_name} in the image with {prediction.probability * 100:.2f}% probability. The system might be compromised."
                send_twilio_message(message_body, location)
                return image, object_count, large_vehicle_count, True  # Indicate an error condition
            # Check for "fire" or "accident" tags and send a Twilio message
            if prediction.tag_name.lower() in ["fire", "accident"]:
                message_body = f"Detected {prediction.tag_name} in the image with {prediction.probability * 100:.2f}% probability. There is an emergency."
                send_twilio_message(message_body, location)

    return image, object_count, large_vehicle_count, False  # No error condition

def process_video(video_bytes, resize_width, resize_height, prediction_client, project_id, model_name, output_queue, location, max_frames=5):
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    # Calculate the total number of frames in the video
    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate frame_skip to extract exactly max_frames frames
    frame_skip = max(1, total_frames // max_frames)

    # Extract frames using video_to_frames with calculated frame_skip
    frames = video_to_frames(temp_video_path, resize_width, resize_height, frame_skip=frame_skip)

    # Ensure exactly max_frames frames are extracted
    frames = frames[:max_frames]

    processed_frames = 0
    total_object_count = 0
    total_large_vehicle_count = 0
    error_detected = False

    def process_frame(frame):
        nonlocal total_object_count, total_large_vehicle_count, processed_frames, error_detected
        image, object_count, large_vehicle_count, error = detect_objects(prediction_client, project_id, model_name, frame, location)
        total_object_count += object_count
        total_large_vehicle_count += large_vehicle_count
        processed_frames += 1
        output_queue.put((image, object_count, large_vehicle_count))
        if error:
            error_detected = True

    # Use ThreadPoolExecutor to process frames in parallel
    with ThreadPoolExecutor(max_workers=max_frames) as executor:
        executor.map(process_frame, frames)

    average_objects_per_frame = total_object_count / processed_frames if processed_frames else 0
    return total_object_count, total_large_vehicle_count, average_objects_per_frame, error_detected

def get_weather_condition(api_key, location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        weather = data['weather'][0]['main']
        return weather.lower()
    else:
        raise Exception(f"Error fetching weather data: {data['message']}")

def process_lane(lane_name, video_bytes, resize_width, resize_height, prediction_client, project_id, model_name, output_queue, event, next_event, lock, model, scaler, feature_names, lane_width, weather_condition, time_of_day_mapping, weather_condition_mapping, max_time_minutes, time_of_day, location):
    fixed_sleep_time = 60  # Fixed sleep time in seconds for error conditions
    while True:
        event.wait()  # Wait for the event to be set

        # Calculate the lane time
        total_object_count, total_large_vehicle_count, average_objects_per_frame, error_detected = process_video(
            video_bytes, resize_width, resize_height, prediction_client, project_id, model_name, output_queue, location
        )

        if error_detected:
            lane_time = fixed_sleep_time
        else:
            vehicle_count = average_objects_per_frame  # Example value

            # Map 'time_of_day' and 'weather_condition' to numerical values
            time_of_day_num = time_of_day_mapping[time_of_day]
            weather_condition_num = weather_condition_mapping[weather_condition]

            # Create a DataFrame with the features
            features = pd.DataFrame([[vehicle_count, lane_width, weather_condition_num, total_large_vehicle_count, time_of_day_num]], columns=['vehicle_count', 'lane_width', 'weather_condition', 'large_vehicle_count', 'time_of_day'])

            # Ensure all expected columns are present, even if they are not in the current data
            for col in feature_names:
                if col not in features.columns:
                    features[col] = 0

            # Reorder the columns to match the training data
            features = features[feature_names]

            # Scale the features
            scaled_features = scaler.transform(features)

            # Use the trained model to predict the lane time
            lane_time = model.predict(scaled_features)[0][0]

            # Ensure the lane time does not exceed the maximum time
            max_time_seconds = max_time_minutes * 60
            lane_time = min(lane_time, max_time_seconds)

            # Convert lane_time to a regular Python float
            lane_time = float(lane_time)

        output_queue.put((f"Calculated {lane_name}", total_object_count, total_large_vehicle_count, average_objects_per_frame, lane_time))

        # Signal that the next lane can start processing up to the calculation part
        next_event.set()

        # Wait for the next lane to calculate its time
        next_event.wait()

        # Clear the event for the next iteration
        event.clear()

        # Acquire the lock to ensure only one lane is sleeping at a time
        with lock:
            start_time = time.time()  # Record the start time
            output_queue.put((f"{lane_name} start time", start_time))
            # Simulate the lane being open for the calculated time
            time.sleep(lane_time)

            end_time = time.time()  # Record the end time
            output_queue.put((f"{lane_name} end time", end_time))

        # Signal that this lane has finished its sleep time
        next_event.clear()

def process_zebra_crossing(video_bytes1, video_bytes2, resize_width, resize_height, prediction_client, project_id, model_name, output_queue, event, next_event, lock, model, crossing_width,location):
    fixed_sleep_time = 30  # Fixed sleep time in seconds for error conditions
    while True:
        event.wait()  # Wait for the event to be set

        # Process the first zebra crossing video to detect humans
        _, _, average_objects_per_frame1, error_detected1 = process_video(video_bytes1, resize_width, resize_height, prediction_client, project_id, model_name, output_queue,location)

        # Process the second zebra crossing video to detect humans
        _, _, average_objects_per_frame2, error_detected2 = process_video(video_bytes2, resize_width, resize_height, prediction_client, project_id, model_name, output_queue,location)

        if error_detected1 or error_detected2:
            crossing_time = fixed_sleep_time
        else:
            # Take the maximum of the average number of humans detected from both videos
            max_avg_humans = max(average_objects_per_frame1, average_objects_per_frame2)

            # Create a DataFrame for the input data with the appropriate feature names
            input_data = pd.DataFrame([[max_avg_humans, crossing_width]], columns=['number_of_persons', 'lane_width'])

            # Predict the time required for them to cross the zebra crossing using the trained model
            crossing_time = model.predict(input_data)[0]

        # Put the zebra crossing summary message into the output queue
        output_queue.put((f"Calculated Zebra Crossing", max_avg_humans, crossing_time))

        # Signal that the next lane can start processing up to the calculation part
        next_event.set()

        # Wait for the next lane to calculate its time
        next_event.wait()

        # Clear the event for the next iteration
        event.clear()

        # Acquire the lock to ensure only one lane is sleeping at a time
        with lock:
            start_time = time.time()  # Record the start time
            output_queue.put((f"Zebra Crossing start time", start_time))
            # Simulate the zebra crossing being open for the calculated time
            time.sleep(crossing_time)

            end_time = time.time()  # Record the end time
            output_queue.put((f"Zebra Crossing end time", end_time))

        # Signal that this zebra crossing has finished its sleep time
        next_event.clear()

def get_time_of_day():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return 'morning'
    elif 12 <= current_hour < 17:
        return 'noon'
    elif 17 <= current_hour < 21:
        return 'evening'
    elif 21 <= current_hour < 24:
        return 'night_early'
    else:
        return 'night_late'

def main():
    try:
        load_dotenv()

        # Vehicle detection environment variables
        vehicle_prediction_endpoint = os.getenv("VEHICLE_PREDICTION_ENDPOINT")
        vehicle_prediction_key = os.getenv("VEHICLE_PREDICTION_KEY")
        vehicle_project_id = os.getenv("VEHICLE_PROJECT_ID")
        vehicle_model_name = os.getenv("VEHICLE_MODEL_NAME")

        # Human detection environment variables
        human_prediction_endpoint = os.getenv("HUMAN_PREDICTION_ENDPOINT")
        human_prediction_key = os.getenv("HUMAN_PREDICTION_KEY")
        human_project_id = os.getenv("HUMAN_PROJECT_ID")
        human_model_name = os.getenv("HUMAN_MODEL_NAME")

        # Weather API key
        weather_api_key = os.getenv("WEATHER_API_KEY")

        # Initialize the prediction clients
        vehicle_credentials = ApiKeyCredentials(in_headers={"Prediction-key": vehicle_prediction_key})
        vehicle_predictor = CustomVisionPredictionClient(vehicle_prediction_endpoint, vehicle_credentials)

        human_credentials = ApiKeyCredentials(in_headers={"Prediction-key": human_prediction_key})
        human_predictor = CustomVisionPredictionClient(human_prediction_endpoint, human_credentials)

        # Load the scaler and the neural network model for lane time prediction
        scaler_path = 'models/scaler.pkl'
        scaler = joblib.load(scaler_path)

        lane_model_path = 'models/traffic_clearance_time_model.h5'
        lane_model = tf.keras.models.load_model(lane_model_path)
        lane_model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

        # Load the trained model for zebra crossing time prediction
        zebra_crossing_model_path = 'ZebraCrossing/models/zebra_crossing_model.pkl'
        zebra_crossing_model = joblib.load(zebra_crossing_model_path)

        # Define the mapping for 'time_of_day' and 'weather_condition'
        time_of_day_mapping = {'morning': 0, 'noon': 1, 'evening': 2, 'night_early': 3, 'night_late': 4}
        weather_condition_mapping = {
            'clear': 0,
            'clouds': 1,
            'drizzle': 2,
            'rain': 3,
            'thunderstorm': 4,
            'snow': 5,
            'mist': 6,
            'smoke': 7,
            'haze': 8,
            'dust': 9,
            'fog': 10,
            'sand': 11,
            'ash': 12,
            'squall': 13,
            'tornado': 14
        }

        # Load the feature names used during training
        feature_names = ['vehicle_count', 'lane_count', 'weather_condition', 'large_vehicle_count', 'time_of_day']

        # Get the current location
        location = st.text_input("Enter your location", "Bhubaneswar")

        # Get the current weather condition
        weather_condition = get_weather_condition(weather_api_key, location)
        st.write(f"The current weather condition in {location} is: {weather_condition}")

        # Debug: Print the current configuration
        st.write(f"Max upload size: {st.config.get_option('server.maxUploadSize')} MB")

        # Load lane videos
        video_file1 = st.file_uploader("Upload First Video (Lane 1)", type=['mp4', 'avi'])
        video_file2 = st.file_uploader("Upload Second Video (Lane 2)", type=['mp4', 'avi'])
        video_file3 = st.file_uploader("Upload Third Video (Lane 3)", type=['mp4', 'avi'])
        resize_width = st.number_input("Enter resize width", value=320)
        resize_height = st.number_input("Enter resize height", value=240)

        # Add input fields for lane width
        lane1_width = st.number_input("Enter lane width for Lane 1", value=10)
        lane2_width = st.number_input("Enter lane width for Lane 2", value=10)
        lane3_width = st.number_input("Enter lane width for Lane 3", value=10)

        # Add input field for maximum lane open time
        max_time_minutes = st.number_input("Enter maximum lane open time (in minutes)", value=2)

        # Get the current time of day
        time_of_day = get_time_of_day()

        # Load zebra crossing videos
        zebra_crossing_video1 = st.file_uploader("Upload First Zebra Crossing Video", type=['mp4', 'avi'])
        zebra_crossing_video2 = st.file_uploader("Upload Second Zebra Crossing Video", type=['mp4', 'avi'])
        zebra_crossing_width = st.number_input("Enter zebra crossing width (meters)", min_value=1.0, value=3.5)

        if video_file1 and video_file2 and video_file3 and zebra_crossing_video1 and zebra_crossing_video2:
            # Read video files into memory
            video_bytes1 = video_file1.read()
            video_bytes2 = video_file2.read()
            video_bytes3 = video_file3.read()
            zebra_crossing_bytes1 = zebra_crossing_video1.read()
            zebra_crossing_bytes2 = zebra_crossing_video2.read()

            # Create a queue for thread-safe communication
            output_queue = queue.Queue()

            # Create events for synchronization
            lane1_event = threading.Event()
            lane2_event = threading.Event()
            lane3_event = threading.Event()
            zebra_crossing_event = threading.Event()
            lock = threading.Lock()

            # Create threads for processing each lane and zebra crossing
            lane1_thread = threading.Thread(target=process_lane, args=("Lane 1", video_bytes1, resize_width, resize_height, vehicle_predictor, vehicle_project_id, vehicle_model_name, output_queue, lane1_event, lane2_event, lock, lane_model, scaler, feature_names, lane1_width, weather_condition, time_of_day_mapping, weather_condition_mapping, max_time_minutes, time_of_day,location))
            lane2_thread = threading.Thread(target=process_lane, args=("Lane 2", video_bytes2, resize_width, resize_height, vehicle_predictor, vehicle_project_id, vehicle_model_name, output_queue, lane2_event, lane3_event, lock, lane_model, scaler, feature_names, lane2_width, weather_condition, time_of_day_mapping, weather_condition_mapping, max_time_minutes, time_of_day,location))
            lane3_thread = threading.Thread(target=process_lane, args=("Lane 3", video_bytes3, resize_width, resize_height, vehicle_predictor, vehicle_project_id, vehicle_model_name, output_queue, lane3_event, zebra_crossing_event, lock, lane_model, scaler, feature_names, lane3_width, weather_condition, time_of_day_mapping, weather_condition_mapping, max_time_minutes, time_of_day,location))
            zebra_crossing_thread = threading.Thread(target=process_zebra_crossing, args=(zebra_crossing_bytes1, zebra_crossing_bytes2, resize_width, resize_height, human_predictor, human_project_id, human_model_name, output_queue, zebra_crossing_event, lane1_event, lock, zebra_crossing_model, zebra_crossing_width,location))

            # Start the threads
            lane1_thread.start()
            lane2_thread.start()
            lane3_thread.start()
            zebra_crossing_thread.start()

            # Start with lane 1
            lane1_event.set()

            # Main loop to update the Streamlit interface
            while True:
                # Get the next message from the queue
                message = output_queue.get()

                if isinstance(message[0], str) and message[0].endswith("start time"):
                    # Lane or zebra crossing start time message
                    name, start_time = message
                    st.write(f"{name} started at {time.ctime(start_time)}")
                elif isinstance(message[0], str) and message[0].endswith("end time"):
                    # Lane or zebra crossing end time message
                    name, end_time = message
                    st.write(f"{name} ended at {time.ctime(end_time)}")
                elif isinstance(message[0], str) and message[0].startswith("Calculated Zebra Crossing"):
                    # Zebra crossing summary message
                    name, max_humans, crossing_time = message
                    st.write(f"Processing {name}...")
                    st.write(f"Maximum number of humans detected in {name}: {max_humans}")
                    st.write(f"{name} will be open for {crossing_time} seconds.")
                elif isinstance(message[0], str):
                    # Lane summary message
                    name, total_object_count, total_large_vehicle_count, average_objects_per_frame, time_required = message
                    st.write(f"Processing {name}...")
                    st.write(f"Total number of objects detected in {name}: {total_object_count}")
                    st.write(f"Total number of large vehicles detected in {name}: {total_large_vehicle_count}")
                    st.write(f"Average number of objects detected per frame in {name}: {average_objects_per_frame:.2f}")
                    st.write(f"{name} will be open for {time_required} seconds.")
                else:
                    # Frame message
                    image, object_count, large_vehicle_count = message
                    st.image(image, caption=f"Frame with {object_count} objects and {large_vehicle_count} large vehicles")
    except Exception as ex:
        st.write(f"An error occurred: {ex}")

if __name__ == "__main__":
    main()