import cv2
import numpy as np

def video_to_frames(video_path, resize_width=None, resize_height=None, frame_skip=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frames = []
    frame_count = 0
    extracted_frame_count = 0
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Skip frames based on frame_skip parameter
        if frame_count % frame_skip == 0:
            # Resize the frame if dimensions are provided
            if resize_width and resize_height:
                frame = cv2.resize(frame, (resize_width, resize_height))
            
            # Append the frame to the list
            frames.append(frame)
            extracted_frame_count += 1
            
            # Print the current frame number and its shape
            print(f"Processing frame {frame_count}, shape: {frame.shape}")

        frame_count += 1

    # Release the video capture object
    cap.release()
    
    print(f"Total number of frames extracted: {extracted_frame_count}")
    return frames