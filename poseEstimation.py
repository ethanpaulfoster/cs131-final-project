#POSE ESTIMATION SAVING TO ARRAY

import cv2
from ultralytics import YOLO
from IPython.display import display, clear_output
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch

def estimate_pose(video_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n-pose.pt')

    data_frames = []
    keypoints = []
    boxes = []

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Variable to store the bounding box of the previous detection
    prev_bbox = None  # Format: [x1, y1, x2, y2]
    m = 50  # Buffer in pixels

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            data_frames.append(frame)
            # If a previous bounding box exists, crop the frame around that region plus a buffer
            if prev_bbox is not None:
                x1, y1, x2, y2 = prev_bbox
                # Calculate new bounds with buffer, and ensure they are within frame dimensions
                x1, y1 = max(0, x1-m), max(0, y1-m)
                x2, y2 = min(frame.shape[1], x2+m), min(frame.shape[0], y2+m)
                crop_frame = frame[y1:y2, x1:x2]
                conf=0.01
            else:
                crop_frame = frame
                conf=0.5

            # Run YOLOv8 inference on the (cropped) frame
            results = model.predict(crop_frame, max_det=1, conf=conf, classes=0)
            if results[0].keypoints:
                keypoints.append(results[0].keypoints.xyn)
                #boxes.append(results[0].boxes.xyxy)
            else:
                keypoints.append(None)

            # Visualize the results on the (cropped) frame
            annotated_frame = results[0].plot()

            if prev_bbox is not None:
                # Adjust annotated_frame to fit into the original frame (for visualization, optional)
                frame[y1:y2, x1:x2] = annotated_frame
                annotated_frame = frame

            # Update prev_bbox using the detection results
            if len(results[0]) > 0:  # Check if there is at least one detection
                bbox = results[0].boxes[0].xyxy[0].numpy()  # Convert to numpy array
                if prev_bbox is not None:
                    # Adjust the bounding box to original frame's coordinates if cropped
                    prev_bbox = [int(bbox[0]) + x1, int(bbox[1]) + y1, int(bbox[2]) + x1, int(bbox[3]) + y1]
                else:
                    # Directly use the detected bounding box if not cropped
                    prev_bbox = bbox.astype(int)
            if prev_bbox is not None:
                boxes.append(torch.tensor(prev_bbox))
            else:
                boxes.append(torch.empty(4, dtype=int))

        else:
            break
            
    return data_frames, keypoints, boxes   