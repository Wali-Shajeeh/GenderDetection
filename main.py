import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import os
from datetime import datetime
import pymongo
import pandas as pd

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# MongoDB connection (replace with your MongoDB Atlas connection string)
mongo_uri = "mongodb+srv://syedwalishajeehrizvi:TAD110@cluster0.xsihefe.mongodb.net/"
client = pymongo.MongoClient(mongo_uri)
db = client["gender_detection"]
collection = db["detections"]

# Load YOLOv8 model for person detection
yolo_model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model

# Load Hugging Face gender classification model
feature_extractor = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
gender_model.eval()  # Set to evaluation mode

# Function to generate Excel file from MongoDB data
def generate_excel(output_path):
    # Fetch all detections
    detections = list(collection.find())
    if not detections:
        print("No detections found in MongoDB.")
        return

    # Prepare data for Excel
    data = {
        "Timestamp": [det["timestamp"] for det in detections],
        "Gender": [det["gender"] for det in detections]
    }
    df = pd.DataFrame(data)

    # Aggregate data: count people by gender and timestamp
    summary = df.groupby(["Timestamp", "Gender"]).size().unstack(fill_value=0)
    summary = summary.reset_index()

    # Save to Excel
    excel_path = os.path.join("outputs", f"gender_summary_{os.path.basename(output_path).replace('.mp4', '')}.xlsx")
    summary.to_excel(excel_path, index=False)
    print(f"Excel file saved to {excel_path}")

# Function to process webcam feed, display, record, and store data
def process_webcam():
    # Open webcam (default camera, index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check if the camera is connected and accessible.")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Webcam resolution: {width}x{height}, FPS: {fps}")

    # Define output video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("outputs", f"gender_detection_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: Could not initialize video writer.")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Check webcam connection.")
            break

        # Person detection with YOLOv8
        results = yolo_model(frame, classes=[0], conf=0.5)  # Class 0 is 'person'

        # Process detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Crop person region
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                # Convert to RGB and preprocess for gender model
                person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                inputs = feature_extractor(images=person_img_rgb, return_tensors="pt")

                # Gender classification
                with torch.no_grad():
                    logits = gender_model(**inputs).logits
                    predicted_idx = logits.argmax(-1).item()
                    label = gender_model.config.id2label[predicted_idx]

                # Store detection in MongoDB
                detection = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "gender": label,
                    "video_file": output_path
                }
                collection.insert_one(detection)

                # Draw bounding box and label
                color = (0, 255, 0) if label == "Male" else (255, 0, 0)  # Green for Male, Red for Female
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)

        # Write frame to output video
        out.write(frame)

        # Display the frame with detection results
        cv2.imshow('Live Gender Detection', frame)

        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Exiting... Video saved to {output_path}")
            generate_excel(output_path)  # Generate Excel file
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    client.close()  # Close MongoDB connection

if __name__ == "__main__":
    try:
        process_webcam()
    except Exception as e:
        print(f"An error occurred: {e}")
        cv2.destroyAllWindows()
        client.close()