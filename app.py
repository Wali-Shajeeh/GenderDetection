# mongo_client = pymongo.MongoClient("mongodb+srv://syedwalishajeehrizvi:TAD110@cluster0.xsihefe.mongodb.net/")


from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import pymongo
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# MongoDB setup
try:
    mongo_client = pymongo.MongoClient("mongodb+srv://syedwalishajeehrizvi:TAD110@cluster0.xsihefe.mongodb.net/")
    db = mongo_client["gender_detection_db"]
    collection = db["detections"]
    logger.info("Connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Load models
try:
    yolo_model = YOLO('yolov8n.pt')
    feature_extractor = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification")
    gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
    gender_model.eval()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Webcam setup
cap = None
client_connected = False

def initialize_webcam():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info("Webcam initialized")
        return True
    except Exception as e:
        logger.error(f"Webcam initialization failed: {e}")
        return False

def release_webcam():
    global cap
    try:
        if cap is not None:
            cap.release()
            cap = None
            logger.info("Webcam released")
    except Exception as e:
        logger.error(f"Webcam release failed: {e}")

def generate_frames():
    global client_connected
    while client_connected:
        if cap is None or not cap.isOpened():
            if not initialize_webcam():
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
                continue

        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            continue

        # Person detection
        try:
            results = yolo_model(frame, classes=[0], conf=0.5)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size == 0:
                        continue

                    # Gender classification
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    inputs = feature_extractor(images=person_img_rgb, return_tensors="pt")
                    with torch.no_grad():
                        logits = gender_model(**inputs).logits
                        predicted_idx = logits.argmax(-1).item()
                        label = gender_model.config.id2label[predicted_idx]

                    # Store in MongoDB
                    detection = {
                        "timestamp": datetime.now(),
                        "gender": label,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    }
                    collection.insert_one(detection)
                    logger.info(f"Detection saved: {detection}")

                    # Draw bounding box and label
                    color = (0, 255, 0) if label == "Male" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, color, 2)
        except Exception as e:
            logger.error(f"Error during detection: {e}")

        # Encode frame for streaming
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")

@app.route('/video_feed')
def video_feed():
    global client_connected
    client_connected = True
    logger.info("Client connected to video feed")
    response = Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    response.call_on_close(lambda: on_client_disconnect())
    return response

def on_client_disconnect():
    global client_connected
    client_connected = False
    release_webcam()
    logger.info("Client disconnected, webcam released")

@app.route('/status', methods=['GET'])
def status():
    logger.info("Received status request")
    return jsonify({
        "client_connected": client_connected
    })

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        release_webcam()
        mongo_client.close()