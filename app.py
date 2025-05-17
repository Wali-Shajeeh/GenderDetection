from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import logging
from datetime import datetime
import base64
import pymongo
from pymongo.errors import ConnectionFailure, PyMongoError
import certifi
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress PyMongo logs (set to WARNING or higher)
logging.getLogger('pymongo').setLevel(logging.WARNING)

app = Flask(__name__)
# CORS for Vercel frontend
CORS(app, resources={r"/*": {"origins": ["https://distributed-computing-22yutz8al-wali-shajeehs-projects.vercel.app/", "*"]}})
socketio = SocketIO(app, cors_allowed_origins=["https://distributed-computing-22yutz8al-wali-shajeehs-projects.vercel.app/", "*"],
                    ping_timeout=120, ping_interval=30, engineio_logger=True)

# Initialize MongoDB client with increased timeouts
try:
    mongo_client = pymongo.MongoClient(
        "mongodb+srv://syedwalishajeehrizvi:TAD110@cluster0.xsihefe.mongodb.net/?retryWrites=true&w=majority&tls=true",
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        maxPoolSize=50,
        tls=True,
        tlsCAFile=certifi.where()
    )
    db = mongo_client["gender_detection"]
    collection = db["detections"]
    # Test connection
    mongo_client.admin.command('ping')
    logger.info("Connected to MongoDB successfully")
except (ConnectionFailure, PyMongoError) as e:
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

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected to SocketIO")
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.warning("Client disconnected")

@socketio.on('image')
def handle_image(data):
    try:
        # Validate image data
        if not data:
            logger.warning("Received empty image data")
            emit('result', {'message': 'Error: Empty image data'})
            return
        try:
            img_data = base64.b64decode(data)
            logger.debug("Image data decoded, size: %d bytes", len(img_data))
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            emit('result', {'message': 'Error: Invalid image data'})
            return
        if len(img_data) == 0:
            logger.warning("Decoded image buffer is empty")
            emit('result', {'message': 'Error: Empty image'})
            return

        # Decode image
        try:
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.debug("Image decoded, shape: %s", str(frame.shape) if frame is not None else "None")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            emit('result', {'message': 'Error: Failed to process image'})
            return
        if frame is None or frame.size == 0:
            logger.error("Decoded image is empty or invalid")
            emit('result', {'message': 'Error: Invalid image'})
            return

        # Person detection
        try:
            logger.debug("Starting YOLO detection")
            results = yolo_model(frame, classes=[0], conf=0.2)
            confidences = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []
            logger.debug("YOLO detection completed, boxes: %d, confidences: %s", len(results[0].boxes), str(confidences))
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}\n{traceback.format_exc()}")
            emit('result', {'message': 'Error: Detection failed'})
            return

        person_detected = False
        label = "No person detected"
        for result in results:
            try:
                boxes = result.boxes.xyxy.cpu().numpy()
                logger.debug("Processing YOLO boxes, count: %d", len(boxes))
            except Exception as e:
                logger.error(f"Failed to process YOLO boxes: {e}\n{traceback.format_exc()}")
                continue
            if len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box[:4])
                try:
                    person_img = frame[y1:y2, x1:x2]
                    logger.debug("Person image cropped, shape: %s", str(person_img.shape))
                except Exception as e:
                    logger.warning(f"Failed to crop person image: {e}")
                    continue
                if person_img.size == 0:
                    logger.warning("Empty person image cropped")
                    continue

                # Gender classification
                try:
                    logger.debug("Starting gender classification")
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    inputs = feature_extractor(images=person_img_rgb, return_tensors="pt")
                    with torch.no_grad():
                        logits = gender_model(**inputs).logits
                        predicted_idx = logits.argmax(-1).item()
                        label = gender_model.config.id2label[predicted_idx]
                    person_detected = True
                    logger.debug("Gender classification completed: %s", label)
                except Exception as e:
                    logger.error(f"Gender classification failed: {e}\n{traceback.format_exc()}")
                    continue

                # Store in MongoDB
                detection = {
                    "timestamp": datetime.now().isoformat(),
                    "gender": label,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                }
                try:
                    logger.debug("Saving detection to MongoDB")
                    collection.insert_one(detection)
                    logger.info(f"Detection saved to MongoDB: {detection}")
                except PyMongoError as e:
                    logger.error(f"Failed to save detection to MongoDB: {e}")
                    # Continue to emit result even if MongoDB fails

                break

        # Send result
        try:
            result_text = f"Person is {label}" if person_detected else label
            logger.debug("Sending result: %s", result_text)
            emit('result', {'message': result_text})
            logger.info("Result sent successfully")
        except Exception as e:
            logger.error(f"Failed to send result: {e}\n{traceback.format_exc()}")
            emit('result', {'message': 'Error: Failed to send result'})

    except Exception as e:
        logger.error(f"Error processing image: {e}\n{traceback.format_exc()}")
        emit('result', {'message': 'Error: Server error'})

@app.route('/status', methods=['GET'])
def status():
    logger.debug("Status request received")
    return jsonify({"status": "running"})

@app.route('/detections', methods=['GET'])
def get_detections():
    try:
        detections = list(collection.find({}, {'_id': 0}))
        logger.debug("Fetched detections from MongoDB")
        return jsonify(detections)
    except PyMongoError as e:
        logger.error(f"Failed to fetch detections: {e}")
        return jsonify({"error": "Failed to fetch detections"}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting server on port 5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
