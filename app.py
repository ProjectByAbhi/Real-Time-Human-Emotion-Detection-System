
from flask import Flask, render_template, jsonify, request
from collections import deque
import cv2
import numpy as np
import os
import threading
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)


# ==========================================
# 1. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# 2. EMOTION CLASSES
# ==========================================

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ======================================
# 3. PREDICTION MANAGEMENT
# ======================================

prediction_lock = threading.Lock()

# Store the latest 10 prediction probability arrays
prediction_history = deque(maxlen=10)

latest_prediction = {
    "emotion": "No face detected",
    "confidence": 0.0,
    "probabilities": {
        name: 0.0 for name in EMOTIONS
    }
}


def update_prediction(emotion, confidence, predictions):
    global latest_prediction

    current_probabilities = np.asarray(
        predictions,
        dtype=np.float32
    )

    with prediction_lock:

        # Add current prediction to history
        prediction_history.append(
            current_probabilities.copy()
        )

        # Average probabilities from recent predictions
        smoothed_predictions = np.mean(
            np.stack(prediction_history, axis=0),
            axis=0
        )

        # Select emotion using averaged probabilities
        index = int(np.argmax(smoothed_predictions))

        smoothed_emotion = EMOTIONS[index]

        smoothed_confidence = float(
            smoothed_predictions[index] * 100
        )

        # Prepare smoothed probability values
        probability_data = {
            name: float(smoothed_predictions[i])
            for i, name in enumerate(EMOTIONS)
        }

        # Update latest prediction
        latest_prediction = {
            "emotion": smoothed_emotion,
            "confidence": smoothed_confidence,
            "probabilities": probability_data
        }

        return smoothed_emotion, smoothed_confidence


def reset_prediction():
    global latest_prediction

    with prediction_lock:

        # Clear previous prediction history
        prediction_history.clear()

        latest_prediction = {
            "emotion": "No face detected",
            "confidence": 0.0,
            "probabilities": {
                name: 0.0 for name in EMOTIONS
            }
        }


def get_prediction():
    with prediction_lock:
        return {
            "emotion": latest_prediction["emotion"],
            "confidence": latest_prediction["confidence"],
            "probabilities": latest_prediction["probabilities"].copy()
        }


# ==========================================
# 4. BUILD CNN MODEL
# ==========================================

model = Sequential([
    Input(shape=(48, 48, 1)),

    Conv2D(64, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (5, 5), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(512, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(512, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(256),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.25),

    Dense(512),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.25),

    Dense(7, activation="softmax")
])


# ==========================================
# 5. LOAD TRAINED MODEL WEIGHTS
# ==========================================

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "emotion_model_v2_weights.npz"
)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Model weights not found: {WEIGHTS_PATH}"
    )

with np.load(
    WEIGHTS_PATH,
    allow_pickle=False
) as weights_data:

    weight_keys = sorted(
        weights_data.files,
        key=lambda key: int(key.split("_")[1])
    )

    weights = [
        weights_data[key]
        for key in weight_keys
    ]

model.set_weights(weights)

print("Emotion model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# ==========================================
# 6. LOAD HAAR CASCADE FACE DETECTOR
# ==========================================

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        f"Haar Cascade file not found: {CASCADE_PATH}"
    )

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError(
        "Could not load Haar Cascade file."
    )

print("Face detector loaded successfully!")



# ==========================================
# 7. BROWSER FRAME PROCESSING
# ==========================================

def process_frame(frame):
    original_height, original_width = frame.shape[:2]

    resize_scale = 1.0
    max_width = 320

    if original_width > max_width:
        resize_scale = original_width / max_width

        resized_height = int(original_height / resize_scale)

        frame = cv2.resize(
            frame,
            (max_width, resized_height),
            interpolation=cv2.INTER_AREA
        )
        
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Select only the largest detected face
    if len(faces) > 0:
        faces = sorted(
            faces,
            key=lambda face: face[2] * face[3],
            reverse=True
        )[:1]

    # Handle no face detected
    if len(faces) == 0:
        reset_prediction()

        result = get_prediction()
        result["face_box"] = None

        return result

    for (x, y, w, h) in faces:

        roi = gray[y:y + h, x:x + w]

        face = cv2.resize(
            roi,
            (48, 48)
        )

        face = face.astype("float32") / 255.0

        face = np.expand_dims(
            face,
            axis=(0, -1)
        )

        # Predict emotion using existing CNN
        predictions = model(
            face,
            training=False
        ).numpy()[0]

        index = int(np.argmax(predictions))

        raw_emotion = EMOTIONS[index]

        raw_confidence = float(
            predictions[index] * 100
        )

        # Apply existing 10-frame probability smoothing
        emotion, confidence = update_prediction(
            raw_emotion,
            raw_confidence,
            predictions
        )
        # Store face coordinates and prediction
        face_box = {
            "x": int(x * resize_scale),
            "y": int(y * resize_scale),
            "width": int(w * resize_scale),
            "height": int(h * resize_scale),
            "emotion": emotion,
            "confidence": confidence
        }

    result = get_prediction()
    result["face_box"] = face_box

    return result


# ==========================================
# 8. PROCESS BROWSER CAMERA FRAME
# ==========================================

@app.route("/process_frame", methods=["POST"])
def process_browser_frame():

    try:
        if "frame" not in request.files:
            return jsonify({
                "error": "No camera frame received."
            }), 400

        file = request.files["frame"]

        image_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        frame = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Invalid camera frame."
            }), 400

        prediction = process_frame(frame)

        return jsonify(prediction)

    except Exception as error:
        print("Frame processing error:", error)

        return jsonify({
            "error": "Unable to process camera frame."
        }), 500


# ==========================================
# 9. PREDICTION API
# ==========================================

@app.route("/predictions")
def predictions():

    return jsonify(get_prediction())


# ==========================================
# 10. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 11. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    print("Starting EmotionSense...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
