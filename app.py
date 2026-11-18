
from flask import Flask, render_template, Response, jsonify
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
# 7. SHARED CAMERA MANAGER
# ==========================================

class CameraManager:

    def __init__(self):

        self.stop_event = threading.Event()

        self.thread = None

        self.state_lock = threading.RLock()
        self.frame_lock = threading.Lock()

        self.latest_frame = None

        # Laptop camera
        self.camera_index = 0

        # Camera status
        self.status = "stopped"
        self.error_message = None

    # --------------------------------------
    # UPDATE CAMERA STATUS
    # --------------------------------------

    def set_status(self, status, error=None):

        with self.state_lock:
            self.status = status
            self.error_message = error

    # --------------------------------------
    # GET CAMERA STATUS
    # --------------------------------------

    def get_status(self):

        with self.state_lock:
            return {
                "status": self.status,
                "error": self.error_message,
                "running": (
                    self.status == "running"
                    or self.status == "starting"
                    or self.status == "reconnecting"
                )
            }

    # --------------------------------------
    # START CAMERA
    # --------------------------------------

    def start(self):

        with self.state_lock:

            if self.thread is not None and self.thread.is_alive():

                print("Camera is already running.")
                return False

            self.stop_event.clear()

            self.error_message = None
            self.status = "starting"

            self.thread = threading.Thread(
                target=self.capture_loop,
                daemon=True,
                name="EmotionSenseCamera"
            )

            self.thread.start()

        print("Camera background thread started.")

        return True

    # --------------------------------------
    # CHECK CAMERA STATUS
    # --------------------------------------

    def is_running(self):

        return (
            not self.stop_event.is_set()
            and self.thread is not None
            and self.thread.is_alive()
        )

    # --------------------------------------
    # GET LATEST FRAME
    # --------------------------------------

    def get_frame(self):

        with self.frame_lock:
            return self.latest_frame

    # --------------------------------------
    # STOP CAMERA
    # --------------------------------------

    def stop(self):

        print("Stopping camera...")

        self.stop_event.set()

        thread = self.thread

        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5)

            if thread.is_alive():
                print(
                    "Warning: Camera thread has not stopped yet."
                )

        with self.frame_lock:
            self.latest_frame = None

        reset_prediction()

        self.set_status("stopped")

        print("Camera stop requested.")

    # --------------------------------------
    # PROCESS FRAME
    # --------------------------------------

    def process_frame(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Select only the largest detected face
        if len(faces) > 0:
            faces = sorted(
                faces,
                key=lambda face: face[2] * face[3],
                reverse=True
            )[:1]

        if len(faces) == 0:
            reset_prediction()

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
            
            # Predict emotion using CNN
            predictions = model(
                face,
                training=False
            ).numpy()[0]

            index = int(np.argmax(predictions))

            raw_emotion = EMOTIONS[index]

            raw_confidence = float(
                predictions[index] * 100
            )

            # Apply 10-frame probability smoothing
            emotion, confidence = update_prediction(
                raw_emotion,
                raw_confidence,
                predictions
            )

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display emotion
            label = f"{emotion}: {confidence:.1f}%"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame

    # --------------------------------------
    # CAMERA CAPTURE LOOP
    # --------------------------------------

    def capture_loop(self):

        camera = None
        connection_failures = 0
        max_connection_failures = 10

        try:

            while not self.stop_event.is_set():

                self.set_status("reconnecting")

                print("Connecting to camera...")

                camera = cv2.VideoCapture(
                    self.camera_index,
                    cv2.CAP_DSHOW
                )

                if not camera.isOpened():

                    connection_failures += 1

                    print(
                        f"Camera unavailable. Attempt "
                        f"{connection_failures}/{max_connection_failures}"
                    )

                    camera.release()
                    camera = None

                    if connection_failures >= max_connection_failures:

                        self.set_status(
                            "error",
                            "Unable to connect to the camera."
                        )

                        self.stop_event.set()
                        break

                    self.stop_event.wait(1)
                    continue

                # Camera configuration
                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    1280
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    720
                )

                camera.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )

                self.set_status("running")

                connection_failures = 0

                print("Camera connected successfully!")

                failed_reads = 0

                while not self.stop_event.is_set():

                    success, frame = camera.read()

                    if not success:

                        failed_reads += 1

                        print(
                            f"Camera read failed: {failed_reads}"
                        )

                        if failed_reads >= 5:

                            print(
                                "Camera disconnected. Reconnecting..."
                            )

                            break

                        self.stop_event.wait(0.05)
                        continue

                    failed_reads = 0

                    # Process frame and detect emotions
                    frame = self.process_frame(frame)

                    # Encode frame as JPEG
                    success, buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            85
                        ]
                    )

                    if success:

                        with self.frame_lock:
                            self.latest_frame = buffer.tobytes()

                # Release camera before reconnecting
                if camera is not None:

                    camera.release()
                    camera = None

                if not self.stop_event.is_set():

                    self.set_status("reconnecting")
                    self.stop_event.wait(1)

        except Exception as error:

            print("Camera processing error:", error)

            self.set_status(
                "error",
                str(error)
            )

            self.stop_event.set()

        finally:

            if camera is not None:
                camera.release()

            with self.frame_lock:
                self.latest_frame = None

            reset_prediction()

            if self.status != "error":
                self.set_status("stopped")

            print("Camera background thread stopped.")


# Initialize shared camera manager
camera_manager = CameraManager()


# ==========================================
# 8. GENERATE VIDEO FRAMES
# ==========================================

def generate_frames():

    camera_manager.start()

    try:

        while camera_manager.is_running():

            frame_bytes = camera_manager.get_frame()

            if frame_bytes is None:

                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    except GeneratorExit:

        print("Browser video connection closed.")

    finally:

        print("Video stream ended.")


# ==========================================
# 9. VIDEO FEED ROUTE
# ==========================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ==========================================
# 10. START CAMERA ROUTE
# ==========================================

@app.route("/start_camera", methods=["POST"])
def start_camera():

    started = camera_manager.start()

    return jsonify({
        "status": "starting" if started else "already_running",
        "camera": camera_manager.get_status()
    })


# ==========================================
# 11. STOP CAMERA ROUTE
# ==========================================

@app.route("/stop_camera", methods=["POST"])
def stop_camera():

    camera_manager.stop()

    return jsonify({
        "status": "stopped",
        "camera": camera_manager.get_status()
    })


# ==========================================
# 12. PREDICTION API
# ==========================================

@app.route("/predictions")
def predictions():

    return jsonify(get_prediction())


# ==========================================
# 13. CAMERA STATUS API
# ==========================================

@app.route("/camera_status")
def camera_status():

    return jsonify(camera_manager.get_status())


# ==========================================
# 14. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 15. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    print("Starting EmotionSense...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
