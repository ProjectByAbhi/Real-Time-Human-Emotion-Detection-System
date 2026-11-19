import os
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EmotionSense | Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "emotion_model_v2_weights.npz")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")


# ==========================================
# 2. LOAD CNN MODEL AND FACE DETECTOR
# ==========================================

@st.cache_resource(show_spinner="Loading EmotionSense CNN model...")
def load_model_assets():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade file not found: {CASCADE_PATH}")

    # Keep the existing CNN architecture unchanged.
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

        Dense(7, activation="softmax"),
    ])

    with np.load(WEIGHTS_PATH, allow_pickle=False) as weights_data:
        weight_keys = sorted(
            weights_data.files,
            key=lambda key: int(key.split("_")[1]),
        )
        weights = [weights_data[key] for key in weight_keys]

    model.set_weights(weights)

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        raise RuntimeError("Could not load Haar Cascade detector.")

    model_lock = threading.Lock()

    print("Emotion model loaded successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Face detector loaded successfully!")

    return model, face_detector, model_lock


try:
    model, face_detector, model_lock = load_model_assets()
except Exception as error:
    st.error(f"Unable to load the emotion model: {error}")
    st.stop()


# ==========================================
# 3. RESPONSIVE APPLICATION STYLING
# ==========================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --page-bg: #0b1120;
        --header-bg: #111c30;
        --panel-bg: #172338;
        --panel-light: #1e2d44;
        --border: #2b3d56;
        --text: #f1f5f9;
        --muted: #9aabc0;
        --accent: #38bdf8;
        --accent-end: #818cf8;
        --green: #22c55e;
    }

    html, body, [class*="css"], button, input, textarea, select {
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--page-bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: var(--page-bg);
    }

    [data-testid="stMainBlockContainer"] {
        width: 100% !important;
        max-width: 1480px !important;
        padding: clamp(1rem, 2.6vw, 2.5rem)
                 clamp(1rem, 3.2vw, 3rem)
                 2.5rem !important;
    }

    /* Brand header */
    .brandbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        width: 100%;
        min-height: 72px;
        box-sizing: border-box;
        padding: 15px 22px;
        margin: 0 0 30px;
        background: var(--header-bg);
        border: 1px solid var(--border);
        border-radius: 15px;
    }

    .brand-name {
        display: flex;
        align-items: center;
        gap: 9px;
        color: var(--text);
        font-size: clamp(20px, 2vw, 27px);
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.7px;
        white-space: nowrap;
    }

    .brand-name .brand-accent {
        color: var(--accent);
    }

    .brand-icon {
        font-size: 1.05em;
        line-height: 1;
    }

    .live-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        flex-shrink: 0;
        padding: 8px 13px;
        border: 1px solid rgba(34, 197, 94, 0.18);
        border-radius: 999px;
        background: rgba(18, 60, 55, 0.9);
        color: var(--green);
        font-size: 11px;
        line-height: 1.2;
        font-weight: 700;
        letter-spacing: 0.35px;
        white-space: nowrap;
    }

    .live-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 9px rgba(34, 197, 94, 0.5);
    }

    /* Hero / page heading */
    .hero {
        margin: 0 0 30px;
    }

    .brand-category {
        margin-bottom: 10px;
        color: var(--accent);
        font-size: 10px;
        line-height: 1.4;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0 0 10px;
        color: var(--text);
        font-size: clamp(26px, 3vw, 39px);
        line-height: 1.16;
        font-weight: 800;
        letter-spacing: -1.15px;
    }

    .hero-subtitle {
        max-width: 720px;
        margin: 0;
        color: var(--muted);
        font-size: clamp(13px, 1.15vw, 15px);
        line-height: 1.65;
        font-weight: 400;
    }

    /* Section headings */
    .panel-heading {
        margin: 0 0 5px;
        color: var(--text);
        font-size: clamp(20px, 1.8vw, 25px);
        line-height: 1.25;
        font-weight: 700;
        letter-spacing: -0.45px;
    }

    .panel-subtitle {
        margin: 0 0 15px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    /* WebRTC camera component */
    div[data-testid="stCustomComponentV1"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
        padding: 7px;
        overflow: hidden;
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 15px;
    }

    div[data-testid="stCustomComponentV1"] iframe {
        width: 100% !important;
        max-width: 100% !important;
        border: 0 !important;
        border-radius: 10px;
    }

    /* Emotion result card */
    .analysis-card,
    .probability-card {
        box-sizing: border-box;
        width: 100%;
        padding: clamp(16px, 1.7vw, 22px);
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 15px;
    }

    .analysis-card {
        margin-bottom: 17px;
    }

    .result-label {
        color: var(--muted);
        font-size: 10px;
        line-height: 1.5;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .result-emotion {
        margin: 8px 0 4px;
        color: var(--accent);
        font-size: clamp(27px, 2.5vw, 35px);
        line-height: 1.18;
        font-weight: 800;
        letter-spacing: -0.7px;
        overflow-wrap: anywhere;
    }

    .result-confidence {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    /* Probability bars */
    .probability-card h3 {
        margin: 0 0 20px;
        color: var(--text);
        font-size: clamp(17px, 1.5vw, 20px);
        line-height: 1.3;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    .prob-row {
        margin-top: 16px;
    }

    .prob-row:first-of-type {
        margin-top: 0;
    }

    .prob-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 7px;
        color: var(--text);
        font-size: 12px;
        line-height: 1.4;
        font-weight: 500;
    }

    .prob-meta strong {
        flex-shrink: 0;
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }

    .prob-track {
        width: 100%;
        height: 8px;
        overflow: hidden;
        background: #2b3d56;
        border-radius: 999px;
    }

    .prob-fill {
        height: 100%;
        min-width: 0;
        background: linear-gradient(90deg, var(--accent), var(--accent-end));
        border-radius: 999px;
        transition: width 0.35s ease;
    }

    /* Camera status */
    .camera-status {
        display: flex;
        align-items: center;
        gap: 8px;
        box-sizing: border-box;
        width: 100%;
        margin-top: 14px;
        padding: 11px 14px;
        border: 1px solid rgba(34, 197, 94, 0.16);
        border-radius: 12px;
        background: rgba(18, 60, 55, 0.85);
        color: var(--green);
        font-size: 11px;
        line-height: 1.45;
        font-weight: 700;
    }

    .camera-status.is-idle {
        border-color: var(--border);
        background: var(--panel-bg);
        color: var(--muted);
    }

    .camera-status.is-error {
        border-color: rgba(248, 113, 113, 0.25);
        background: rgba(127, 29, 29, 0.22);
        color: #fca5a5;
    }

    /* Footer */
    .custom-footer {
        width: 100%;
        box-sizing: border-box;
        margin-top: 30px;
        padding: 17px 8px 5px;
        border-top: 1px solid var(--border);
        color: #8394aa;
        font-size: 11px;
        line-height: 1.6;
        text-align: center;
    }

    /* Streamlit alerts / separators */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: var(--border) !important;
    }

    /* Tablet and mobile layout */
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 100% !important;
            padding: 1.25rem 1.25rem 2rem !important;
        }

        .brandbar {
            margin-bottom: 24px;
        }

        .hero {
            margin-bottom: 24px;
        }
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            width: 100% !important;
            padding: 0.85rem 0.8rem 1.6rem !important;
        }

        /* Stack camera above analysis on narrow screens */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 1.15rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }

        .brandbar {
            min-height: 58px;
            gap: 8px;
            padding: 11px 13px;
            margin-bottom: 21px;
            border-radius: 12px;
        }

        .brand-name {
            gap: 6px;
            font-size: clamp(18px, 5vw, 22px);
            letter-spacing: -0.55px;
        }

        .brand-icon {
            font-size: 0.95em;
        }

        .live-pill {
            padding: 7px 9px;
            font-size: 9px;
            letter-spacing: 0.15px;
        }

        .live-dot {
            width: 6px;
            height: 6px;
        }

        .hero {
            margin-bottom: 22px;
        }

        .brand-category {
            margin-bottom: 8px;
            font-size: 9px;
            letter-spacing: 1.7px;
        }

        .hero-title {
            max-width: 100%;
            margin-bottom: 8px;
            font-size: clamp(25px, 7vw, 31px);
            line-height: 1.17;
            letter-spacing: -0.8px;
        }

        .hero-subtitle {
            font-size: 13px;
            line-height: 1.55;
        }

        .panel-heading {
            font-size: 21px;
        }

        .panel-subtitle {
            margin-bottom: 11px;
            font-size: 12px;
        }

        div[data-testid="stCustomComponentV1"] {
            padding: 5px;
            border-radius: 12px;
        }

        .analysis-card,
        .probability-card {
            padding: 16px;
            border-radius: 13px;
        }

        .analysis-card {
            margin-bottom: 13px;
        }

        .result-emotion {
            font-size: 29px;
        }

        .probability-card h3 {
            margin-bottom: 17px;
            font-size: 18px;
        }

        .prob-row {
            margin-top: 14px;
        }

        .prob-meta,
        .prob-meta strong {
            font-size: 11px;
        }

        .prob-track {
            height: 7px;
        }

        .camera-status {
            margin-top: 11px;
            padding: 10px 12px;
            font-size: 10px;
        }

        .custom-footer {
            margin-top: 23px;
            font-size: 10px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.markdown(
    """
    <div class="brandbar">
        <div class="brand-name">
            <span class="brand-icon">😊</span>
            <span>Emotion<span class="brand-accent">Sense</span></span>
        </div>
        <div class="live-pill">
            <span class="live-dot"></span>
            EMOTION AI
        </div>
    </div>

    <div class="hero">
        <div class="brand-category">COMPUTER VISION</div>
        <h1 class="hero-title">Real-Time Emotion Detection</h1>
        <p class="hero-subtitle">
            Facial expression recognition powered by Deep Learning and Computer Vision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 5. SHARED PREDICTION STATE
# ==========================================

prediction_lock = threading.Lock()
callback_lock = threading.Lock()
prediction_history = deque(maxlen=10)

shared_state = {
    "last_processed": 0.0,
    "emotion": "Waiting for face...",
    "confidence": 0.0,
    "probabilities": {emotion: 0.0 for emotion in EMOTIONS},
    "face_box": None,
    "error": None,
}


# ==========================================
# 6. DRAW FACE DETECTION RESULT
# ==========================================

def draw_face_result(image, face_box, emotion, confidence):
    if face_box is None:
        cv2.putText(
            image,
            "Waiting for face...",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )
        return image

    x = face_box["x"]
    y = face_box["y"]
    w = face_box["width"]
    h = face_box["height"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    label = f"{emotion}: {confidence:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    label_y = max(y - 10, text_height + 15)

    cv2.rectangle(
        image,
        (x, label_y - text_height - 12),
        (x + text_width + 12, label_y + baseline),
        (0, 180, 0),
        -1,
    )

    cv2.putText(
        image,
        label,
        (x + 6, label_y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    return image


# ==========================================
# 7. PROCESS WEBCAM FRAMES
# ==========================================

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    original_height, original_width = image.shape[:2]

    resize_scale = 1.0
    max_width = 320

    if original_width > max_width:
        resize_scale = original_width / max_width
        resized_height = int(original_height / resize_scale)

        detection_frame = cv2.resize(
            image,
            (max_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = image.copy()

    # Limit inference to approximately two predictions per second.
    with callback_lock:
        current_time = time.monotonic()

        if current_time - shared_state["last_processed"] >= 0.5:
            shared_state["last_processed"] = current_time
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

            try:
                with model_lock:
                    faces = face_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=3,
                        minSize=(30, 30),
                    )

                    if len(faces) > 0:
                        faces = sorted(
                            faces,
                            key=lambda face: face[2] * face[3],
                            reverse=True,
                        )[:1]

                if len(faces) == 0:
                    prediction_history.clear()
                    result = {
                        "emotion": "Waiting for face...",
                        "confidence": 0.0,
                        "probabilities": {name: 0.0 for name in EMOTIONS},
                        "face_box": None,
                        "error": None,
                    }
                else:
                    x, y, w, h = faces[0]
                    roi = gray[y:y + h, x:x + w]

                    face = cv2.resize(roi, (48, 48))
                    face = face.astype("float32") / 255.0
                    face = np.expand_dims(face, axis=(0, -1))

                    # Run CNN inference with the existing model.
                    with model_lock:
                        predictions = model(
                            face,
                            training=False,
                        ).numpy()[0]

                    prediction_history.append(predictions.copy())
                    smoothed_predictions = np.mean(
                        np.stack(list(prediction_history), axis=0),
                        axis=0,
                    )

                    index = int(np.argmax(smoothed_predictions))
                    emotion = EMOTIONS[index]
                    confidence = float(smoothed_predictions[index] * 100)

                    probabilities = {
                        name: float(smoothed_predictions[i])
                        for i, name in enumerate(EMOTIONS)
                    }

                    face_box = {
                        "x": int(x * resize_scale),
                        "y": int(y * resize_scale),
                        "width": int(w * resize_scale),
                        "height": int(h * resize_scale),
                    }

                    result = {
                        "emotion": emotion,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "face_box": face_box,
                        "error": None,
                    }

            except Exception as error:
                print("Frame processing error:", error)
                result = {
                    "emotion": "Processing error",
                    "confidence": 0.0,
                    "probabilities": {name: 0.0 for name in EMOTIONS},
                    "face_box": None,
                    "error": str(error),
                }

            with prediction_lock:
                shared_state["emotion"] = result["emotion"]
                shared_state["confidence"] = result["confidence"]
                shared_state["probabilities"] = result["probabilities"]
                shared_state["face_box"] = result["face_box"]
                shared_state["error"] = result["error"]

    with prediction_lock:
        emotion = shared_state["emotion"]
        confidence = shared_state["confidence"]
        face_box = shared_state["face_box"]

    image = draw_face_result(image, face_box, emotion, confidence)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


# ==========================================
# 8. DASHBOARD LAYOUT
# ==========================================

camera_column, analysis_column = st.columns([1.25, 1], gap="large")

with camera_column:
    st.markdown(
        '<div class="panel-heading">Live Camera</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-subtitle">Real-time facial expression analysis</div>',
        unsafe_allow_html=True,
    )

    webrtc_context = webrtc_streamer(
        key="emotionsense-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
    )

with analysis_column:
    st.markdown(
        '<div class="panel-heading">Emotion Analysis</div>',
        unsafe_allow_html=True,
    )
    emotion_display = st.empty()
    probability_display = st.empty()
    status_display = st.empty()


# ==========================================
# 9. RENDER EMOTION ANALYSIS
# ==========================================

def render_emotion_analysis(emotion, confidence, probabilities):
    emotion_html = (
        '<div class="analysis-card">'
        '<div class="result-label">CURRENT EXPRESSION</div>'
        f'<div class="result-emotion">{emotion}</div>'
        f'<div class="result-confidence">{confidence:.1f}% confidence</div>'
        '</div>'
    )

    emotion_display.markdown(
        emotion_html,
        unsafe_allow_html=True,
    )

    probability_html = [
        '<div class="probability-card">',
        '<h3>Emotion Probabilities</h3>',
    ]

    for name in EMOTIONS:
        value = float(probabilities.get(name, 0.0))
        if not np.isfinite(value):
            value = 0.0

        percentage = max(0.0, min(100.0, value * 100))

        probability_html.append(
            '<div class="prob-row">'
            '<div class="prob-meta">'
            f'<span>{name}</span>'
            f'<strong>{percentage:.1f}%</strong>'
            '</div>'
            '<div class="prob-track">'
            f'<div class="prob-fill" style="width:{percentage:.1f}%"></div>'
            '</div>'
            '</div>'
        )

    probability_html.append('</div>')

    probability_display.markdown(
        "".join(probability_html),
        unsafe_allow_html=True,
    )


# ==========================================
# 10. LIVE DASHBOARD UPDATE
# ==========================================

if webrtc_context.state.playing:
    status_display.markdown(
        '<div class="camera-status">'
        '<span class="live-dot"></span>'
        'CAMERA CONNECTED · LIVE ANALYSIS'
        '</div>',
        unsafe_allow_html=True,
    )

    while webrtc_context.state.playing:
        with prediction_lock:
            current_emotion = shared_state["emotion"]
            current_confidence = shared_state["confidence"]
            current_probabilities = shared_state["probabilities"].copy()
            current_error = shared_state["error"]

        render_emotion_analysis(
            current_emotion,
            current_confidence,
            current_probabilities,
        )

        if current_error:
            status_display.error(f"Frame processing error: {current_error}")

        time.sleep(0.25)
else:
    status_display.markdown(
        '<div class="camera-status is-idle">'
        'Start the camera and allow browser permission.'
        '</div>',
        unsafe_allow_html=True,
    )

    render_emotion_analysis(
        "Camera stopped",
        0.0,
        {name: 0.0 for name in EMOTIONS},
    )


# ==========================================
# 11. FOOTER
# ==========================================

st.markdown(
    '<div class="custom-footer">'
    'EmotionSense · Real-Time Human Emotion Detection · Powered by Abhishek Rai'
    '</div>',
    unsafe_allow_html=True,
)
