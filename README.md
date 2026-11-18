# Real-Time Human Emotion Detection System

**EmotionSense** is a real-time facial emotion detection system that uses Deep Learning and Computer Vision to recognize human emotions through a browser webcam.

The application captures frames from the user's browser camera, sends them to a Flask backend for face detection and emotion prediction, and displays the results on an interactive dashboard.

## Features

- Real-time face detection using OpenCV and a Haar Cascade classifier.
- Facial emotion classification using a Convolutional Neural Network (CNN).
- Browser-based webcam access.
- Supports 7 emotion categories.
- Displays detected emotion, confidence, and probability scores.
- Draws a face detection box with the predicted emotion label.
- Start and stop camera functionality.
- Responsive web dashboard.

## Emotions Detected

The system recognizes seven facial emotion categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Technologies Used

| Technology         | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Python             | Backend development                      |
| TensorFlow / Keras | Deep Learning model                      |
| OpenCV             | Face detection and image processing      |
| Flask              | Web application and frame-processing API |
| HTML               | Webpage structure                        |
| CSS                | Dashboard styling                        |
| JavaScript         | Browser camera and frontend interaction  |
| NumPy              | Numerical operations                     |

## Project Structure

```text
Real-Time-Human-Emotion-Detection-System/
│
├── app.py
├── requirements.txt
├── emotion_model_v2_weights.npz
├── haarcascade_frontalface_default.xml
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ProjectByAbhi/Real-Time-Human-Emotion-Detection-System.git
```

### 2. Open the Project Folder

```bash
cd Real-Time-Human-Emotion-Detection-System
```

### 3. Create and Activate a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

Visit:

```text
http://127.0.0.1:5000
```

Allow camera access when your browser prompts you, then click **Start Camera** to begin emotion detection.

> Camera access requires a secure context: localhost during local development, or HTTPS when deployed.

## Model Information

- Model Type: Convolutional Neural Network (CNN)
- Input Image Size: 48 × 48 pixels
- Image Format: Grayscale
- Number of Classes: 7
- Model Weights: `emotion_model_v2_weights.npz`

## How It Works

1. JavaScript requests access to the webcam in the user's browser.
2. The browser captures video frames and sends them to the Flask `/process_frame` endpoint.
3. OpenCV detects the face using the Haar Cascade classifier.
4. The detected face is converted to grayscale and resized to 48 × 48 pixels.
5. The CNN model predicts probabilities for the seven emotion categories.
6. The backend smooths recent predictions and returns the emotion, confidence, probabilities, and face coordinates.
7. The dashboard displays the prediction and draws the face detection box with its emotion label.

## Deployment

The application is intended for deployment on Render using Gunicorn.

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

The public deployment link will be added here after deployment is completed.

## Future Improvements

- Improve emotion classification accuracy.
- Enhance detection under different lighting conditions.
- Explore more advanced face detection techniques.
- Improve prediction speed and performance.

## Author

**Abhishek Rai**

GitHub: [ProjectByAbhi](https://github.com/ProjectByAbhi)

---

_EmotionSense – Real-Time Human Emotion Detection System_
