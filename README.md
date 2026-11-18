# Real-Time Human Emotion Detection System

**EmotionSense** is a real-time facial emotion detection system that uses Deep Learning and Computer Vision to recognize human emotions through a webcam.

The system detects faces using OpenCV and predicts facial emotions using a Convolutional Neural Network (CNN). It displays the detected emotion and confidence scores through an interactive web dashboard.

## Features

- Real-time face detection using OpenCV.
- Facial emotion classification using a CNN model.
- Webcam-based emotion detection.
- Supports 7 emotion categories.
- Displays emotion confidence and probability scores.
- Interactive web dashboard.
- Start and stop camera functionality.
- Responsive user interface.

## Emotions Detected

The system recognizes seven facial emotions:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Technologies Used

| Technology         | Purpose                             |
| ------------------ | ----------------------------------- |
| Python             | Backend development                 |
| TensorFlow / Keras | Deep Learning model                 |
| OpenCV             | Face detection and image processing |
| Flask              | Web application                     |
| HTML               | Webpage structure                   |
| CSS                | Dashboard styling                   |
| JavaScript         | Real-time frontend interaction      |
| NumPy              | Numerical operations                |

## Project Structure

```text
Emotion_Detection_CNN/
│
├── app.py
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

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries

```bash
pip install flask tensorflow==2.15.0 opencv-python==4.10.0.84 numpy
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

Visit:

```text
http://127.0.0.1:5000
```

Allow camera access and click **Start Camera** to begin real-time emotion detection.

## Model Information

- Model Type: Convolutional Neural Network (CNN)
- Input Image Size: 48 × 48 pixels
- Image Format: Grayscale
- Number of Classes: 7
- Model Weights: `emotion_model_v2_weights.npz`

## How It Works

1. The webcam captures live video.
2. OpenCV detects faces using the Haar Cascade classifier.
3. The detected face is converted into grayscale.
4. The face image is resized to 48 × 48 pixels.
5. The CNN model predicts the emotion.
6. The Flask backend provides prediction results to the frontend.
7. The dashboard displays the detected emotion and probability scores.

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
