# Real-Time Human Emotion Detection System

### EmotionSense – Real-Time Facial Emotion Recognition Using CNN

A Deep Learning and Computer Vision-based application that detects human faces and predicts facial expressions in real time using a Convolutional Neural Network (CNN).

## 🚀 Live Demo

[**EmotionSense – Live Demo ↗**](https://real-time-human-emotion-detection-system-sumdwycybq5onyrcv7d2i.streamlit.app/)

## 💻 GitHub Repository

[**Real-Time Human Emotion Detection System ↗**](https://github.com/ProjectByAbhi/Real-Time-Human-Emotion-Detection-System)


## 📌 About the Project

EmotionSense is a Real-Time Human Emotion Detection System developed using Python, TensorFlow, Keras, and OpenCV.

The system uses a trained Convolutional Neural Network (CNN) to recognize seven facial expressions from live webcam input.

The original application is developed using **Python and Flask**. An additional **Streamlit interface** is provided for an interactive live demonstration using the same trained model and facial emotion detection pipeline.

The Streamlit interface is an additional demonstration interface and does not replace the original Flask application.

## ✨ Features

- Real-time face detection using OpenCV Haar Cascade.
- Facial emotion classification using a trained CNN model.
- Detection of seven different facial expressions.
- Live webcam input through the browser.
- Face detection bounding box with emotion label.
- Confidence score for the predicted emotion.
- Probability distribution for all seven emotions.
- Separate Flask and Streamlit application interfaces.

## 😊 Emotions Detected

The model recognizes the following seven emotions:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## 🛠️ Technologies Used

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Main programming language           |
| TensorFlow            | Deep Learning framework             |
| Keras                 | CNN model development               |
| OpenCV                | Face detection and image processing |
| NumPy                 | Numerical operations                |
| Flask                 | Original web application            |
| Streamlit             | Additional live demo interface      |
| streamlit-webrtc      | Browser webcam integration          |
| HTML, CSS, JavaScript | Flask web interface                 |

## 🧠 Model Information

The project uses a Convolutional Neural Network (CNN) for facial emotion classification.

| Parameter           | Details                      |
| ------------------- | ---------------------------- |
| Model Type          | Convolutional Neural Network |
| Input Image Size    | 48 × 48 pixels               |
| Image Format        | Grayscale                    |
| Output Classes      | 7 Emotions                   |
| Activation Function | Softmax                      |
| Framework           | TensorFlow / Keras           |

The trained model weights are stored in `emotion_model_v2_weights.npz`.

## ⚙️ How It Works

1. The application receives video frames from the webcam.
2. OpenCV Haar Cascade detects the face.
3. The detected face is converted into grayscale.
4. The face image is resized to 48 × 48 pixels.
5. The processed image is passed to the trained CNN model.
6. The model predicts probabilities for seven emotions.
7. The predicted emotion and confidence score are displayed on the dashboard.

## 🏗️ Project Architecture

This repository contains two application interfaces.

### 1. Original Flask Application

The main implementation of the project, developed using Flask, Python, TensorFlow, and OpenCV.

It processes webcam frames through the Flask backend and displays facial emotion predictions on its web interface.

**Run command:**

```bash
python app.py
```

### 2. Additional Streamlit Demo

A separate demonstration interface developed using Streamlit and Streamlit WebRTC.

It uses the same CNN architecture, trained model weights, Haar Cascade classifier, and emotion classification approach as the original project.

**Run command:**

```bash
streamlit run streamlit_app.py
```

Both interfaces are part of the same project. Streamlit does not replace the original Flask implementation.

## 📂 Project Structure

```text
Real-Time-Human-Emotion-Detection-System/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .python-version
│
├── emotion_model_v2_weights.npz
├── haarcascade_frontalface_default.xml
│
├── templates/
├── static/
│
├── screenshots/
│   └── homepage.png
│
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ProjectByAbhi/Real-Time-Human-Emotion-Detection-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Real-Time-Human-Emotion-Detection-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Choose either of the following interfaces.

### Option 1: Run the Original Flask Application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

### Option 2: Run the Streamlit Demo

```bash
streamlit run streamlit_app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

Allow camera access when prompted by your browser.

## 🌐 Deployment

The additional Streamlit demonstration interface is deployed using **Streamlit Community Cloud**.

**Live Application:** [EmotionSense – Live Demo ↗](https://real-time-human-emotion-detection-system-sumdwycybq5onyrcv7d2i.streamlit.app/)

The original Flask application remains available in the repository as the main implementation and can be run locally using `python app.py`.

## 🔮 Future Improvements

- Improve emotion classification accuracy.
- Optimize model inference speed.
- Improve performance under different lighting conditions.
- Explore advanced CNN architectures.
- Enhance the user interface and visualization.

## 👨‍💻 Author

**Abhishek Rai**

MCA Student | Artificial Intelligence & Machine Learning

- GitHub: [ProjectByAbhi ↗](https://github.com/ProjectByAbhi)
- Project Repository: [Emotion Detection System ↗](https://github.com/ProjectByAbhi/Real-Time-Human-Emotion-Detection-System)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Thank you for visiting EmotionSense!**
