document.addEventListener("DOMContentLoaded", () => {
  // ======================================
  // DOM ELEMENTS
  // ======================================

  const videoFeed = document.getElementById("video-feed");
  const faceOverlay = document.getElementById("face-overlay");
  const overlayContext = faceOverlay.getContext("2d");
  const startButton = document.getElementById("start-camera");
  const stopButton = document.getElementById("stop-camera");
  const placeholder = document.getElementById("camera-placeholder");

  const emotionElement = document.getElementById("detected-emotion");
  const confidenceElement = document.getElementById("emotion-confidence");

  const cameraStatus = document.getElementById("camera-status");
  const cameraStatusDot = document.getElementById("camera-status-dot");

  const panelStatus = document.getElementById("camera-panel-status");
  const panelStatusDot = document.getElementById("camera-panel-status-dot");

  if (
    !videoFeed ||
    !startButton ||
    !stopButton ||
    !placeholder ||
    !emotionElement ||
    !confidenceElement ||
    !cameraStatus ||
    !cameraStatusDot ||
    !panelStatus ||
    !panelStatusDot
  ) {
    console.error("EmotionSense: Required HTML elements are missing.");
    return;
  }

  // ======================================
  // APPLICATION STATE
  // ======================================

  let cameraRunning = false;
  let cameraStream = null;
  let frameCaptureTimer = null;
  let cameraStarting = false;
  let cameraStopping = false;
  let predictionTimer = null;
  let predictionController = null;
  let predictionRequestRunning = false;
  let cameraFailureHandled = false;

  const emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
  ];

  // ======================================
  // UPDATE CAMERA STATUS
  // ======================================

  function updateCameraStatus(status, message) {
    cameraStatus.textContent = message;
    panelStatus.textContent = message;

    cameraStatus.prepend(cameraStatusDot);
    panelStatus.prepend(panelStatusDot);

    cameraStatus.dataset.state = status;
    panelStatus.dataset.state = status;

    cameraStatusDot.dataset.state = status;
    panelStatusDot.dataset.state = status;
  }

  // ======================================
  // RESET EMOTION BARS
  // ======================================

  function resetEmotionBars() {
    emotions.forEach((emotion) => {
      const name = emotion.toLowerCase();
      const bar = document.getElementById(`bar-${name}`);
      const value = document.getElementById(`value-${name}`);

      if (bar) {
        bar.style.transition = "none";
        bar.style.width = "0%";
      }

      if (value) {
        value.textContent = "0.0%";
      }
    });
  }

  // ======================================
  // RESET DASHBOARD
  // ======================================

  function resetDashboard(message) {
    emotionElement.textContent = message;
    confidenceElement.textContent = "0.0% confidence";
    resetEmotionBars();
  }

  // ======================================
  // UPDATE EMOTION BARS
  // ======================================

  function updateEmotionBars(probabilities) {
    emotions.forEach((emotion) => {
      const name = emotion.toLowerCase();
      const bar = document.getElementById(`bar-${name}`);
      const value = document.getElementById(`value-${name}`);

      if (!bar || !value) return;

      const probability = Number(probabilities[emotion]);

      if (!Number.isFinite(probability)) return;

      const percentage = Math.max(0, Math.min(100, probability * 100));

      bar.style.transition = "width 0.45s ease";
      bar.style.width = `${percentage}%`;
      value.textContent = `${percentage.toFixed(1)}%`;
    });
  }

  // ======================================
  // STOP PREDICTION POLLING
  // ======================================

  function stopPredictionUpdates() {
    if (predictionTimer !== null) {
      clearTimeout(predictionTimer);
      predictionTimer = null;
    }

    if (frameCaptureTimer !== null) {
      clearInterval(frameCaptureTimer);
      frameCaptureTimer = null;
    }

    if (predictionController) {
      predictionController.abort();
      predictionController = null;
    }

    predictionRequestRunning = false;
  }

  // ======================================
  // CAMERA PLACEHOLDER AND VIDEO
  // ======================================

  function showCameraPlaceholder() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }

    videoFeed.srcObject = null;
    videoFeed.style.display = "none";
    faceOverlay.style.display = "none";
    overlayContext.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
    placeholder.style.display = "flex";
  }

  function drawFaceBox(faceBox) {
    overlayContext.clearRect(0, 0, faceOverlay.width, faceOverlay.height);

    if (!faceBox) return;

    const scaleX = faceOverlay.width / videoFeed.videoWidth;
    const scaleY = faceOverlay.height / videoFeed.videoHeight;

    const x = faceBox.x * scaleX;
    const y = faceBox.y * scaleY;
    const width = faceBox.width * scaleX;
    const height = faceBox.height * scaleY;

    // Draw green face rectangle
    overlayContext.strokeStyle = "#22c55e";
    overlayContext.lineWidth = 3;
    overlayContext.strokeRect(x, y, width, height);

    // Emotion label
    const emotion = faceBox.emotion || "Detecting...";
    const confidence = Number(faceBox.confidence) || 0;
    const label = `${emotion}: ${confidence.toFixed(1)}%`;

    // Draw label background
    overlayContext.font = "bold 18px Arial";
    const textWidth = overlayContext.measureText(label).width;

    overlayContext.fillStyle = "#22c55e";
    overlayContext.fillRect(x, Math.max(0, y - 30), textWidth + 16, 30);

    // Draw label text
    overlayContext.fillStyle = "#ffffff";
    overlayContext.fillText(label, x + 8, Math.max(20, y - 9));
  }

  // ======================================
  // UPDATE PREDICTIONS
  // ======================================

  async function updatePredictions() {
    if (!cameraRunning || predictionRequestRunning) return;
    if (!cameraStream || videoFeed.readyState < 2) return;

    predictionRequestRunning = true;

    const controller = new AbortController();
    predictionController = controller;

    try {
      const canvas = document.createElement("canvas");
      canvas.width = videoFeed.videoWidth;
      canvas.height = videoFeed.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(videoFeed, 0, 0);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.85);
      });

      if (!blob) {
        throw new Error("Unable to capture camera frame.");
      }

      const formData = new FormData();
      formData.append("frame", blob, "frame.jpg");

      const response = await fetch("/process_frame", {
        method: "POST",
        body: formData,
        cache: "no-store",
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Frame processing failed: ${response.status}`);
      }

      const data = await response.json();

      if (!cameraRunning) return;
      if (!data || typeof data !== "object") return;

      // Draw face detection rectangle
      faceOverlay.width = videoFeed.videoWidth;
      faceOverlay.height = videoFeed.videoHeight;

      drawFaceBox(data.face_box);

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.emotion === "No face detected") {
        resetDashboard("Waiting for face...");
        return;
      }

      if (typeof data.emotion === "string" && emotions.includes(data.emotion)) {
        emotionElement.textContent = data.emotion;
      }

      const confidence = Number(data.confidence);

      if (Number.isFinite(confidence)) {
        const safeConfidence = Math.max(0, Math.min(100, confidence));
        confidenceElement.textContent = `${safeConfidence.toFixed(1)}% confidence`;
      }

      if (data.probabilities && typeof data.probabilities === "object") {
        updateEmotionBars(data.probabilities);
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error("Emotion prediction error:", error);
      }
    } finally {
      if (predictionController === controller) {
        predictionController = null;
        predictionRequestRunning = false;
      }
    }
  }

  // ======================================
  // START PREDICTION POLLING
  // ======================================

  function startPredictionUpdates() {
    stopPredictionUpdates();

    async function pollPredictions() {
      if (!cameraRunning) return;

      await updatePredictions();

      if (cameraRunning) {
        predictionTimer = setTimeout(pollPredictions, 500);
      }
    }

    pollPredictions();
  }

  // ======================================
  // STOP CAMERA
  // ======================================

  async function stopCamera(message = "Camera stopped") {
    if (cameraStopping) return;

    cameraStopping = true;
    cameraRunning = false;
    cameraStarting = false;

    stopPredictionUpdates();

    startButton.disabled = true;
    stopButton.disabled = true;

    showCameraPlaceholder();
    resetDashboard(message);
    updateCameraStatus("offline", "OFFLINE");

    try {
      console.log("EmotionSense: Browser camera stopped.");
    } catch (error) {
      console.error("Camera stop error:", error);
    } finally {
      cameraStopping = false;
      startButton.disabled = false;
      stopButton.disabled = true;
    }
  }

  // ======================================
  // CAMERA FAILURE
  // ======================================

  async function handleCameraFailure() {
    if (!cameraRunning || cameraFailureHandled) return;

    cameraFailureHandled = true;

    console.error("EmotionSense: Camera stream disconnected.");

    await stopCamera("Camera disconnected");
    updateCameraStatus("error", "ERROR");
  }

  // ======================================
  // START CAMERA
  // ======================================

  startButton.addEventListener("click", async () => {
    if (cameraRunning || cameraStarting || cameraStopping) return;

    cameraStarting = true;
    cameraFailureHandled = false;

    startButton.disabled = true;
    stopButton.disabled = true;

    updateCameraStatus("starting", "STARTING");
    resetDashboard("Waiting for face...");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Browser webcam access is not supported.");
      }

      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
        },
        audio: false,
      });

      videoFeed.srcObject = cameraStream;
      videoFeed.style.display = "block";
      faceOverlay.style.display = "block";
      placeholder.style.display = "none";

      await videoFeed.play();

      cameraRunning = true;
      cameraStarting = false;

      stopButton.disabled = false;

      cameraStream
        .getVideoTracks()[0]
        .addEventListener("ended", handleCameraFailure);

      startPredictionUpdates();

      updateCameraStatus("live", "LIVE");

      console.log("EmotionSense: Browser camera started.");
    } catch (error) {
      console.error("Camera startup error:", error);

      cameraRunning = false;
      cameraStarting = false;

      showCameraPlaceholder();
      resetDashboard("Camera access failed");
      updateCameraStatus("error", "ERROR");

      startButton.disabled = false;
      stopButton.disabled = true;
    }
  });

  // ======================================
  // VIDEO ELEMENT ERROR
  // ======================================

  videoFeed.addEventListener("error", () => {
    handleCameraFailure();
  });

  // ======================================
  // STOP BUTTON
  // ======================================

  stopButton.addEventListener("click", async () => {
    if (!cameraRunning || cameraStopping) return;

    await stopCamera("Camera stopped");
  });

  // ======================================
  // INITIAL PAGE STATE
  // ======================================

  showCameraPlaceholder();

  startButton.disabled = false;
  stopButton.disabled = true;

  resetDashboard("Camera stopped");
  updateCameraStatus("offline", "OFFLINE");

  console.log("EmotionSense: Dashboard initialized.");
});
