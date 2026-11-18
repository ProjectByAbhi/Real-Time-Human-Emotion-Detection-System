document.addEventListener("DOMContentLoaded", () => {
  // ======================================
  // DOM ELEMENTS
  // ======================================

  const videoFeed = document.getElementById("video-feed");
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
    videoFeed.removeAttribute("src");

    videoFeed.style.display = "none";
    placeholder.style.display = "flex";
  }

  function showCameraFeed() {
    videoFeed.style.display = "block";
    placeholder.style.display = "none";
  }

  // ======================================
  // UPDATE PREDICTIONS
  // ======================================

  async function updatePredictions() {
    if (!cameraRunning || predictionRequestRunning) return;

    predictionRequestRunning = true;

    const controller = new AbortController();
    predictionController = controller;

    try {
      const response = await fetch("/predictions", {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed: ${response.status}`);
      }

      const data = await response.json();

      // Ignore results received after stopping
      if (!cameraRunning) return;

      if (!data || typeof data !== "object") return;

      // Handle the situation when no face is detected
      if (data.emotion === "No face detected") {
        resetDashboard("Waiting for face...");
        return;
      }

      // Update detected emotion
      if (typeof data.emotion === "string" && emotions.includes(data.emotion)) {
        emotionElement.textContent = data.emotion;
      }

      // Update confidence
      const confidence = Number(data.confidence);

      if (Number.isFinite(confidence)) {
        const safeConfidence = Math.max(0, Math.min(100, confidence));
        confidenceElement.textContent = `${safeConfidence.toFixed(1)}% confidence`;
      }

      // Update emotion probabilities
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
      const response = await fetch("/stop_camera", {
        method: "POST",
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Camera stop failed: ${response.status}`);
      }

      console.log("EmotionSense: Camera stopped.");
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

  startButton.addEventListener("click", () => {
    if (cameraRunning || cameraStarting || cameraStopping) return;

    cameraStarting = true;
    cameraRunning = true;
    cameraFailureHandled = false;

    startButton.disabled = true;
    stopButton.disabled = false;

    updateCameraStatus("starting", "STARTING");
    resetDashboard("Waiting for face...");

    showCameraFeed();

    videoFeed.src = `/video_feed?t=${Date.now()}`;

    startPredictionUpdates();

    cameraStarting = false;

    updateCameraStatus("live", "LIVE");

    console.log("EmotionSense: Camera started.");
  });

  // ======================================
  // VIDEO STREAM ERROR
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
