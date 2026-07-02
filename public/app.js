import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const APP_VERSION = "2026.07.02.1";
const MODEL_PATH = "/models/hand_landmarker.task";
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

const PAUSE_MS = 1500;
const MAX_SAMPLE_MS = 20000;
const MIN_POINTS = 10;
const ERASE_RADIUS = 32;
const BRUSH_SIZE = 9;
const RECORD_FPS = 24;
const RECORD_BITS_PER_SECOND = 500000;
const RECORDER_STOP_TIMEOUT_MS = 1500;
const UPLOAD_TIMEOUT_MS = 60000;
const CANVAS_BLOB_TIMEOUT_MS = 2500;
const FILE_READER_TIMEOUT_MS = 10000;
const CLEAR_COOLDOWN_MS = 1400;
const BANGLA_VOWELS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ"];

const TIP_IDS = [4, 8, 12, 16, 20];
const PIP_IDS = [3, 6, 10, 14, 18];
const INDEX_TIP = 8;
const HAND_CONN = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const els = {
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  stage: document.getElementById("stage"),
  blocked: document.getElementById("blocked"),
  blockedDetail: document.getElementById("blockedDetail"),
  cameraStatus: document.getElementById("cameraStatus"),
  uploadStatus: document.getElementById("uploadStatus"),
  versionStatus: document.getElementById("versionStatus"),
  pauseBar: document.getElementById("pauseBar"),
  saveFlash: document.getElementById("saveFlash"),
  username: document.getElementById("username"),
  label: document.getElementById("label"),
  uploadKey: document.getElementById("uploadKey"),
  saveBtn: document.getElementById("saveBtn"),
  clearBtn: document.getElementById("clearBtn"),
  installBtn: document.getElementById("installBtn"),
  modeReadout: document.getElementById("modeReadout"),
  pointsReadout: document.getElementById("pointsReadout"),
  savedReadout: document.getElementById("savedReadout"),
};

const overlayCtx = els.overlay.getContext("2d");
const inkCanvas = document.createElement("canvas");
const inkCtx = inkCanvas.getContext("2d");
const recordCanvas = document.createElement("canvas");
const recordCtx = recordCanvas.getContext("2d");

const state = {
  landmarker: null,
  cameraStream: null,
  recorder: null,
  recorderMime: "",
  recorderExt: "webm",
  recorderSession: 0,
  recordedChunks: [],
  running: false,
  saving: false,
  clearing: false,
  clearUntil: 0,
  operationId: 0,
  uploadController: null,
  sampleStarted: false,
  sampleStartedAt: 0,
  lastPenDownAt: 0,
  prevPoint: null,
  mode: "Pen up",
  savedCount: 0,
  hasInk: false,
  autoSaveBlocked: false,
  feature: null,
  deferredInstall: null,
};

class FeatureExtractor {
  constructor() {
    this.reset();
  }

  reset() {
    this.points = [];
    this.strokes = [];
    this.strokeId = 0;
    this.wasDown = false;
  }

  update(x, y, fw, fh, down) {
    if (down) {
      if (!this.wasDown) {
        this.strokeId += 1;
      }
      this.points.push([x / fw, y / fh]);
      this.strokes.push(this.strokeId);
    }
    this.wasDown = down;
  }

  hasData() {
    return this.points.length >= MIN_POINTS;
  }

  pointCount() {
    return this.points.length;
  }

  extract() {
    const vecs = [];
    for (let i = 0; i < this.points.length; i += 1) {
      const [p, q] = this.points[i];
      let dp = 0;
      let dq = 0;
      let sinA = 0;
      let cosA = 1;
      let sinB = 0;
      let cosB = 1;

      if (i > 0) {
        const [pp, pq] = this.points[i - 1];
        dp = p - pp;
        dq = q - pq;
        const d = Math.hypot(dp, dq) + 1e-9;
        cosA = dp / d;
        sinA = dq / d;

        if (i >= 2) {
          const [p2, q2] = this.points[i - 2];
          const dp2 = pp - p2;
          const dq2 = pq - q2;
          const d2 = Math.hypot(dp2, dq2) + 1e-9;
          const ca2 = dp2 / d2;
          const sa2 = dq2 / d2;
          sinB = sinA * ca2 - cosA * sa2;
          cosB = cosA * ca2 + sinA * sa2;
        }
      }

      const same = i < this.points.length - 1 && this.strokes[i] === this.strokes[i + 1] ? 1 : 0;
      vecs.push([dp, dq, sinA, cosA, sinB, cosB, same, 1 - same].map(round6));
    }

    return {
      points_normalized: this.points.map(([x, y]) => [round6(x), round6(y)]),
      stroke_ids: [...this.strokes],
      num_strokes: this.strokeId,
      num_points: this.points.length,
      features_8d: vecs,
      feature_names: [
        "delta_p",
        "delta_q",
        "sin_alpha",
        "cos_alpha",
        "sin_beta",
        "cos_beta",
        "same_stroke",
        "diff_stroke",
      ],
    };
  }
}

function round6(value) {
  return Math.round(value * 1000000) / 1000000;
}

function setPill(el, text, kind = "neutral") {
  el.textContent = text;
  el.className = `pill ${kind}`;
}

function setUpload(text, kind = "neutral") {
  setPill(els.uploadStatus, text, kind);
}

function setCamera(text, kind = "neutral") {
  setPill(els.cameraStatus, text, kind);
}

function setVersion() {
  if (els.versionStatus) {
    setPill(els.versionStatus, `v${APP_VERSION}`, "neutral");
  }
}

function loadSettings() {
  els.username.value = localStorage.getItem("airwriting.username") || els.username.value;
  const savedLabel = localStorage.getItem("airwriting.label");
  els.label.value = BANGLA_VOWELS.includes(savedLabel) ? savedLabel : BANGLA_VOWELS[0];
  els.uploadKey.value = localStorage.getItem("airwriting.uploadKey") || "";
}

function persistSettings() {
  localStorage.setItem("airwriting.username", els.username.value.trim() || "user1");
  localStorage.setItem("airwriting.label", normalizedLabel());
  localStorage.setItem("airwriting.uploadKey", els.uploadKey.value);
}

function normalizedLabel() {
  return BANGLA_VOWELS.includes(els.label.value) ? els.label.value : BANGLA_VOWELS[0];
}

function assertCurrentOperation(operationId) {
  if (operationId !== state.operationId) {
    throw new Error("Operation cancelled");
  }
}

function resizeCanvas(canvas, w, h, preserve = false) {
  if (canvas.width === w && canvas.height === h) {
    return;
  }
  if (preserve && canvas.width && canvas.height) {
    const old = document.createElement("canvas");
    old.width = canvas.width;
    old.height = canvas.height;
    old.getContext("2d").drawImage(canvas, 0, 0);
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(old, 0, 0, w, h);
  } else {
    canvas.width = w;
    canvas.height = h;
  }
}

function resizeSurfaces() {
  const w = els.video.videoWidth || 640;
  const h = els.video.videoHeight || 480;
  els.stage.style.aspectRatio = `${w} / ${h}`;
  resizeCanvas(els.overlay, w, h);
  resizeCanvas(inkCanvas, w, h, state.hasInk);
  resizeCanvas(recordCanvas, w, h);
}

async function initLandmarker() {
  setCamera("Loading model");
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  try {
    state.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_PATH,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  } catch (error) {
    state.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  }
}

async function initCamera() {
  setCamera("Opening camera");
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not expose camera access.");
  }
  state.cameraStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 24, max: 30 },
    },
    audio: false,
  });
  els.video.srcObject = state.cameraStream;
  await els.video.play();
  resizeSurfaces();
  setCamera("Camera on", "good");
}

function fingersUp(lms, isRight) {
  const up = [];
  up.push(isRight ? lms[TIP_IDS[0]].x < lms[PIP_IDS[0]].x : lms[TIP_IDS[0]].x > lms[PIP_IDS[0]].x);
  for (let i = 1; i < TIP_IDS.length; i += 1) {
    up.push(lms[TIP_IDS[i]].y < lms[PIP_IDS[i]].y);
  }
  return up;
}

function classifyGesture(lms, isRight) {
  const up = fingersUp(lms, isRight);
  if (up[1] && !up.slice(2).some(Boolean)) {
    return "Draw";
  }
  if (up.slice(1).every(Boolean)) {
    return "Erase";
  }
  return "Pen up";
}

function getHandedness(result) {
  const handed = result.handednesses || result.handedness || [];
  const first = handed[0]?.[0];
  const name = first?.categoryName || first?.displayName || "Right";
  return name === "Right";
}

function displayPoint(lm) {
  return {
    x: Math.max(0, Math.min(els.overlay.width, (1 - lm.x) * els.overlay.width)),
    y: Math.max(0, Math.min(els.overlay.height, lm.y * els.overlay.height)),
  };
}

function beginSample(now) {
  if (state.sampleStarted) {
    return;
  }
  state.sampleStarted = true;
  state.sampleStartedAt = now;
  state.lastPenDownAt = now;
  state.recordedChunks = [];
  startRecorder();
}

function pickMime() {
  if (!window.MediaRecorder) {
    return "";
  }
  const candidates = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return candidates.find((mime) => MediaRecorder.isTypeSupported(mime)) || "";
}

function startRecorder() {
  if (!window.MediaRecorder) {
    return;
  }
  const recorderSession = state.recorderSession + 1;
  const mimeType = pickMime();
  const stream = recordCanvas.captureStream
    ? recordCanvas.captureStream(RECORD_FPS)
    : state.cameraStream;
  const options = {
    videoBitsPerSecond: RECORD_BITS_PER_SECOND,
  };
  if (mimeType) {
    options.mimeType = mimeType;
  }

  try {
    state.recorderSession = recorderSession;
    state.recorderMime = mimeType || "video/webm";
    state.recorderExt = state.recorderMime.includes("mp4") ? "mp4" : "webm";
    state.recorder = new MediaRecorder(stream, options);
    state.recorder.ondataavailable = (event) => {
      if (state.recorderSession !== recorderSession) {
        return;
      }
      if (event.data?.size) {
        state.recordedChunks.push(event.data);
      }
    };
    state.recorder.start(250);
  } catch (error) {
    state.recorder = null;
  }
}

function currentVideoBlob() {
  if (!state.recordedChunks.length) {
    return null;
  }
  return new Blob(state.recordedChunks, { type: state.recorderMime || "video/webm" });
}

function stopRecorder({ discard = false } = {}) {
  if (!state.recorder || state.recorder.state === "inactive") {
    state.recorder = null;
    if (discard) {
      state.recorderSession += 1;
      state.recordedChunks = [];
      return Promise.resolve(null);
    }
    return Promise.resolve(currentVideoBlob());
  }
  const recorder = state.recorder;
  const recorderSession = state.recorderSession;
  if (discard) {
    state.recorderSession += 1;
    state.recordedChunks = [];
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeoutId);
      if (state.recorder === recorder) {
        state.recorder = null;
      }
      if (discard || state.recorderSession !== recorderSession) {
        resolve(null);
        return;
      }
      resolve(currentVideoBlob());
    };
    const timeoutId = window.setTimeout(finish, RECORDER_STOP_TIMEOUT_MS);
    recorder.addEventListener("stop", finish, { once: true });
    recorder.addEventListener("error", finish, { once: true });
    try {
      if (!discard && typeof recorder.requestData === "function") {
        recorder.requestData();
      }
      recorder.stop();
    } catch (error) {
      finish();
    }
  });
}

function updateFromDetection(result, now) {
  if (state.saving || state.clearing) {
    return null;
  }
  if (now < state.clearUntil) {
    state.prevPoint = null;
    state.mode = "Pen up";
    els.modeReadout.textContent = "Pen up";
    return null;
  }

  const landmarks = result?.landmarks?.[0] || null;
  let mode = "Pen up";
  let penDown = false;

  if (landmarks) {
    mode = classifyGesture(landmarks, getHandedness(result));
    const index = displayPoint(landmarks[INDEX_TIP]);

    if (mode === "Draw") {
      penDown = true;
      beginSample(now);
      inkCtx.save();
      inkCtx.strokeStyle = "#ffffff";
      inkCtx.fillStyle = "#ffffff";
      inkCtx.lineWidth = BRUSH_SIZE * 2;
      inkCtx.lineCap = "round";
      inkCtx.lineJoin = "round";
      if (state.prevPoint) {
        inkCtx.beginPath();
        inkCtx.moveTo(state.prevPoint.x, state.prevPoint.y);
        inkCtx.lineTo(index.x, index.y);
        inkCtx.stroke();
      }
      inkCtx.beginPath();
      inkCtx.arc(index.x, index.y, BRUSH_SIZE, 0, Math.PI * 2);
      inkCtx.fill();
      inkCtx.restore();
      state.prevPoint = index;
      state.lastPenDownAt = now;
      state.hasInk = true;
      state.autoSaveBlocked = false;
      state.feature.update(index.x, index.y, els.overlay.width, els.overlay.height, true);
    } else {
      state.prevPoint = null;
      state.feature.update(0, 0, els.overlay.width, els.overlay.height, false);
      if (mode === "Erase") {
        const wrist = displayPoint(landmarks[0]);
        inkCtx.save();
        inkCtx.globalCompositeOperation = "destination-out";
        inkCtx.beginPath();
        inkCtx.arc(wrist.x, wrist.y, ERASE_RADIUS, 0, Math.PI * 2);
        inkCtx.fill();
        inkCtx.restore();
      }
    }
  } else {
    state.prevPoint = null;
    state.feature.update(0, 0, els.overlay.width, els.overlay.height, false);
  }

  state.mode = mode;
  els.modeReadout.textContent = mode;
  els.pointsReadout.textContent = String(state.feature.pointCount());

  const canAutoSave = state.sampleStarted && state.feature.hasData() && !state.autoSaveBlocked;
  const elapsed = now - state.sampleStartedAt;
  const pauseProgress = canAutoSave && !penDown
    ? Math.min((now - state.lastPenDownAt) / PAUSE_MS, 1)
    : 0;

  els.pauseBar.style.width = `${pauseProgress * 100}%`;
  els.pauseBar.style.backgroundColor = pauseProgress >= 1 ? "var(--cyan)" : "var(--green)";

  if (canAutoSave && (pauseProgress >= 1 || elapsed >= MAX_SAMPLE_MS)) {
    saveSample(true);
  }

  return landmarks;
}

function drawSkeleton(ctx, landmarks, color = "#38d879") {
  if (!landmarks) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONN) {
    const pa = displayPoint(landmarks[a]);
    const pb = displayPoint(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (const lm of landmarks) {
    const p = displayPoint(lm);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawVisible(landmarks) {
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  overlayCtx.drawImage(inkCanvas, 0, 0);
  drawSkeleton(overlayCtx, landmarks);
}

function drawRecordFrame(landmarks) {
  const w = recordCanvas.width;
  const h = recordCanvas.height;
  recordCtx.clearRect(0, 0, w, h);
  recordCtx.save();
  recordCtx.translate(w, 0);
  recordCtx.scale(-1, 1);
  recordCtx.drawImage(els.video, 0, 0, w, h);
  recordCtx.restore();
  recordCtx.drawImage(inkCanvas, 0, 0);
  drawSkeleton(recordCtx, landmarks, "#46c4d9");
}

function tick(now) {
  if (!state.running) {
    return;
  }

  resizeSurfaces();
  let result = null;
  if (state.landmarker && els.video.readyState >= 2) {
    result = state.landmarker.detectForVideo(els.video, now);
  }
  const landmarks = updateFromDetection(result, now);
  drawVisible(landmarks);
  drawRecordFrame(landmarks);
  requestAnimationFrame(tick);
}

function canvasToBlob(canvas, type = "image/png") {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (blob) => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeoutId);
      resolve(blob || null);
    };
    const timeoutId = window.setTimeout(() => finish(null), CANVAS_BLOB_TIMEOUT_MS);
    canvas.toBlob(finish, type);
  });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    if (!blob) {
      resolve(null);
      return;
    }
    let settled = false;
    const reader = new FileReader();
    const finish = (callback) => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeoutId);
      callback();
    };
    const timeoutId = window.setTimeout(() => {
      reader.abort();
      finish(() => reject(new Error("Media preparation timed out")));
    }, FILE_READER_TIMEOUT_MS);
    reader.onload = () => finish(() => resolve(reader.result));
    reader.onerror = () => finish(() => reject(reader.error || new Error("Media preparation failed")));
    reader.onabort = () => finish(() => reject(new Error("Media preparation cancelled")));
    reader.readAsDataURL(blob);
  });
}

function makeSampleId(date = new Date()) {
  const pad = (value) => String(value).padStart(2, "0");
  const stamp = [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
  ].join("") + "_" + [pad(date.getHours()), pad(date.getMinutes()), pad(date.getSeconds())].join("");
  return `${stamp}_${Math.random().toString(36).slice(2, 8)}`;
}

async function uploadSample({ imageBlob, videoBlob, metadata, operationId }) {
  setUpload("Preparing image", "warn");
  const imageDataUrl = await blobToDataUrl(imageBlob);
  assertCurrentOperation(operationId);
  let videoDataUrl = null;
  if (videoBlob) {
    setUpload("Preparing video", "warn");
    try {
      videoDataUrl = await blobToDataUrl(videoBlob);
    } catch (error) {
      console.warn("Video skipped", error);
      videoBlob = null;
    }
    assertCurrentOperation(operationId);
  }
  const uploadKey = els.uploadKey.value;
  const payload = {
    username: metadata.username,
    label: metadata.label,
    sampleId: metadata.sample_id,
    createdAt: metadata.created_at,
    image: imageDataUrl ? {
      dataUrl: imageDataUrl,
      mime: "image/png",
      extension: "png",
    } : null,
    video: videoDataUrl ? {
      dataUrl: videoDataUrl,
      mime: videoBlob.type || state.recorderMime || "video/webm",
      extension: state.recorderExt,
    } : null,
    metadata,
  };

  const headers = {
    "Content-Type": "application/json",
  };
  if (uploadKey) {
    headers["x-upload-secret"] = uploadKey;
  }

  assertCurrentOperation(operationId);
  setUpload("Uploading", "warn");
  const controller = new AbortController();
  state.uploadController = controller;
  const timeoutId = window.setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);
  let response;
  try {
    response = await fetch("/api/upload", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Upload cancelled or timed out. Check Vercel logs and GitHub token settings.");
    }
    throw error;
  } finally {
    if (state.uploadController === controller) {
      state.uploadController = null;
    }
    window.clearTimeout(timeoutId);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `Upload failed (${response.status})`);
  }
  return data;
}

async function saveSample(auto = false) {
  if (state.saving || state.clearing) {
    return;
  }
  if (!state.feature) {
    setUpload("Camera still loading", "warn");
    return;
  }
  if (!state.feature.hasData() && !state.hasInk) {
    setUpload("Nothing to save", "warn");
    return;
  }

  const operationId = state.operationId + 1;
  state.operationId = operationId;
  state.saving = true;
  setButtonsDisabled(true);
  setUpload(auto ? "Auto-saving" : "Saving", "warn");

  try {
    const createdAt = new Date();
    const sampleId = makeSampleId(createdAt);
    setUpload("Stopping recorder", "warn");
    const videoBlob = await stopRecorder();
    if (operationId !== state.operationId) {
      return;
    }
    setUpload("Preparing drawing", "warn");
    const imageBlob = await canvasToBlob(inkCanvas);
    if (operationId !== state.operationId) {
      return;
    }
    const metadata = {
      app: "airwriting-pwa",
      app_version: APP_VERSION,
      sample_id: sampleId,
      username: els.username.value.trim() || "user1",
      label: normalizedLabel(),
      created_at: createdAt.toISOString(),
      auto_saved: auto,
      frame_size: [els.overlay.width, els.overlay.height],
      media: {
        image: imageBlob ? `${sampleId}.png` : null,
        video: videoBlob ? `${sampleId}.${state.recorderExt}` : null,
        video_mime: videoBlob?.type || null,
      },
      trajectory: state.feature.extract(),
      user_agent: navigator.userAgent,
    };

    const result = await uploadSample({ imageBlob, videoBlob, metadata, operationId });
    if (operationId !== state.operationId) {
      return;
    }
    state.savedCount += 1;
    els.savedReadout.textContent = String(state.savedCount);
    setUpload("Uploaded", "good");
    showFlash();
    resetDrawing();
    console.info("Saved sample", result.paths, result.commit);
  } catch (error) {
    if (operationId === state.operationId) {
      // Hold auto-save until the user draws again, otherwise the expired
      // pause timer re-triggers a failing save on every frame.
      state.autoSaveBlocked = true;
      setUpload(error.message || "Upload failed", "bad");
    }
  } finally {
    if (operationId === state.operationId) {
      state.saving = false;
      setButtonsDisabled(false);
    }
  }
}

function setButtonsDisabled(disabled) {
  els.saveBtn.disabled = disabled;
  els.clearBtn.disabled = false;
}

function resetDrawing() {
  inkCtx.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  state.feature?.reset();
  state.prevPoint = null;
  state.mode = "Pen up";
  state.hasInk = false;
  state.autoSaveBlocked = false;
  state.sampleStarted = false;
  state.sampleStartedAt = 0;
  state.lastPenDownAt = 0;
  state.recordedChunks = [];
  state.recorder = null;
  els.pauseBar.style.width = "0";
  els.modeReadout.textContent = "Pen up";
  els.pointsReadout.textContent = "0";
}

function showFlash() {
  els.saveFlash.hidden = false;
  window.setTimeout(() => {
    els.saveFlash.hidden = true;
  }, 1000);
}

function clearSample() {
  if (state.clearing) {
    return;
  }
  state.operationId += 1;
  state.uploadController?.abort();
  state.uploadController = null;
  state.saving = false;
  state.clearing = true;
  state.clearUntil = performance.now() + CLEAR_COOLDOWN_MS;
  const hadSample = state.hasInk
    || Boolean(state.feature?.hasData())
    || state.sampleStarted
    || state.recordedChunks.length > 0
    || Boolean(state.recorder);
  stopRecorder({ discard: true }).catch(() => null);
  resetDrawing();
  drawVisible(null);
  setUpload(hadSample ? "Cleared" : "Nothing to clear", "neutral");
  window.setTimeout(() => {
    if (performance.now() >= state.clearUntil) {
      state.clearing = false;
    }
  }, CLEAR_COOLDOWN_MS);
  state.clearing = false;
  setButtonsDisabled(false);
}

function bindEvents() {
  for (const input of [els.username, els.label, els.uploadKey]) {
    input.addEventListener("input", persistSettings);
    input.addEventListener("change", persistSettings);
  }
  els.saveBtn.addEventListener("click", () => saveSample(false));
  els.clearBtn.addEventListener("click", () => {
    clearSample();
  });

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    state.deferredInstall = event;
    els.installBtn.hidden = false;
  });

  els.installBtn.addEventListener("click", async () => {
    if (!state.deferredInstall) {
      return;
    }
    state.deferredInstall.prompt();
    await state.deferredInstall.userChoice.catch(() => null);
    state.deferredInstall = null;
    els.installBtn.hidden = true;
  });

  window.addEventListener("online", () => setUpload("Ready", "good"));
  window.addEventListener("offline", () => setUpload("Offline", "bad"));
}

async function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    try {
      const registration = await navigator.serviceWorker.register("/sw.js");
      await registration.update().catch(() => null);
    } catch (error) {
      console.warn("Service worker registration failed", error);
    }
  }
}

async function boot() {
  state.feature = new FeatureExtractor();
  setVersion();
  loadSettings();
  bindEvents();
  window.lucide?.createIcons();
  await registerServiceWorker();

  try {
    await initLandmarker();
    await initCamera();
    state.running = true;
    setUpload(navigator.onLine ? "Ready" : "Offline", navigator.onLine ? "good" : "bad");
    requestAnimationFrame(tick);
  } catch (error) {
    setCamera("Camera blocked", "bad");
    els.blocked.hidden = false;
    els.blockedDetail.textContent = error.message || "Camera permission was not granted.";
  }
}

boot();
