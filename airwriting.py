#!/usr/bin/env python3
"""
Air-Writing Data Collector  (VCRec / AWCV paper method)

Desktop:      python3 airwriting.py [--name NAME] [--label A]
Web/Mobile:   python3 airwriting.py --web [--port 5000] [--ssl]
              Then open the printed URL on any device on the same WiFi.

Output layout:
  output/{user}/{label}/
      {label}_{n:04d}_{timestamp}.mp4   raw camera video
      {label}_{n:04d}_{timestamp}.png   trajectory canvas
      {label}_{n:04d}_{timestamp}.json  8-D features + metadata
"""

import argparse
import base64
import json
import math
import os
import socket
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── paths & video ──────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
VIDEO_CODEC     = "mp4v"
VIDEO_EXT       = ".mp4"

# ── gesture thresholds ─────────────────────────────────────────────────────
PAUSE_FRAMES = 45       # ~1.5 s at 30 fps  → auto-save trigger
MIN_POINTS   = 10       # minimum trajectory points for a valid sample
ERASE_RADIUS = 30

# ── mediapipe landmark indices ─────────────────────────────────────────────
TIP_IDS   = [4, 8, 12, 16, 20]
PIP_IDS   = [3, 6, 10, 14, 18]
INDEX_TIP = 8

# ── desktop UI constants ───────────────────────────────────────────────────
PANEL_H = 60
INSTR_W = 220

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ═══════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s) or "x"


def make_label_dir(username: str, label: str) -> str:
    """output/{user}/{label}/  — created on demand."""
    d = os.path.join(BASE_OUTPUT_DIR, _safe(username), _safe(label))
    os.makedirs(d, exist_ok=True)
    return d


def count_existing(label_dir: str) -> int:
    if not os.path.isdir(label_dir):
        return 0
    return sum(1 for f in os.listdir(label_dir) if f.endswith(".json"))


def save_sample(canvas: np.ndarray,
                video_frames: list,
                feat,               # FeatureExtractor
                label: str,
                username: str,
                fps: float,
                frame_size: tuple,
                extra_meta: dict = None) -> dict:
    """Write PNG + MP4 + JSON for one sample.  Returns {image, video, json, n}."""
    label_dir = make_label_dir(username, label)
    n         = count_existing(label_dir) + 1
    ts        = time.strftime("%Y%m%d_%H%M%S")
    base      = f"{_safe(label)}_{n:04d}_{ts}"

    img_path  = os.path.join(label_dir, base + ".png")
    vid_path  = os.path.join(label_dir, base + VIDEO_EXT)
    json_path = os.path.join(label_dir, base + ".json")

    # trajectory image
    saved_img = None
    if np.any(canvas):
        cv2.imwrite(img_path, canvas)
        saved_img = img_path

    # video
    saved_vid = None
    if video_frames:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        w = cv2.VideoWriter(vid_path, fourcc, fps, frame_size)
        if w.isOpened():
            for f in video_frames:
                w.write(f)
            w.release()
            saved_vid = vid_path

    # 8-D trajectory features
    traj = feat.extract() if feat.has_data() else {}
    meta = {
        "label":       label,
        "username":    username,
        "sample_no":   n,
        "image":       os.path.basename(saved_img) if saved_img else None,
        "video":       os.path.basename(saved_vid) if saved_vid else None,
        "fps":         fps,
        "frame_count": len(video_frames),
        "frame_size":  list(frame_size),
        "created_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "trajectory":  traj,
        **(extra_meta or {}),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"  [saved] {label!r} #{n:04d} → {label_dir}/")
    return {"image": saved_img, "video": saved_vid, "json": json_path, "n": n}


# ═══════════════════════════════════════════════════════════════════════════
# 8-D Fingertip Feature Extractor  (VCRec paper, Sec. IV-B)
# x_t = [Δp, Δq, sin α, cos α, sin β, cos β, I(same), I(diff)]
# ═══════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    def __init__(self):
        self.reset()

    def reset(self):
        self._pts      = []
        self._strokes  = []
        self._sid      = 0
        self._was_down = False

    def update(self, x: int, y: int, fw: int, fh: int, down: bool):
        if down:
            if not self._was_down:
                self._sid += 1
            self._pts.append((x / fw, y / fh))
            self._strokes.append(self._sid)
        self._was_down = down

    def has_data(self) -> bool:
        return len(self._pts) >= MIN_POINTS

    def point_count(self) -> int:
        return len(self._pts)

    def extract(self) -> dict:
        pts, stk = self._pts, self._strokes
        n = len(pts)
        vecs = []
        for i in range(n):
            p, q = pts[i]
            if i == 0:
                dp = dq = sin_a = sin_b = 0.0
                cos_a = cos_b = 1.0
            else:
                pp, pq = pts[i - 1]
                dp, dq = p - pp, q - pq
                d = math.hypot(dp, dq) + 1e-9
                cos_a, sin_a = dp / d, dq / d
                if i >= 2:
                    p2, q2 = pts[i - 2]
                    dp2, dq2 = pp - p2, pq - q2
                    d2 = math.hypot(dp2, dq2) + 1e-9
                    ca2, sa2 = dp2 / d2, dq2 / d2
                    sin_b = sin_a * ca2 - cos_a * sa2
                    cos_b = cos_a * ca2 + sin_a * sa2
                else:
                    sin_b, cos_b = 0.0, 1.0
            same = 1 if (i < n - 1 and stk[i] == stk[i + 1]) else 0
            vecs.append([round(v, 6) for v in
                         [dp, dq, sin_a, cos_a, sin_b, cos_b, same, 1 - same]])
        return {
            "points_normalized": [[round(x, 6), round(y, 6)] for x, y in pts],
            "stroke_ids":  stk,
            "num_strokes": self._sid,
            "num_points":  n,
            "features_8d": vecs,
            "feature_names": [
                "delta_p", "delta_q",
                "sin_alpha", "cos_alpha",
                "sin_beta",  "cos_beta",
                "same_stroke", "diff_stroke",
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Hand gesture helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fingers_up(lms, is_right: bool) -> list:
    up = []
    if is_right:
        up.append(lms[TIP_IDS[0]].x < lms[PIP_IDS[0]].x)
    else:
        up.append(lms[TIP_IDS[0]].x > lms[PIP_IDS[0]].x)
    for t, p in zip(TIP_IDS[1:], PIP_IDS[1:]):
        up.append(lms[t].y < lms[p].y)
    return up


def classify_gesture(lms, is_right: bool) -> str:
    """Returns 'DRAW' | 'ERASE' | 'PEN_UP'."""
    up = _fingers_up(lms, is_right)
    if up[1] and not any(up[2:]):
        return "DRAW"
    if all(up[1:]):
        return "ERASE"
    return "PEN_UP"


def draw_skeleton(img, lms, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lms]
    for a, b in HAND_CONN:
        cv2.line(img, pts[a], pts[b], (80, 200, 80), 1)
    for pt in pts:
        cv2.circle(img, pt, 3, (0, 255, 0), -1)


# ═══════════════════════════════════════════════════════════════════════════
# Desktop collector  (OpenCV window)
# ═══════════════════════════════════════════════════════════════════════════

def run_desktop(username: str, start_label: str):
    latest = {"result": None, "lock": threading.Lock()}

    def on_result(result, _img, _ts):
        with latest["lock"]:
            latest["result"] = result

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return
    ret, frame0 = cap.read()
    if not ret:
        print("Cannot read from webcam.")
        return

    h, w  = frame0.shape[:2]
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1:
        fps = 30.0

    label = start_label

    S = dict(
        canvas      = np.zeros((h, w, 3), dtype=np.uint8),
        frames      = [],
        feat        = FeatureExtractor(),
        prev_pt     = None,
        brush       = 8,
        mode        = "PEN_UP",
        pen_up_fr   = 0,
        detected    = False,
        ts          = 0,
        saved_msg   = 0.0,
        n_saved     = count_existing(make_label_dir(username, label)),
    )

    def reset():
        S["canvas"]    = np.zeros((h, w, 3), dtype=np.uint8)
        S["frames"]    = []
        S["feat"].reset()
        S["prev_pt"]   = None
        S["pen_up_fr"] = 0
        S["detected"]  = False

    def do_save():
        res = save_sample(S["canvas"], S["frames"], S["feat"],
                          label, username, fps, (w, h))
        S["n_saved"] = res["n"]
        reset()
        S["saved_msg"] = time.time() + 2.0

    def change_label():
        nonlocal label
        cap.release()
        cv2.destroyAllWindows()
        new = input(f"New label [{label}]: ").strip()
        if new:
            label = new
        S["n_saved"] = count_existing(make_label_dir(username, label))
        print(f"Label → {label!r}  (existing: {S['n_saved']})")
        cap2 = cv2.VideoCapture(0)
        return cap2

    cv2.namedWindow("Air Writing Collector", cv2.WINDOW_NORMAL)

    print(f"\n=== Air Writing Collector ===")
    print(f"  User : {username}")
    print(f"  Label: {label!r}")
    print(f"  [s] Save   [n] Clear   [l] Change label   [+/-] Brush   [q] Quit")
    print(f"  Gestures: index only = DRAW | fist = PEN UP | open palm = ERASE")
    print(f"  Writing, then pausing 1.5 s auto-detects end of letter.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if cv2.getWindowProperty("Air Writing Collector", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        frame = cv2.flip(frame, 1)
        S["frames"].append(frame.copy())
        S["ts"] += 1

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_img, S["ts"])

        with latest["lock"]:
            result = latest["result"]

        S["mode"] = "PEN_UP"
        pen_down  = False

        if result and result.hand_landmarks:
            lms_     = result.hand_landmarks[0]
            is_right = result.handedness[0][0].display_name == "Right"
            g        = classify_gesture(lms_, is_right)

            tx = int(lms_[INDEX_TIP].x * w)
            ty = int(lms_[INDEX_TIP].y * h)

            if ty > PANEL_H:
                if g == "DRAW":
                    S["mode"] = "DRAW"
                    pen_down  = True
                    if S["prev_pt"]:
                        cv2.line(S["canvas"], S["prev_pt"], (tx, ty), (255, 255, 255), S["brush"] * 2)
                        cv2.circle(S["canvas"], (tx, ty), S["brush"], (255, 255, 255), -1)
                    cv2.circle(frame, (tx, ty), S["brush"], (255, 255, 255), -1)
                    S["prev_pt"] = (tx, ty)
                elif g == "ERASE":
                    S["mode"] = "ERASE"
                    S["prev_pt"] = None
                    wx = int(lms_[0].x * w)
                    wy = int(lms_[0].y * h)
                    cv2.circle(S["canvas"], (wx, wy), ERASE_RADIUS, (0, 0, 0), -1)
                    cv2.circle(frame, (wx, wy), ERASE_RADIUS, (0, 80, 220), 2)
                else:
                    S["prev_pt"] = None
            else:
                S["prev_pt"] = None

            S["feat"].update(tx, ty, w, h, pen_down)
            draw_skeleton(frame, lms_, w, h)
        else:
            S["prev_pt"] = None
            S["feat"].update(0, 0, w, h, False)

        # auto-save after pause
        if S["feat"].has_data() and not S["detected"]:
            if not pen_down:
                S["pen_up_fr"] += 1
            else:
                S["pen_up_fr"] = 0
            if S["pen_up_fr"] >= PAUSE_FRAMES:
                S["detected"] = True
                print(f"  Pause detected ({S['feat'].point_count()} pts) – press [s] to save or [n] to clear")
        elif not S["feat"].has_data():
            S["pen_up_fr"] = 0

        # compose display
        gray = cv2.cvtColor(S["canvas"], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        display = cv2.add(
            cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(S["canvas"], S["canvas"], mask=mask),
        )

        # top HUD bar
        cv2.rectangle(display, (0, 0), (w, PANEL_H), (20, 20, 20), -1)
        mc = (0,255,100) if S["mode"]=="DRAW" else (0,100,255) if S["mode"]=="ERASE" else (160,160,160)
        cv2.putText(display, f"Label: {label}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2)
        cv2.putText(display, f"#{S['n_saved']}  Mode: {S['mode']}  Pts: {S['feat'].point_count()}",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mc, 1)

        # right panel
        panel = np.zeros((h, INSTR_W, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        py = PANEL_H + 20
        for line, col in [
            ("DRAW:  index finger", (0, 220, 80)),
            ("ERASE: open palm",    (0, 80, 220)),
            ("PEN UP: fist",        (160, 160, 160)),
            ("", (0,0,0)),
            ("[s] Save", (200, 220, 200)),
            ("[n] Clear", (200, 220, 200)),
            ("[l] Label", (200, 220, 200)),
            ("[+/-] Brush", (200, 220, 200)),
            ("[q] Quit", (200, 220, 200)),
        ]:
            if line:
                cv2.putText(panel, line, (8, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
            py += 20

        # pause progress bar in panel
        py += 10
        prog = min(S["pen_up_fr"] / PAUSE_FRAMES, 1.0)
        bw   = INSTR_W - 20
        cv2.rectangle(panel, (8, py), (8 + bw, py + 10), (40, 40, 40), -1)
        col = (0, 200, 255) if prog >= 1.0 else (0, 180, 80)
        cv2.rectangle(panel, (8, py), (8 + int(bw * prog), py + 10), col, -1)
        cv2.putText(panel, "Pause →auto", (8, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

        if S["detected"]:
            cv2.putText(panel, "READY TO SAVE", (4, py + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

        combined = np.hstack([display, panel])

        if time.time() < S["saved_msg"]:
            cw = combined.shape[1]
            cv2.putText(combined, "SAVED!", (cw // 2 - 70, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 120), 3)

        cv2.imshow("Air Writing Collector", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if S["feat"].has_data() or np.any(S["canvas"]):
                do_save()
        elif key == ord("n"):
            reset()
            print("  Cleared.")
        elif key == ord("l"):
            cap = change_label()
            cv2.namedWindow("Air Writing Collector", cv2.WINDOW_NORMAL)
        elif key in (ord("+"), ord("=")):
            S["brush"] = min(S["brush"] + 2, 40)
        elif key == ord("-"):
            S["brush"] = max(S["brush"] - 2, 2)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Session ended.")


# ═══════════════════════════════════════════════════════════════════════════
# Web / Mobile collector  (Flask server)
# ═══════════════════════════════════════════════════════════════════════════

_WEB_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Air Writing Collector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:monospace;overflow-x:hidden}
#top{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:8px 10px;
     background:#1a1a1a;border-bottom:1px solid #333}
#top label{font-size:13px;display:flex;align-items:center;gap:4px}
#top input{background:#2a2a2a;border:1px solid #555;color:#eee;
           padding:4px 8px;border-radius:4px;font-size:13px;width:110px}
.btn{background:#2a2a2a;border:1px solid #555;color:#eee;
     padding:7px 16px;border-radius:4px;cursor:pointer;font-size:13px}
.btn.green{border-color:#3a7;color:#5fa}
.btn.red{border-color:#a33;color:#f88}
#status{font-size:12px;color:#8f8;margin-left:auto}
#count{font-size:12px;color:#aaa}
#prog{height:5px;background:#222}
#bar{height:100%;background:#2a8;width:0%;transition:width .1s}
#wrap{position:relative;width:100%;max-width:800px;margin:0 auto;display:block}
video,#ovr{width:100%;display:block}
#ovr{position:absolute;top:0;left:0;pointer-events:none}
#hint{padding:6px 10px;font-size:11px;color:#666;text-align:center}
#flash{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
       font-size:48px;font-weight:bold;color:#0f6;display:none;
       text-shadow:0 0 20px #0f6;pointer-events:none}
</style>
</head>
<body>
<div id="top">
  <label>User <input id="uname" value="user1"></label>
  <label>Label <input id="lbl" value="A" style="width:70px"></label>
  <button class="btn green" onclick="doSave()">&#10003; Save [S]</button>
  <button class="btn red"   onclick="doClear()">&#10005; Clear [N]</button>
  <span id="count">Saved: 0</span>
  <span id="status">Connecting…</span>
</div>
<div id="prog"><div id="bar"></div></div>
<div id="wrap">
  <video id="vid" autoplay playsinline muted></video>
  <canvas id="ovr"></canvas>
</div>
<div id="hint">index finger = DRAW &nbsp;|&nbsp; fist = PEN UP (pause 1.5 s → auto-save) &nbsp;|&nbsp; open palm = ERASE</div>
<div id="flash">SAVED!</div>

<script>
const vid=document.getElementById('vid');
const ovr=document.getElementById('ovr');
const ctx=ovr.getContext('2d');
const statusEl=document.getElementById('status');
const barEl=document.getElementById('bar');
const countEl=document.getElementById('count');
const flash=document.getElementById('flash');

const CONN=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];

let busy=false, animId=null;

async function startCamera(){
  try{
    const s=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
    vid.srcObject=s;
    vid.onloadedmetadata=()=>{
      ovr.width=vid.videoWidth; ovr.height=vid.videoHeight;
      statusEl.textContent='Ready';
      loop();
    };
  }catch(e){statusEl.textContent='Camera error: '+e.message;}
}

const grab=document.createElement('canvas');
const gctx=grab.getContext('2d');

function loop(){animId=requestAnimationFrame(async()=>{
  if(!busy&&vid.videoWidth){
    busy=true;
    grab.width=vid.videoWidth; grab.height=vid.videoHeight;
    gctx.drawImage(vid,0,0);
    grab.toBlob(async blob=>{
      try{
        const fd=new FormData(); fd.append('frame',blob,'f.jpg');
        const r=await fetch('/process',{method:'POST',body:fd});
        const d=await r.json();
        renderOverlay(d);
        barEl.style.width=(Math.min(d.pause_progress||0,1)*100)+'%';
        barEl.style.background=d.pause_progress>=1?'#0cc':'#2a8';
        statusEl.textContent=d.mode||'';
        if(d.auto_saved){countEl.textContent='Saved: '+d.count; showFlash();}
      }catch(_){}
      busy=false;
    },'image/jpeg',0.75);
  }
  loop();
});}

function renderOverlay(d){
  ctx.clearRect(0,0,ovr.width,ovr.height);
  const W=ovr.width, H=ovr.height;
  if(d.traj_b64){
    const img=new Image();
    img.onload=()=>{ctx.globalAlpha=.75;ctx.drawImage(img,0,0,W,H);ctx.globalAlpha=1;};
    img.src='data:image/png;base64,'+d.traj_b64;
  }
  if(d.lms){
    ctx.strokeStyle='#4f4'; ctx.lineWidth=1.5;
    for(const[a,b]of CONN){
      ctx.beginPath();
      ctx.moveTo(d.lms[a][0]*W,d.lms[a][1]*H);
      ctx.lineTo(d.lms[b][0]*W,d.lms[b][1]*H);
      ctx.stroke();
    }
    ctx.fillStyle='#0f0';
    for(const pt of d.lms){
      ctx.beginPath(); ctx.arc(pt[0]*W,pt[1]*H,3,0,Math.PI*2); ctx.fill();
    }
  }
}

function showFlash(){
  flash.style.display='block';
  setTimeout(()=>flash.style.display='none',1200);
}

async function doSave(){
  const r=await fetch('/save',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:document.getElementById('uname').value,
                         label:document.getElementById('lbl').value})});
  const d=await r.json();
  if(d.ok){countEl.textContent='Saved: '+d.count; showFlash();}
  else statusEl.textContent=d.error||'nothing to save';
}
async function doClear(){
  await fetch('/clear',{method:'POST'});
  barEl.style.width='0%';
  statusEl.textContent='Cleared';
}
document.addEventListener('keydown',e=>{
  if(e.key==='s'||e.key==='S') doSave();
  if(e.key==='n'||e.key==='N') doClear();
});
startCamera();
</script>
</body>
</html>
"""


def run_web(username_default: str, label_default: str,
            host: str, port: int, use_ssl: bool):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed.  Run:  pip install flask")
        return

    app = Flask(__name__)

    # shared per-session state (single user)
    S = dict(
        lock       = threading.Lock(),
        canvas     = None,
        frames     = [],
        feat       = FeatureExtractor(),
        prev_pt    = None,
        pen_up_fr  = 0,
        detected   = False,
        username   = username_default,
        label      = label_default,
        fps        = 15.0,
        frame_size = (640, 480),
        count      = count_existing(make_label_dir(username_default, label_default)),
        ts         = 0,
    )

    lm_latest = {"result": None, "lock": threading.Lock()}

    def on_result(result, _img, _ts):
        with lm_latest["lock"]:
            lm_latest["result"] = result

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

    def _reset(st, h, w):
        st["canvas"]    = np.zeros((h, w, 3), dtype=np.uint8)
        st["frames"]    = []
        st["feat"].reset()
        st["prev_pt"]   = None
        st["pen_up_fr"] = 0
        st["detected"]  = False

    @app.route("/")
    def index():
        return _WEB_HTML

    @app.route("/process", methods=["POST"])
    def process():
        blob = request.files.get("frame")
        if blob is None:
            return jsonify({"error": "no frame"}), 400

        buf   = np.frombuffer(blob.read(), np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "decode fail"}), 400

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        with S["lock"]:
            if S["canvas"] is None or S["canvas"].shape[:2] != (h, w):
                _reset(S, h, w)
                S["frame_size"] = (w, h)

            S["frames"].append(frame.copy())
            S["ts"] += 1
            ts = S["ts"]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_img, ts)

        with lm_latest["lock"]:
            result = lm_latest["result"]

        mode      = "PEN_UP"
        pen_down  = False
        lms_out   = None
        auto_saved = False

        with S["lock"]:
            if result and result.hand_landmarks:
                lms_     = result.hand_landmarks[0]
                is_right = result.handedness[0][0].display_name == "Right"
                g        = classify_gesture(lms_, is_right)

                tx = int(lms_[INDEX_TIP].x * w)
                ty = int(lms_[INDEX_TIP].y * h)

                if g == "DRAW":
                    mode     = "DRAW"
                    pen_down = True
                    if S["prev_pt"]:
                        cv2.line(S["canvas"], S["prev_pt"], (tx, ty), (255, 255, 255), 8)
                        cv2.circle(S["canvas"], (tx, ty), 4, (255, 255, 255), -1)
                    S["prev_pt"] = (tx, ty)
                elif g == "ERASE":
                    mode = "ERASE"
                    S["prev_pt"] = None
                    wx = int(lms_[0].x * w)
                    wy = int(lms_[0].y * h)
                    cv2.circle(S["canvas"], (wx, wy), ERASE_RADIUS, (0, 0, 0), -1)
                else:
                    S["prev_pt"] = None

                S["feat"].update(tx, ty, w, h, pen_down)
                lms_out = [[round(l.x, 4), round(l.y, 4)] for l in lms_]
            else:
                S["prev_pt"] = None
                S["feat"].update(0, 0, w, h, False)

            # auto-save on pause
            if S["feat"].has_data() and not S["detected"]:
                if not pen_down:
                    S["pen_up_fr"] += 1
                else:
                    S["pen_up_fr"] = 0
                if S["pen_up_fr"] >= PAUSE_FRAMES:
                    S["detected"] = True
                    res = save_sample(
                        S["canvas"], S["frames"], S["feat"],
                        S["label"], S["username"],
                        S["fps"], S["frame_size"],
                    )
                    S["count"] = res["n"]
                    auto_saved = True
                    _reset(S, h, w)
            elif not S["feat"].has_data():
                S["pen_up_fr"] = 0

            pause_prog = min(S["pen_up_fr"] / PAUSE_FRAMES, 1.0)
            canvas_copy = S["canvas"].copy()
            count = S["count"]

        traj_b64 = None
        if np.any(canvas_copy):
            _, enc   = cv2.imencode(".png", canvas_copy)
            traj_b64 = base64.b64encode(enc.tobytes()).decode()

        return jsonify({
            "mode":          mode,
            "lms":           lms_out,
            "traj_b64":      traj_b64,
            "pause_progress": pause_prog,
            "auto_saved":    auto_saved,
            "count":         count,
        })

    @app.route("/save", methods=["POST"])
    def save_endpoint():
        data = request.get_json(silent=True) or {}
        with S["lock"]:
            uname = data.get("username") or S["username"]
            lbl   = data.get("label")    or S["label"]
            S["username"] = uname
            S["label"]    = lbl

            if not S["feat"].has_data() and (S["canvas"] is None or not np.any(S["canvas"])):
                return jsonify({"ok": False, "error": "nothing to save"})

            h, w = S["frame_size"][1], S["frame_size"][0]
            canvas = S["canvas"] if S["canvas"] is not None else np.zeros((h, w, 3), np.uint8)
            res = save_sample(canvas, S["frames"], S["feat"],
                              lbl, uname, S["fps"], S["frame_size"])
            S["count"] = res["n"]
            _reset(S, h, w)

        return jsonify({"ok": True, "count": res["n"]})

    @app.route("/clear", methods=["POST"])
    def clear_endpoint():
        with S["lock"]:
            if S["canvas"] is not None:
                h, w = S["canvas"].shape[:2]
                _reset(S, h, w)
        return jsonify({"ok": True})

    # SSL (needed for camera on mobile)
    ssl_ctx = None
    proto   = "http"
    if use_ssl:
        try:
            from OpenSSL import crypto
            import tempfile

            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 2048)
            cert = crypto.X509()
            cert.get_subject().CN = "airwriting"
            cert.set_serial_number(1)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(k)
            cert.sign(k, "sha256")

            cf = tempfile.NamedTemporaryFile(suffix=".crt", delete=False)
            kf = tempfile.NamedTemporaryFile(suffix=".key", delete=False)
            cf.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            kf.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
            cf.close(); kf.close()
            ssl_ctx = (cf.name, kf.name)
            proto   = "https"
            print("[SSL] Self-signed certificate generated.")
        except ImportError:
            print("[WARN] pyOpenSSL not found – running HTTP (mobile may block camera).")
            print("       Fix: pip install pyOpenSSL  then add --ssl")

    # local IP for convenience
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n=== Air Writing Web Collector ===")
    print(f"  Open on desktop : {proto}://localhost:{port}/")
    print(f"  Open on mobile  : {proto}://{local_ip}:{port}/")
    print(f"  (Mobile must be on the same WiFi network)")
    if not use_ssl:
        print(f"  [!] Add --ssl for HTTPS — required for camera on iOS/Android Chrome")
    print(f"  Ctrl-C to stop\n")

    app.run(host=host, port=port, debug=False, ssl_context=ssl_ctx, threaded=True)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Air-Writing Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python airwriting.py                        # desktop webcam collector
  python airwriting.py --name Alice --label A # skip name prompt, start at label A
  python airwriting.py --web --port 5000      # web server for mobile
  python airwriting.py --web --ssl            # HTTPS (needed for iOS camera)
""")
    ap.add_argument("--web",   action="store_true", help="Run Flask web server (mobile-friendly)")
    ap.add_argument("--ssl",   action="store_true", help="Enable HTTPS (requires: pip install pyOpenSSL)")
    ap.add_argument("--host",  default="0.0.0.0",   help="Web host (default: 0.0.0.0)")
    ap.add_argument("--port",  default=5000, type=int, help="Web port (default: 5000)")
    ap.add_argument("--name",  default="",  help="Username (skip interactive prompt)")
    ap.add_argument("--label", default="A", help="Starting label (default: A)")
    args = ap.parse_args()

    username = args.name.strip()
    if not username:
        while True:
            username = input("Enter your name: ").strip()
            if username:
                break
            print("Name cannot be empty.")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    if args.web:
        run_web(username, args.label, args.host, args.port, args.ssl)
    else:
        run_desktop(username, args.label)


if __name__ == "__main__":
    main()
