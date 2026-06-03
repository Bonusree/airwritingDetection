import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
import os
import time
import subprocess

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# BGR colors
COLORS = [
    ("Blue",   (255,   0,   0)),
    ("Green",  (  0, 200,   0)),
    ("Red",    (  0,   0, 255)),
    ("Yellow", (  0, 220, 220)),
    ("White",  (255, 255, 255)),
]

PANEL_H     = 70
SWATCH_W    = 44
SWATCH_H    = 44
SWATCH_GAP  = 8
ERASE_RADIUS = 30
INSTR_W     = 220

# Button geometry (set after we know frame width)
BTN_H = 44
BTN_W = 80

TIP_IDS  = [4, 8, 12, 16, 20]
PIP_IDS  = [3, 6, 10, 14, 18]
INDEX_TIP = 8


def fingers_up(landmarks, is_right_hand):
    up = []
    if is_right_hand:
        up.append(landmarks[TIP_IDS[0]].x < landmarks[PIP_IDS[0]].x)
    else:
        up.append(landmarks[TIP_IDS[0]].x > landmarks[PIP_IDS[0]].x)
    for t, p in zip(TIP_IDS[1:], PIP_IDS[1:]):
        up.append(landmarks[t].y < landmarks[p].y)
    return up


def swatch_rect(i):
    x1 = SWATCH_GAP + i * (SWATCH_W + SWATCH_GAP)
    y1 = (PANEL_H - SWATCH_H) // 2
    return x1, y1, x1 + SWATCH_W, y1 + SWATCH_H


def button_rects(frame_w):
    # Save button
    sx = frame_w - 2 * (BTN_W + 12)
    # Clear button
    cx = frame_w - (BTN_W + 12)
    y1 = (PANEL_H - BTN_H) // 2
    return (sx, y1, sx + BTN_W, y1 + BTN_H), (cx, y1, cx + BTN_W, y1 + BTN_H)


def draw_instructions(display, canvas_w, mode):
    h, total_w = display.shape[:2]
    gestures = [
        ("DRAW",    "Index finger only", (0, 255, 100)),
        ("ERASE",   "Open palm",         (0, 100, 255)),
        ("PEN UP",  "Closed fist",       (180, 180, 180)),
    ]

    # Dark background for the right panel
    cv2.rectangle(display, (canvas_w, 0), (total_w, h), (25, 25, 25), -1)
    cv2.line(display, (canvas_w, 0), (canvas_w, h), (80, 80, 80), 1)

    px = canvas_w + 14

    cv2.putText(display, "Hand Gestures", (px, PANEL_H + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)
    cv2.line(display, (canvas_w + 8, PANEL_H + 38),
             (total_w - 8, PANEL_H + 38), (70, 70, 70), 1)

    y = PANEL_H + 60
    for gmode, desc, color in gestures:
        active = (gmode == mode)
        c = color if active else (90, 90, 90)
        weight = 2 if active else 1
        prefix = "> " if active else "  "
        cv2.putText(display, prefix + gmode, (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, weight)
        cv2.putText(display, "   " + desc, (px, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        y += 56


def draw_ui(frame, color_idx, brush_size, mode, hover):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, PANEL_H), (30, 30, 30), -1)

    # Color swatches
    for i, (name, color) in enumerate(COLORS):
        x1, y1, x2, y2 = swatch_rect(i)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        if i == color_idx:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
        elif hover == ("color", i):
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (180, 180, 180), 1)

    # Mode label
    mode_color = (0, 255, 100) if mode == "DRAW" else (0, 100, 255) if mode == "ERASE" else (180, 180, 180)
    cx_label = len(COLORS) * (SWATCH_W + SWATCH_GAP) + 20
    cv2.putText(frame, f"Mode: {mode}", (cx_label, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    cv2.putText(frame, f"Brush: {brush_size}px  [+/-]", (cx_label, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Save / Clear buttons
    save_r, clear_r = button_rects(w)
    for rect, label, hkey in [(save_r, "Save", "save"), (clear_r, "Clear", "clear")]:
        bx1, by1, bx2, by2 = rect
        bg = (60, 120, 60) if label == "Save" else (120, 60, 60)
        if hover == hkey:
            bg = tuple(min(c + 40, 255) for c in bg)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        tx = bx1 + (BTN_W - tw) // 2
        cv2.putText(frame, label, (tx, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)


def blend_canvas(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return cv2.add(cv2.bitwise_and(frame, frame, mask=mask_inv),
                   cv2.bitwise_and(canvas, canvas, mask=mask))


def draw_hand(frame, landmarks, w, h):
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 80), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 255, 0), -1)


def point_in_rect(x, y, r):
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


def _ask_save_path(default_name):
    """Open a native save-as dialog. Returns filepath string or None if cancelled."""
    try:
        abs_path = os.path.join(os.path.abspath(OUTPUT_DIR), default_name)
        result = subprocess.run(
            [
                "zenity", "--file-selection", "--save",
                "--confirm-overwrite",
                "--title=Save Drawing",
                f"--filename={abs_path}",
                "--file-filter=PNG files (*.png) | *.png",
                "--file-filter=All files | *",
            ],
            capture_output=True, text=True,
        )
        path = result.stdout.strip()
        if result.returncode != 0 or not path:
            return None
        if not path.lower().endswith(".png"):
            path += ".png"
        return path
    except FileNotFoundError:
        # zenity not available — fall back to terminal input
        name = input(f"Filename [{default_name}]: ").strip()
        if not name:
            name = default_name
        if not name.lower().endswith(".png"):
            name += ".png"
        return os.path.join(OUTPUT_DIR, name)


def save_drawing(canvas):
    if not np.any(canvas):
        print("Nothing to save — canvas is empty.")
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    default_name = f"drawing_{cv2.getTickCount()}.png"

    fname = _ask_save_path(default_name)
    if not fname:
        print("Save cancelled.")
        return None

    cv2.imwrite(fname, canvas)
    print(f"Saved: {fname}")
    return fname


def main():
    latest = {"result": None, "lock": threading.Lock()}

    def on_result(result, output_image, timestamp_ms):
        with latest["lock"]:
            latest["result"] = result

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from webcam.")
        return

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    color_idx = 0
    brush_size = 8
    prev_pt = None
    mode = "PEN UP"
    ts = 0
    hover = None
    frame_count = 0  # ignore mouse events until the window is stable
    saved_msg_until = 0

    # Mouse state
    mouse = {"x": -1, "y": -1, "clicked": False}

    def on_mouse(event, mx, my, flags, param):
        mouse["x"], mouse["y"] = mx, my
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["clicked"] = True

    cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Air Writing", on_mouse)
    # Show a black frame first so the window exists before WND_PROP_VISIBLE is checked
    cv2.imshow("Air Writing", np.zeros((h, w, 3), dtype=np.uint8))
    cv2.waitKey(1)

    print("\n=== Air Writing ===")
    print("Gestures:")
    print("  Index finger only  → DRAW")
    print("  Closed fist        → PEN UP")
    print("  Open hand (palm)   → ERASE")
    print(f"Drawings saved to: {OUTPUT_DIR}/")
    print("Keys: [+/-] brush size  [q] quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Exit if the window was closed via the X button
        try:
            if cv2.getWindowProperty("Air Writing", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        frame = cv2.flip(frame, 1)
        ts += 1
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, ts)

        with latest["lock"]:
            result = latest["result"]

        # --- Handle mouse clicks in the panel ---
        save_r, clear_r = button_rects(w)
        mx, my = mouse["x"], mouse["y"]

        # Determine hover state for rendering
        hover = None
        if 0 <= my < PANEL_H:
            for i in range(len(COLORS)):
                if point_in_rect(mx, my, swatch_rect(i)):
                    hover = ("color", i)
                    break
            if point_in_rect(mx, my, save_r):
                hover = "save"
            elif point_in_rect(mx, my, clear_r):
                hover = "clear"

        if mouse["clicked"] and frame_count > 30:
            mouse["clicked"] = False
            if 0 <= my < PANEL_H:
                for i in range(len(COLORS)):
                    if point_in_rect(mx, my, swatch_rect(i)):
                        color_idx = i
                        print(f"Color: {COLORS[i][0]}")
                        break
                if point_in_rect(mx, my, save_r):
                    if save_drawing(canvas):
                        saved_msg_until = time.time() + 2
                elif point_in_rect(mx, my, clear_r):
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    print("Canvas cleared.")

        # --- Hand tracking ---
        mode = "PEN UP"

        if result and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            is_right = result.handedness[0][0].display_name == "Right"
            up = fingers_up(landmarks, is_right)

            tip_x = int(landmarks[INDEX_TIP].x * w)
            tip_y = int(landmarks[INDEX_TIP].y * h)

            # Gesture detection
            # up = [thumb, index, middle, ring, pinky]
            fist      = not any(up[1:])          # all 4 fingers closed
            open_hand = all(up[1:])              # all 4 fingers open
            index_only = up[1] and not any(up[2:])  # only index extended

            # Only draw/erase below the UI panel
            if tip_y > PANEL_H:
                if fist:
                    # Closed fist → PEN UP
                    mode = "PEN UP"
                    prev_pt = None
                    cv2.circle(frame, (tip_x, tip_y), brush_size + 6, (200, 200, 200), 2)

                elif open_hand:
                    # Open palm → ERASE
                    mode = "ERASE"
                    prev_pt = None
                    # Use wrist landmark as erase centre for open hand
                    wx = int(landmarks[0].x * w)
                    wy = int(landmarks[0].y * h)
                    cv2.circle(canvas, (wx, wy), ERASE_RADIUS, (0, 0, 0), -1)
                    cv2.circle(frame,  (wx, wy), ERASE_RADIUS, (0, 0, 200), 2)

                elif index_only:
                    # Index only → DRAW
                    mode = "DRAW"
                    _, color = COLORS[color_idx]
                    cv2.circle(frame, (tip_x, tip_y), brush_size, color, -1)
                    if prev_pt:
                        cv2.line(canvas, prev_pt, (tip_x, tip_y), color, brush_size * 2)
                        cv2.circle(canvas, (tip_x, tip_y), brush_size, color, -1)
                    prev_pt = (tip_x, tip_y)

                else:
                    prev_pt = None
            else:
                prev_pt = None

            draw_hand(frame, landmarks, w, h)
        else:
            prev_pt = None

        output = blend_canvas(cv2.GaussianBlur(frame, (51, 51), 0), canvas)
        draw_ui(output, color_idx, brush_size, mode, hover)

        if time.time() < saved_msg_until:
            msg = "Data Saved"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            (tw, th), _ = cv2.getTextSize(msg, font, scale, thickness)
            tx = (w - tw) // 2
            ty = (h + PANEL_H) // 2
            cv2.putText(output, msg, (tx + 2, ty + 2), font, scale, (0, 0, 0), thickness + 2)
            cv2.putText(output, msg, (tx, ty), font, scale, (0, 255, 120), thickness)

        display = np.zeros((h, w + INSTR_W, 3), dtype=np.uint8)
        display[:, :w] = output
        draw_instructions(display, w, mode)

        cv2.imshow("Air Writing", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            brush_size = min(brush_size + 2, 40)
        elif key == ord("-"):
            brush_size = max(brush_size - 2, 2)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Air Writing closed.")


if __name__ == "__main__":
    main()
