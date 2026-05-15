"""
LAPTOP SCRIPT — Real-time ASL inference with MediaPipe hand tracking.
Compatible with mediapipe >= 0.10 (Tasks API).

Requirements:
    pip install tensorflow opencv-python numpy mediapipe

On first run, downloads hand_landmarker.task (~8 MB) automatically.

Usage:
    python Livestream.py
    python Livestream.py --camera 1       # try 1, 2 for iVCam
    python Livestream.py --no-landmarks   # hide finger skeleton
    python Livestream.py --debug          # save every crop to debug_crops/

Controls:
    q  — quit
    s  — save screenshot
    l  — toggle landmark overlay
    p  — toggle 28x28 preview
    d  — toggle debug crop saving
    m  — cycle preprocessing mode (clahe / equalize / stretch / thresh)
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
ASL_MODEL_PATH  = os.path.join(SCRIPT_DIR, "asl_model.tflite")
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
DEBUG_DIR       = os.path.join(SCRIPT_DIR, "debug_crops")
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

INPUT_SIZE  = 28
CONF_THRESH = 0.5
SMOOTH_N    = 7
PAD_FRAC    = 0.20   # tighter crop — less arm/background noise

ALPHABET = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'
}
IDX_TO_LETTER = {i: ALPHABET[i] for i in range(25) if i in ALPHABET}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
FINGERTIPS = {4, 8, 12, 16, 20}

PREP_MODES = ['clahe', 'equalize', 'stretch', 'thresh']


# ── Preprocessing modes ───────────────────────────────────────────────────────
def preprocess(crop_bgr, mode='stretch'):
    """
    Convert BGR hand crop → 28x28 float32 [0,1].

    Sign MNIST images have hands with clear contrast on neutral background.
    'stretch' aggressively normalises histogram so the hand fills full 0-255.
    'thresh'  binarises — most similar to Sign MNIST silhouettes.
    'equalize'/'clahe' are gentler alternatives.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    if mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

    elif mode == 'equalize':
        gray = cv2.equalizeHist(gray)

    elif mode == 'stretch':
        lo, hi = gray.min(), gray.max()
        if hi > lo:
            gray = ((gray.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)

    elif mode == 'thresh':
        # Otsu threshold → binary image (hand=white on black like Sign MNIST)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ── Utilities ─────────────────────────────────────────────────────────────────
def ensure_hand_model(path, url):
    if os.path.exists(path):
        return
    print(f"Downloading hand_landmarker.task (~8 MB)...")
    urllib.request.urlretrieve(url, path)
    print("Download complete.")


def load_asl_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ASL model not found: {path}")
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def predict_asl(interp, inp_det, out_det, img_28x28):
    tensor = img_28x28.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)
    interp.set_tensor(inp_det[0]['index'], tensor)
    interp.invoke()
    probs    = interp.get_tensor(out_det[0]['index'])[0]  # already softmax
    pred_idx = int(np.argmax(probs))
    return IDX_TO_LETTER.get(pred_idx, '?'), float(probs[pred_idx]), probs


def hand_bbox(landmarks, frame_h, frame_w, pad_frac=PAD_FRAC):
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    pad_x = (max(xs) - min(xs)) * pad_frac
    pad_y = (max(ys) - min(ys)) * pad_frac
    x1 = max(0, int(min(xs) - pad_x))
    y1 = max(0, int(min(ys) - pad_y))
    x2 = min(frame_w, int(max(xs) + pad_x))
    y2 = min(frame_h, int(max(ys) + pad_y))
    return x1, y1, x2, y2


def draw_landmarks(frame, landmarks, frame_h, frame_w):
    pts = {i: (int(lm.x * frame_w), int(lm.y * frame_h))
           for i, lm in enumerate(landmarks)}
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 255), 2)
    for i, (x, y) in pts.items():
        r = 6 if i in FINGERTIPS else 3
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), r, (0, 180, 255), 1)


def draw_top5(frame, probs, origin=(10, 60)):
    top5   = np.argsort(probs)[::-1][:5]
    ox, oy = origin
    bar_w  = 120
    bar_h  = 14
    gap    = 20
    for rank, idx in enumerate(top5):
        y     = oy + rank * gap
        conf  = float(probs[idx])
        color = (0, 220, 80) if rank == 0 else (80, 140, 200)
        cv2.rectangle(frame, (ox, y), (ox + bar_w, y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (ox, y), (ox + int(bar_w * conf), y + bar_h), color, -1)
        cv2.putText(frame,
                    f"{IDX_TO_LETTER.get(idx,'?')} {conf*100:.1f}%",
                    (ox + bar_w + 6, y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-landmarks', action='store_true')
    parser.add_argument('--no-preview',   action='store_true')
    parser.add_argument('--debug',        action='store_true',
                        help='Save every hand crop to debug_crops/')
    parser.add_argument('--prep', default='stretch',
                        choices=PREP_MODES,
                        help='Preprocessing mode (default: stretch)')
    args = parser.parse_args()

    show_landmarks = not args.no_landmarks
    show_preview   = not args.no_preview
    debug_mode     = args.debug
    prep_mode_idx  = PREP_MODES.index(args.prep)

    if debug_mode:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug crops → {DEBUG_DIR}")

    ensure_hand_model(HAND_MODEL_PATH, HAND_MODEL_URL)
    interp, inp_det, out_det = load_asl_model(ASL_MODEL_PATH)
    print(f"ASL model loaded: {ASL_MODEL_PATH}")

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera opened.")
    print("q=quit  s=save  l=landmarks  p=preview  d=debug  m=cycle-prep-mode")

    recent_preds = []
    smoothed     = '?'
    last_probs   = np.ones(25, dtype=np.float32) / 25
    fps          = 0.0
    fps_timer    = time.time()
    frame_count  = 0
    timestamp_ms = 0
    debug_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        timestamp_ms += 33

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_detector.detect_for_video(mp_img, timestamp_ms)

        hand_detected = len(result.hand_landmarks) > 0
        prep_mode     = PREP_MODES[prep_mode_idx]

        if hand_detected:
            landmarks  = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name  # 'Left' or 'Right'
            x1, y1, x2, y2 = hand_bbox(landmarks, h, w)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # Mirror left hand so it matches right-hand training data
                if handedness == 'Left':
                    crop = cv2.flip(crop, 1)

                processed = preprocess(crop, mode=prep_mode)
                letter, conf, last_probs = predict_asl(interp, inp_det, out_det, processed)

                if debug_mode:
                    save_img = (processed * 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(DEBUG_DIR, f"{debug_count:05d}_{letter}_{conf:.2f}.png"),
                        save_img
                    )
                    debug_count += 1

                recent_preds.append(letter)
                if len(recent_preds) > SMOOTH_N:
                    recent_preds.pop(0)
                smoothed = max(set(recent_preds), key=recent_preds.count)

                box_color = (0, 255, 80) if conf >= CONF_THRESH else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Handedness label on bbox
                cv2.putText(frame, handedness[0],  # L or R
                            (x2 - 22, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 80), 2)

                # 28x28 preview
                if show_preview:
                    preview_sz  = 84
                    preview     = cv2.resize(
                        (processed * 255).astype(np.uint8),
                        (preview_sz, preview_sz), interpolation=cv2.INTER_NEAREST
                    )
                    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                    px1, py1    = x2 + 4, y1
                    px2, py2    = px1 + preview_sz, py1 + preview_sz
                    if px2 < w and py2 < h:
                        frame[py1:py2, px1:px2] = preview_bgr
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)
                        cv2.putText(frame, "28x28 input", (px1, py2 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

                if show_landmarks:
                    draw_landmarks(frame, landmarks, h, w)

                pred_text = (f"{smoothed}  {conf*100:.0f}%"
                             if conf >= CONF_THRESH else f"?  {conf*100:.0f}%")
                cv2.putText(frame, pred_text, (x1, max(y1 - 12, 30)),
                            cv2.FONT_HERSHEY_DUPLEX, 1.6, box_color, 3)
        else:
            recent_preds.clear()
            cv2.putText(frame, "No hand detected", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 200), 2)

        draw_top5(frame, last_probs)

        # Status
        prep_label = f"prep:{prep_mode}"
        dbg_label  = " DBG" if debug_mode else ""
        cv2.putText(frame,
                    f"FPS {fps:.1f}  Hand:{'YES' if hand_detected else 'NO '}  {prep_label}{dbg_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(frame, "q=quit  s=save  l=landmarks  p=preview  d=debug  m=prep-mode",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps         = frame_count / (now - fps_timer)
            fps_timer   = now
            frame_count = 0

        cv2.imshow("ASL Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('p'):
            show_preview = not show_preview
        elif key == ord('d'):
            debug_mode = not debug_mode
            if debug_mode:
                os.makedirs(DEBUG_DIR, exist_ok=True)
            print(f"Debug: {'ON → ' + DEBUG_DIR if debug_mode else 'OFF'}")
        elif key == ord('m'):
            prep_mode_idx = (prep_mode_idx + 1) % len(PREP_MODES)
            recent_preds.clear()
            print(f"Prep mode: {PREP_MODES[prep_mode_idx]}")

    cap.release()
    hand_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
