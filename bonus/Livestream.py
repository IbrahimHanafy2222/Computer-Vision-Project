"""
ASL Real-Time Recognition — MediaPipe + selectable model backend.

Requirements:
    pip install tensorflow opencv-python numpy mediapipe scikit-learn joblib

On first run, downloads hand_landmarker.task (~8 MB) automatically.

Usage:
    python Livestream.py                        # interactive model prompt
    python Livestream.py --model cnn            # Robust CNN  (Sign MNIST 28×28)
    python Livestream.py --model optionA        # Real-Photo CNN (ASL Alphabet 64×64)
    python Livestream.py --model optionB        # Landmark MLP  (no pixels)
    python Livestream.py --model cnn --camera 1

Controls:
    q  — quit
    s  — save screenshot
    l  — toggle landmark overlay
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MODEL_PATHS = {
    "cnn":     os.path.join(SCRIPT_DIR, "asl_model.tflite"),
    "optionA": os.path.join(SCRIPT_DIR, "webcam_trials", "models", "optionA.tflite"),
    "optionB": os.path.join(SCRIPT_DIR, "webcam_trials", "models", "optionB.pkl"),
}

# ── Label maps ────────────────────────────────────────────────────────────────
# CNN / Option B (25 classes — Sign MNIST, no J no Z)
ALPHABET_25 = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'
}
# Option A (26 classes — A-Z real photos)
ALPHABET_26 = {i: chr(65 + i) for i in range(26)}

# ── Hand skeleton ─────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
FINGERTIPS = {4, 8, 12, 16, 20}

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTH_N  = 7
CONF_THRESH = 0.5
PAD_FRAC  = 0.20


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_tflite(path):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def load_optionB(path):
    import joblib
    bundle = joblib.load(path)
    return bundle['mlp'], bundle['scaler'], bundle['classes']


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_cnn(crop_bgr, input_size):
    """Stretch histogram → resize → float32 [0,1]."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = gray.min(), gray.max()
    if hi > lo:
        gray = ((gray.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
    resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0, resized


def preprocess_optionA(crop_bgr, input_size=64):
    """RGB resize → float32 [0,1] for real-photo model."""
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0, resized


def landmarks_to_features(landmarks):
    """21 MediaPipe landmarks → 63-dim normalised feature vector."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                          # translate: wrist at origin
    scale = np.linalg.norm(pts[9])        # wrist→middle-MCP distance
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()                   # (63,)


# ── Inference ─────────────────────────────────────────────────────────────────
def infer_tflite(interp, inp_det, out_det, img_array, channels):
    if channels == 1:
        tensor = img_array.reshape(1, img_array.shape[0], img_array.shape[1], 1)
    else:
        tensor = img_array.reshape(1, img_array.shape[0], img_array.shape[1], 3)
    interp.set_tensor(inp_det[0]['index'], tensor)
    interp.invoke()
    probs    = interp.get_tensor(out_det[0]['index'])[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx]), probs


def infer_optionB(mlp, scaler, classes, features):
    feat_scaled = scaler.transform(features.reshape(1, -1))
    probs       = mlp.predict_proba(feat_scaled)[0]
    pred_idx    = int(np.argmax(probs))
    # classes is an array of label strings (e.g. ['A','B',...])
    letter      = classes[pred_idx] if pred_idx < len(classes) else '?'
    return letter, float(probs[pred_idx]), probs, classes


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks, h, w):
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 255), 2)
    for i, (x, y) in pts.items():
        r = 6 if i in FINGERTIPS else 3
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), r, (0, 180, 255), 1)


def draw_top5(frame, probs, label_map, origin=(10, 60)):
    if isinstance(label_map, np.ndarray):
        get_letter = lambda i: label_map[i] if i < len(label_map) else '?'
    else:
        get_letter = lambda i: label_map.get(i, '?')
    top5   = np.argsort(probs)[::-1][:5]
    ox, oy = origin
    for rank, idx in enumerate(top5):
        y      = oy + rank * 20
        conf   = float(probs[idx])
        color  = (0, 220, 80) if rank == 0 else (80, 140, 200)
        cv2.rectangle(frame, (ox, y), (ox + 120, y + 14), (40, 40, 40), -1)
        cv2.rectangle(frame, (ox, y), (ox + int(120 * conf), y + 14), color, -1)
        cv2.putText(frame, f"{get_letter(idx)} {conf*100:.1f}%",
                    (ox + 126, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def hand_bbox(landmarks, h, w):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    pad_x = (max(xs) - min(xs)) * PAD_FRAC
    pad_y = (max(ys) - min(ys)) * PAD_FRAC
    return (max(0, int(min(xs) - pad_x)), max(0, int(min(ys) - pad_y)),
            min(w, int(max(xs) + pad_x)), min(h, int(max(ys) + pad_y)))


# ── Startup model selection ───────────────────────────────────────────────────
def choose_model(arg_model):
    if arg_model:
        return arg_model
    print("\n┌─────────────────────────────────────────┐")
    print("│   ASL Recognition — Select Model        │")
    print("├─────────────────────────────────────────┤")
    print("│  1.  Robust CNN     (Sign MNIST  28×28) │")
    print("│  2.  Option A CNN   (Real photos 64×64) │")
    print("│  3.  Option B MLP   (Landmarks,  fast)  │")
    print("└─────────────────────────────────────────┘")
    mapping = {'1': 'cnn', '2': 'optionA', '3': 'optionB'}
    while True:
        choice = input("Enter 1 / 2 / 3: ").strip()
        if choice in mapping:
            return mapping[choice]
        print("Invalid — enter 1, 2, or 3.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  choices=['cnn', 'optionA', 'optionB'], default=None)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-landmarks', action='store_true')
    args = parser.parse_args()

    model_key  = choose_model(args.model)
    model_path = MODEL_PATHS[model_key]
    show_landmarks = not args.no_landmarks

    # Check model file exists
    if not os.path.exists(model_path):
        print(f"\nModel file not found: {model_path}")
        if model_key == 'optionA':
            print("Train first: bonus/webcam_trials/train_option_a.ipynb")
        elif model_key == 'optionB':
            print("Train first: bonus/webcam_trials/train_option_b.ipynb")
        sys.exit(1)

    # Download hand model if needed
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task (~8 MB)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

    # Load selected model
    print(f"\nLoading model: {model_key} — {model_path}")
    if model_key == 'cnn':
        interp, inp_det, out_det = load_tflite(model_path)
        input_size = inp_det[0]['shape'][1]   # 28
        label_map  = ALPHABET_25
        channels   = 1
        n_classes  = 25
    elif model_key == 'optionA':
        interp, inp_det, out_det = load_tflite(model_path)
        input_size = inp_det[0]['shape'][1]   # 64
        label_map  = ALPHABET_26
        channels   = inp_det[0]['shape'][3]   # 3 (RGB) or 1
        n_classes  = 26
    else:  # optionB
        mlp, scaler, classes = load_optionB(model_path)
        label_map  = classes
        n_classes  = len(classes)

    print(f"Model ready. Input: {input_size if model_key != 'optionB' else 'landmarks'}  Classes: {n_classes}")

    # MediaPipe hand detector
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
        sys.exit(f"Cannot open camera {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera opened.  q=quit  s=save  l=landmarks\n")

    recent_preds = []
    smoothed     = '?'
    last_probs   = np.ones(n_classes, dtype=np.float32) / n_classes
    fps, fps_timer, frame_count = 0.0, time.time(), 0
    timestamp_ms = 0

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

        if hand_detected:
            landmarks  = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name
            x1, y1, x2, y2 = hand_bbox(landmarks, h, w)

            letter, conf = '?', 0.0

            if model_key == 'optionB':
                features              = landmarks_to_features(landmarks)
                letter, conf, probs, _ = infer_optionB(mlp, scaler, classes, features)
                last_probs            = probs

            else:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    if handedness == 'Left':
                        crop = cv2.flip(crop, 1)

                    if model_key == 'cnn':
                        img_arr, preview_gray = preprocess_cnn(crop, input_size)
                    else:
                        img_arr, preview_rgb  = preprocess_optionA(crop, input_size)

                    pred_idx, conf, probs = infer_tflite(
                        interp, inp_det, out_det, img_arr, channels
                    )
                    last_probs = probs

                    if isinstance(label_map, dict):
                        letter = label_map.get(pred_idx, '?')
                    else:
                        letter = label_map[pred_idx] if pred_idx < len(label_map) else '?'

            recent_preds.append(letter)
            if len(recent_preds) > SMOOTH_N:
                recent_preds.pop(0)
            smoothed = max(set(recent_preds), key=recent_preds.count)

            box_color = (0, 255, 80) if conf >= CONF_THRESH else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, handedness[0], (x2 - 22, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 80), 2)

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

        draw_top5(frame, last_probs, label_map)

        model_label = {'cnn': 'Robust CNN', 'optionA': 'Option A (Real CNN)',
                       'optionB': 'Option B (Landmarks)'}[model_key]
        cv2.putText(frame,
                    f"FPS {fps:.1f}  [{model_label}]  Hand:{'YES' if hand_detected else 'NO'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(frame, "q=quit  s=save  l=landmarks",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps, fps_timer, frame_count = frame_count / (now - fps_timer), now, 0

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

    cap.release()
    hand_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
