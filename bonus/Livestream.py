"""
LAPTOP SCRIPT — Real-time ASL inference using iVCam (iPhone) or any webcam.

Requirements:
    pip install tensorflow opencv-python numpy

Usage:
    python laptop_livestream.py
    
    - Make sure iVCam PC client is running and iPhone app is connected.
    - Press 'q' to quit.
    - Press 's' to save a screenshot.
"""

import cv2
import numpy as np
import tensorflow as tf
import time

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "asl_model.tflite"   # path to your converted model
CAMERA_INDEX = 0                     # try 0, 1, 2 until iVCam appears
ROI_SIZE     = 300                   # square ROI drawn on screen (pixels)
INPUT_SIZE   = 28                    # model input (28x28)
CONF_THRESH  = 0.6                   # minimum confidence to display prediction
SMOOTH_N     = 5                     # prediction smoothing over N frames

ALPHABET = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'
}
# Map model output index (0-24) → letter
IDX_TO_LETTER = ALPHABET  # output index = original dataset label (0-24, J=9 absent)

# ── Load TFLite model ───────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(roi_gray):
    """
    roi_gray: grayscale ROI, any size → resized to 28x28, normalized, inferred.
    Returns: (letter, confidence)
    """
    img = cv2.resize(roi_gray, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)  # (1, 28, 28, 1)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # (25,)

    probs = np.exp(output) / np.sum(np.exp(output))
    pred_idx  = np.argmax(probs)
    conf      = probs[pred_idx]
    letter    = IDX_TO_LETTER.get(pred_idx, '?')
    return letter, float(conf)

# ── Open camera ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Could not open camera index {CAMERA_INDEX}. Try changing CAMERA_INDEX.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Camera opened. Press 'q' to quit, 's' to save screenshot.")

# ── Smoothing buffer ────────────────────────────────────────────────────────
recent_preds = []
fps_timer    = time.time()
frame_count  = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── Draw ROI box in center ──────────────────────────────────────────────
    x1 = w // 2 - ROI_SIZE // 2
    y1 = h // 2 - ROI_SIZE // 2
    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE

    roi = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ── Inference ──────────────────────────────────────────────────────────
    letter, conf = predict(roi_gray)

    # Smooth predictions
    recent_preds.append(letter)
    if len(recent_preds) > SMOOTH_N:
        recent_preds.pop(0)
    smoothed = max(set(recent_preds), key=recent_preds.count)

    # ── Draw UI ────────────────────────────────────────────────────────────
    color = (0, 255, 0) if conf >= CONF_THRESH else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Show the 28x28 preprocessed patch (scaled up for visibility)
    preview = cv2.resize(roi_gray, (112, 112))
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    frame[y2 - 120: y2 - 8, x1: x1 + 112] = preview_bgr

    # Prediction text
    pred_text = f"{smoothed}  ({conf*100:.1f}%)" if conf >= CONF_THRESH else f"? ({conf*100:.1f}%)"
    cv2.putText(frame, pred_text, (x1, y1 - 15),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)

    # FPS counter
    frame_count += 1
    if time.time() - fps_timer >= 1.0:
        fps = frame_count / (time.time() - fps_timer)
        fps_timer   = time.time()
        frame_count = 0
    cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in dir() else "FPS: --",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.putText(frame, "Place hand in box", (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("ASL Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(fname, frame)
        print(f"📸 Saved {fname}")

cap.release()
cv2.destroyAllWindows()