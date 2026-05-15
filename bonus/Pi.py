"""
RASPBERRY PI SCRIPT — Real-time ASL inference using TFLite + USB camera.

Setup on Pi (run these once):
    sudo apt update
    sudo apt install -y python3-opencv libatlas-base-dev
    pip3 install tflite-runtime numpy

    # Get tflite-runtime wheel for Pi 4 (Python 3.9 — adjust if different):
    pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
    # OR simply: pip3 install tflite-runtime (works on Pi OS Bullseye+)

Copy files to Pi:
    scp asl_model_int8.tflite pi@<PI_IP>:~/asl/
    scp pi_inference.py       pi@<PI_IP>:~/asl/

Run:
    ssh pi@<PI_IP>
    cd ~/asl
    python3 pi_inference.py

Press 'q' to quit (if display connected), or Ctrl+C for headless mode.
"""

import cv2
import numpy as np
import time

# Use tflite_runtime on Pi (much lighter than full TF)
try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("⚠️  tflite_runtime not found, falling back to full TensorFlow")

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "asl_model_int8.tflite"   # INT8 is faster on Pi
CAMERA_INDEX  = 0                          # USB cam is usually /dev/video0
ROI_SIZE      = 240                        # smaller ROI for Pi performance
INPUT_SIZE    = 28
CONF_THRESH   = 0.6
SMOOTH_N      = 7                          # more smoothing — Pi may drop frames
HEADLESS      = False                      # set True if Pi has no display

ALPHABET = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'
}
IDX_TO_LETTER = {i: ALPHABET[k] for i, k in enumerate(sorted(ALPHABET.keys()))}

# ── Load model ───────────────────────────────────────────────────────────────
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model loaded. Input: {input_details[0]['shape']}, Output: {output_details[0]['shape']}")

def predict(roi_gray):
    img = cv2.resize(roi_gray, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, INPUT_SIZE, INPUT_SIZE, 1)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    probs    = np.exp(output) / np.sum(np.exp(output))
    pred_idx = np.argmax(probs)
    conf     = probs[pred_idx]
    return IDX_TO_LETTER[pred_idx], float(conf)

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)   # lower res for Pi
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"❌ Cannot open camera {CAMERA_INDEX}")
    exit()

print("✅ Camera ready. Starting inference loop...")

recent_preds  = []
fps_timer     = time.time()
frame_count   = 0
fps           = 0.0

# ── Benchmarking for paper ───────────────────────────────────────────────────
inference_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Frame grab failed")
        continue

    h, w = frame.shape[:2]
    x1 = w // 2 - ROI_SIZE // 2
    y1 = h // 2 - ROI_SIZE // 2
    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE

    # Clamp ROI to frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi      = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Time the inference (for paper benchmarking)
    t0 = time.time()
    letter, conf = predict(roi_gray)
    inference_ms = (time.time() - t0) * 1000
    inference_times.append(inference_ms)

    # Smoothing
    recent_preds.append(letter)
    if len(recent_preds) > SMOOTH_N:
        recent_preds.pop(0)
    smoothed = max(set(recent_preds), key=recent_preds.count)

    # FPS
    frame_count += 1
    elapsed = time.time() - fps_timer
    if elapsed >= 2.0:
        fps       = frame_count / elapsed
        fps_timer  = time.time()
        frame_count = 0
        avg_inf = sum(inference_times[-20:]) / len(inference_times[-20:])
        print(f"FPS: {fps:.1f} | Inference: {avg_inf:.1f} ms | Pred: {smoothed} ({conf*100:.1f}%)")

    if not HEADLESS:
        color = (0, 255, 0) if conf >= CONF_THRESH else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        pred_text = f"{smoothed} ({conf*100:.0f}%)" if conf >= CONF_THRESH else "?"
        cv2.putText(frame, pred_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2)
        cv2.putText(frame, f"FPS:{fps:.1f} | Inf:{inference_ms:.0f}ms",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("ASL - Pi Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Headless: just run, Ctrl+C to stop
        pass

# ── Print benchmark summary for paper ────────────────────────────────────────
if inference_times:
    print("\n📊 Benchmark Summary (for paper):")
    print(f"  Total frames processed : {len(inference_times)}")
    print(f"  Avg inference time     : {np.mean(inference_times):.2f} ms")
    print(f"  Min inference time     : {np.min(inference_times):.2f} ms")
    print(f"  Max inference time     : {np.max(inference_times):.2f} ms")
    print(f"  Avg FPS (inference)    : {1000/np.mean(inference_times):.1f}")

cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()