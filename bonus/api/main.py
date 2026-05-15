import io
import os
import sys
import threading
import asyncio
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR  = Path(__file__).parent
# MODELS_DIR env var lets HF Spaces / Docker point to a flat model folder.
# Default: one level up (local dev layout inside bonus/api/).
_MDIR     = Path(os.environ.get("MODELS_DIR", str(BASE_DIR.parent)))
MODEL_PATH    = _MDIR / "asl_model.tflite"
HAND_MODEL    = _MDIR / "hand_landmarker.task"
_OB_FLAT      = _MDIR / "optionB.pkl"
_OB_NESTED    = _MDIR / "webcam_trials" / "models" / "optionB.pkl"
OPTION_B_PATH = _OB_FLAT if _OB_FLAT.exists() else _OB_NESTED

ALPHABET = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y',
}

# ── CNN (upload tab) ────────────────────────────────────────────────────
_interp     = None
_inp        = None
_out        = None
_model_lock = threading.Lock()

# ── MLP / Option B (live webcam tab) ────────────────────────────────────
_mlp        = None
_mlp_scaler = None
_mlp_classes = None


def _load_cnn():
    global _interp, _inp, _out
    if not MODEL_PATH.exists():
        print(f"WARNING: CNN model not found at {MODEL_PATH}", file=sys.stderr)
        return
    _interp = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    _interp.allocate_tensors()
    _inp = _interp.get_input_details()
    _out = _interp.get_output_details()
    print(f"Loaded CNN: {MODEL_PATH.name}")


def _load_mlp():
    global _mlp, _mlp_scaler, _mlp_classes
    if not OPTION_B_PATH.exists():
        print("INFO: optionB.pkl not found — live webcam will use CNN fallback", file=sys.stderr)
        return
    try:
        import joblib
        bundle = joblib.load(OPTION_B_PATH)
        _mlp        = bundle["mlp"]
        _mlp_scaler = bundle["scaler"]
        _mlp_classes = bundle["classes"]
        print(f"Loaded MLP: {OPTION_B_PATH.name}  ({len(_mlp_classes)} classes)")
    except Exception as exc:
        print(f"WARNING: could not load MLP: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_cnn()
    _load_mlp()
    yield


app = FastAPI(title="ASL Vision API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ── Page routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active": "home"})


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "active": "predict"})


# ── Upload → CNN ─────────────────────────────────────────────────────────

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if _interp is None:
        raise HTTPException(503, "Model not loaded — place asl_model.tflite in bonus/")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Upload must be an image file")

    raw = await file.read()
    try:
        img = PILImage.open(io.BytesIO(raw)).convert("L").resize((28, 28), PILImage.LANCZOS)
    except Exception as exc:
        raise HTTPException(400, f"Cannot decode image: {exc}")

    arr = (np.array(img, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)
    with _model_lock:
        _interp.set_tensor(_inp[0]["index"], arr)
        _interp.invoke()
        probs = _interp.get_tensor(_out[0]["index"])[0].copy()

    ranked = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)
    top5 = [
        {"letter": ALPHABET[i], "confidence": round(p * 100, 2)}
        for i, p in ranked if i in ALPHABET
    ][:5]

    return JSONResponse({
        "letter":     top5[0]["letter"],
        "confidence": top5[0]["confidence"],
        "top5":       top5,
    })


# ── WebSocket livestream: MediaPipe → landmarks → MLP (or CNN fallback) ──

def _make_hand_detector():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _landmarks_to_features(lms) -> np.ndarray:
    """21 landmarks → 63-dim normalised vector (same as Livestream.py optionB)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    pts -= pts[0]                      # wrist at origin
    scale = np.linalg.norm(pts[9])    # wrist → middle-MCP distance
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def _classify_mlp(lms) -> list:
    feats  = _landmarks_to_features(lms).reshape(1, -1)
    scaled = _mlp_scaler.transform(feats)
    probs  = _mlp.predict_proba(scaled)[0]
    ranked = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)
    return [
        {"letter": str(_mlp_classes[i]), "confidence": round(p * 100, 1)}
        for i, p in ranked
    ][:5]


def _classify_cnn(frame_rgb: np.ndarray, x1, y1, x2, y2) -> list:
    h, w   = frame_rgb.shape[:2]
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    crop = frame_rgb[py1:py2, px1:px2]
    arr  = (
        np.array(
            PILImage.fromarray(crop).convert("L").resize((28, 28), PILImage.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    ).reshape(1, 28, 28, 1)
    with _model_lock:
        _interp.set_tensor(_inp[0]["index"], arr)
        _interp.invoke()
        probs = _interp.get_tensor(_out[0]["index"])[0].copy()
    ranked = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)
    return [
        {"letter": ALPHABET[i], "confidence": round(p * 100, 1)}
        for i, p in ranked if i in ALPHABET
    ][:5]


def _process_frame(jpeg_bytes: bytes, detector) -> dict:
    try:
        img_pil   = PILImage.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        frame_rgb = np.array(img_pil, dtype=np.uint8)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_img)

        if not result.hand_landmarks:
            return {"hand_detected": False}

        lms = result.hand_landmarks[0]

        # Bounding box from landmarks
        xs  = [lm.x for lm in lms]
        ys  = [lm.y for lm in lms]
        pad = 0.1
        x1  = max(0.0, min(xs) - pad)
        y1  = max(0.0, min(ys) - pad)
        x2  = min(1.0, max(xs) + pad)
        y2  = min(1.0, max(ys) + pad)

        # Classify: MLP preferred (uses landmarks directly), CNN fallback
        if _mlp is not None:
            top5  = _classify_mlp(lms)
            model = "MLP"
        elif _interp is not None:
            top5  = _classify_cnn(frame_rgb, x1, y1, x2, y2)
            model = "CNN"
        else:
            return {"hand_detected": False, "error": "No model loaded"}

        # Return landmarks as [[x,y], ...] for client-side skeleton drawing
        landmarks = [[lm.x, lm.y] for lm in lms]

        return {
            "hand_detected": True,
            "letter":        top5[0]["letter"],
            "confidence":    top5[0]["confidence"],
            "top5":          top5,
            "bbox":          {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "landmarks":     landmarks,   # 21 × [x, y] normalised 0-1
            "model":         model,
        }

    except Exception as exc:
        return {"hand_detected": False, "error": str(exc)}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    if _interp is None and _mlp is None:
        await websocket.close(code=1011, reason="No model loaded")
        return

    if not HAND_MODEL.exists():
        await websocket.accept()
        await websocket.send_json({"error": "hand_landmarker.task not found in bonus/"})
        await websocket.close()
        return

    await websocket.accept()
    detector = _make_hand_detector()
    loop     = asyncio.get_event_loop()

    try:
        while True:
            jpeg = await websocket.receive_bytes()
            out  = await loop.run_in_executor(None, _process_frame, jpeg, detector)
            await websocket.send_json(out)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        detector.close()


if __name__ == "__main__":
    import uvicorn, threading, webbrowser
    url = "http://localhost:8000"
    threading.Timer(1.5, webbrowser.open, args=[url]).start()
    print(f"\n  \033[1;36mASL Vision\033[0m  →  \033[4;94m{url}\033[0m\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
