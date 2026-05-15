---
title: ASL Vision
emoji: 🤟
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# ASL Vision — Hugging Face Spaces Deployment

FastAPI web app for real-time American Sign Language recognition.
Live webcam tab uses MediaPipe landmarks + sklearn MLP (98.9% accuracy).
Upload tab uses a TFLite CNN (Sign MNIST, 28×28 grayscale).

---

## File Structure Required in the HF Space Repo

```
README.md                  ← this file (must be at root with frontmatter)
Dockerfile                 ← bonus/api/Dockerfile
requirements.txt           ← bonus/api/requirements.txt
main.py                    ← bonus/api/main.py
templates/                 ← bonus/api/templates/
static/                    ← bonus/api/static/
asl_model.tflite           ← bonus/asl_model.tflite
hand_landmarker.task       ← bonus/hand_landmarker.task
optionB.pkl                ← bonus/webcam_trials/models/optionB.pkl
```

---

## Step-by-Step Deployment

### 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Give it a name (e.g. `asl-vision`)
3. **SDK → Docker**
4. Set visibility to **Public**
5. Click **Create Space**

### 2 — Clone the Space repo

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/asl-vision
cd asl-vision
```

### 3 — Copy files into the repo

From your project root:

```bash
# App source
cp bonus/api/Dockerfile         .
cp bonus/api/requirements.txt   .
cp bonus/api/main.py            .
cp -r bonus/api/templates       .
cp -r bonus/api/static          .

# Model files (flat, next to main.py)
cp bonus/asl_model.tflite       .
cp bonus/hand_landmarker.task   .
cp bonus/webcam_trials/models/optionB.pkl .
```

Create the README (this file must be at root with the YAML frontmatter above).

### 4 — Commit and push

```bash
git add .
git commit -m "Deploy ASL Vision"
git push
```

HF Spaces will automatically build the Docker image and deploy.
Build takes ~5–10 minutes (downloading TensorFlow + MediaPipe).

### 5 — Access the app

Once the build goes green, your app is live at:
```
https://YOUR_USERNAME-asl-vision.hf.space
```

---

## Local Development

```bash
cd bonus/api
pip install -r requirements.txt
python main.py
# Opens http://localhost:8000 automatically
```

---

## Notes

- **WebSocket**: HF Spaces Docker supports WebSocket — the Live Webcam tab works.
- **Port**: HF Spaces Docker requires port `7860`. The `Dockerfile` sets this. Local dev uses `8000`.
- **optionB.pkl version**: if you get a sklearn/numpy version warning on load, retrain in a local Colab-matching environment or `pip install scikit-learn==1.6.1`.
- **Cold start**: first request after inactivity may be slow (model loading). Subsequent requests are fast.
