# CIE 552 — Computer Vision Project
## American Sign Language Recognition System

> **Live Demo:** https://ibrahimhanafy2222-vision-project.hf.space

---

## Team

| Name | ID | Email |
|---|---|---|
| Ibrahim Hanafy | 202200518 | s-ibrahim.hanafy@zewailcity.edu.eg |
| Abdullah Ahmed | 202200206 | s-abdullah.abdelsamiaa@zewailcity.edu.eg |
| Mai Ibrahim | 202200504 | s-mai.ibrahim@zewailcity.edu.eg |

**Course Instructor:** Dr. Mohamed Tolba  
**Teaching Assistants:** Retaj Yousri · Mennah Mohamed  
**Institution:** Zewail City of Science and Technology — Spring 2026

---

## Problem Statement

Over 70 million deaf and hard-of-hearing people worldwide rely on sign language as their primary means of communication. Automated ASL recognition can power real-time translation tools, assistive devices, and educational platforms — making the world more accessible. This project builds and compares multiple computer vision pipelines for recognising the 26 letters of the American Sign Language alphabet from still images and live webcam video.

---

## What Was Built

### Core Task — Model Comparison Pipeline

Five classification architectures were trained and rigorously compared on the **Sign Language MNIST** dataset (27,455 training / 7,172 test grayscale 28×28 images, 24 classes — J and Z excluded as they require motion):

| Model | Test Accuracy |
|---|---|
| Custom CNN (baseline) | — |
| CNN + Data Augmentation | best CNN result |
| SVM + HOG features | — |
| MobileNetV2 (feature extraction head) | — |
| EfficientNetB0 (feature extraction head) | — |
| EfficientNetB0 (fine-tuned) | — |

All confusion matrices, training curves, and a full metrics CSV are saved in `results/`.

#### Notebooks
- `notebooks/01_training_pipelines.ipynb` — trains all five models end-to-end
- `notebooks/02_evaluation_and_comparison.ipynb` — side-by-side evaluation, plots, metrics table

---

### Bonus A — Robust CNN for Real-World Webcam

**File:** `bonus/webcam_trials/train_option_a.ipynb`

The Sign MNIST dataset uses cropped, grayscale, perfectly-lit hand images. Real webcam frames look nothing like this (backgrounds, lighting variation, skin tone diversity). Bonus A addresses the domain gap by training a CNN with aggressive data augmentation:

- Random brightness / contrast / zoom / horizontal flip
- `RandomBrightness(factor, value_range=(0,1))` — corrected from default (0,255) which caused all-black images
- `tf.clip_by_value` post-augmentation to keep pixel values in [0,1]
- Exported to **TFLite** (`bonus/asl_model.tflite`) for fast inference

---

### Bonus B — Landmark-Based MLP (98.9% Accuracy)

**Files:** `bonus/webcam_trials/train_option_b.ipynb`, `bonus/Livestream.py`

Instead of classifying raw pixels, Bonus B uses **MediaPipe** to extract the 3D coordinates of 21 hand keypoints (landmarks), then classifies those 63 numbers with a scikit-learn MLP classifier:

1. MediaPipe detects and tracks the hand in each frame
2. 21 landmarks (x, y, z each) → 63-dimensional feature vector
3. Normalised: wrist moved to origin, scale divided by wrist→middle-MCP distance (makes it rotation/scale-invariant)
4. MLP trained on landmark features → **98.9% accuracy**, appearance-invariant
5. Model saved as `bonus/webcam_trials/models/optionB.pkl` (joblib bundle: mlp + scaler + classes)

`Livestream.py` demonstrates live inference: opens webcam, runs MediaPipe, classifies, overlays prediction on screen.

---

### Bonus C — Model Optimisation

**File:** `bonus/Optimization.ipynb`

Explores post-training optimisation strategies:
- TFLite conversion (float32 → int8 quantisation)
- Size/accuracy trade-off analysis

---

### Bonus D — FastAPI Web Application + Docker Deployment

**Files:** `bonus/api/`, deployed at https://ibrahimhanafy2222-vision-project.hf.space

A production-grade web application with two modes:

#### Upload Tab — CNN Classification
- User uploads any hand sign photo
- Server converts to 28×28 grayscale, runs TFLite CNN
- Returns top-5 predictions with confidence bars

#### Live Webcam Tab — MediaPipe + MLP
- Browser streams JPEG frames over WebSocket to server
- Server runs MediaPipe HandLandmarker → extracts 21 landmarks
- MLP classifies landmarks (preferred, 98.9%) with CNN as fallback
- Server returns: predicted letter, confidence, top-5, bounding box, 21 landmark coordinates
- Browser draws live skeleton overlay (joints + connections) and floating prediction badge

#### Tech Stack
- **FastAPI** — async Python web framework
- **Jinja2** — HTML templating
- **WebSocket** — real-time bidirectional frame streaming
- **MediaPipe Tasks** — hand detection and landmark extraction
- **TensorFlow Lite** — fast CNN inference
- **scikit-learn** — MLP classifier
- **Docker** — containerised deployment
- **Hugging Face Spaces** — free cloud hosting

---

## File Structure

```
Project/
├── README.md                          ← this file
├── Computer Vision Term Project.pdf   ← project brief
├── .gitignore
│
├── Data/                              ← Sign Language MNIST dataset (not in git)
│
├── notebooks/
│   ├── 01_training_pipelines.ipynb    ← train all 5 models
│   └── 02_evaluation_and_comparison.ipynb
│
├── results/                           ← confusion matrices, training curves, metrics
│   ├── metrics_table.csv
│   ├── cm_*.png
│   └── curves_*.png
│
└── bonus/
    ├── asl_model.tflite               ← exported CNN (Bonus A)
    ├── hand_landmarker.task           ← MediaPipe hand model binary
    ├── Livestream.py                  ← live demo script (Bonus B)
    ├── Optimization.ipynb             ← TFLite optimisation (Bonus C)
    │
    ├── webcam_trials/
    │   ├── train_option_a.ipynb       ← augmented CNN training (Bonus A)
    │   ├── train_option_b.ipynb       ← landmark MLP training (Bonus B)
    │   ├── train_robust_cnn.ipynb     ← robust CNN experiments
    │   └── models/
    │       └── optionB.pkl            ← trained MLP bundle
    │
    └── api/                           ← web app (Bonus D)
        ├── Dockerfile
        ├── requirements.txt
        ├── main.py                    ← FastAPI application
        ├── README.md                  ← HF Spaces deployment guide
        ├── templates/
        │   ├── base.html
        │   ├── index.html             ← homepage
        │   └── predict.html           ← prediction UI
        └── static/
            ├── css/style.css
            ├── js/app.js
            └── favicon.svg
```

---

## How to Run Locally

### Core Notebooks
```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn pillow opencv-python
jupyter notebook notebooks/01_training_pipelines.ipynb
```

### Live Webcam Demo (Bonus B)
```bash
pip install mediapipe scikit-learn opencv-python joblib
python bonus/Livestream.py
```

### Web Application (Bonus D)
```bash
cd bonus/api
pip install -r requirements.txt
python main.py
# Opens http://localhost:8000 automatically
```

---

## Results

All evaluation artifacts are in `results/`:
- `metrics_table.csv` — accuracy, precision, recall, F1 for all models
- `cm_*.png` — confusion matrices
- `curves_*.png` — training/validation accuracy and loss curves
- `sample_*.png` — misclassification examples
