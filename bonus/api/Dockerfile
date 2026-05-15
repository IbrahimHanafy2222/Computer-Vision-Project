FROM python:3.11-slim

WORKDIR /app

# System libs required by MediaPipe / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source + model files
COPY . .

# Tell main.py where models live (flat next to main.py in this image)
ENV MODELS_DIR=/app

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
