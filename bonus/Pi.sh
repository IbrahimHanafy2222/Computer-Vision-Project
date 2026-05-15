#!/bin/bash
# ── Raspberry Pi 4B Setup Script ─────────────────────────────────────────────
# Run this once on your Pi after a fresh Pi OS install.
# Tested on: Raspberry Pi OS Bullseye (64-bit), Python 3.9+

echo "🔧 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "📦 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libc-ares-dev \
    libeigen3-dev \
    libopenblas-dev \
    v4l-utils

echo "📦 Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy

# Install tflite-runtime (much lighter than full TensorFlow)
echo "📦 Installing tflite-runtime..."
pip3 install tflite-runtime

# Verify camera is detected
echo "🎥 Checking for USB camera..."
v4l2-ctl --list-devices || echo "⚠️  v4l2-ctl not found or no camera detected"
ls /dev/video* 2>/dev/null || echo "⚠️  No /dev/video* devices found — plug in USB cam"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy your model and script to the Pi:"
echo "     scp asl_model_int8.tflite pi@<PI_IP>:~/asl/"
echo "     scp pi_inference.py       pi@<PI_IP>:~/asl/"
echo ""
echo "  2. SSH into Pi and run:"
echo "     cd ~/asl && python3 pi_inference.py"
echo ""
echo "  3. If camera index 0 doesn't work, try 1 or 2 in the script config."