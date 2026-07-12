#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  python3-venv \
  python3-picamera2 \
  python3-opencv \
  python3-numpy \
  v4l-utils \
  libopenblas0

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-runtime.txt

echo
echo "Runtime is ready."
echo "Activate it with: source .venv/bin/activate"
echo "Export a model with: python tools/export_yolo_onnx.py --weights yolo26n.pt --imgsz 320 --no-end2end --output models/yolo26n_320_classic.onnx"
