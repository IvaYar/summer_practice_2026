$ErrorActionPreference = "Stop"

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

Write-Host ""
Write-Host "Dev environment is ready."
Write-Host "Export a model with: .\.venv\Scripts\python.exe tools\export_yolo_onnx.py --weights yolo26n.pt --imgsz 320 --no-end2end --output models\yolo26n_320_classic.onnx"
