$ErrorActionPreference = "Stop"

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

Write-Host ""
Write-Host "Dev environment is ready."
Write-Host "Export a model with: .\.venv\Scripts\python.exe tools\export_yolo_onnx.py --weights yolo11n.pt --imgsz 320 --output models\yolo11n_320.onnx"
