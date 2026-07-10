param(
    [string]$Video = "C:\Users\jarom\Desktop\car_project\vehicle-oncoming-detector\test_videos\own_test.mp4",
    [string]$Dataset = "datasets\vehicle_pseudo",
    [string]$Teacher = "yolo26n.pt",
    [int]$LabelImageSize = 640,
    [double]$LabelConfidence = 0.18,
    [int]$SampleEvery = 15,
    [int]$MaxFrames = 800,
    [int]$TrainImageSize = 320,
    [int]$Epochs = 40,
    [int]$Batch = 2,
    [string]$Output = "models\vehicle_yolo26n_320.onnx"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path "logs" "vehicle_training_$stamp.log"

Start-Transcript -Path $log -Force
try {
    Write-Host "Repo: $RepoRoot"
    Write-Host "Video: $Video"
    Write-Host "Dataset: $Dataset"
    Write-Host "Output: $Output"
    Write-Host "Started: $(Get-Date -Format o)"

    if (-not (Test-Path -LiteralPath $Video)) {
        throw "Video file not found: $Video"
    }

    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        py -3 -m venv .venv
    }

    .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
    .\.venv\Scripts\python.exe -m pip install -r requirements-training.txt

    .\.venv\Scripts\python.exe tools\build_vehicle_dataset.py `
        --video $Video `
        --output $Dataset `
        --teacher $Teacher `
        --label-imgsz $LabelImageSize `
        --conf $LabelConfidence `
        --sample-every $SampleEvery `
        --max-frames $MaxFrames `
        --overwrite

    .\.venv\Scripts\python.exe tools\train_vehicle_model.py `
        --data (Join-Path $Dataset "dataset.yaml") `
        --weights $Teacher `
        --imgsz $TrainImageSize `
        --epochs $Epochs `
        --batch $Batch `
        --device cpu `
        --output $Output

    Write-Host "Finished: $(Get-Date -Format o)"
    Write-Host "Model: $Output"
}
finally {
    Stop-Transcript
}
