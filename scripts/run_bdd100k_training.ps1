param(
    [Parameter(Mandatory=$true)]
    [string]$BddRoot,
    [string]$Dataset = "datasets\bdd100k_vehicle",
    [string]$Weights = "yolo26n.pt",
    [int]$MaxTrain = 10000,
    [int]$MaxVal = 2000,
    [int]$ImageSize = 320,
    [int]$Epochs = 30,
    [int]$Batch = 2,
    [string]$Output = "models\bdd_vehicle_yolo26n_320.onnx"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path "logs" "bdd100k_training_$stamp.log"

Start-Transcript -Path $log -Force
try {
    Write-Host "Repo: $RepoRoot"
    Write-Host "BDD root: $BddRoot"
    Write-Host "Dataset: $Dataset"
    Write-Host "Output: $Output"
    Write-Host "Started: $(Get-Date -Format o)"

    if (-not (Test-Path -LiteralPath $BddRoot)) {
        throw "BDD100K root not found: $BddRoot"
    }

    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        py -3 -m venv .venv
    }

    .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
    .\.venv\Scripts\python.exe -m pip install -r requirements-training.txt

    .\.venv\Scripts\python.exe tools\convert_bdd100k_vehicles.py `
        --bdd-root $BddRoot `
        --output $Dataset `
        --classes car,bus,truck `
        --max-train $MaxTrain `
        --max-val $MaxVal `
        --copy-mode hardlink `
        --overwrite

    .\.venv\Scripts\python.exe tools\train_vehicle_model.py `
        --data (Join-Path $Dataset "dataset.yaml") `
        --weights $Weights `
        --imgsz $ImageSize `
        --epochs $Epochs `
        --batch $Batch `
        --device cpu `
        --workers 0 `
        --name bdd_vehicle_yolo26n `
        --output $Output

    Write-Host "Finished: $(Get-Date -Format o)"
    Write-Host "Model: $Output"
}
finally {
    Stop-Transcript
}
