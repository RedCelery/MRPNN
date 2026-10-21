param(
    [int]$Samples = 1000,
    [int]$LowSpp = 512,
    [int]$HighSpp = 4096,
    [int]$Bounces = 2048,
    [string]$Output = 'training/datasets/smoke'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$pythonPath = if ($env:MRPNN_PYTHON) { $env:MRPNN_PYTHON } else {
    'D:\Hong\FacePoke\.venv\Scripts\python.exe'
}
if (!(Test-Path $pythonPath)) { throw 'Set MRPNN_PYTHON to Python with PyTorch and NumPy.' }
cmake --build build-5080 --config Release --target DatasetGen -j 8
if ($LASTEXITCODE -ne 0) { throw 'DatasetGen build failed.' }
& $pythonPath training/generate_dataset.py --samples $Samples --spp $LowSpp $HighSpp `
    --bounces $Bounces --output $Output
if ($LASTEXITCODE -ne 0) { throw "Dataset generation/validation failed: $LASTEXITCODE" }
