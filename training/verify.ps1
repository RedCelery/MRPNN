param([int]$Samples = 256)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$pythonPath = if ($env:MRPNN_PYTHON) { $env:MRPNN_PYTHON } else {
    'D:\Hong\FacePoke\.venv\Scripts\python.exe'
}
if (!(Test-Path $pythonPath)) {
    throw 'Set MRPNN_PYTHON to a Python interpreter with PyTorch and NumPy.'
}
& $pythonPath training\verify.py --samples $Samples
if ($LASTEXITCODE -ne 0) { throw "Verification failed: $LASTEXITCODE" }
