param(
    [int]$Steps = 600,
    [int]$Batch = 256,
    [double]$LearningRate = 0.0003,
    [string]$Output = 'training/runs/physical-smoke'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$pythonPath = if ($env:MRPNN_PYTHON) { $env:MRPNN_PYTHON } else {
    'D:\Hong\FacePoke\.venv\Scripts\python.exe'
}
if (!(Test-Path $pythonPath)) { throw 'Set MRPNN_PYTHON to Python with PyTorch, NumPy and matplotlib.' }
& $pythonPath training/fit.py --steps $Steps --batch $Batch --lr $LearningRate --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Physical fit or checkpoint export failed.' }
$header = (Resolve-Path (Join-Path $Output 'NNWeight.cuh')).Path.Replace('\', '/')
cmake -S . -B build-5080 "-DMRPNN_FITTED_WEIGHT_HEADER=$header"
if ($LASTEXITCODE -ne 0) { throw 'Fitted CUDA target configure failed.' }
cmake --build build-5080 --config Release --target FittedNetworkProbe NetworkProbe -j 8
if ($LASTEXITCODE -ne 0) { throw 'Fitted CUDA target build failed.' }
& $pythonPath training/verify_export.py --run $Output
if ($LASTEXITCODE -ne 0) { throw 'Fitted CUDA export verification failed.' }
& training/verify.ps1
if ($LASTEXITCODE -ne 0) { throw 'Published weight regression failed.' }
