param([int]$Width=256,[int]$NetworkSpp=256,[int]$ReferenceSpp=2048)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$pythonPath=if($env:MRPNN_PYTHON){$env:MRPNN_PYTHON}else{'D:\Hong\FacePoke\.venv\Scripts\python.exe'}
$header=(Resolve-Path 'training/runs/physical-smoke/NNWeight.cuh').Path.Replace('\','/')
cmake -S . -B build-5080 "-DMRPNN_FITTED_WEIGHT_HEADER=$header"
if($LASTEXITCODE -ne 0){throw 'Configure failed'}
cmake --build build-5080 --config Release --target RenderCompare FittedRenderCompare -j 8
if($LASTEXITCODE -ne 0){throw 'Render target build failed'}
& $pythonPath training/visualize.py --width $Width --nn-spp $NetworkSpp --mc-spp $ReferenceSpp
if($LASTEXITCODE -ne 0){throw 'Rendering failed'}
Write-Output 'Open training/runs/physical-smoke/render-comparison/render-comparison.html'
