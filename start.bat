@echo off
cd /d "%~dp0"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
set "SCENE=%~1"
if not defined SCENE set "SCENE=TestCase\CLOUD03"
if not exist "build-5080\Release\VolumeRender.exe" (
    echo Build VolumeRender first.
    pause
    exit /b 1
)
"build-5080\Release\VolumeRender.exe" "%SCENE%"
if errorlevel 1 pause
