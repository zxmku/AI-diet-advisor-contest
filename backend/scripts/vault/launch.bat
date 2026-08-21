@echo off
cd /d "%~dp0"
if not exist "launch.ps1" (
    echo ERROR: launch.ps1 not found in this folder.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "launch.ps1"
pause
