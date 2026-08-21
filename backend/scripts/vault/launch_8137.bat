@echo off
cd /d "%~dp0"
if not exist "launch_8137.ps1" (
    echo ERROR: launch_8137.ps1 缺失
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "launch_8137.ps1"
pause
