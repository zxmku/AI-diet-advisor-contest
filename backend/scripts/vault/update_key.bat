@echo off
cd /d "%~dp0"
if not exist "update_key.ps1" (
    echo ERROR: update_key.ps1 not found in this folder.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "update_key.ps1"
pause
