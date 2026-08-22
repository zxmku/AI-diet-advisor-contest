@echo off
chcp 65001 >nul
title HealthPick 一键启动
echo.
echo   HealthPick 健康优选 - 一键启动进入交互端...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\launcher.ps1"
pause
