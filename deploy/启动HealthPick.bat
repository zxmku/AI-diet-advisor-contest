@echo off
chcp 65001 >nul
title HealthPick 一键启动器
echo.
echo  HealthPick 健康优选 - 正在启动...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher.ps1"
pause
