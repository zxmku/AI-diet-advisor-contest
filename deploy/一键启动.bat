@echo off
chcp 65001 >nul
title HealthPick 本地启动（兜底）
echo.
echo  ==============================================================
echo   ☁️  优先使用 README 顶部的「线上体验」链接（云端，免安装）
echo   本脚本仅作本地兜底：需本机已装 Python 3.10-3.12
echo   本地运行受环境差异影响，体验可能不如云端稳定
echo  ==============================================================
echo.
echo   HealthPick 健康优选 - 本地一键启动进入交互端...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher.ps1"
pause
