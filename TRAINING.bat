@echo off
title Pre-training I-JEPA - Parte 1
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0training.ps1"
echo.
echo  Premi un tasto per chiudere.
pause >nul
