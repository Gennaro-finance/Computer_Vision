@echo off
title Estrazione dataset periapicale
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0estrai_dataset.ps1"
echo.
echo  Premi un tasto per chiudere.
pause >nul
