@echo off
title Parte 1 - passi preparatori
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0parte1.ps1"
echo.
echo  Premi un tasto per chiudere.
pause >nul
