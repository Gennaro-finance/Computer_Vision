@echo off
setlocal enabledelayedexpansion
title Setup progetto Computer Vision - Progetto 8
color 0F
cd /d "%~dp0"

echo ==============================================================
echo  SETUP PROGETTO 8 - Computer Vision
echo  %~dp0
echo ==============================================================
echo.
echo  Questo script prepara tutto da solo. Puo' richiedere qualche
echo  minuto: la parte lunga e' scaricare PyTorch, circa 2.5 GB.
echo  Non chiudere questa finestra.
echo.

REM ---------------------------------------------------------------- Python
echo [1/6] Cerco Python...
set "PYCMD="
for %%P in ("py -3.13" "py -3.12" "py -3.11" "py -3" "python") do (
    if not defined PYCMD (
        %%~P -c "import sys" >nul 2>&1
        if !errorlevel! equ 0 set "PYCMD=%%~P"
    )
)

if not defined PYCMD (
    echo    ERRORE: nessun Python trovato.
    echo    Installalo da python.org e rilancia questo file.
    goto :fine
)
for /f "delims=" %%V in ('%PYCMD% -c "import sys;print('%%d.%%d.%%d'%%sys.version_info[:3])"') do set "PYVER=%%V"
echo    OK - %PYCMD% versione !PYVER!
echo.

REM ------------------------------------------------------------------ venv
echo [2/6] Ambiente virtuale...
if exist ".venv\Scripts\python.exe" (
    echo    OK - .venv esiste gia, lo riuso
) else (
    echo    creazione in corso...
    %PYCMD% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo    ERRORE: creazione del venv fallita.
        goto :fine
    )
    echo    OK - .venv creato
)
set "VPY=%~dp0.venv\Scripts\python.exe"
echo.

REM ------------------------------------------------------------- pacchetti
echo [3/6] Aggiorno pip...
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
echo    OK
echo.

echo [4/6] Installo PyTorch con CUDA 12.6 - questa e' la parte lunga...
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --quiet --disable-pip-version-check
if !errorlevel! neq 0 (
    echo    La build CUDA non e' andata. Provo quella CPU...
    "%VPY%" -m pip install torch torchvision --quiet --disable-pip-version-check
    if !errorlevel! neq 0 (
        echo    ERRORE: installazione di PyTorch fallita.
        echo    Controlla la connessione a internet e rilancia.
        goto :fine
    )
    echo    OK - installata la build CPU
) else (
    echo    OK - PyTorch con CUDA installato
)
echo.

echo [5/6] Installo le altre dipendenze...
"%VPY%" -m pip install numpy pillow matplotlib --quiet --disable-pip-version-check
"%VPY%" -m pip install jupyter ipykernel nbconvert --quiet --disable-pip-version-check
"%VPY%" -m ipykernel install --user --name cv-periapical-jepa --display-name "Python (cv-periapical-jepa)" >nul 2>&1
echo    OK - numpy, pillow, matplotlib, jupyter
echo.

REM ------------------------------------------------- cartelle e config IDE
echo [6/6] Cartelle e configurazioni PyCharm...
if not exist "data\periapical" mkdir "data\periapical"
if not exist "runs" mkdir "runs"
if not exist "notebooks" mkdir "notebooks"
if exist "ide\pycharm\runConfigurations" (
    if not exist ".idea\runConfigurations" mkdir ".idea\runConfigurations"
    copy /Y "ide\pycharm\runConfigurations\*.xml" ".idea\runConfigurations\" >nul 2>&1
    echo    OK - configurazioni copiate in .idea
) else (
    echo    cartella ide\pycharm assente, salto
)
echo.

REM -------------------------------------------------------------- verifica
echo ==============================================================
echo  VERIFICA DELL'AMBIENTE
echo ==============================================================
echo.
"%VPY%" verify_setup.py
echo.

echo ==============================================================
echo  FATTO
echo ==============================================================
echo.
echo  Interprete da impostare in PyCharm:
echo    %~dp0.venv\Scripts\python.exe
echo.
echo  Settings ^> Project ^> Python Interpreter ^> Add Local ^> Existing
echo.
echo  Se PyCharm era aperto, chiudilo e riaprilo per vedere
echo  le configurazioni di avvio numerate da 1 a 8.
echo.

:fine
echo.
echo  Premi un tasto per chiudere questa finestra.
pause >nul
endlocal
