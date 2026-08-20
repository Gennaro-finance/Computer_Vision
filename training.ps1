# training.ps1 - avvia il pre-training vero (Parte 1)
#
# Riprende automaticamente da dove si era interrotto: se il PC si spegne o
# chiudi la finestra, rilancia e riparte dall'ultimo checkpoint.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venv = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venv)) {
    Write-Host " Manca l'ambiente virtuale. Lancia prima AVVIA.bat" -ForegroundColor Red
    Read-Host " Premi INVIO per chiudere"; exit 1
}

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$log = Join-Path $logDir "training.log"

Write-Host "==============================================================" -ForegroundColor White
Write-Host " PRE-TRAINING - Parte 1" -ForegroundColor White
Write-Host "==============================================================" -ForegroundColor White
Write-Host ""
Write-Host " Cosa guardare mentre gira:" -ForegroundColor White
Write-Host "   rango=.../...  (...% del sano)" -ForegroundColor Cyan
Write-Host "     deve restare ALTO. Se scende sotto il 15% e ci resta," -ForegroundColor DarkGray
Write-Host "     lo script si ferma da solo: e' il collasso." -ForegroundColor DarkGray
Write-Host ""
Write-Host "   [k-NN probe] ogni 20 epoche" -ForegroundColor Cyan
Write-Host "     deve superare il valore della maggioritaria. E' il segnale" -ForegroundColor DarkGray
Write-Host "     che le rappresentazioni servono davvero a qualcosa." -ForegroundColor DarkGray
Write-Host ""
Write-Host "   La loss che scende NON basta: puo' scendere anche mentre" -ForegroundColor DarkGray
Write-Host "   il modello collassa." -ForegroundColor DarkGray
Write-Host ""
Write-Host " Puoi interrompere quando vuoi con Ctrl+C: i checkpoint sono" -ForegroundColor White
Write-Host " salvati a ogni epoca e rilanciando riprende da li'." -ForegroundColor White
Write-Host ""
Write-Host " Log: $log" -ForegroundColor DarkGray
Write-Host "==============================================================" -ForegroundColor White
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $venv "-u" "train_ssl.py" "--resume" 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
$sw.Stop()

Write-Host ""
Write-Host "==============================================================" -ForegroundColor White
Write-Host (" Sessione terminata dopo {0:hh\:mm\:ss}" -f $sw.Elapsed) -ForegroundColor White
Write-Host " Per riprendere: rilancia questo stesso file." -ForegroundColor White
Write-Host " Log completo: $log" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor White
