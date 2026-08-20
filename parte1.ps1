# parte1.ps1 - esegue in sequenza i passi preparatori della Parte 1
# e salva tutto in logs\parte1.log
#
# Nota di implementazione: in PowerShell una funzione restituisce TUTTO cio'
# che finisce nel flusso di output, non solo il `return`. La versione
# precedente usava Tee-Object dentro una funzione, quindi l'intero output di
# Python entrava nel valore di ritorno e il confronto sul codice di uscita
# vedeva un array di centinaia di righe invece di un numero. Qui il codice di
# uscita viaggia in una variabile di script e l'output va esplicitamente a
# schermo con Out-Host.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venv = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venv)) {
    Write-Host " Manca l'ambiente virtuale. Lancia prima AVVIA.bat" -ForegroundColor Red
    Read-Host " Premi INVIO per chiudere"
    exit 1
}

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$script:log = Join-Path $logDir "parte1.log"
if (Test-Path $script:log) { Remove-Item $script:log -Force }
$script:codice = 0

function Passo {
    param([int]$num, [string]$titolo, [string[]]$argomenti)

    $sep = "=" * 70
    $intestazione = "`n$sep`n PASSO $num - $titolo`n$sep"
    Write-Host $intestazione -ForegroundColor Cyan
    Add-Content -Path $script:log -Value $intestazione

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $venv "-u" @argomenti 2>&1 | Tee-Object -FilePath $script:log -Append | Out-Host
    $script:codice = $LASTEXITCODE
    $sw.Stop()

    $esito = if ($script:codice -eq 0) { "OK" } else { "ERRORE (codice $script:codice)" }
    $riga = " -> $esito  in $([int]$sw.Elapsed.TotalSeconds)s"
    Write-Host $riga -ForegroundColor $(if ($script:codice -eq 0) { "Green" } else { "Red" })
    Add-Content -Path $script:log -Value $riga
}

Write-Host "==============================================================" -ForegroundColor White
Write-Host " PARTE 1 - passi preparatori" -ForegroundColor White
Write-Host " Il log completo finisce in logs\parte1.log" -ForegroundColor DarkGray
Write-Host "==============================================================" -ForegroundColor White

$passi = @(
    @{ n = 1; t = "Verifica ambiente e dataset"; a = @("verify_setup.py") },
    @{ n = 2; t = "Lettura delle annotazioni";   a = @("data.py", "--inspect") },
    @{ n = 3; t = "Statistiche bbox";            a = @("data.py", "--bbox-stats") },
    @{ n = 4; t = "Creazione degli split";       a = @("data.py", "--splits") },
    @{ n = 5; t = "Smoke test del pre-training"; a = @("train_ssl.py", "--smoke", "--epochs", "1", "--batch-size", "8") }
)

$fallito = $false
foreach ($p in $passi) {
    if ($p.n -eq 5) {
        Write-Host "`n Il primo avvio del training e' lento: Windows deve creare i" -ForegroundColor DarkGray
        Write-Host " processi di caricamento dati, e ognuno reimporta PyTorch." -ForegroundColor DarkGray
    }
    Passo $p.n $p.t $p.a
    if ($script:codice -ne 0) { $fallito = $true; break }
}

Write-Host "`n==============================================================" -ForegroundColor White
if ($fallito) {
    Write-Host " QUALCOSA SI E' FERMATO - vedi sopra" -ForegroundColor Yellow
} else {
    Write-Host " PARTE 1 PREPARATA" -ForegroundColor Green
}
Write-Host ""
Write-Host " Log completo:" -ForegroundColor White
Write-Host "   $script:log" -ForegroundColor Cyan
Write-Host ""
Write-Host " Mandalo cosi' com'e'." -ForegroundColor White
Write-Host "==============================================================" -ForegroundColor White
