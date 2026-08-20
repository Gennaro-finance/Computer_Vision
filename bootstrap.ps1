# bootstrap.ps1 - prepara l'ambiente locale in un comando solo.
#
# Uso (dalla cartella del progetto):
#     powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
#
# Idempotente: se lo rilanciate non rompe niente, riusa il venv esistente.
# Opzioni:
#     -SkipInstall   salta pip install (se avete gia' installato)
#     -Cpu           installa la build CPU di torch invece di quella CUDA

param(
    [switch]$SkipInstall,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Good($msg)     { Write-Host "  OK   $msg" -ForegroundColor Green }
function Bad($msg)      { Write-Host "  FAIL $msg" -ForegroundColor Red }
function Note($msg)     { Write-Host "       $msg" -ForegroundColor DarkGray }

Write-Host "==============================================================" -ForegroundColor White
Write-Host " Progetto 8 - Setup ambiente locale" -ForegroundColor White
Write-Host " $root" -ForegroundColor DarkGray
Write-Host "==============================================================" -ForegroundColor White

# ---------------------------------------------------------------- 1. Python
Step 1 "Cerco un interprete Python"

$py = $null
foreach ($cand in @("py -3.13", "py -3.12", "py -3.11", "py -3", "python")) {
    $parts = $cand.Split(" ")
    $exe = $parts[0]
    $args = if ($parts.Length -gt 1) { $parts[1..($parts.Length-1)] } else { @() }
    try {
        $v = & $exe @args -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) {
            $py = $cand
            Good "trovato $cand -> Python $v"
            break
        }
    } catch { }
}

if (-not $py) {
    Bad "Nessun Python trovato nel PATH."
    Note "Installatelo da python.org e rilanciate questo script."
    exit 1
}

# ------------------------------------------------------------------ 2. venv
Step 2 "Virtual environment"

$venvPy = Join-Path $root ".venv\Scripts\python.exe"

if (Test-Path $venvPy) {
    Good ".venv esiste gia', lo riuso"
} else {
    Note "creazione di .venv (qualche secondo)..."
    $parts = $py.Split(" ")
    & $parts[0] @($parts[1..($parts.Length-1)]) -m venv .venv
    if (-not (Test-Path $venvPy)) {
        Bad "creazione del venv fallita"
        exit 1
    }
    Good ".venv creato"
}

# -------------------------------------------------------------- 3. pacchetti
Step 3 "Dipendenze"

if ($SkipInstall) {
    Note "saltato (-SkipInstall)"
} else {
    & $venvPy -m pip install --upgrade pip --quiet
    Good "pip aggiornato"

    if ($Cpu) {
        Note "installo torch (build CPU)..."
        & $venvPy -m pip install torch torchvision --quiet
    } else {
        Note "installo torch con CUDA 12.6 (puo' richiedere qualche minuto, ~2.5 GB)..."
        & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --quiet
    }
    if ($LASTEXITCODE -ne 0) {
        Bad "installazione di torch fallita"
        Note "riprovate con:  .\bootstrap.ps1 -Cpu"
        exit 1
    }
    Good "torch installato"

    & $venvPy -m pip install numpy pillow matplotlib --quiet
    Good "numpy, pillow, matplotlib installati"

    Note "installo il supporto notebook (jupyter, ipykernel, nbconvert)..."
    & $venvPy -m pip install jupyter ipykernel nbconvert --quiet
    Good "supporto notebook installato"

    # Registra il kernel: cosi' il notebook in notebooks/ gira dentro PyCharm
    # usando l'interprete del progetto invece di uno di sistema.
    & $venvPy -m ipykernel install --user --name cv-periapical-jepa `
        --display-name "Python (cv-periapical-jepa)" 2>$null | Out-Null
    Good "kernel Jupyter registrato come 'Python (cv-periapical-jepa)'"
}

# ---------------------------------------------------------------- 4. cartelle
Step 4 "Cartelle di lavoro"

foreach ($d in @("data\periapical", "runs", "notebooks")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
Good "data\, runs\, notebooks\ pronte"

# ------------------------------------------------------- 4b. config PyCharm
Step "4b" "Configurazioni di avvio per PyCharm"

$src = Join-Path $root "ide\pycharm\runConfigurations"
$dst = Join-Path $root ".idea\runConfigurations"

if (Test-Path $src) {
    if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Path $dst -Force | Out-Null }
    Copy-Item "$src\*.xml" $dst -Force
    $n = (Get-ChildItem "$dst\*.xml").Count
    Good "$n configurazioni copiate in .idea\runConfigurations"
    Note "compariranno nel menu in alto a destra di PyCharm"
    Note "(se PyCharm era gia' aperto, chiudetelo e riapritelo)"
} else {
    Note "cartella ide\pycharm non trovata, salto"
}

# ---------------------------------------------------------------- 5. verifica
Step 5 "Verifica dell'ambiente"
Write-Host ""

& $venvPy verify_setup.py
$verifyCode = $LASTEXITCODE

# ------------------------------------------------------------------ epilogo
Write-Host ""
Write-Host "==============================================================" -ForegroundColor White

if ($verifyCode -eq 0) {
    Write-Host " SETUP COMPLETATO" -ForegroundColor Green
} else {
    Write-Host " SETUP INCOMPLETO - vedi sopra" -ForegroundColor Yellow
    Write-Host " (se manca solo il dataset e' normale: e' il passo successivo)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host " Per lavorare da terminale, attivate il venv:" -ForegroundColor White
Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host " In PyCharm, impostate l'interprete su:" -ForegroundColor White
Write-Host "     $venvPy" -ForegroundColor Cyan
Write-Host "     (Settings > Project > Python Interpreter > Add Local > Existing)" -ForegroundColor DarkGray
Write-Host "==============================================================" -ForegroundColor White

exit $verifyCode
