# Catena di esperimenti: I-JEPA PURO, come il progetto lo aveva dichiarato.
#
# PERCHE' ESISTE. Nessuno dei sette esperimenti fatti fino al 22 ago girava
# su una configurazione pulita: tutti avevano dentro almeno una modifica
# introdotta strada facendo (SIGReg aggiunto, learning rate e momentum
# cambiati reagendo a un falso allarme, un terzo delle epoche, batch con 32
# immagini distinte invece di 128). Quelle sette ablation esplorano quindi il
# vicinato di UNA CONFIGURAZIONE MODIFICATA, non di quella del progetto.
# Finche' questo run non gira, alla domanda "cosa fa I-JEPA su questo
# dataset?" non sappiamo rispondere.
#
# CONFIGURAZIONE: 300 epoche, SIGREG_LAMBDA=0, SSL_LR=1.5e-4,
# SSL_EMA_START=0.996, PREDICTOR_DIM=96, CROPS_PER_ITEM=1. Restano solo le
# correzioni di DIFETTI: crop a scala preservata, soglie della guardia
# ricalibrate, salvataggio atomico, fix del flip e della memoria nel probe.
#
# PROTEZIONE HARDWARE. La macchina e' un portatile che il 21 ago si e' spento
# tre volte durante i nostri addestramenti (Kernel-Power 41 alle 16:46, 20:34
# e 23:55). Qui: 12 worker invece di 16, guardiano termico prima di ogni
# epoca, e RIPRESA AUTOMATICA - se il pre-training si interrompe si rilancia
# da solo con --resume, fino a 6 tentativi. Con i checkpoint atomici una
# interruzione costa un'epoca, non il run.
#
# Uso:  powershell -File catena.ps1 [-OraAvvio 1300] [-Subito]

param([int]$OraAvvio = 1300, [switch]$Subito)

Set-Location "C:\Users\39346\PycharmProjects\cv-periapical-jepa"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$py = ".\.venv\Scripts\python.exe"
$L = @("2", "7", "11")

if (-not $Subito) {
    while ([int](Get-Date -Format "HHmm") -lt $OraAvvio) { Start-Sleep -Seconds 60 }
}
"### AVVIO $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# ---- 1. metrica che il Task prescrive, mai misurata. Gira sui latenti gia'
#         cachati: costa poco e puo' cambiare le conclusioni.
"`n### [1/5] PR-AUC su PAI 5 - metrica del Task per la classe minoritaria"
& $py -u exp_scala.py --no-geom-only

# ---- 2. il run che manca al progetto
"`n### [2/5] I-JEPA PURO - 300 epoche, configurazione dichiarata"
$tentativo = 0
while ($tentativo -lt 6) {
    if ($tentativo -eq 0) {
        & $py -u train_ssl.py --variant vit_small --epochs 300 --batch-size 128 --tag puro
    } else {
        "--- ripresa $tentativo dopo interruzione $(Get-Date -Format 'HH:mm')"
        & $py -u train_ssl.py --variant vit_small --epochs 300 --batch-size 128 --tag puro --resume
    }
    if ($LASTEXITCODE -eq 0) { break }
    $tentativo++
    Start-Sleep -Seconds 120     # lascia raffreddare prima di riprovare
}

# ---- 3. latenti dall'encoder puro
"`n### [3/5] latenti dall'encoder puro"
& $py -u train_downstream.py --cache --arm ijepa --variant vit_small --layers @L --ckpt-tag puro

# ---- 4. il deliverable dell'obiettivo 4
"`n### [4/5] OBIETTIVO 4 - griglia completa"
& $py -u train_downstream.py --grid --arm ijepa --variant vit_small --layers @L

# ---- 5. ablation della novita'
"`n### [5/5] ablation alpha della novita'"
& $py -u train_downstream.py --sweep-alpha --arm ijepa --variant vit_small --layers @L

"`n### FINITO $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
