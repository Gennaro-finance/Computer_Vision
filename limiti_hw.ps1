# Limiti hardware per far girare gli esperimenti senza spegnimenti.
#
# DA ESEGUIRE IN UN POWERSHELL DA AMMINISTRATORE. Senza elevazione
# nvidia-smi risponde "the current user does not have permission to change
# clocks" e non fa niente.
#
# --------------------------------------------------------------------------
# PERCHE' IL CLOCK E NON LA POTENZA
#
# La strada pulita sarebbe abbassare il limite di potenza:
#     nvidia-smi -pl 90
# ma su questa scheda risponde "not supported in current scope": il
# produttore l'ha bloccato nel vBIOS e l'elevazione non lo sblocca. Il
# limite corrente e' 175 W contro un PREDEFINITO di 60 W, quindi la scheda
# gira a quasi tre volte il suo consumo di riferimento.
#
# Il blocco del clock invece risponde "the current user does not have
# permission", che e' un problema di PERMESSI, non di scope: da
# amministratore funziona. E la potenza segue il clock circa come f * V^2,
# con V che a sua volta cresce col clock: dimezzare il clock taglia la
# potenza molto piu' della meta'.
#
# DIFFERENZA COL FRENO. Il freno in utils.py limita il CICLO DI LAVORO: la
# GPU fa 175 W per il 70% del tempo e 0 per il resto. Il consumo MEDIO
# scende, il PICCO no. Quello che spegne questa macchina e' il picco: 15
# eventi Kernel-Power 41, nessun bugcheck, nessun WHEA - e' una caduta di
# alimentazione, non un errore. Con la batteria al 74.2% di salute
# (64.819 su 87.395 mWh) il pacco non riesce piu' a tamponare i transitori
# che l'alimentatore da 330 W non copre da solo, con un i9-13980HX
# nell'altro piatto della bilancia.
#
# Il clock lock toglie il picco. E' la differenza fra sperare e impedire.
#
# --------------------------------------------------------------------------
# USO
#     .\limiti_hw.ps1                 applica i limiti
#     .\limiti_hw.ps1 -Clock 1800     applica con un tetto piu' alto
#     .\limiti_hw.ps1 -Ripristina     rimette tutto come prima
#
# I limiti sul clock NON sopravvivono al riavvio: dopo un riavvio vanno
# riapplicati. Il limite sulla CPU invece resta, ed e' il motivo per cui
# c'e' -Ripristina.

param(
    [int]$Clock = 1500,        # tetto sul clock SM in MHz (supportati: 210-3105)
    [int]$Cpu = 70,            # tetto sullo stato massimo del processore, in %
    [switch]$Ripristina
)

$ErrorActionPreference = "Continue"

function Elevato {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Elevato)) {
    Write-Output ""
    Write-Output "SERVE UN POWERSHELL DA AMMINISTRATORE."
    Write-Output "Tasto destro su PowerShell -> Esegui come amministratore, poi:"
    Write-Output ""
    Write-Output "    cd '$PSScriptRoot'"
    Write-Output "    .\limiti_hw.ps1"
    Write-Output ""
    Write-Output "Se rifiuta di eseguire lo script:"
    Write-Output "    Set-ExecutionPolicy -Scope Process Bypass -Force"
    Write-Output ""
    exit 1
}

if ($Ripristina) {
    Write-Output "Ripristino i valori di fabbrica."
    nvidia-smi -rgc
    nvidia-smi -rmc
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
    powercfg /setactive SCHEME_CURRENT
    Write-Output "Fatto. GPU e CPU senza tetto."
    exit 0
}

Write-Output "=== PRIMA ==="
nvidia-smi --query-gpu=clocks.max.sm,clocks.sm,power.draw,temperature.gpu `
           --format=csv

Write-Output ""
Write-Output "=== APPLICO ==="

# Il minimo si lascia basso: bloccare anche il minimo impedirebbe alla
# scheda di scendere quando e' ferma, e resterebbe a consumare fra un
# esperimento e l'altro.
Write-Output "GPU: clock SM limitato a $Clock MHz (minimo libero a 210)"
nvidia-smi -lgc 210,$Clock

# La memoria si lascia stare: il suo contributo al consumo e' piccolo e
# abbassarla rallenta molto un carico che legge 3 GB di latenti.

Write-Output "CPU: stato massimo del processore al $Cpu%"
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX $Cpu
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX $Cpu
powercfg /setactive SCHEME_CURRENT

Write-Output ""
Write-Output "=== DOPO ==="
nvidia-smi --query-gpu=clocks.max.sm,clocks.sm,power.draw,temperature.gpu `
           --format=csv

Write-Output ""
Write-Output "Ora lancia gli esperimenti passando per la sorveglianza:"
Write-Output ""
Write-Output "    python sorveglia.py --tetto 95 -- python exp_diversita.py --tag _casuale"
Write-Output ""
Write-Output "CALIBRAZIONE. Il tetto di $Clock MHz e' un punto di partenza"
Write-Output "prudente, non una misura. Lancia l'esperimento piu' corto e"
Write-Output "guarda il picco che sorveglia.py riporta alla fine:"
Write-Output "  - picco molto sotto 95 W  -> rilancia con -Clock 1800, poi 2100"
Write-Output "  - picco vicino a 95 W     -> vai cosi'"
Write-Output "  - spegnimento comunque    -> -Clock 1200, e -Cpu 60"
Write-Output ""
Write-Output "I limiti sul clock cadono al riavvio: riapplicali dopo ogni"
Write-Output "riavvio. Quello sulla CPU resta finche' non usi -Ripristina."
