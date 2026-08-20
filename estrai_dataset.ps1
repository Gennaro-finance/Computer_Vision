# estrai_dataset.ps1 - estrae dal ZIP Mendeley solo le due cartelle che servono.
#
# L'archivio pesa ~8.9 GB ma di utile ce ne sono 0.86: le immagini aumentate
# e i .rar duplicati interni non servono e non vengono toccati.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "==============================================================" -ForegroundColor White
Write-Host " Estrazione dataset periapicale" -ForegroundColor White
Write-Host "==============================================================" -ForegroundColor White
Write-Host ""

# ------------------------------------------------------------- trova il ZIP
$candidati = @()
foreach ($dir in @("$env:USERPROFILE\Downloads", "$root\data", $root)) {
    if (Test-Path $dir) {
        $candidati += Get-ChildItem -Path $dir -Filter "*.zip" -File -ErrorAction SilentlyContinue |
                      Where-Object { $_.Name -match "eriapical" -or $_.Name -match "anoramic" }
    }
}

if ($candidati.Count -eq 0) {
    Write-Host " Non trovo l'archivio." -ForegroundColor Red
    Write-Host " Cercavo un file .zip con 'periapical' o 'panoramic' nel nome, in:"
    Write-Host "   $env:USERPROFILE\Downloads"
    Write-Host " Scaricalo da https://data.mendeley.com/datasets/kx52tk2ddj/3"
    Read-Host "`n Premi INVIO per chiudere"
    exit 1
}

$zipPath = ($candidati | Sort-Object Length -Descending)[0].FullName
Write-Host " Archivio: $zipPath"
Write-Host (" Dimensione: {0:N2} GB" -f ((Get-Item $zipPath).Length / 1GB))
Write-Host ""

# ------------------------------------------------------------- destinazione
$dst = Join-Path $root "data\periapical"
New-Item -ItemType Directory -Path $dst -Force | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)

try {
    # Solo queste due sottocartelle. I nomi sono quelli REALI dentro
    # l'archivio: la cartella delle annotazioni si chiama "Image Annots",
    # non "Image Annotations".
    $vogliamo = @("/Original JPG Images/", "/Image Annots/")

    $entries = @()
    foreach ($e in $zip.Entries) {
        if ($e.Length -eq 0 -and $e.Name -eq "") { continue }
        $n = $e.FullName -replace "\\", "/"
        foreach ($w in $vogliamo) {
            if ($n -like "*$w*" -and $n -notlike "*Augmentation*") {
                $entries += $e
                break
            }
        }
    }

    Write-Host (" File da estrarre: {0}" -f $entries.Count)
    Write-Host " (le immagini aumentate e i .rar interni vengono ignorati)"
    Write-Host ""

    $i = 0; $saltati = 0
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    foreach ($e in $entries) {
        $i++
        $n = $e.FullName -replace "\\", "/"

        # tiene solo <Cartella>/<file>, scartando i livelli superiori
        $parti = $n.Split("/")
        $rel = ($parti[-2], $parti[-1]) -join "\"
        $out = Join-Path $dst $rel

        $cartella = Split-Path -Parent $out
        if (-not (Test-Path $cartella)) { New-Item -ItemType Directory -Path $cartella -Force | Out-Null }

        # se esiste gia' identico, salta: cosi' rilanciare lo script riprende
        if ((Test-Path $out) -and ((Get-Item $out).Length -eq $e.Length)) {
            $saltati++
        } else {
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $out, $true)
        }

        if ($i % 1000 -eq 0) {
            $pct = [int](100 * $i / $entries.Count)
            $el = $sw.Elapsed.TotalSeconds
            $stima = if ($i -gt 0) { [int]($el / $i * ($entries.Count - $i)) } else { 0 }
            Write-Host ("  {0,6}/{1}  {2,3}%   trascorsi {3:N0}s   mancano ~{4:N0}s" -f `
                        $i, $entries.Count, $pct, $el, $stima)
        }
    }
    $sw.Stop()
    Write-Host ""
    Write-Host (" Estratti {0} file in {1:N0}s ({2} gia' presenti)" -f `
                $entries.Count, $sw.Elapsed.TotalSeconds, $saltati) -ForegroundColor Green
}
finally {
    $zip.Dispose()
}

# ---------------------------------------------------------------- controllo
Write-Host ""
Write-Host "--- Controllo finale ---"
$jpg = (Get-ChildItem -Path $dst -Filter "*.jpg" -Recurse -File -ErrorAction SilentlyContinue).Count
$xml = (Get-ChildItem -Path $dst -Filter "*.xml" -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host ("  JPG: {0}   (attesi 3924)" -f $jpg)
Write-Host ("  XML: {0}   (attesi 17004)" -f $xml)
Write-Host ""
Get-ChildItem -Path $dst -Directory | ForEach-Object { Write-Host ("  cartella: {0}" -f $_.Name) }

if ($jpg -ge 3900 -and $xml -ge 16900) {
    Write-Host ""
    Write-Host " DATASET PRONTO" -ForegroundColor Green
    Write-Host " Prossimo passo: in PyCharm lancia '3 - Dati statistiche bbox'"
} else {
    Write-Host ""
    Write-Host " Conteggi sotto le attese: rilancia questo script, riprende da dove era." -ForegroundColor Yellow
}
