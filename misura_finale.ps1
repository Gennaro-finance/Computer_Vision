$py = ".venv\Scripts\python"
$t0 = Get-Date
Write-Output "[1/3] griglia sull'encoder JEPA (epoca 59) - cache gia' pronta"
& $py -u train_downstream.py --grid --variant vit_small --layers 2 7 11 *> logs\M1_grid_jepa.log
Write-Output "[2/3] cache dei latenti dell'encoder CASUALE, stesso protocollo"
& $py -u train_downstream.py --cache --variant vit_small --layers 2 7 11 --random --tag _casuale *> logs\M2_cache_casuale.log
Write-Output "[3/3] griglia sull'encoder casuale"
& $py -u train_downstream.py --grid --variant vit_small --layers 2 7 11 --tag _casuale *> logs\M3_grid_casuale.log
Write-Output ("FINITO in {0:N1} minuti" -f ((Get-Date) - $t0).TotalMinutes)
