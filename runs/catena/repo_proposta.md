# Proposta di riordino del repository

Generata da `sistema_repo.py` dopo la catena notturna.
**Niente di quanto segue e' stato cancellato**: e' un elenco da approvare.

## Com'e' andata la notte

Tutti gli stadi della catena sono andati a buon fine.

## Sospetti orfani

Script tracciati che nessun altro file tracciato nomina. **Non e' una condanna**: un file lanciato solo da riga di comando risulta orfano ed e' vivo. Serve il tuo giudizio.

| file | byte | verdetto (da mettere a mano) |
|---|---:|---|
| `ESTRAI_DATASET.bat` | 198 | |
| `PARTE1.bat` | 187 | |
| `TRAINING.bat` | 191 | |
| `attendi_e_sistema.py` | 3,346 | |
| `exp_accordo.py` | 4,491 | |
| `exp_protocollo.py` | 10,190 | |
| `exp_rumore.py` | 9,190 | |
| `exp_spostamento.py` | 4,728 | |
| `misura_finale.ps1` | 665 | |
| `notebooks/_build_colab.py` | 21,481 | |
| `stato.py` | 4,992 | |
| `sweep_collasso.py` | 4,313 | |

## Cose che NON propongo di togliere, e perche'

- `runs/archivio/` — sono i risultati superati, e sono la prova documentale della sezione *errori fatti e chiusi*. Toglierli renderebbe non verificabile proprio la parte piu' onesta del racconto.
- `runs/PROVENIENZA.md` — dice quale checkpoint ha prodotto quale file. E' cio' che ha permesso di scoprire il braccio costruito su un checkpoint ignoto.
- i `logs/` in whitelist — senza, l'affermazione *ogni numero e' ricalcolabile* diventa falsa.

## Il push non e' stato fatto

Il commit e' pronto sul ramo `catena-300`. Quando la proposta qui sopra ti convince:

```bash
git push -u origin catena-300
```

Il remote e' `Gennaro-finance/Computer_Vision` e siete in tre: conviene aprire una PR invece di fondere dritto su `main`, cosi' gli altri due vedono cosa cambia.

