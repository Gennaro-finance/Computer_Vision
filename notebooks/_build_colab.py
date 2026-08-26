"""Genera notebooks/colab.ipynb.

Il notebook si costruisce da qui invece di scriverlo a mano: un .ipynb e'
JSON con il codice spezzato riga per riga dentro liste, e modificarlo
direttamente e' il modo piu' rapido per produrre un file che Colab rifiuta
di aprire. Questo script e' anche la traccia di cosa contiene.
"""

import json
import os

CELLE = []


def md(testo):
    CELLE.append({"cell_type": "markdown", "metadata": {},
                  "source": testo.strip().splitlines(keepends=True)})


def code(testo):
    CELLE.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": testo.strip().splitlines(keepends=True)})


md("""
# Progetto 8 - I-JEPA periapicale su Colab

Notebook di lavoro per quando l'hardware locale non e' utilizzabile.

**Da mettere su Google Drive una volta sola:**

| cosa | dimensione | dove | serve? |
|---|---|---|---|
| il dataset | 860 MB | `MyDrive/periapical/data/` | **si', sempre** |
| un checkpoint | 175 MB | `MyDrive/periapical/checkpoints/` | solo con `CASUALE = False` |

`splits.json` **non** va portato: e' versionato nel repo e arriva col clone.
Anche i risultati gia' misurati arrivano da GitHub.

Per lo sweep di alpha basta il dataset: gira sull'encoder casuale, che e'
il riferimento e detiene i due primati assoluti.

Esegui le celle in ordine. **La cella 5 verifica che l'ambiente sia
corretto**: se passa, i risultati saranno confrontabili con quelli gia'
ottenuti in locale.
""")

md("## 1 - Che GPU ci hanno dato")
code("""
!nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv

import torch
print(f"torch {torch.__version__}   CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"scheda : {p.name}")
    print(f"memoria: {p.total_memory/1e9:.1f} GB")
else:
    print("NESSUNA GPU. Runtime > Cambia tipo di runtime > Acceleratore hardware > GPU")
""")

md("## 2 - Google Drive")
code("""
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/periapical'    # cambia se l'hai messo altrove
for p in (BASE, BASE + '/data', BASE + '/checkpoints', BASE + '/risultati'):
    os.makedirs(p, exist_ok=True)
print('cartelle pronte sotto', BASE)
""")

md("""
## 3 - Il codice, da GitHub

Se hai gia' clonato in una sessione precedente, questa cella aggiorna e basta.

Niente comandi con `!` qui dentro: IPython non li trasforma in modo
affidabile quando stanno dentro un `if`, e il fallimento e' oscuro. Con
`subprocess` il comando che ha fallito e il suo errore si leggono per
intero.
""")
code("""
REPO = 'https://github.com/Gennaro-finance/Computer_Vision.git'
DEST = '/content/progetto'

import os, subprocess


def esegui(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print('COMANDO FALLITO:', ' '.join(cmd))
        print((r.stdout or '') + (r.stderr or ''))
        raise SystemExit('vedi sopra')
    return (r.stdout or '').strip()


if os.path.isdir(os.path.join(DEST, '.git')):
    esegui(['git', 'pull', '--quiet'], cwd=DEST)
    print('repo aggiornato')
else:
    esegui(['git', 'clone', '--quiet', REPO, DEST])
    print('repo clonato')

os.chdir(DEST)
print('cartella:', os.getcwd())
print('commit  :', esegui(['git', 'log', '--oneline', '-1']))
""")

md("""
## 4 - Collegare i dati

**L'unica cosa davvero necessaria e' il dataset.**

- `splits.json` arriva **col clone**: e' versionato nel repo, non serve
  portarlo su Drive.
- Il **checkpoint serve solo** se nella cella 7 metti `CASUALE = False`.
  Per lo sweep di alpha non serve: gira sull'encoder casuale, che e' il
  riferimento e detiene i due primati assoluti.

Il dataset si collega con un symlink invece di copiarlo: 860 MB ricopiati a
ogni sessione sono minuti persi, e Drive li serve direttamente.
""")
code("""
import os, glob

os.makedirs('/content/progetto/runs/checkpoints', exist_ok=True)
problemi = []

# --- dataset (necessario)
if not os.path.exists('/content/progetto/data'):
    if os.path.isdir(BASE + '/data'):
        os.symlink(BASE + '/data', '/content/progetto/data')
    else:
        problemi.append(
            'dataset assente. Carica la cartella `data/` in ' + BASE + '/data')

n_img = len(glob.glob('/content/progetto/data/**/*.jpg', recursive=True))
n_xml = len(glob.glob('/content/progetto/data/**/*.xml', recursive=True))
print(f'dataset    : {n_img} immagini, {n_xml} xml   (attese 3924 immagini)')
if n_img == 0:
    problemi.append('la cartella data/ e\\' vuota o il symlink e\\' rotto')
elif n_img != 3924:
    print('  numero diverso dal previsto: i risultati potrebbero non combaciare')

# --- split (dal repo, non da Drive)
sp_path = '/content/progetto/runs/splits.json'
print(f'splits.json: {"presente nel repo" if os.path.isfile(sp_path) else "ASSENTE"}')
if not os.path.isfile(sp_path):
    problemi.append('splits.json mancante: verifica che il clone sia riuscito')

# --- checkpoint (facoltativi)
if os.path.isdir(BASE + '/checkpoints'):
    for f in os.listdir(BASE + '/checkpoints'):
        dst = '/content/progetto/runs/checkpoints/' + f
        if not os.path.exists(dst):
            os.symlink(BASE + '/checkpoints/' + f, dst)
trovati = os.listdir('/content/progetto/runs/checkpoints')
print(f'checkpoint : {trovati if trovati else "nessuno (va bene: servono solo con CASUALE = False)"}')

print()
if problemi:
    for p in problemi:
        print('DA SISTEMARE:', p)
else:
    print('tutto a posto, prosegui')
""")

md("""
## 5 - Verifica dell'ambiente

**La cella che conta.** Ricalcola quantita' note: se combaciano, l'ambiente
e' corretto.
""")
code("""
import sys
sys.path.insert(0, '/content/progetto')
from data import parse_annotations, load_splits
from globals import NUM_CLASSES

recs = parse_annotations(verbose=False)
sp = load_splits()

ok = True
print(f"{'controllo':34s} {'atteso':>10s} {'trovato':>10s}")
print('-' * 60)
for k, atteso in (('train', 2746), ('val', 588), ('test', 590)):
    trovato = len(sp[k])
    buono = trovato == atteso
    ok &= buono
    print(f"immagini in {k:22s} {atteso:10d} {trovato:10d}  "
          f"{'ok' if buono else 'DIVERSO'}")

by = {r['image_id']: r for r in recs}
y = [l['grade'] for i in sp['test'] for l in by[i]['lesions']]
q = max(y.count(g) for g in (3, 4, 5)) / len(y)
pavimento = (2 * q / (1 + q)) / NUM_CLASSES

for nome, atteso, trovato, tol in (('lesioni nel test', 1013, len(y), 0),
                                   ('pavimento macro-F1', 0.2589, pavimento, 1e-3)):
    buono = abs(trovato - atteso) <= tol
    ok &= buono
    print(f"{nome:34s} {atteso:10.4f} {trovato:10.4f}  "
          f"{'ok' if buono else 'DIVERSO'}")

print()
print('AMBIENTE CORRETTO' if ok
      else 'QUALCOSA NON COMBACIA - non fidarti dei risultati')
""")

md("""
## 6 - Cache dei ritagli

Una volta per sessione. Taglia le lesioni dalle panoramiche e le tiene in
memoria come uint8: senza, ogni misura riaprirebbe le immagini intere da
disco, che era il collo di bottiglia in locale (oltre due minuti a sonda
contro 0.02 s).
""")
code("""
from data import cache_crop
import time

for split in ('train', 'val'):
    t = time.time()
    d = cache_crop(recs, sp[split], split)
    print(f"  {split:6s} {d['image'].shape[0]:5d} ritagli in {time.time()-t:5.1f}s")
""")

md("""
## 7 - Estrazione dei latenti

Da un checkpoint, oppure dall'encoder **casuale** - che nei nostri
esperimenti e' il riferimento, e detiene i due primati assoluti (PR-AUC
0.8813 e F1 su PAI 5 di 0.790).

Sono ~3 GB: restano in `/content`, non su Drive.
""")
code("""
CASUALE  = True              # encoder con pesi casuali (il riferimento)
CKPT_TAG = 'completa_best'   # usato solo se CASUALE = False
TAG      = '_casuale' if CASUALE else '_jepa'

from train_downstream import cache_latents
cache_latents('vit_small', layers=[2, 7, 11], casuale=CASUALE,
              ckpt_tag='' if CASUALE else CKPT_TAG, tag=TAG)
""")

md("""
## 8 - L'esperimento

Sotto c'e' lo sweep di alpha, l'unico rimasto in sospeso. Per la griglia
completa dell'obiettivo 4 usa invece `run_grid(...)` da `train_downstream`.

Protocollo in due fasi: screening su 3 seed, poi le due migliori su 5 seed
**disgiunti**. I seed non si riusano perche' selezionare su una misura
rumorosa e poi riportare quella stessa misura gonfia il vincitore di circa
una deviazione - dello stesso ordine dell'effetto cercato.
""")
code("""
ALPHAS    = [0.25, 0.50, 0.75, 1.00]
SCREENING = [0, 1, 2]
FINALI    = [10, 11, 12, 13, 14]      # disgiunti dai precedenti

import numpy as np, torch
from train_downstream import load_latents, train_head
from evaluation import evaluate_split

cached = load_latents('vit_small', layers=[2, 7, 11], tag=TAG)


def misura(alpha, seeds):
    v = {'macro_f1': [], 'pr_auc': [], 'recall5': [], 'prec5': []}
    for s in seeds:
        clf, _ = train_head(cached, 'balanced_tokens', 'flat',
                            seed=s, bts_alpha=alpha)
        r = evaluate_split(clf, cached['data']['test'], 'flat')
        del clf
        torch.cuda.empty_cache()
        v['macro_f1'].append(r['macro_f1'])
        v['pr_auc'].append(r['pr_auc_pai5'])
        v['recall5'].append(r['recall_pai5'])
        v['prec5'].append(r['precision_pai5'])
    return {k: (float(np.mean(x)), float(np.std(x))) for k, x in v.items()}


def riga(a, m):
    return (f"{a:6.2f} {m['pr_auc'][0]:9.4f}+-{m['pr_auc'][1]:.4f} "
            f"{m['macro_f1'][0]:9.4f}+-{m['macro_f1'][1]:.4f} "
            f"{m['recall5'][0]:8.4f} {m['prec5'][0]:8.3f}")


testata = f"{'alpha':>6s} {'PR-AUC5':>17s} {'macro-F1':>17s} {'rec5':>8s} {'prec5':>8s}"
print('FASE 1 - screening,', len(SCREENING), 'seed')
print(testata)
print('-' * 60)
scr = {}
for a in ALPHAS:
    scr[a] = misura(a, SCREENING)
    print(riga(a, scr[a]), flush=True)

migliori = sorted(ALPHAS, key=lambda a: -scr[a]['pr_auc'][0])[:2]
print('\\nFASE 2 - le due migliori', migliori, '- seed DISGIUNTI')
print(testata)
print('-' * 60)
fin = {}
for a in migliori:
    fin[a] = misura(a, FINALI)
    print(riga(a, fin[a]), flush=True)
""")

md("""
## 9 - Salvare su Drive

**Falla subito.** La sessione Colab si chiude e `/content` sparisce.
""")
code("""
import json, datetime

nome = f"sweep_alpha{TAG}_{datetime.datetime.now():%Y%m%d_%H%M}.json"
with open(f'{BASE}/risultati/{nome}', 'w') as f:
    json.dump({'screening': {str(k): v for k, v in scr.items()},
               'finali': {str(k): v for k, v in fin.items()},
               'seed_screening': SCREENING, 'seed_finali': FINALI,
               'encoder': TAG}, f, indent=2)
print('salvato su Drive:', nome)
""")

md("""
---

### Riferimenti gia' misurati

Stesso encoder casuale, stessa testa `flat`, stesso test, 5 seed. Servono a
capire subito se un numero nuovo e' buono:

| metodo | PR-AUC5 | macro-F1 | F1 PAI5 | rec5 | prec5 |
|---|---|---|---|---|---|
| `balanced_tokens` (alpha 0.5) | **0.8813** | 0.7676 | **0.790** | 0.7696 | 0.814 |
| `none` | 0.8758 | **0.7705** | 0.777 | 0.7232 | 0.840 |
| `oversample` | 0.8636 | 0.7603 | 0.771 | 0.7929 | 0.752 |

Pavimento (classificatore costante): **0.2589**.

### Se la sessione cade

Colab stacca dopo ~12 ore, o prima se resta inattiva. Le celle 1-7 vanno
rieseguite (~10 minuti, quasi tutto per i latenti); la 8 riparte da capo.

Per esperimenti lunghi, salva i risultati **parziali** dentro il ciclo
invece che alla fine - e' lo stesso motivo per cui la griglia in locale
scrive una riga alla volta.

### Cosa NON portare su Drive

I risultati gia' misurati sono nel repo (`runs/results_*.json`), quindi
arrivano col clone. La cache dei latenti si rigenera in pochi minuti e pesa
3 GB per braccio: lasciala in `/content`.
""")

NOTEBOOK = {
    "cells": CELLE,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

if __name__ == "__main__":
    qui = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(qui, "colab.ipynb")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, ensure_ascii=False, indent=1)
    print(f"{dest}: {len(CELLE)} celle")
