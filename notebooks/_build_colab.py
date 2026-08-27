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
| `periapical_data.zip` | 850 MB | `MyDrive/periapical/` | **si', sempre** |
| un checkpoint | 175 MB | `MyDrive/periapical/checkpoints/` | solo con `CASUALE = False` |

Carica l'**archivio**, non la cartella espansa: il dataset sono 20.928 file,
e il caricatore di Drive con tanti file piccoli e' lentissimo e spesso
fallisce a meta'. Un file solo si carica una volta e si scompatta in Colab
in un paio di minuti.

`splits.json` **non** va portato: e' versionato nel repo e arriva col clone.
Anche i risultati gia' misurati arrivano da GitHub.

Per i cinque esperimenti della sezione 8 basta il dataset: girano tutti
sull'encoder casuale, che e' il riferimento e detiene i due primati
assoluti.

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
  Per i cinque esperimenti della sezione 8 non serve: girano tutti
  sull'encoder casuale, che e' il riferimento e detiene i due primati
  assoluti.

Il dataset si prende dall'archivio su Drive e si scompatta sul disco locale
di Colab. Costa un paio di minuti per sessione, ma poi le letture sono
veloci: Drive via FUSE e' lento proprio sui tanti file piccoli, che e'
esattamente la forma di questo dataset.
""")
code("""
import os, glob

os.makedirs('/content/progetto/runs/checkpoints', exist_ok=True)
problemi = []

# --- dataset (necessario)
# Due modi, in ordine di preferenza:
#   1. un archivio unico su Drive, scompattato sul disco locale
#   2. la cartella gia' espansa su Drive, collegata con symlink
# Il primo e' molto piu' veloce: il dataset sono 20.928 file, e Drive - sia
# nel caricamento dal browser sia in lettura via FUSE - va male con tanti
# file piccoli. Un archivio da 850 MB si carica una volta e si scompatta in
# un paio di minuti sul disco locale di Colab, che poi legge veloce.
DATA = '/content/progetto/data'

# L'archivio si cerca OVUNQUE sotto BASE, non in un percorso fisso: e'
# facilissimo caricarlo dentro la sottocartella sbagliata, e in quel caso
# la ricerca in un punto solo fallisce in modo fuorviante - ripiega sul
# symlink a una cartella che contiene lo zip e nessuna immagine, e riporta
# "0 immagini" come se il dataset mancasse.
zip_trovati = sorted(glob.glob(BASE + '/**/*.zip', recursive=True))

# Un collegamento rimasto da un tentativo precedente punta alla cartella
# sbagliata e impedisce ogni correzione: si toglie.
if os.path.islink(DATA) and len(glob.glob(DATA + '/**/*.jpg', recursive=True)) == 0:
    os.unlink(DATA)
    print('rimosso un collegamento precedente che non conteneva immagini')

if not os.path.exists(DATA):
    if zip_trovati:
        import zipfile, time
        scelto = zip_trovati[0]
        print(f'archivio: {scelto.replace(BASE, "")} '
              f'({os.path.getsize(scelto)/1e6:.0f} MB)')
        t = time.time()
        with zipfile.ZipFile(scelto) as z:
            z.extractall(DATA)
        print(f'scompattato in {time.time()-t:.0f}s')
    elif glob.glob(BASE + '/data/**/*.jpg', recursive=True):
        os.symlink(BASE + '/data', DATA)
        print('cartella data/ collegata da Drive')
    else:
        problemi.append(
            'dataset assente. Carica periapical_data.zip da qualsiasi parte '
            'dentro ' + BASE)

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
# Cartella e percorso di ricerca si rimettono a posto QUI, non si danno per
# scontati: dopo un riavvio della sessione la cartella corrente torna a
# /content e sys.path perde il progetto, quindi `from data import ...`
# fallisce con un ModuleNotFoundError che non spiega niente.
import os, sys

os.chdir('/content/progetto')
if '/content/progetto' not in sys.path:
    sys.path.insert(0, '/content/progetto')

from data import parse_annotations, load_splits
from globals import DATA_ROOT, NUM_CLASSES

print('cartella :', os.getcwd())
print('DATA_ROOT:', DATA_ROOT)
if not os.path.isdir(DATA_ROOT):
    raise SystemExit(
        'DATA_ROOT non esiste. Se punta a /kaggle/... rilancia la cella 3 '
        '(git pull) e riavvia la sessione: la correzione e nel repo.')

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
## 8 - Gli esperimenti

Lo sweep di alpha e' **chiuso** (massimo interno a 0.50, vedi i riferimenti
in fondo). Restano i cinque della lista, ognuno nella sua cella. Sono
indipendenti: puoi lanciarne uno solo, e ognuno **riparte da dove si era
fermato** se la sessione cade.

| cella | esperimento | a cosa risponde | GPU |
|---|---|---|---|
| 8a | controlli | il merito e' del ribilanciamento o dell'augmentation? e dei passi di gradiente in piu'? | si', ~2-4 h |
| 8b | pooling | il collo di bottiglia e' l'aggregazione dei token? | si', ~1-2 h |
| 8c | curve PR | che precisione resta a recall 0.80 su PAI 5? | si', ~30-60 min |
| 8d | diversita' | quante viste INDIPENDENTI valgono quelle assegnate? | poca, ~5 min |
| 8e | figure | ridisegna tutte le figure dai JSON | no |

I tempi sono stime su T4 ricavate dai tempi misurati in locale (una testa a
100 epoche: 102 s per `none`, 166 s per `balanced_tokens` su RTX 4080).
Prendili come ordine di grandezza, non come promesse.

`--carico 100` toglie il freno termico: serve solo al portatile di casa, su
Colab e' tempo buttato.

**Se hai poco tempo, l'ordine di valore e' 8d, 8c, 8a, 8b.** La 8d costa
cinque minuti e da' il numero che spiega il risultato su alpha.
""")

md("""
### 8a - I due controlli (punti 2 e 4)

`balanced_token_sampling` fa **due cose insieme**: genera viste
(augmentation) e ne da' di piu' alle classi rare (ribilanciamento). Finche'
restano insieme non si sa quale delle due produce il risultato.

- `random_tokens` tiene le viste e toglie il ribilanciamento, a **budget di
  viste identico**. La differenza `balanced - random` isola il
  ribilanciamento; `random - none` isola l'augmentation.
- Il secondo controllo pareggia il **numero totale di istanze**: a parita'
  di epoche la novita' fa 6894 passi contro i 4719 delle baseline, cioe' il
  **46% di gradiente in piu'**. Qui le baseline ricevono le epoche che
  servono a pareggiare, 146 invece di 100.

Il secondo controllo peggiora deliberatamente il confronto per la novita'.
Se sopravvive, il risultato vale piu' di prima.
""")
code("""
!python exp_controlli.py --tag {TAG} --carico 100
""")

md("""
### 8b - Pooling gated e top-k (punto 5)

L'encoder e' congelato, quindi tutto cio' che si addestra sta in due pezzi:
**come i token della bbox diventano un vettore** (pooling) e come quel
vettore diventa tre logit (testa). Se l'aggregazione butta via
l'informazione, nessuna testa la recupera.

- `gated` - punteggio non lineare per token, stile MIL (Ilse et al. 2018):
  `a ~ w' (tanh(Vh) * sigmoid(Uh))`. Esprime *questo token conta SE anche
  quest'altra caratteristica c'e'*, che sul PAI e' la struttura del
  problema: una regione conta se e' insieme grande **e** scura. Con
  nascosto 128 sono 0.3M parametri contro i 5.3M dell'attention che
  sostituisce, quindi e' **piu' leggero**, non piu' pesante.
- `topk` - solo i k token piu' forti. E' l'ipotesi **opposta** alla
  novita': la novita' dice *non appoggiarti a pochi token*, il top-k dice
  *appoggiati solo ai migliori*. Se vince, l'argomento della novita' si
  indebolisce - ed e' meglio saperlo prima della presentazione che durante.

Selezione su **validation**; sul test va solo la configurazione scelta.
""")
code("""
!python exp_pooling.py --tag {TAG} --carico 100
""")

md("""
### 8c - Curve PR su PAI 5 (punto 3)

La PR-AUC e' un integrale, e integrali uguali vengono da curve diverse. Su
una minoritaria clinica non serve *tutta la curva*: serve **la precisione
che resta quando pretendi di trovare l'80% dei PAI 5**. Quella si legge
sulla curva, non nell'area.

Le curve dei 5 seed si mediano in verticale su una griglia comune di
recall, perche' ogni seed produce un numero diverso di punti a valori di
recall diversi e mediarle punto a punto non si puo'.
""")
code("""
!python exp_curve_pr.py --tag {TAG} --carico 100
""")

md("""
### 8d - Diversita' dei token (punto 6)

**E' il numero che spiega il risultato dello sweep di alpha**, e costa
cinque minuti.

Lo sweep dice che dare 7 viste a un PAI 5 rende **meno** che dargliene 3.
L'interpretazione e' che le viste siano ridondanti. Qui diventa una misura:
dal campionamento statistico, k osservazioni con correlazione interna rho
non valgono k campioni indipendenti, ne valgono

    n_eff = k / (1 + (k - 1) * rho)

che e' l'inverso del design effect. `rho` e' l'ICC multivariato - varianza
fra lesioni su varianza totale.

Misurato in locale sull'encoder casuale: **rho = 0.985 su PAI 5**, cioe'
**7 viste valgono 1.01 esempi indipendenti**. Col pooling addestrato rho
scende a 0.923 e n_eff sale a 1.07: la conclusione tiene.

Si esegue due volte - media mascherata (senza parametri) e pooling
addestrato - perche' la prima obiezione a questa misura e' *hai usato un
aggregatore finto*.
""")
code("""
!python exp_diversita.py --tag {TAG}
!python exp_diversita.py --tag {TAG} --addestrato
""")

md("""
### 8e - Le figure

Nessuna GPU: legge i JSON in `runs/` e salta le figure di cui non trova i
dati, invece di fallire.
""")
code("""
!python figure_finali.py

from IPython.display import Image, display
import glob, os
for f in sorted(glob.glob('runs/figures/fin*.png')):
    print(os.path.basename(f))
    display(Image(f))
""")

md("""
## 9 - Salvare su Drive

**Falla subito.** La sessione Colab si chiude e `/content` sparisce.

Copia tutti i JSON e le figure in una cartella con la data, cosi' due
sessioni diverse non si sovrascrivono a vicenda.
""")
code("""
import datetime, glob, os, shutil

quando = f"{datetime.datetime.now():%Y%m%d_%H%M}"
dest = f'{BASE}/risultati/{quando}{TAG}'
os.makedirs(dest, exist_ok=True)

copiati = 0
for pattern in ('runs/controlli_*.json', 'runs/pooling_*.json',
                'runs/curve_pr_*.json', 'runs/diversita_*.json',
                'runs/sweep_alpha_*.json', 'runs/results_*.json',
                'runs/figures/*.png'):
    for f in glob.glob(pattern):
        shutil.copy2(f, dest)
        copiati += 1

print(f'{copiati} file copiati in', dest.replace(BASE, 'Drive:'))
for f in sorted(os.listdir(dest)):
    print('  ', f)
""")

md("""
### 9b - Rimandarli nel repo (facoltativo, consigliato)

I JSON sono piccoli e sono **i dati della presentazione**: nel repo sono al
sicuro e versionati, su Drive no. Serve un token GitHub con permesso
`repo`; se non ce l'hai, scarica i file da Drive e committali da casa.

Il token e' una credenziale: incollalo, esegui, e **non salvare il
notebook** dopo averlo fatto.
""")
code("""
TOKEN = ''      # incolla qui, poi NON salvare il notebook

if TOKEN:
    import subprocess
    REPO_TOK = REPO.replace('https://', f'https://{TOKEN}@')
    passi = (['git', 'config', 'user.email', 'colab@local'],
             ['git', 'config', 'user.name', 'Colab'],
             ['git', 'add', 'runs/'],
             ['git', 'commit', '-m', f'risultati da Colab {quando}'],
             ['git', 'push', REPO_TOK, 'HEAD:main'])
    for cmd in passi:
        r = subprocess.run(cmd, cwd=DEST, capture_output=True, text=True)
        mostra = ' '.join(cmd[:3]).replace(TOKEN, '***')
        print(mostra, '->', r.returncode)
        if r.returncode and 'nothing to commit' not in (r.stdout or ''):
            print(((r.stdout or '') + (r.stderr or '')).replace(TOKEN, '***'))
            break
    else:
        print('fatto')
else:
    print('nessun token: scarica i file da Drive e committali da casa')
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
| `focal` | 0.8734 | 0.7577 | - | 0.7964 | 0.751 |
| `class_weighted` | 0.8706 | 0.7533 | - | 0.7875 | 0.753 |
| `oversample` | 0.8636 | 0.7603 | 0.771 | 0.7929 | 0.752 |

Pavimento (classificatore costante): **0.2589**.

**Sweep di alpha, chiuso.** Screening a 3 seed: 0.25 -> 0.8793, **0.50 ->
0.8814**, 0.75 -> 0.8747, 1.00 -> 0.8689. Il massimo e' **interno**: alpha
0.50 batte alpha 1.00 di +0.0125 a 2.2 errori standard. Rimisurato su 5
seed disgiunti: 0.50 -> 0.8797, 0.25 -> 0.8775. Alpha 0.50 misurato tre
volte in modo indipendente da' 0.8813 / 0.8814 / 0.8797, scarto 0.0017.

**Diversita' dei token** (encoder casuale, media mascherata): rho = 0.966 /
0.974 / **0.985** per PAI 3 / 4 / 5. Sette viste di un PAI 5 valgono
**1.01** esempi indipendenti. Col pooling addestrato rho = 0.824 / 0.890 /
0.923, e n_eff = 1.07.

### Attenzione al confronto fra encoder

La novita' e' prima sull'encoder **casuale**, seconda sul completo, **ultima
sullo spinto** (0.8711, dietro tutte e quattro le baseline). Il vantaggio
dipende dall'encoder, e va detto: non e' *la novita' vince*, e' *la novita'
vince dove il pre-training non ha appiattito le rappresentazioni*.

### Se la sessione cade

Colab stacca dopo ~12 ore, o prima se resta inattiva. Le celle 1-7 vanno
rieseguite (~10 minuti, quasi tutto per i latenti). Le celle 8a e 8b invece
**riprendono da dove erano**: rileggono il loro JSON in `runs/` e saltano le
celle gia' misurate, purche' il protocollo coincida - stessi seed, stessa
testa, stesso alpha. Con un protocollo diverso ripartono da capo invece di
mescolare due esperimenti nella stessa tabella.

Le 8c, 8d ed 8e durano poco e si rifanno interamente.

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
