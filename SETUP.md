# Setup — dai file al primo training

Guida operativa per il flusso scelto: **PyCharm in locale + training su Kaggle**.
Seguite i passi in ordine. Il tempo totale è di circa un'ora, dominata dal
download del dataset.

---

## Passo 1 — Ambiente Python locale

### Un comando solo

Aprite PowerShell nella cartella del progetto e lanciate:

```powershell
cd $env:USERPROFILE\PycharmProjects\cv-periapical-jepa
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

Lo script crea il virtual environment, installa PyTorch con CUDA 12.6 più le
altre dipendenze, prepara le cartelle di lavoro ed esegue la verifica completa
dell'ambiente. Richiede qualche minuto, quasi tutto per scaricare torch
(~2,5 GB). È idempotente: se lo rilanciate riusa il venv esistente.

Il `-ExecutionPolicy Bypass` serve perché per impostazione predefinita
PowerShell non esegue script non firmati. Non cambia nulla in modo permanente:
vale solo per quella invocazione.

Opzioni: `-SkipInstall` salta pip se avete già installato, `-Cpu` installa la
build CPU se la CUDA dà problemi.

### Se preferite farlo a mano

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy pillow matplotlib
python verify_setup.py
```

Se l'attivazione viene bloccata da un errore di *execution policy*, lanciate
una volta sola:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Nota:** `matplotlib` non risulta installato nel vostro Python globale e
serve per le figure. Nel venv lo installate sopra, quindi il problema non si
pone.

---

## Passo 1b — Configurare PyCharm

**PyCharm è l'IDE giusto per questo progetto** e ce l'avete già installato
(2025.1.3.1). Dalla versione 2025.1 JetBrains ha unificato Community e
Professional in un prodotto solo, e **il supporto ai Jupyter notebook — con
esecuzione, debug, rendering degli output e code assistance — è nel tier
gratuito**. Serve esattamente per il notebook in `notebooks/`, quindi non
dovete installare né comprare nulla.

Le guidelines del corso citano VSCode come opzione, ma per questo progetto
PyCharm è preferibile: il codice è distribuito su otto moduli che si
importano a vicenda, e refactoring, navigazione fra simboli, gestione del
venv e Git integrato vi fanno risparmiare tempo reale. Aggiungerne un secondo
non porta nulla.

### 1. Interprete

*File → Settings → Project → Python Interpreter → Add Local Interpreter →
Existing* e puntate a:

```
%USERPROFILE%\PycharmProjects\cv-periapical-jepa\.venv\Scripts\python.exe
```

Se saltate questo passo, PyCharm continua a usare il Python globale e vi
troverete import che funzionano da terminale ma non nell'IDE.

### 2. Configurazioni di avvio — già pronte

Il progetto include otto configurazioni pronte. Sono versionate in
`ide/pycharm/runConfigurations/` e `bootstrap.ps1` le copia in
`.idea/runConfigurations/`, da dove PyCharm le carica. Se PyCharm era già
aperto durante il bootstrap, chiudetelo e riapritelo.

Nel menu a tendina in alto a destra le troverete nel giusto ordine di
esecuzione:

| | Cosa fa |
|---|---|
| 1 - Verifica ambiente | controllo completo, non serve il dataset |
| 2 - Dati ispeziona | struttura reale del dataset |
| 3 - Dati statistiche bbox | **la verifica che decide l'architettura** |
| 4 - Dati crea split | split + assert anti-leakage |
| 5 - SSL smoke test | 20 step, conferma che gira |
| 6 - SSL pre-training | training vero, con resume |
| 7 - Latenti cache | estrazione una-tantum |
| 8 - Downstream ablation | griglia metodi × teste × seed |

Sono versionate apposta: così tutti e tre lanciate le stesse cose con gli
stessi parametri, e nessuno sbaglia un flag da riga di comando. Il resto di
`.idea/` è ignorato da git, quindi le vostre preferenze personali restano
vostre.

### 3. Notebook

`bootstrap.ps1` installa `jupyter` e `ipykernel` nel venv e registra il kernel
come *Python (cv-periapical-jepa)*. Aprendo
`notebooks/cv_project_p8_kaggle.ipynb` in PyCharm, selezionate quel kernel.

Il notebook è pensato per Kaggle (la prima cella clona la repo), ma le celle
Data, Network ed Evaluation girano anche in locale.

### 4. Due impostazioni che evitano grattacapi

- *Settings → Editor → Code Style → Python*: lunghezza riga **88**, coerente
  con il codice già scritto.
- *Settings → Version Control → Git*: abilitate **"Add files to Git
  automatically"** solo se vi fidate del `.gitignore` — meglio no, per non
  rischiare di committare per sbaglio il dataset o un checkpoint da 400 MB.

### Verifica

`bootstrap.ps1` la esegue già, ma potete rilanciarla in qualsiasi momento:

```powershell
python verify_setup.py
```

Controlla versione di Python, venv attivo, pacchetti, GPU e VRAM, import dei
moduli, e poi esegue quattro test di correttezza su casi con esito noto:
rilevamento del collasso, forward I-JEPA, metriche ordinali e gestione dello
sbilanciamento. Nessuno richiede il dataset.

Finché non avete scaricato i dati vedrete `[ !! ]` sull'ultima riga: è atteso,
non è un errore.

---

## Passo 2 — Scaricare il dataset

Il dataset va scaricato a mano dal browser (non è su Kaggle e non ha un link
diretto stabile):

**https://data.mendeley.com/datasets/kx52tk2ddj/3**

Scaricate l'archivio e scompattatelo in:

```
data\periapical\
    Original JPG Images\
    Image Annots\                 <- NON "Image Annotations"
    Augmentation JPG Images\      <- presente ma NON usata, vedi sotto
```

**Attenzione al nome della cartella delle annotazioni.** L'archivio Mendeley
la chiama `Image Annots`, non `Image Annotations` come lascerebbe pensare la
descrizione del dataset. `globals.py` cerca esattamente questi nomi.

Dentro lo zip le tre cartelle stanno annidate sotto
`Periapical Dataset/Periapical Lesions/`. Non serve spostarle a mano:

```powershell
powershell -ExecutionPolicy Bypass -File .\estrai_dataset.ps1
```

trova lo zip nei Download, estrae solo le due cartelle che servono e le mette
al posto giusto.

> **Perché la cartella di augmentation non si usa.** Contiene 17.004 immagini
> derivate dalle 3.926 originali per scaling, mirroring e flipping. Splittarle
> a caso metterebbe varianti geometriche della stessa radiografia in train e in
> test, gonfiando le metriche. Il pre-training self-supervised applica già le
> proprie augmentation on-the-fly, quindi quelle pre-generate non aggiungono
> informazione: aggiungono solo occasioni di sbagliare.

Il dataset **non va committato** su GitHub — è escluso dal `.gitignore`. Il
brief chiede "Dataset (or the link to it)": mettete il link Mendeley nel
README.

---

## Passo 3 — I tre comandi da lanciare per primi

In quest'ordine. Il secondo è quello che decide l'architettura del progetto.

```powershell
python data.py --inspect
```

Stampa la struttura reale del dataset e il contenuto del primo XML. Serve a
verificare che il parser legga davvero le vostre annotazioni. Se compare
"Etichette non riconosciute", adattate `_parse_grade()` in `data.py`.

```powershell
python data.py --bbox-stats
```

**Questo è il comando più importante.** Misura quanto è grande una lesione in
token a ogni risoluzione candidata e marca come `INUTILIZZABILE` quelle sotto
soglia. Vi aspettate che le configurazioni con panoramica ridimensionata
falliscano e che passi solo il tile a risoluzione nativa. Se anche il tile
nativo non passa, riducete `PATCH_SIZE` o lavorate su crop più stretti attorno
alla lesione.

Salvate la figura che produce: è materiale diretto per la slide *Proposed
Method*, perché documenta un problema di design che il brief non menziona.

```powershell
python data.py --splits
```

Crea gli split a livello di immagine e verifica con degli assert che nessuna
immagine compaia in due split. Controllate anche che la distribuzione PAI
stampata sia coerente tra train, val e test.

### Smoke test del training

```powershell
python train_ssl.py --smoke --epochs 1 --batch-size 8
```

Venti step, giusto per confermare che la pipeline gira end-to-end prima di
impegnare ore di GPU.

---

## Passo 4 — Repository GitHub

È un deliverable obbligatorio: il form chiede il link alla repo con codice,
dataset (o link), presentazione e README dettagliato.

Create la repo su GitHub — **privata** finché non consegnate, poi rendetela
pubblica o aggiungete i docenti come collaboratori. Nome suggerito:
`cv-periapical-jepa`.

Poi, dalla cartella del progetto:

```powershell
git init
git add .
git commit -m "Scaffold iniziale: I-JEPA, tiling, split anti-leakage, metriche"
git branch -M main
git remote add origin https://github.com/UTENTE/cv-periapical-jepa.git
git push -u origin main
```

Aggiungete subito gli altri due membri come collaboratori
(*Settings → Collaborators*).

### Come lavorare in tre senza conflitti

Un branch per persona, secondo la divisione dei compiti:

```powershell
git checkout -b feature/ssl-pretraining      # Persona A
git checkout -b feature/downstream-novelty   # Persona B
git checkout -b feature/baselines-eval       # Persona C
```

I file sono già separati per ruolo, quindi i conflitti reali si limitano a
`globals.py`. Regola pratica: **chi tocca `globals.py` lo annuncia nel gruppo
prima di farlo.**

Prima di ogni commit di notebook, ripulite gli output — altrimenti ogni run
genera un diff enorme e i merge diventano ingestibili:

```powershell
jupyter nbconvert --clear-output --inplace notebooks\cv_project_p8_kaggle.ipynb
```

---

## Passo 5 — Kaggle

### 5a. Caricare il dataset

1. [kaggle.com](https://www.kaggle.com) → *Datasets* → *New Dataset*
2. Caricate le cartelle `Original JPG Images` e `Image Annotations`
   (**non** quella di augmentation: risparmiate tempo e spazio)
3. Titolo: `panoramic-periapical-lesions`
4. Visibilità **privata**

Il caricamento richiede un po'. Fatelo partire e passate ad altro.

### 5b. Creare il notebook

1. *Code* → *New Notebook*
2. *File* → *Import Notebook* → caricate
   `notebooks/cv_project_p8_kaggle.ipynb`
3. Pannello destro → *Input* → *Add Input* → il dataset appena caricato
4. Pannello destro → *Accelerator* → **GPU T4 x2**
5. Pannello destro → *Persistence* → *Files only* (mantiene `/kaggle/working`
   tra le sessioni: è ciò che rende utile il resume)

**Preferite T4 alla P100.** La P100 è architettura Pascal, non supporta bene
fp16 e vi fa perdere il vantaggio della mixed precision. Su T4 l'AMP è
efficace. La prima cella del notebook ve lo dice esplicitamente.

### 5c. Collegare la repo

Nella prima cella del notebook, sostituite `REPO_URL` con la vostra.

Se la repo è privata, servite un Personal Access Token: GitHub → *Settings →
Developer settings → Personal access tokens → Fine-grained*, permesso di sola
lettura sulla repo. Su Kaggle salvatelo in *Add-ons → Secrets* come
`GITHUB_TOKEN` e usate:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
REPO_URL = f"https://{token}@github.com/UTENTE/cv-periapical-jepa.git"
```

Non incollate mai il token direttamente in una cella: finirebbe nella
cronologia git.

### 5d. Verificare i path

Su Kaggle il dataset finisce in `/kaggle/input/<slug-del-dataset>`. Se il vostro
slug è diverso da `panoramic-periapical-lesions`, aggiustate `DATA_ROOT` in
`globals.py`. La cella *Globals* del notebook lo stampa: controllatelo.

---

## Passo 6 — Budget di calcolo

Kaggle free dà ~30 h di GPU a settimana, sessioni fino a 12 h. Stima per
ViT-Tiny a 224², batch 64, ~24k tile:

| Epoche | Tempo stimato | Sta in una sessione? |
|---|---|---|
| 300 | ~6,3 h | sì |
| 600 | ~12,5 h | no — serve il resume |
| 800 | ~16,7 h | no — serve il resume |

I checkpoint si salvano a ogni epoca, quindi la disconnessione non è un
disastro: rilanciate con `resume=True` e riparte da dov'era. Ma serve
*Persistence: Files only*, altrimenti `/kaggle/working` viene azzerato e i
checkpoint spariscono.

Con tre persone avete tre account Kaggle e quindi ~90 h/settimana complessive:
distribuite i run (per esempio A fa il pre-training, C in parallelo il braccio
ImageNet frozen sul proprio account).

---

## Checklist prima di considerare il setup concluso

- [ ] Le quattro autoverifiche dei moduli passano in locale
- [ ] `python data.py --inspect` legge le annotazioni senza etichette ignote
- [ ] `python data.py --bbox-stats` conferma che il tile nativo è utilizzabile
- [ ] `python data.py --splits` passa gli assert anti-leakage
- [ ] Il conteggio PAI 4 stampato è vicino a 1.817 (era un valore derivato)
- [ ] Repo GitHub creata, primo push fatto, collaboratori aggiunti
- [ ] Dataset su Kaggle, notebook importato, GPU T4 e Persistence attivi
- [ ] Smoke test SSL completato su Kaggle
- [ ] Mail ai docenti sulla discrepanza di scadenza (6 settembre nella lista
      progetti contro 7 giorni prima nelle exam guidelines → 4 settembre)

L'ultimo punto non è tecnico ma è il più urgente: mandatela oggi.
