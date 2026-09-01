# Self-Supervised Latent Representations for Imbalanced Apical Periodontitis Grading

Progetto 8 — Computer Vision A.A. 2025-2026, Prof. Irene Amerini
Sapienza Università di Roma · ALCOR Lab

| | |
|---|---|
| **Gruppo** | Nome 1 (matricola) · Nome 2 (matricola) · Nome 3 (matricola) |
| **Sessione** | 11 settembre 2026 |
| **Framework** | PyTorch |

---

## Overview

Pipeline a due stadi per il grading automatico della periodontite apicale su
radiografie panoramiche:

1. **Pre-training self-supervised** — una Vision-JEPA (context encoder,
   target encoder aggiornato via EMA, predictor shallow) impara a predire le
   rappresentazioni latenti di patch anatomiche mascherate, senza etichette.
2. **Classificazione downstream** — encoder congelato, attention pooling sui
   token interni alla bbox della lesione, testa leggera che predice il grado
   PAI (3, 4 o 5) su una distribuzione fortemente sbilanciata.

La novità metodologica affronta la scarsità della classe PAI 5 agendo nello
spazio latente (`imbalance.py`).

## Le due decisioni di progetto che contano

### 1. Tile a risoluzione nativa, non panoramiche ridimensionate

Una panoramica inquadra l'intera arcata (~2900 px); una lesione periapicale è
di pochi millimetri (~50 px). Ridimensionando l'immagine intera a 224×224 la
lesione diventa **~4 px, cioè meno di un patch token da 16×16**: il latente
estratto alla bbox descriverebbe mandibola generica, non patologia.

Per questo si lavora a tile su risoluzione nativa. **Verificatelo sui vostri
dati prima di addestrare qualsiasi cosa:**

```bash
python data.py --bbox-stats
```

Il tool riporta la copertura in token per ogni risoluzione candidata e segna
come `INUTILIZZABILE` quelle sotto soglia.

### 2. I-JEPA come primario, SIGReg come assicurazione

L'obiettivo 1 del brief richiede esplicitamente il target encoder aggiornato
via EMA — cioè I-JEPA. LeJEPA (ref [2] del brief) rimuove precisamente EMA,
stop-gradient e teacher-student, sostituendoli con SIGReg.

La scelta implementata: **I-JEPA come metodo primario** (obiettivo 1
soddisfatto alla lettera) e **SIGReg come braccio di confronto**. Quest'ultimo
è anche l'assicurazione sul collasso: se I-JEPA diverge sulle ~4k immagini,
averlo già pronto vale giorni.

## Dati

[Panoramic Radiographs with Periapical Lesions Dataset](https://data.mendeley.com/datasets/kx52tk2ddj/3)
(Do et al., *Data in Brief* 54:110486, 2024) — Hanoi Medical University.

| | |
|---|---|
| Immagini originali | 3.926 |
| Lesioni etichettate | 6.029 (≈1,54 per immagine) |
| PAI 3 / PAI 4 / PAI 5 | 3.691 / 1.817\* / 521 |
| Sbilanciamento | 7,08 : 1 |

\* derivato per sottrazione; `data.py` verifica se i conti tornano davvero.

**La cartella `Augmentation JPG Images` non viene usata.** Contiene 17.004
immagini derivate dalle 3.926 originali: splittarle a caso metterebbe
varianti geometriche della stessa radiografia in train e test. Il SSL applica
già le proprie augmentation on-the-fly.

## Struttura del codice

Segue la struttura concettuale richiesta dal corso:

| File | Sezione | Contenuto |
|---|---|---|
| `globals.py` | Globals | iperparametri, path, config I-JEPA e tiling |
| `utils.py` | Utils | seed, checkpoint, **monitoraggio del collasso**, k-NN probe |
| `data.py` | Data | parsing XML, statistiche bbox, split anti-leakage, tile e crop |
| `network.py` | Network | ViT, I-JEPA, masking a blocchi, attention pooling, teste |
| `imbalance.py` | Network / Train | baseline per lo sbilanciamento e **la novità** |
| `train_ssl.py` | Train | pre-training stadio 1 con resume |
| `train_downstream.py` | Train | caching dei latenti + stadio 2 + griglia di ablation |
| `evaluation.py` | Evaluation | Macro-F1, PR-AUC, confusion matrix, **kappa quadratico** |

Oltre alla pipeline il repository contiene gli script che hanno prodotto i
numeri riportati. **Ogni file dichiara in prima riga del proprio docstring a
quale famiglia appartiene**, così le due specie non si confondono:

| Marca | Cosa significa | File |
|---|---|---|
| `PIPELINE` | il codice consegnato, quello che gira | i dieci file della tabella qui sopra |
| `MISURAZIONE` | misura una metrica su un braccio | `exp_curve_pr` `exp_diversita` `exp_rumore` `exp_sonde` `exp_traiettoria_testa` |
| `CONFRONTO` | confronta una metrica fra bracci o configurazioni | `exp_fixedk` `exp_fewshot` `exp_novita_K` `exp_testa` `exp_alpha` `exp_controlli` `exp_mascheramento` `exp_pooling` `exp_stratificata` `exp_protocollo` `exp_accordo` `exp_spostamento` `sweep_collasso` |
| `DIAGNOSI` | indaga una discrepanza fra misure | `exp_scarto` |
| `INFRASTRUTTURA` | supporto, non richiesto dalla traccia | `catena` `sorveglia` `stato` `sistema_repo` `attendi_e_sistema` `figure_finali` `make_figures` |

Gli script marcati `MISURAZIONE`, `CONFRONTO` e `DIAGNOSI` **non fanno parte
della pipeline**: hanno prodotto i risultati e restano nel repository perché
ogni numero riportato sia rieseguibile, non perché servano a farla girare.

## Come si esegue

```bash
pip install -r requirements.txt
```

### 1. Verificare i dati — sempre per primo

```bash
python data.py --inspect       # struttura reale del dataset
python data.py --bbox-stats    # la lesione copre abbastanza token?
python data.py --splits        # split a livello immagine + assert anti-leakage
```

### 2. Pre-training self-supervised

```bash
python train_ssl.py --variant vit_tiny --epochs 300
python train_ssl.py --resume          # dopo un timeout Kaggle
python train_ssl.py --smoke           # 20 step, per verificare che giri
```

I checkpoint sono salvati a ogni epoca: 300 epoche ≈ 6 h, 600 ≈ 12,5 h, e la
sessione Kaggle si stacca a 12 h. Il resume non è opzionale.

**Guardate il monitoraggio, non la loss.** La loss I-JEPA può scendere
regolarmente mentre gli embedding collassano a una costante — predire un
target costante è banale. Ogni epoca vengono loggati deviazione standard e
rango effettivo, e ogni 20 epoche un k-NN probe.

### 3. Caching dei latenti — una volta sola

```bash
python train_downstream.py --cache --arm ijepa
python train_downstream.py --cache --arm imagenet   # il braccio critico
python train_downstream.py --cache --arm random     # il pavimento
```

Da qui in poi ogni esperimento sullo sbilanciamento gira **in secondi**, anche
su CPU: l'ablation dell'obiettivo 4 diventa praticamente gratuito.

### 4. Classificazione e ablation

```bash
python train_downstream.py --method balanced_tokens --head ordinal
python train_downstream.py --grid     # metodi × teste × seed
```

### 5. Verifiche autonome

Ogni modulo ha un `__main__` che si autoverifica su casi con esito noto:

```bash
python utils.py        # rilevamento del collasso su 4 casi noti
python network.py      # forward I-JEPA, conteggio parametri, maschera bbox
python imbalance.py    # pesi e loss sulla distribuzione reale
python evaluation.py   # metriche su predizioni con risultato atteso
```

## Metriche

Il brief vieta implicitamente l'accuracy globale, e ha ragione: con il 61% di
PAI 3, predire sempre la maggioritaria dà 61% di accuracy e zero utilità
clinica.

Riportate: **Macro-F1**, **PR-AUC per classe**, **confusion matrix**, recall
per classe, balanced accuracy.

**In più: kappa di Cohen quadratico pesato.** Il PAI è una scala *ordinale*
(3 < 4 < 5) e confondere PAI 3 con PAI 5 è clinicamente peggio che confondere
4 con 5. La Macro-F1 non lo coglie — nei test in `evaluation.py` valuta
l'errore a due gradi *meglio* di quello a un grado (0.641 contro 0.619),
mentre il kappa quadratico lo penalizza correttamente (0.429 contro 0.857).
È l'argomento per cui c'è anche una testa ordinale accanto a quella piatta.

## Risultati

Tutti i valori: **5 semi, split di test**, encoder **congelato**. Metrica
primaria **PR-AUC sulla classe PAI 5**, come chiede la traccia. La soglia di
significatività è **|z| ≥ 2,31** — quantile di Student a 8 gradi di libertà,
non 1,96: con cinque ripetizioni la normale sottostima la coda.

### Il risultato centrale: il protocollo della traccia non misura l'encoder

Il grado PAI è in larga parte la **dimensione** della lesione, quindi il numero
di token dentro la bounding box (19 / 34 / 77 in media per PAI 3 / 4 / 5) è
quasi l'etichetta. Conseguenza misurata:

| Cosa legge il classificatore | Contenuto dell'immagine | Macro-F1 |
|---|---|---|
| **la sola maschera one-hot**, pixel azzerati | nessuno | **0,7708** |
| encoder **casuale**, vettore latente intero | tutto | 0,7705 |
| encoder **I-JEPA**, vettore latente intero | tutto | 0,7663 |
| due soglie sul lato in px della bbox, nessuna rete | nessuno | 0,7567 |

Quattro modi diversi di **non** usare l'encoder danno lo stesso numero. Non è
che gli encoder siano equivalenti: è che quel protocollo non li interroga.

### Protocollo a conteggio fisso — PR-AUC su PAI 5

Si tiene il centro della bbox e si prendono i **K token più vicini**, con lo
stesso K per ogni classe: la cardinalità non può più portare l'etichetta,
la localizzazione resta. Cambia **solo quali token si aggregano**.

| Protocollo | Conteggio | Localizzazione | Casuale | I-JEPA | Δ | z |
|---|---|---|---|---|---|---|
| `P1_bbox` — la traccia | presente | presente | 0,8758 | 0,8785 | +0,0027 | +0,84 |
| `P2b` griglia fissa | tolto | **tolta** | 0,3621 | 0,3792 | +0,0171 | +0,78 |
| **`P3_K16`** | tolto | presente | 0,3861 | **0,4713** | **+0,0851** | **+5,11** |
| **`P3_K36`** | tolto | presente | 0,3823 | **0,5031** | **+0,1209** | **+8,72** |
| **`P3_K64`** | tolto | presente | 0,3740 | **0,4553** | **+0,0813** | **+5,60** |

Le prime due righe sono i **controlli** e devono pareggiare: `P2b` toglie anche
la localizzazione, e lì I-JEPA non vince. Il vantaggio compare in una casella
su tre di un disegno a due fattori — è ciò che lo rende interpretabile.

Con una testa **MIL a livello di istanza** (un MLP per token, media delle
probabilità) su `P3_K16` il margine sale a **+0,1158 di macro-F1, z = 6,49**:
è il più grande misurato nel progetto.

### Ablation sullo sbilanciamento — obiettivo 4

Encoder casuale, protocollo `P1_bbox`, testa flat, 5 semi:

| Metodo | PR-AUC PAI 5 ↑ | Recall PAI 5 | Precisione PAI 5 |
|---|---|---|---|
| **Balanced token sampling** (la novità) | **0,8826 ± 0,0050** | 0,771 | **0,789** |
| Focal loss | 0,8730 ± 0,0117 | 0,791 | 0,784 |
| Cross-entropy pesata | 0,8706 ± 0,0091 | 0,798 | 0,737 |
| Cross-entropy semplice | 0,8676 ± 0,0065 | 0,755 | 0,774 |
| Oversampling | 0,8658 ± 0,0142 | 0,798 | 0,728 |

**+0,0168 su oversampling (z = 2,50)** e +0,0150 sulla CE semplice (z = 4,11),
e senza lo scambio recall-contro-precisione degli altri metodi.

Da dichiarare: sulla **F1** della classe rara vince `focal` di misura
(0,7863 contro 0,7794). Il primato della novità è sulla metrica primaria, non
su tutte.

Sotto conteggio fisso (`P3_K16`, encoder I-JEPA) la novità **si comporta al
contrario** che nel protocollo della traccia: lì lo danneggia (−0,0074),
qui lo aiuta di **+0,0437 di PR-AUC5, z = 2,93**, portandolo a 0,5860 di
macro-F1 — il valore più alto raggiunto nel progetto.

### Sweep di α — il massimo è interno

| α | Istanze per epoca | PR-AUC PAI 5 |
|---|---|---|
| 0,25 | 6.421 | 0,8793 |
| **0,50** | 6.894 | **0,8814** |
| 0,75 | 7.840 | 0,8747 |
| 1,00 | 10.015 | 0,8689 |

Ribilanciare di più **non** è meglio: le viste sono sottoinsiemi della stessa
lesione, quindi correlate — con ICC ρ = 0,9864 sette viste valgono 1,01
campioni indipendenti. Protocollo in due fasi: screening a 3 semi, poi le due
migliori rimisurate su **5 semi disgiunti**, perché selezionare e riportare
sugli stessi semi gonfia il vincitore di circa una deviazione.

### Il pre-training impara? Misurato durante l'addestramento

La stessa sonda ha letto **due** protocolli ogni dieci epoche, 28 misure su
279 epoche, sullo stesso encoder negli stessi istanti:

| Serie | prime 10 sonde | ultime 10 | Δ | z |
|---|---|---|---|---|
| qualità (conteggio fisso) | 0,5406 | 0,5482 | +0,0075 | +1,15 |
| **scorciatoia (bbox)** | 0,7539 | 0,7271 | **−0,0268** | **−11,27** |
| rango effettivo | 2,91 | 14,79 | ×5,1 | — |

La rappresentazione continua a cambiare, la qualità tiene, e la leggibilità
della bounding box **cala a undici errori standard**.

Il ciclo ha coperto **289 delle 300 epoche configurate** (fermato dal nostro
limitatore di potenza). Le ultime 11 non cambiano nulla di misurabile, e non è
un'assunzione: abbiamo estratto e valutato anche l'encoder dell'**epoca 288**,
che nessun criterio ha selezionato, e batte comunque il casuale del **+16 %**
sulla metrica primaria (z = 3,15).

### Limiti dichiarati

- Il checkpoint consegnato (epoca 69) è migliore **sul criterio con cui è stato
  scelto** e peggiore su letture cieche alla dimensione.
- Lo sweep di α ha usato un encoder fisso, per isolare il parametro.
- Il `±` riportato è la dispersione **fra semi**: risponde a «è riproducibile?».
  Non cattura l'errore di campionamento del test, che con 112 lesioni PAI 5
  vale circa **0,021**, quasi dieci volte tanto. Differenze sotto quella soglia
  sono riproducibili, non dimostrate generalizzabili.
- `SMOTE latente` non è fra le baseline misurate nella griglia finale.

### Presentazione

`Project8_CV_2025-2026.pptx` — 16 slide, generate da `build_deck.js`
(`npm install && node build_deck.js`).

## Riferimenti

1. Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023. · [codice](https://github.com/facebookresearch/ijepa)
2. Balestriero, R., LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv:2511.08544, 2025. · [codice](https://github.com/rbalestr-lab/lejepa)
3. Do, H.V. et al. *A Dataset of apical periodontitis lesions in panoramic radiographs for deep-learning based classification and detection.* Data in Brief 54:110486, 2024.
