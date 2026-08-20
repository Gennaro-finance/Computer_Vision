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

<!-- Da compilare dopo gli esperimenti. -->

### Bracci di confronto — la domanda vera del progetto

| Encoder | Macro-F1 ↑ | Recall PAI 5 ↑ | PR-AUC PAI 5 ↑ | Kappa quad. ↑ |
|---|---|---|---|---|
| Random (pavimento) | | | | |
| **ImageNet frozen** | | | | |
| **JEPA in-domain** | | | | |
| Supervisionato da zero | | | | |

Il confronto tra le righe 2 e 3 è la tesi del progetto. Se il JEPA in-domain
non batte il transfer da ImageNet, *quello* è il risultato da riportare.

### Ablation sullo sbilanciamento

| Metodo | Testa | Macro-F1 ↑ | Recall PAI 5 ↑ | Kappa quad. ↑ |
|---|---|---|---|---|
| Cross-entropy | | | | |
| CE pesata | | | | |
| Focal loss | | | | |
| Oversampling | | | | |
| SMOTE latente | | | | |
| **Balanced token sampling** | | | | |

Media ± deviazione standard su 5 seed: con 7:1 di sbilanciamento i margini
sono stretti e un singolo run non distingue nulla.

## Riferimenti

1. Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023. · [codice](https://github.com/facebookresearch/ijepa)
2. Balestriero, R., LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv:2511.08544, 2025. · [codice](https://github.com/rbalestr-lab/lejepa)
3. Do, H.V. et al. *A Dataset of apical periodontitis lesions in panoramic radiographs for deep-learning based classification and detection.* Data in Brief 54:110486, 2024.
