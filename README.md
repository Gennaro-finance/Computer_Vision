# Self-Supervised Latent Representations for Imbalanced Apical Periodontitis Grading

**Il protocollo di valutazione richiesto dalla traccia non misura l'encoder.**
La sola bounding box, senza un pixel, dà macro-F1 0,7708 — quanto un encoder
con pesi casuali che vede l'immagine intera. Tolto quel canale, I-JEPA vince
fino a **+32 % di PR-AUC sulla classe rara**.

| | |
|---|---|
| **Progetto** | 8 — Computer Vision A.A. 2025-2026, Prof. Irene Amerini · Sapienza, ALCOR Lab |
| **Gruppo** | Nome 1 (matricola) · Nome 2 (matricola) · Nome 3 (matricola) |
| **Sessione** | 11 settembre 2026 |
| **Framework** | PyTorch |
| **Presentazione** | `Project8_CV_2025-2026.pptx` (21 slide, generata da `build_deck.js`) |

---

## Il problema, e cosa abbiamo trovato

Il grado PAI è in larga parte la **dimensione** della lesione. La traccia
chiede di usare la bounding box per estrarre i vettori latenti — ma la bbox
seleziona **19 / 34 / 77 token** in media per PAI 3 / 4 / 5, quindi il loro
numero è quasi l'etichetta. È un caso da manuale di **bag-size bias**.

| Cosa legge il classificatore | Immagine | Macro-F1 |
|---|---|---|
| la sola maschera one-hot, pixel azzerati | nessuna | **0,7708** |
| encoder **casuale**, vettore latente intero | tutta | 0,7705 |
| encoder **I-JEPA**, vettore latente intero | tutta | 0,7663 |
| due soglie sul lato in px della bbox, nessuna rete | nessuna | 0,7567 |

Quattro modi diversi di **non** usare l'encoder danno lo stesso numero.

**La correzione**: si tiene il *centro* della bbox e si prendono i **K token
più vicini**, stesso K per ogni classe. La cardinalità non porta più
l'etichetta, la localizzazione resta. Cambia solo quali token si aggregano —
stesso ritaglio, stessa risoluzione, stessi token dell'encoder, stessi semi.

---

## Risultati

5 semi, split di test, encoder **congelato**. Metrica primaria **PR-AUC su
PAI 5**. Soglia |z| ≥ 2,31 (Student, 8 g.d.l., non 1,96: con cinque
ripetizioni la normale sottostima la coda).

### Obiettivo 2 — le rappresentazioni congelate

| Protocollo | Conteggio | Localizz. | Casuale | I-JEPA | Δ | z |
|---|---|---|---|---|---|---|
| `P1_bbox` — la traccia | presente | presente | 0,8758 | 0,8785 | +0,0027 | +0,84 |
| `P2b` griglia fissa | tolto | **tolta** | 0,3621 | 0,3792 | +0,0171 | +0,78 |
| **`P3_K16`** | tolto | presente | 0,3861 | **0,4713** | **+0,0851** | **+5,11** |
| **`P3_K36`** | tolto | presente | 0,3823 | **0,5031** | **+0,1209** | **+8,72** |
| **`P3_K64`** | tolto | presente | 0,3740 | **0,4553** | **+0,0813** | **+5,60** |

Le prime due righe sono **controlli** e devono pareggiare: `P2b` toglie anche
la localizzazione, e lì I-JEPA non vince. Il vantaggio compare in una casella
su tre di un disegno a due fattori — è ciò che lo rende interpretabile, e
significativo su tutti e tre i valori di K.

Con una testa **MIL a livello di istanza** su `P3_K16` il margine sale a
**+0,1158 di macro-F1 (z = 6,49)**, il più grande del progetto.

### Obiettivi 3 e 4 — la novità e la sua ablation

`balanced_token_sampling`: per le classi rare si campionano **sottoinsiemi
diversi** dei token della stessa lesione, e ogni sottoinsieme è un'istanza di
training. Non duplica vettori identici come l'oversampling, non inventa punti
come SMOTE.

| Metodo | PR-AUC PAI 5 ↑ | Recall 5 | Precisione 5 |
|---|---|---|---|
| **Balanced token sampling** | **0,8826 ± 0,0050** | 0,771 | **0,789** |
| Focal loss | 0,8730 ± 0,0117 | 0,791 | 0,784 |
| Cross-entropy pesata | 0,8706 ± 0,0091 | 0,798 | 0,737 |
| Cross-entropy semplice | 0,8676 ± 0,0065 | 0,755 | 0,774 |
| Oversampling | 0,8658 ± 0,0142 | 0,798 | 0,728 |

**+0,0168 su oversampling (z = 2,50)**, senza lo scambio recall-contro-precisione
degli altri. Sulla F1 della classe rara vince `focal` di misura (0,7863 contro
0,7794): il primato è sulla metrica primaria, non su tutte.

Lo **sweep di α** ha il massimo interno — 0,8814 a α = 0,5 contro 0,8689 a
α = 1,0. Ribilanciare di più non è meglio: le viste sono sottoinsiemi della
stessa lesione, e con ICC ρ = 0,9864 sette viste valgono 1,01 campioni
indipendenti.

E la novità **si comporta al contrario nei due protocolli**: nel protocollo
della traccia danneggia I-JEPA, sotto conteggio fisso lo aiuta di +0,0437
(z = 2,93).

### Obiettivo 1 — il pre-training impara?

La stessa sonda ha letto due protocolli ogni dieci epoche, 28 misure su 279:

| Serie | prime 10 sonde | ultime 10 | Δ | z |
|---|---|---|---|---|
| qualità (conteggio fisso) | 0,5406 | 0,5482 | +0,0075 | +1,15 |
| **scorciatoia (bbox)** | 0,7539 | 0,7271 | **−0,0268** | **−11,27** |
| rango effettivo | 2,91 | 14,79 | ×5,1 | — |

La rappresentazione continua a cambiare, la qualità tiene, e la leggibilità
della bounding box cala a undici errori standard.

---

## Dati

Radiografie panoramiche con bounding box e grado PAI per lesione, **split a
livello di paziente** (l'id viene dal campo `<filename>` degli XML: dividere
per lesione farebbe passare informazione fra train e test).

| Split | Immagini | Lesioni | PAI 3 | PAI 4 | PAI 5 |
|---|---|---|---|---|---|
| train | 2.746 | 4.719 | 3.017 | 1.229 | 473 |
| validation | 588 | 1.009 | 617 | 268 | 124 |
| test | 590 | 1.013 | 643 | 258 | 112 |

Fonte: Do, H. V. et al., *Data in Brief* 54:110486 (2024) ·
[Mendeley Data, DOI 10.17632/kx52tk2ddj.3](https://data.mendeley.com/datasets/kx52tk2ddj/3)

---

## Come si esegue

```bash
pip install -r requirements.txt
python data.py --inspect --bbox-stats --splits     # dati e split, sempre per primo
python train_ssl.py --variant vit_small --epochs 300 --tag finale
python train_downstream.py --cache --layers 2 7 11 --ckpt-tag finale_best --tag _geo_finale
python train_downstream.py --cache --layers 2 7 11 --random --tag _casuale
python train_downstream.py --grid --layers 2 7 11 --tag _geo_finale
python exp_fixedk.py --tag _casuale _geo_finale    # il risultato centrale
python verify_claims.py                            # ricalcola ogni numero dai file salvati
```

Su GPU con alimentazione al limite, anteporre
`python sorveglia.py --tetto 95 --tetto-temp 86 --` a qualunque comando.

---

## Struttura del codice

Segue la struttura concettuale richiesta dal corso. **Ogni file dichiara in
prima riga del proprio docstring a quale famiglia appartiene**, così la
pipeline non si confonde con gli script che hanno prodotto i numeri.

| Marca | Significato | File |
|---|---|---|
| `PIPELINE` | il codice consegnato, quello che gira | `globals` `utils` `data` `network` `imbalance` `train_ssl` `train_downstream` `evaluation` `run_all` `verify_claims` |
| `MISURAZIONE` | misura una metrica su un braccio | 5 script `exp_*` |
| `CONFRONTO` | confronta una metrica fra bracci | 13 script `exp_*` |
| `DIAGNOSI` | indaga una discrepanza fra misure | `exp_scarto` |
| `INFRASTRUTTURA` | supporto, non richiesto dalla traccia | `catena` `sorveglia` `stato` `figure_finali` … |

L'architettura: ViT-S/16 (12 blocchi, dim 384, 6 teste) come **context
encoder** e **target encoder** a EMA, più un **predictor** deliberatamente
stretto — 4 blocchi a dim 96, il 2,4 % di un encoder. Finestra 224 px →
griglia 14×14 di patch da 16 px → **196 token**, numero fisso per costruzione.
A valle si concatenano i blocchi 2, 7 e 11 (1.152 dim) e si addestrano solo
attention pooling e testa.

---

## Limiti dichiarati

- Il ciclo ha coperto **289 delle 300 epoche** configurate, fermato dal
  limitatore di potenza. Le ultime 11 non cambiano nulla di misurabile, e non
  è un'assunzione: abbiamo valutato anche l'encoder dell'**epoca 288**, che
  nessun criterio ha selezionato, e batte comunque il casuale del **+16 %**
  sulla metrica primaria (z = 3,15).
- Il checkpoint consegnato (epoca 69) è migliore **sul criterio con cui è
  stato scelto** e peggiore su letture cieche alla dimensione.
- Lo sweep di α ha usato un encoder fisso, per isolare il parametro.
- Il `±` riportato è la dispersione **fra semi**: non cattura l'errore di
  campionamento del test, che con 112 lesioni PAI 5 vale circa **0,021**.
  Differenze sotto quella soglia sono riproducibili, non dimostrate
  generalizzabili.
- `SMOTE latente` non è fra le baseline della griglia finale.

---

## Riferimenti

1. Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* CVPR 2023 · [codice](https://github.com/facebookresearch/ijepa)
2. Ilse, M., Tomczak, J., Welling, M. *Attention-based Deep Multiple Instance Learning.* ICML 2018
3. Do, H. V. et al. *A Dataset of apical periodontitis lesions in panoramic radiographs.* Data in Brief 54:110486, 2024
4. Saito, T., Rehmsmeier, M. *The precision-recall plot is more informative than the ROC plot on imbalanced datasets.* PLoS ONE 10(3), 2015
