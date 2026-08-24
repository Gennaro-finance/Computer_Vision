# Briefing — Progetto 8, Self-Supervised Latent Representations for Imbalanced Apical Periodontitis Grading

Documento di consegna a un assistente AI che riprende il lavoro senza
contesto pregresso. Contiene il compito, cosa esiste, cosa e' stato
misurato e le trappole in cui il progetto e' gia' caduto.

---

## 1. Il compito (dalla traccia)

Quattro obiettivi, tutti obbligatori:

1. **Implementare I-JEPA**: Context Encoder + Target Encoder aggiornato per
   **EMA** + **shallow** Predictor. Pre-training self-supervised sulle
   radiografie, senza etichette.
2. **Valutare le rappresentazioni CONGELATE** sul grading delle lesioni,
   usando le **bounding box fornite** per estrarre i vettori latenti
   corrispondenti alle aree lesionate. L'encoder non si fine-tuna: il brief
   dice "frozen".
3. **Formulare una novita' algoritmica originale** contro lo sbilanciamento
   di classe **nello spazio latente**. Il brief cita come esempio le
   "balanced token-sampling strategies".
4. **Ablation study**: confronto della novita' contro le baseline, con
   metriche **sulla classe minoritaria** e **threshold-agnostic**.

Vincoli operativi: consegna form entro il **6 settembre 2026**,
presentazione **11 settembre 2026**.

### Metrica primaria

**PR-AUC sulla classe PAI 5** (la minoritaria). E' l'unica delle metriche
nominate dal brief che sia insieme specifica per la minoritaria e
indipendente dalla soglia. La macro-F1 e' nell'elenco ma media le tre
classi con lo stesso peso e dipende dall'argmax: si riporta, non decide.
L'accuracy globale e' esplicitamente vietata dal brief.

---

## 2. I dati

Radiografie panoramiche dentali, grading PAI ordinale **3 < 4 < 5**.

| split | immagini | lesioni |
|---|---|---|
| train | 2746 | 4719 |
| val | 588 | 1009 |
| test | 590 | 1013 |

Distribuzione dei gradi: **PAI 3 = 4277, PAI 4 = 1755, PAI 5 = 709**
(sbilanciamento 6.03:1). Split a livello di **paziente** (id estratto dal
campo `<filename>` degli XML), verificato: nessun paziente attraversa due
split.

**Pavimento onesto**: un classificatore che predice sempre la maggioritaria
ottiene macro-F1 = 2q/(1+q)/K = **0.2589** sul test (q = 0.6347). NON
confrontare la macro-F1 con q: e' l'errore che faceva dichiarare "al
livello del caso" un encoder che funzionava.

---

## 3. Cosa esiste (repo `cv-periapical-jepa`, ~4300 righe)

| file | contenuto |
|---|---|
| `globals.py` | tutte le costanti, ciascuna con il perche' del valore |
| `data.py` | parsing XML, split per paziente, `TileDataset` (pre-training), `LesionCropDataset` (downstream, finestra fissa 224 px nativi), cache dei crop |
| `network.py` | ViT, mascheramento a blocchi, Predictor, `IJEPA`, attention pooling sui token dentro la bbox, teste flat e ordinale (CORAL) |
| `imbalance.py` | baseline (CE, pesata, focal, oversampling) e **la novita': `balanced_token_sampling`** |
| `train_ssl.py` | pre-training, pannello diagnostico, cancello anti-spreco, checkpoint migliore |
| `train_downstream.py` | cache dei latenti multi-layer, addestramento teste, griglia dell'ablation |
| `evaluation.py` | metriche implementate a mano, **verificate contro scikit-learn a 4.4e-16** |
| `verify_claims.py` | ricalcola ogni numero dai file salvati; applica il criterio dichiarato |
| `run_all.py`, `make_figures.py` | pipeline e figure |

Repo separata `ijepa-anticollapse` (39 test): quattro interventi
anti-collasso implementati e testati (varianza/covarianza VICReg, schedule
EMA, mascheramento a rapporto controllato, predictor a collo di bottiglia).

### La novita' (obiettivo 3)

`balanced_token_sampling`: l'attention pooling aggrega i token dentro la
bbox. Invece di usarli sempre tutti, per le classi minoritarie si campionano
**sottoinsiemi diversi** degli stessi token, e ogni sottoinsieme diventa
un'istanza di training. E' oversampling nello spazio dei token: non duplica
vettori identici (a differenza dell'oversampling classico) e non genera
pixel sintetici (a differenza di SMOTE) — ogni vista e' genuina.

---

## 4. Risultati misurati

### Obiettivo 4 — ablation, 10 configurazioni x 5 seed, sul test

| metodo | testa | PR-AUC5 | macro-F1 | recall PAI5 |
|---|---|---|---|---|
| none | flat | 0.8800 | 0.7631 | 0.7196 |
| **balanced_tokens** | flat | 0.8787 | 0.7594 | 0.7875 |
| focal | flat | 0.8738 | 0.7567 | 0.7946 |
| none | ordinal | 0.8732 | 0.7631 | 0.7464 |
| class_weighted | flat | 0.8702 | 0.7562 | 0.7750 |
| balanced_tokens | ordinal | 0.8692 | 0.7575 | 0.7661 |

**Il risultato piu' solido del progetto**: applicando il criterio dichiarato
("PAI 3 e 4 non peggiorano, PAI 5 migliora") contro la cross-entropy
semplice, l'unico metodo che lo supera e' **`balanced_tokens/ordinal`**:
recall PAI 5 **+0.046 a 3.2 errori standard**, senza perdere sulle altre
classi.

### Obiettivo 1/2 — il pre-training non batte l'inizializzazione casuale

Encoder migliore (epoca 59, EMA 0.9996): downstream macro-F1 **0.7672 +-
0.0121** e **0.7681 +- 0.0033** a 5 seed sul test, contro **0.7705 +-
0.0111** di un ViT con pesi casuali. Differenza **sotto un errore standard**.

**Indizio non ancora verificato**: sulla PR-AUC di PAI 5 — la metrica
primaria — il pre-training sta sopra il casuale in **14 misure su 15**
(media 0.868 contro 0.861). ATTENZIONE: sono misure ripetute dello stesso
run, quindi **correlate**; serve una verifica a 5 seed indipendenti sul test
prima di dichiararlo.

### Perche' il soffitto e' basso, misurato

- Il grado PAI e' in larga parte **dimensione e scurezza** della
  radiotrasparenza. Lato mediano della bbox: **57 / 80 / 127 px** per PAI
  3 / 4 / 5. **Due sole soglie** su quel numero danno macro-F1 0.7567 e
  kappa 0.7779 sul test, senza usare nessuna rete.
- Un ViT **casuale** codifica gia' intensita' media (R^2 **0.99**) e
  dimensione (R^2 **0.89**) della lesione. Parte gia' vicino al soffitto.
- Le panoramiche dentali sono auto-simili: rango effettivo dei **pixel
  grezzi** = **1.12**, con il 97.3% della norma nella direzione media.
  Predire una regione mascherata non richiede comprensione: la loss scende
  a 0.086 in 9 epoche **sia mascherando il 54% sia l'80%**.

---

## 5. Trappole — leggere prima di proporre qualcosa

Il progetto ha perso giorni ottimizzando contro **tre metriche sbagliate**.

1. **Rango effettivo non centrato**: vale **1.07 sull'encoder casuale** e
   1.12 sui pixel grezzi. Segna "collasso totale" su qualunque cosa, perche'
   la componente media domina. Un allarme di collasso basato su questo e'
   un artefatto. Nei run il rango e' **salito** da 2.9 a 10.2 senza alcun
   beneficio a valle: non e' un obiettivo da inseguire.
2. **k-NN probe**: misura la **geometria**, non l'informazione. E' crollato
   da 0.73 a 0.34 mentre R^2 per intensita' e dimensione restava a 0.99 e
   0.87 **e il downstream saliva**. Un cancello basato sul k-NN interrompe
   run che stanno migliorando.
3. **Sonda lineare su un solo layer**: sui due punti dove si conoscono
   entrambi i numeri ha il **segno invertito** rispetto al downstream vero.

**L'unica misura affidabile** e' il downstream stesso: attention pooling +
testa addestrata sui layer 2+7+11 dell'encoder congelato, valutato su
**validation** (mai sul test, che si usa solo per il numero finale). Costa
63 secondi ed e' implementato in `train_ssl.sonda_downstream`.

### Interventi gia' provati, con esito

| intervento | esito |
|---|---|
| EMA 0.996 -> 0.9996 | **efficace**: sonda da 0.435 a 0.712. Da tenere |
| predictor 384 -> 96 | gia' fatto, e' il 3.2% del riferimento I-JEPA |
| mascheramento 54% -> 80% | **nessun effetto**: loss identica, risultati leggermente peggiori |
| rimozione augmentation fotometriche e jitter di scala | corretto: insegnavano invarianza a intensita' e dimensione, cioe' al segnale |
| VICReg | implementato e testato, **mai lanciato**. E' un'ESTENSIONE: I-JEPA non lo prevede, va dichiarato come tale |

---

## 6. Stato e cosa manca

**Modello scelto**: `runs/checkpoints/ijepa_vit_small_completa_best.pt`,
epoca 59, selezionato sul validation (downstream 0.7700), configurazione del
paper con EMA 0.9996.

Mancano:

1. **Una misura**: griglia a 5 seed sul test con quel checkpoint, piu' il
   braccio casuale con lo stesso protocollo (~70 min). Serve a verificare
   l'indizio sulla PR-AUC.
2. Figure, slide, README con il link Mendeley al dataset.
3. Form entro il 6 settembre.

---

## 7. Come lavorare su questo progetto

- **Ogni numero deve essere verificabile**: `verify_claims.py` ricalcola
  tutto dai file salvati invece di trascriverlo. Non riportare valori a
  memoria.
- **Non allontanarsi dalla traccia.** Iperparametri di I-JEPA (mascheramento,
  EMA, capacita' del predictor) sono liberi. Cambiare l'obiettivo di
  pre-training (VICReg, SIGReg/LeJEPA, DINO) e' un'estensione e va
  dichiarata. Sostituire l'EMA viola l'obiettivo 1.
- **Nessun risultato addolcito.** Se una differenza e' dentro il rumore, si
  dice. Con 5 seed la deviazione standard della macro-F1 e' 0.004-0.011:
  differenze sotto 0.01 non sono dichiarabili.
- L'hardware si spegne da solo sotto carico GPU (5 eventi Kernel-Power 41,
  alimentatore 330 W al limite). I checkpoint sono atomici e il resume
  rifiuta iperparametri diversi da quelli registrati.
