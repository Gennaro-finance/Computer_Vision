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

Le prime righe delle 30 configurazioni misurate (3 encoder x 10 config x 5
seed), ordinate per PR-AUC su PAI 5:

| encoder | metodo | testa | PR-AUC5 | macro-F1 | F1 PAI5 | prec5 |
|---|---|---|---|---|---|---|
| **casuale** | **balanced_tokens** | flat | **0.8813** | 0.7676 | **0.790** | 0.814 |
| spinto | class_weighted | flat | 0.8798 | 0.7632 | 0.778 | 0.785 |
| spinto | oversample | flat | 0.8778 | 0.7607 | 0.774 | 0.789 |
| spinto | none | flat | 0.8772 | 0.7676 | 0.769 | 0.819 |
| casuale | none | flat | 0.8758 | **0.7705** | 0.777 | 0.840 |
| completa | none | flat | 0.8753 | 0.7672 | 0.776 | 0.797 |

**Il risultato piu' solido del progetto**: `balanced_token_sampling` ha
insieme la PR-AUC piu' alta (0.8813) E la F1 sulla classe rara piu' alta
(0.790) di TUTTE E TRENTA le configurazioni, con precisione 0.814 - cioe'
senza il tipico scambio recall-contro-precisione degli altri tre metodi
(focal arriva a recall 0.8054 ma con precisione 0.763).

ATTENZIONE: entrambi i primati sono sull'encoder CASUALE. La novita'
funziona; il pre-training no. Le due cose vanno riportate insieme.

Una versione precedente di questo documento dava la novita' a "+0.046 a
3.2 errori standard": quel valore veniva da un encoder poi sostituito. Sul
confronto attuale il margine in recall e' piu' stretto - va citata la F1 e
la PR-AUC, che sono i primati veri.

### Obiettivo 4 — ablation su alpha, il parametro della novita'

FATTO il 26 ago su Colab (T4), encoder casuale, testa flat, misure sul test.
Protocollo in due fasi: screening a 3 seed su tutte le alpha, poi le due
migliori rimisurate su 5 seed DISGIUNTI da quelli dello screening (con 4
candidati e una deviazione di ~0.006, selezionare e riportare sugli stessi
seed gonfia il vincitore di circa una deviazione, dello stesso ordine
dell'effetto cercato).

alpha regola quante viste riceve ogni classe: n_c = ceil((max/n_c)^alpha).

| alpha | viste [1,2,5] | istanze | PR-AUC5 (3 seed) | recall5 | prec5 |
|---|---|---|---|---|---|
| 0.25 | [1,2,2] | 6.421 | 0.8793 ±0.0049 | 0.7381 | 0.833 |
| **0.50** | [1,2,3] | 6.894 | **0.8814 ±0.0071** | 0.7738 | 0.810 |
| 0.75 | [1,2,5] | 7.840 | 0.8747 ±0.0028 | 0.7946 | 0.756 |
| 1.00 | [1,3,7] | 10.015 | 0.8689 ±0.0069 | 0.7708 | 0.783 |

Fase 2, 5 seed nuovi: alpha 0.50 -> 0.8797 ±0.0043, alpha 0.25 -> 0.8775
±0.0048. Distanza 0.8 errori standard: equivalenti.

**Il massimo e' interno.** alpha 0.50 batte alpha 1.00 di +0.0125 a 2.2
errori standard. Ribilanciare di piu' non e' meglio: le viste sono
sottoinsiemi di token della STESSA lesione, quindi fortemente correlate, e
sette viste di un PAI 5 non fanno sette esempi - fanno un esempio contato
sette volte, che sposta il confine di decisione senza aggiungere
informazione.

**Previsione smentita.** `exp_alpha.py` prevedeva alpha 1.00 come vincitore,
perche' a 10.015 istanze entra nello stesso regime di `oversample` (9.051).
E' invece la peggiore. Il docstring del file conserva la previsione: e'
l'ipotesi che l'esperimento ha falsificato, non un errore da correggere.

**Il confronto piu' pulito per l'obiettivo 3** e' ad alpha 0.75, dove la
novita' raggiunge lo stesso punto di lavoro di `oversample` sullo stesso
encoder e stessa testa - recall 0.7946 contro 0.7929, precisione 0.756
contro 0.752 - ma con PR-AUC 0.8747 contro 0.8636, **+0.0111**. Stesso
ribilanciamento, stesso compromesso recall/precisione, ordinamento
migliore: la differenza e' attribuibile ai sottoinsiemi di token genuini
invece dei duplicati identici.

**Riproducibilita'.** alpha 0.50 e' stato misurato tre volte in modo
indipendente: 0.8813 (griglia, seed 0-4), 0.8814 (screening, seed 0-2),
0.8797 (fase 2, seed 10-14). Scarto massimo 0.0017.

Conseguenza pratica: il default alpha=0.5, scelto senza ottimizzarlo, era
gia' il migliore. Tutti i numeri della griglia restano validi come sono.
Dati in `runs/sweep_alpha_vit_small_L2-7-11_casuale.json`, figura
`runs/figures/fin5_alpha.png`.

### Obiettivo 1/2 — il pre-training non batte l'inizializzazione casuale

VERIFICATO il 25 ago con il braccio casuale misurato collo stesso
protocollo, 10 configurazioni x 5 seed sul test:

    encoder     macro-F1 media (max)   PR-AUC5 media (max)   dev. media
    casuale       0.7584  (0.7705)       0.8719  (0.8813)       0.0096
    spinto        0.7611  (0.7676)       0.8746  (0.8798)       0.0066
    completa      0.7545  (0.7681)       0.8650  (0.8753)       0.0087

Test del segno appaiato, spinto contro casuale: macro-F1 6/10 (p=0.377),
PR-AUC5 8/10 (p=0.055), kappa 7/10 (p=0.172). Nessuno significativo, e
ENTRAMBI I MASSIMI ASSOLUTI restano dell'encoder casuale.

L'indizio sulla PR-AUC riportato in precedenza - "sopra il riferimento in
14 misure su 15" - era un ARTEFATTO: misure ripetute dello stesso run,
correlate, confrontate con un riferimento a un seed su validation. La
verifica indipendente lo ha smentito.

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

Il progetto ha perso giorni ottimizzando contro **quattro metriche
sbagliate**, in quest'ordine.

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
4. **La sonda downstream su validation, a UN SEED** (`sonda_downstream`).
   Diceva che il pre-training vinceva in **22 misure su 23**. La verifica a
   5 seed indipendenti sul test: **vince 2 righe su 10**. Usa un
   sottoinsieme fisso di 2500 lesioni, seed 0 sempre, e valuta su
   validation: e' un errore SISTEMATICO, non rumore. Serve a fermare un run
   che sta chiaramente degradando, MAI a dichiarare un risultato.

**L'unica misura affidabile** e' la **griglia a 5 seed sul test**
(`train_downstream.py --grid`). Ogni conclusione difendibile di questo
progetto passa da li'. Tutto il resto e' diagnostica.

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

1. Le quattro teste rimanenti (norm, mlp e varianti ordinali): le due
   misurate dicono che il LayerNorm non sposta la media ma DIMEZZA la
   dispersione, da 0.0079 a 0.0031.
2. Slide, README con il link Mendeley al dataset.
3. Form entro il 6 settembre.

SWEEP DI ALPHA: chiuso, vedi sezione 4. Il default era gia' il migliore.

CONFRONTO PRINCIPALE: chiuso. Il braccio casuale e' misurato, l'indizio
sulla PR-AUC verificato e smentito. Non serve altra sperimentazione
sull'encoder.

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
