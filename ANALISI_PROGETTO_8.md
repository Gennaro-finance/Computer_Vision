# Progetto 8 — Analisi approfondita

**Self-Supervised Latent Representations for Imbalanced Apical Periodontitis Grading**
Computer Vision A.A. 2025-2026, Prof. Irene Amerini · Sessione 11 settembre 2026 · 3 persone · Kaggle free

---

## Sommario esecutivo

Il progetto è **fattibile in 25 giorni, ma solo con tre decisioni tecniche prese correttamente nei primi tre giorni**. Non è fattibile se lo si affronta come descritto letteralmente nel brief.

Le due scoperte che determinano tutto:

1. **Un problema di scala che il brief nasconde.** Il brief dice di pre-addestrare su "resized panoramic images". Una radiografia panoramica inquadra l'intera dentatura; una lesione periapicale è di pochi millimetri. Ridimensionando la panoramica a 224×224, una lesione diventa più piccola di un singolo patch token da 16×16. Preso alla lettera, il pipeline non può funzionare. Va risolto con un approccio a tile a risoluzione nativa — ed è la prima cosa da verificare, misurando la distribuzione delle bbox.

2. **Un'evidenza che rende il progetto credibile.** Il timore ovvio è che il self-supervised learning su ~4.000 immagini collassi. LeJEPA (Balestriero & LeCun, 2025 — che è la ref [2] del vostro brief) documenta esattamente il contrario: pre-training *in-domain* su Galaxy10, un dataset specialistico di scala comparabile, con backbone piccoli (ConvNeXt-V2 Nano, ResNet-34), che **batte il transfer learning da DINOv3** (82.72% vs 81.60% a supervisione piena). La tesi del vostro progetto ha una conferma indipendente in letteratura, su un dominio diverso ma con la stessa struttura di problema. È il vostro argomento più forte nella sezione State of the Art.

Verdetto in una riga: **più interessante del Progetto 2, e più rischioso**. La differenza sostanziale è che P2 è a stadio singolo e il risultato è garantito; P8 è a due stadi e se il primo stadio fallisce il secondo non ha niente su cui lavorare. Se lo scegliete, serve un gate di decisione al 26 agosto (§9).

---

## 1. I numeri reali del dataset

Fonte: Do, H.V. et al. (2024), *Data in Brief* 54:110486 · [Mendeley Data, DOI 10.17632/kx52tk2ddj.3](https://data.mendeley.com/datasets/kx52tk2ddj/3)

| | |
|---|---|
| Radiografie sottoposte a screening | 16.519 |
| **Immagini con lesioni (pubblicate)** | **3.926** |
| Totale con augmentation fornita | 17.004 |
| **Lesioni etichettate** | **6.029** (≈ 1,54 per immagine) |
| Provenienza | High-quality Dental Treatment Centre, School of Dentistry, Hanoi Medical University |
| Periodo di raccolta | gennaio 2016 – marzo 2021 |
| Annotatori | 3 dentisti esperti |
| Struttura | tre cartelle: *Original JPG Images*, *Augmentation JPG Images*, *Image Annotations* |

### Distribuzione delle classi

| Grado PAI | Lesioni | Quota | Rapporto vs PAI 5 |
|---|---|---|---|
| PAI 3 | 3.691 | 61,2 % | 7,08 : 1 |
| PAI 4 | **1.817** *(derivato)* | 30,1 % | 3,49 : 1 |
| PAI 5 | 521 | 8,6 % | 1 : 1 |

I valori PAI 3 e PAI 5 vengono dal vostro brief; **PAI 4 l'ho ottenuto per sottrazione** da 6.029 e va verificato leggendo gli XML — non l'ho trovato dichiarato in una fonte. È il primo numero da confermare.

Nota: uno sbilanciamento 7:1 è **serio ma non estremo**. Molti lavori sul long-tail affrontano 100:1 o 1000:1. Questo è un dato importante per calibrare le aspettative: significa che le baseline standard (class-weighted CE, focal loss, oversampling) funzioneranno *decentemente*, e quindi la vostra novità deve batterle su un margine ristretto. Non aspettatevi guadagni spettacolari, e progettate la valutazione per rilevare differenze piccole — il che significa **più seed e intervalli di confidenza**, non un singolo run.

---

## 2. Il problema di scala (la cosa più importante di questa analisi)

Il brief, nell'obiettivo 1, dice che la rete "will process unannotated, **resized** panoramic images". Nel Task dice "predict the embeddings of missing patches from **localized contexts**".

Queste due frasi sono in tensione, e la seconda ha ragione.

**L'aritmetica.** Una radiografia panoramica tipica è dell'ordine di 2900×1400 px e inquadra l'intera arcata dentaria. Una lesione periapicale misura pochi millimetri, quindi qualche decina di pixel per lato. Se ridimensionate l'immagine intera a 224×224:

```
fattore di scala ≈ 224 / 2900 ≈ 0,077
lesione da ~50 px  →  ~4 px nell'immagine ridimensionata
patch token ViT     →  16 px
```

**La lesione finisce sotto la dimensione di un singolo token.** Il downstream chiede di "extract the latent vectors corresponding to the lesion areas" usando le bbox: se la bbox copre 4 px, corrisponde a una frazione di un token, e il vettore latente che estraete descrive un pezzo di mandibola generico, non la lesione. Tutto ciò che viene dopo — attention pooling, testa di classificazione, novità sullo sbilanciamento — lavora su feature che non contengono il segnale.

### Come si risolve

**Pre-training a tile su risoluzione nativa.** Non ridimensionate la panoramica: tagliatela in tile da 224×224 (o 256×256) a risoluzione originale, con sovrapposizione. Da 3.926 immagini si ottengono ~20–25k tile, che è anche una quantità di dati di pre-training più sana. Le lesioni restano alla loro scala reale e una bbox copre diversi token.

Varianti da considerare, in ordine di preferenza:

| Approccio | Pro | Contro |
|---|---|---|
| **Tile a risoluzione nativa** | scala corretta, più campioni di pre-training | i tile perdono il contesto globale dell'arcata |
| Tile centrati sulla regione apicale | ancora più efficiente | usa le annotazioni → non è più puro SSL |
| Input a risoluzione più alta (es. 512×256, patch 16 → 512 token) | mantiene il contesto globale | ~2,6× il costo per immagine, lesioni ancora piccole |
| Panoramica intera a 224² | quello che dice il brief | **non funziona**, vedi sopra |

Raccomandazione: **tile a risoluzione nativa** come impostazione primaria, e un braccio di confronto con la panoramica ridimensionata *proprio per dimostrare empiricamente che non funziona*. Quel confronto è materiale eccellente per la presentazione: mostrate che avete individuato un problema di design e lo avete quantificato, invece di seguire il brief alla lettera. Le guidelines premiano esplicitamente "critical thinking and problem solving skills".

### Il primo task, giorno 1

Prima di scrivere una riga di modello, aprite gli XML e calcolate:

- distribuzione delle dimensioni delle immagini (H, W);
- distribuzione delle dimensioni delle bbox in pixel, e **in frazione dell'immagine**;
- quanti token da 16×16 coprirebbe la bbox mediana a ciascuna risoluzione di input candidata (224², 384², 512×256, tile nativi).

È mezza giornata di lavoro e determina l'intera architettura. Fatelo prima di tutto il resto. Se la bbox mediana copre meno di ~4 token nella configurazione scelta, la configurazione è sbagliata.

---

## 3. I-JEPA o LeJEPA? Il brief vi mette in una contraddizione

**Obiettivo 1 del brief**, alla lettera: implementare "a Context Encoder, a Target Encoder updated via Exponential Moving Average (EMA), and a shallow Predictor network".

Quella è la struttura di **I-JEPA** (Assran et al., CVPR 2023).

**LeJEPA** (Balestriero & LeCun, 2025), che il brief cita come ref [2] e nomina nel Task, rimuove precisamente quei componenti: nessuno stop-gradient, nessun teacher-student, nessun EMA, nessuno scheduler di iperparametri. Sostituisce tutto con **SIGReg** (Sketched Isotropic Gaussian Regularization), che vincola gli embedding a distribuirsi come una gaussiana isotropa usando proiezioni randomizzate e statistiche 1-D. Il risultato: un singolo iperparametro di trade-off, complessità lineare in tempo e memoria, ~50 righe di codice, e stabilità dichiarata attraverso architetture e iperparametri.

Se implementate LeJEPA puro, **non soddisfate l'obiettivo 1 come è scritto**.

### La strategia che risolve entrambi i problemi

Implementate **I-JEPA come metodo primario** — context encoder, target encoder con EMA, predictor shallow: l'obiettivo 1 è soddisfatto alla lettera. Poi aggiungete **SIGReg come braccio di confronto** nell'ablation.

Perché è la scelta giusta su quattro fronti:

1. **Conformità:** l'obiettivo 1 richiede l'EMA, e l'avete.
2. **Letteratura:** dimostrate di aver letto la ref [2] del vostro stesso brief, che è un paper di novembre 2025. In una presentazione da 10 minuti questo si nota.
3. **Ablation gratuito:** un asse di confronto in più senza lavoro aggiuntivo di progettazione.
4. **Assicurazione sul collasso** — e questo è il motivo vero. I metodi basati su EMA sono notoriamente instabili fuori dal regime di iperparametri per cui sono stati tarati, e il vostro regime è lontanissimo da quello di I-JEPA (§4). Se I-JEPA collassa su 4k immagini, avere SIGReg già implementato è la differenza tra perdere due giorni e perdere due ore. `pip install lejepa` vi dà la loss direttamente.

**Implementate il braccio SIGReg presto, non alla fine.** È contro-intuitivo — sembra un extra — ma è la mitigazione di rischio più efficace di tutto il progetto.

---

## 4. Il regime di compute: quanto siete lontani dall'originale

| | I-JEPA originale | Voi |
|---|---|---|
| Dataset | ImageNet-1K, 1,28 M immagini | 3.926 immagini (~20–25k tile) |
| Hardware | 16 × A100 80 GB | 1 × T4 / P100 16 GB |
| Batch effettivo | 2.048 | 64–128 |
| Epoche | 300 | ? |
| Backbone | ViT-H/14 (632 M param) | **ViT-Tiny / Small (5–22 M)** |

Siete a ~0,3% dei dati e ~1/100 del compute. Questo **non** significa che il progetto sia irrealizzabile: significa che ogni scelta tarata su ImageNet va rimessa in discussione.

### La buona notizia: il pre-training costa poco a questa scala

Stima per ViT-Ti/16 a 224², batch 64, su T4 con AMP:

```
~24.000 tile / 64  ≈  375 step per epoca
a ~5 it/s          ≈  75 s per epoca
300 epoche         ≈  6,3 ore
```

300 epoche stanno dentro una sessione Kaggle da 12 h. Se ne servono più — e a questa scala di dati è probabile: 600 epoche fanno ~12,5 h, 800 fanno ~16,7 h — sforate la sessione singola, quindi il **checkpointing con resume non è opzionale, è la precondizione**. Prevedetelo dal primo commit, non dopo il primo crash.

**Il collo di bottiglia non è il compute: è la diversità dei dati.**

### Nota pratica sull'hardware Kaggle

Kaggle offre P100 e T4. **Preferite T4**: la P100 è architettura Pascal, non ha supporto bf16 e il fp16 è limitato, quindi perdete il vantaggio dell'AMP. Su T4 usate fp16 mixed precision. Se vi capita la P100, accettate fp32 e riducete il batch.

### Il backbone: piccolo, e non è un compromesso

Il risultato Galaxy10 di LeJEPA usa **ConvNeXt-V2 Nano e ResNet-34** — modelli da 15–22 M parametri — e batte DINOv3 ViT-S/16 in transfer. Non è un ripiego: a questa scala di dati un modello piccolo è la scelta *corretta*, perché un ViT-B su 4k immagini overfitta la struttura anatomica globale senza imparare nulla di locale.

Usate **ViT-Tiny/16** (5 M) o **ViT-Small/16** (22 M). E notate: i checkpoint I-JEPA rilasciati da Meta sono solo ViT-H/14 e ViT-g/16, quindi non esiste un checkpoint piccolo da cui partire. Pre-addestrate da zero, che è comunque quello che il progetto chiede.

---

## 5. Collasso della rappresentazione: come accorgersene in tempo

È il modo in cui questo progetto fallisce, e fallisce **silenziosamente**. La loss di I-JEPA può scendere in modo perfettamente sano mentre tutti gli embedding convergono a una costante: la predizione di un target costante è banalmente facile.

Se vi accorgete del collasso il 5 settembre, il progetto è finito. Quindi strumentate il training dal primo giorno.

**Da loggare a ogni epoca, non solo la loss:**

| Segnale | Cosa indica | Sintomo del collasso |
|---|---|---|
| Deviazione standard degli embedding per dimensione | quanto variano le feature | → 0 |
| Rango effettivo della covarianza degli embedding (participation ratio) | quante direzioni sono usate | crolla verso 1 |
| Norma media degli embedding | scala delle rappresentazioni | deriva verso 0 o esplode |
| **k-NN probe su un piccolo sottoinsieme etichettato** | utilità reale delle feature | resta al livello del caso |

L'ultimo è il più importante e il più economico: ogni 20 epoche, congelate l'encoder, estraete le feature di un sottoinsieme etichettato, e fate un k-NN a 20 vicini. Se dopo 100 epoche il k-NN è al livello della classe maggioritaria (61%), qualcosa non funziona. È il vostro segnale d'allarme precoce, e costa secondi.

Nota interessante da citare in presentazione: LeJEPA rivendica che la sua loss **correla** con la performance del linear probe downstream, cosa che permette la selezione del modello senza probing supervisionato. Per I-JEPA questa proprietà non è garantita — motivo in più per monitorare rango e varianza esplicitamente.

---

## 6. Le trappole di data leakage (tre, non una)

Il brief dice: "Students must handle validation/testing splits strictly at the patient level to avoid data leakage." Ci sono tre problemi distinti, e il brief ne nomina solo uno.

### 6.1 Non esistono identificatori di paziente documentati

Nessuna delle fonti che ho consultato documenta ID paziente nel dataset. In un contesto di screening è plausibile che ci sia una panoramica per paziente, ma **è un'assunzione, non un fatto**.

Come gestirlo: verificate negli XML e nei nomi file se c'è un identificativo. Se non c'è, splittate a livello di **immagine** e dichiarate esplicitamente l'assunzione "una radiografia = un paziente", con la nota che se il dataset contenesse più esami dello stesso paziente ci sarebbe un residuo di leakage non eliminabile. Dichiararlo apertamente in una slide è la cosa corretta e vi copre: state rispettando lo spirito del vincolo con i dati che avete.

### 6.2 La cartella di augmentation è la trappola vera

Il dataset fornisce 17.004 immagini derivate da 3.926 originali tramite scaling, mirroring e flipping. Se splittate i 17.004 casualmente, **varianti geometriche della stessa radiografia finiscono in train e in test**. Le metriche si gonfiano e il risultato è privo di significato.

Regola operativa: **costruite gli split sui 3.926 originali**, poi propagate. Per il pre-training SSL, la mia raccomandazione è più netta: **ignorate del tutto la cartella di augmentation**. Il SSL applica già le proprie augmentation on-the-fly; le immagini pre-aumentate non aggiungono informazione reale, aggiungono solo occasioni di sbagliare lo split.

### 6.3 Più lesioni per immagine

6.029 lesioni in 3.926 immagini: ~1,54 per immagine. Se splittate a livello di **lesione**, lesioni provenienti dalla stessa radiografia finiscono in train e test — stesso paziente, stessa anatomia, stessa qualità d'immagine, stesso annotatore. È leakage, ed è sottile perché il conteggio dei campioni sembra corretto.

Regola: **split a livello di immagine, poi assegnate le lesioni**. Verificate che nessuna immagine compaia in due split, con un assert nel codice, non a occhio.

---

## 7. Metriche: e una scelta di modello che il brief non fa

Il brief richiede metriche threshold-agnostic sulla classe minoritaria — **Macro-F1, Precision-Recall AUC, confusion matrix** — e vieta implicitamente l'accuracy globale. Giusto: con 61% di PAI 3, un classificatore che predice sempre PAI 3 ottiene 61% di accuracy.

Aggiungete: **recall per classe** (la sensibilità su PAI 5 è il numero clinicamente rilevante), balanced accuracy, e per-class PR-AUC.

### Il PAI è una scala ordinale, e questo cambia le cose

PAI 3 < PAI 4 < PAI 5 è un **ordinamento**, non un insieme di categorie senza relazione. Il brief chiede "a lightweight multi class classification head", ma trattare il problema come classificazione piatta a 3 classi butta via informazione e, soprattutto, **usa la metrica sbagliata**: confondere PAI 3 con PAI 5 è un errore clinicamente più grave che confondere PAI 4 con PAI 5, e la Macro-F1 li pesa uguale.

Due conseguenze:

1. **Aggiungete il Cohen's kappa quadratico pesato** e la MAE ordinale. Sono le metriche standard per il grading ordinale in ambito medico, e penalizzano correttamente gli errori a due gradi di distanza.
2. **Una testa ordinale** (CORAL, o cumulative link) invece di una softmax a 3 vie è tecnicamente più appropriata ed è un posto naturale dove collocare o rafforzare la novità.

Discutere l'ordinalità nella presentazione è uno dei modi più rapidi di dimostrare di aver capito il problema clinico e non solo il problema di ML. Tenete comunque la softmax piatta come braccio di confronto, così la scelta è argomentata con numeri.

---

## 8. La novità metodologica (obiettivo 3)

Il brief chiede "an original algorithmic novelty specifically designed to combat class imbalance **in the latent space**", suggerendo "asymmetric embedding regularizers or class balanced patch masking".

**Attenzione a una trappola concettuale:** il pre-training è *self-supervised*, quindi qualsiasi cosa "class-balanced" durante il pre-training richiede etichette che non dovreste usare. Se usate i gradi PAI in fase di pre-training, il claim self-supervised cade. Usare le *posizioni* delle bbox (non i gradi) è una via di mezzo: è supervisione debole, va dichiarata come tale, e va confrontata con l'alternativa non supervisionata.

### Tre opzioni concrete

**N1 — Masking guidato da salienza (stadio di pre-training)**
Invece del block masking casuale di I-JEPA, si sbilancia il campionamento dei target block verso le regioni apicali. Motivazione: sul tile di una panoramica, il masking casuale spende gran parte della capacità predittiva su osso e tessuto irrilevanti; concentrarlo dove si manifesta la patologia rende il compito predittivo più informativo.
- *Versione con supervisione debole:* usa le bbox. Semplice, ma non è più puro SSL.
- *Versione pura, e più originale:* guida il masking con un **proxy di salienza non supervisionato** — energia del gradiente locale, o varianza della densità ossea — che correla con le regioni apicali senza usare annotazioni. È più difendibile e più interessante.

**N2 — Regolarizzatore latente asimmetrico (stadio downstream)**
Un termine che allarga il margine latente della classe minoritaria: margine dipendente dalla classe in stile LDAM (∝ n_c^{-1/4}), applicato ai latenti aggregati per attention pooling, più una penalità sulla varianza intra-classe di PAI 5 per compattarla mentre i centroidi vengono allontanati.
Consolidato, economico, effetto prevedibile. Meno originale.

**N3 — Balanced token sampling nell'attention pooling** ⭐
Il brief nomina esplicitamente le "balanced token-sampling strategies", e questa è la più elegante. L'attention pooling aggrega i token dentro la bbox: per le classi minoritarie si campionano **sottoinsiemi diversi di token dalla stessa lesione**, trattandoli come istanze di training distinte. È oversampling *nello spazio dei token* che non duplica immagini e non genera pixel sintetici — ogni istanza è una vista genuinamente diversa della stessa lesione.

**Raccomandazione:** **N3 come novità primaria** (nominata dal brief, sicuramente implementabile, economica), con la variante a salienza non supervisionata di **N1 come stretch goal** se il tempo lo consente. Il brief chiede *una* novità originale, non tre.

### Le baseline che la novità deve battere

Questa è la parte che decide la credibilità. Con feature congelate, tutti i metodi classici per lo sbilanciamento si applicano direttamente, quindi non avete scuse per non confrontarvi:

| Baseline | Perché è necessaria |
|---|---|
| Cross-entropy semplice | il riferimento minimo |
| Class-weighted CE | il primo riflesso di chiunque |
| Focal loss | lo standard del settore |
| Oversampling della minoritaria | banale ma competitivo |
| **SMOTE nello spazio latente** | **la baseline scomoda** |

L'ultima merita attenzione: con un encoder congelato, i vostri campioni *sono* vettori di feature, quindi SMOTE latente si applica in modo naturale ed è notoriamente forte. Se la vostra novità non batte SMOTE latente, dovete saperlo prima della presentazione, non durante. Riportarlo onestamente vale più che ometterlo.

### Il vantaggio che rende l'ablation quasi gratuito

**Precalcolate e cachate i latenti congelati una volta sola.** Dopo il pre-training, estraete i vettori di tutte le 6.029 lesioni e salvateli su disco. Da quel momento, ogni esperimento sullo sbilanciamento — novità, cinque baseline, sweep di iperparametri, cinque seed per gli intervalli di confidenza — gira in **secondi, anche su CPU**.

Questo trasforma l'ablation da compute-bound a istantaneo, ed è esattamente ciò che serve con 25 giorni. È anche la ragione per cui, nonostante il rischio dello stadio 1, la seconda metà del progetto è molto più tranquilla del Progetto 2.

---

## 9. I bracci di confronto per l'intera pipeline (non negoziabili)

L'abstract del brief afferma che "supervised models overfit to the majority class". Quella è un'**ipotesi da testare**, non una premessa da assumere. Servono tre bracci:

| # | Braccio | Perché |
|---|---|---|
| 1 | **Supervisionato da zero** su crop di lesione | verifica empiricamente il claim dell'abstract |
| 2 | **Feature congelate ImageNet** (ResNet-34 o ViT-S) + stessa testa | ⚠️ **il braccio critico** |
| 3 | **JEPA in-domain congelato** (il metodo proposto) | la vostra tesi |

Il braccio 2 è quello che fa o rompe la storia, ed è il più economico dei tre: sono feature pre-addestrate scaricate e una passata di estrazione. Se il JEPA in-domain su 4k immagini non batte il transfer da ImageNet, **quello è il vostro risultato** — e il confronto Galaxy10 di LeJEPA vi dà il quadro teorico per discutere perché il vostro caso differisce (meno dati, dominio a bassa diversità, un solo centro clinico).

Non saltate il braccio 2 per mancanza di tempo. Un progetto che dice "il nostro metodo ottiene Macro-F1 0.62" senza dire cosa ottiene ImageNet frozen non ha dimostrato niente, e sarà la prima domanda che vi faranno.

---

## 10. Il gate di decisione del 26 agosto

Questa è la parte più importante dal punto di vista pratico. P8 è a due stadi: se lo stadio 1 non produce rappresentazioni utili, lo stadio 2 non ha materia prima e non c'è modo di consegnare.

**Entro il 26 agosto** dovete avere un k-NN probe sulle feature JEPA in-domain che:

- batte chiaramente la baseline della classe maggioritaria (61%), e
- si trova nella stessa fascia delle feature ImageNet congelate (braccio 2).

**Se sì:** procedete, la seconda metà è in discesa grazie al caching dei latenti.

**Se no:** passate al Progetto 2. Avete già lo scaffold, il piano e i dataset individuati dalla conversazione precedente, e il P2 non ha punti di fallimento silenzioso — CORES è post-hoc su un modello supervisionato che si addestra sicuramente. Restano 16 giorni, che per il P2 è stretto ma sufficiente se lavorate in parallelo.

Fissate questa data ora e rispettatela. Il modo tipico in cui questi progetti falliscono non è scegliere quello sbagliato: è insistere su un pipeline che non converge fino a quando non resta tempo per l'alternativa.

---

## 11. Calendario a 25 giorni per il P8

Ricordate la discrepanza sulle scadenze già segnalata: la lista progetti dice **6 settembre**, le exam guidelines impongono il form **7 giorni prima** della presentazione, cioè **4 settembre** per l'11. Puntate al 4 e chiedete conferma ai docenti. Freeze repo: **9 settembre**.

| Giorno | Cosa | Milestone verificabile |
|---|---|---|
| **17–19 ago** | Download dataset. **Statistiche bbox e scelta della risoluzione (§2).** Parsing XML, verifica di PAI 4 = 1.817, ricerca ID paziente. Split a livello di immagine con assert anti-leakage. Pipeline di tiling. | Istogramma delle bbox in token + split verificati |
| **20–25 ago** | I-JEPA: context encoder, target EMA, predictor. Training con **monitoraggio di rango, varianza e k-NN**. In parallelo: braccio 2 (ImageNet frozen) e braccio 1 (supervisionato). | Curve di training sane + k-NN probe > 61% |
| **26 ago** | 🚦 **GATE DI DECISIONE** | k-NN in-domain vs ImageNet frozen |
| **27–31 ago** | Braccio SIGReg/LeJEPA. Estrazione e **caching dei latenti**. Attention pooling + testa. Novità N3. Tutte le baseline sullo sbilanciamento incluso SMOTE latente. | Griglia completa, 5 seed, intervalli di confidenza |
| **1–3 set** | Ablation della novità, confusion matrix, kappa quadratico, testa ordinale vs piatta. README. Prima stesura slide. | Tabelle e figure congelate |
| **4 set** | **Invio form** (data prudenziale) | Form inviato con link repo |
| **5–9 set** | Slide, 3 prove cronometrate da 10 min, anticipo delle domande | Presentazione sotto i 10 min |
| **9 set** | **Freeze repo** | Ultimo commit |
| **11 set** | Presentazione | — |

### Divisione dei compiti

- **Persona A — Dati & SSL:** statistiche bbox, tiling, split anti-leakage, I-JEPA, monitoraggio del collasso. È il ruolo con più rischio: assegnatelo a chi è più solido in PyTorch.
- **Persona B — Downstream & Novità:** attention pooling, testa (piatta e ordinale), N3, tutte le baseline sullo sbilanciamento. Lavora sui latenti cachati, quindi è indipendente da A dopo il 27 agosto.
- **Persona C — Bracci di confronto & Deliverable:** braccio 1 e braccio 2, metriche, kappa, confusion matrix, plot, README, slide. Il braccio 2 è la sua consegna critica.

Il contratto tra A e B è il formato del file dei latenti cachati: `(n_lesioni, dim)` più un CSV con `image_id, lesion_id, pai_grade, split`. Definitelo il giorno 1 su carta. Da quel momento B e C lavorano senza dipendere da A.

---

## 12. P8 contro P2: confronto onesto

| | Progetto 2 (CORES + Depth) | Progetto 8 (JEPA + PAI) |
|---|---|---|
| Stadi | 1 — supervisionato + scoring post-hoc | **2 — SSL poi downstream** |
| Punti di fallimento silenzioso | nessuno | **collasso SSL, scala delle lesioni** |
| Peso dei dati | ~4 GB (NYU + KITTI) | poche centinaia di MB |
| Righe di codice delicate | ~150 (scoring) | **~400–600 (macchinario SSL)** |
| Costo dell'ablation | medio (richiede forward su modelli) | **quasi nullo (latenti cachati)** |
| Risultato garantito? | **sì** — le metriche escono comunque | no — la tesi può non reggere |
| Letteratura di supporto | CORES è del 2024, consolidato | **LeJEPA è di novembre 2025, fresco** |
| Interesse per la presentazione | buono | **alto** (clinico, JEPA, LeCun) |
| Rischio complessivo | **basso** | medio-alto |

**La differenza sostanziale** non è la difficoltà: è che nel P2 anche uno scenario negativo produce una presentazione completa (avete addestrato modelli, avete AUROC e FPR95, avete un ablation), mentre nel P8 un fallimento dello stadio 1 vi lascia senza niente da mostrare al punto 3 dello schema di presentazione.

**Il mio consiglio, dato l'interesse del gruppo:** fatelo, ma con il gate del 26 agosto trattato come vincolante e non come suggerimento. E tenete lo scaffold del P2 nella cartella — non come sfiducia, ma come la cosa che vi permette di scegliere il P8 senza giocarvi la sessione.

Un ultimo punto a favore del P8 che vale la pena dire: l'analisi sulla scala delle lesioni (§2) è un'osservazione che il brief non contiene e che la maggior parte dei gruppi non farà. Presentare quel ragionamento — "abbiamo misurato, la configurazione ovvia non poteva funzionare, ecco perché e ecco cosa abbiamo fatto" — è precisamente il tipo di contributo che le guidelines dicono di premiare.

---

## Riferimenti

1. Assran, M. et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* CVPR 2023. · [codice](https://github.com/facebookresearch/ijepa)
2. Balestriero, R., LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv:2511.08544, 2025. · [codice](https://github.com/rbalestr-lab/lejepa)
3. Do, H.V. et al. *A Dataset of apical periodontitis lesions in panoramic radiographs for deep-learning based classification and detection.* Data in Brief 54:110486, 2024. · [Mendeley Data](https://data.mendeley.com/datasets/kx52tk2ddj/3)
