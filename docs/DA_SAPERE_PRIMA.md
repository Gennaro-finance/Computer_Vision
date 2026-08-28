# Da sapere prima di esporre

Documento unico per la discussione dell'11 settembre. Contiene le slide
nell'ordine in cui vanno dette, ogni numero che pronuncerai con il file da
cui viene, tutti gli iperparametri, e i concetti che devi saper spiegare
senza guardare.

Regola d'uso: **se un numero è in questo file, devi saperlo difendere. Se
non lo sai difendere, toglilo dalle slide.**

Aggiornato il 28 agosto 2026. Il risultato centrale ha cambiato segno due
volte: prima con l'ablation cieca alla dimensione, poi con il protocollo a
K fisso, che è la versione definitiva perché lascia invariata la finestra
di osservazione.

---

# PARTE 1 — Le diciannove slide

Diciannove slide, tempo obiettivo 16-19 minuti. Le slide contrassegnate ⭐ sono quelle su cui
si vince o si perde: se il tempo stringe, si taglia altrove.

## Atto 1 — Il problema, e il tetto che si porta dentro (3 slide, ~2 min)

**1. Il compito e la metrica** — 30 s
I quattro obiettivi del brief. Metrica primaria **PR-AUC su PAI 5**: l'unica
fra quelle nominate che sia insieme specifica per la minoritaria e
indipendente dalla soglia. Dichiara subito che **l'accuracy globale è
vietata dal brief**, e perché: PAI 3 è il 61%, un classificatore costante
fa 61%.

**2. I dati** — 30 s
4.719 lesioni di training, 3017 / 1229 / 473 per PAI 3 / 4 / 5.
Sbilanciamento 6,4:1. Serio ma **non estremo**: le baseline funzioneranno
decentemente e i margini saranno stretti. Dirlo qui compra credibilità per
tutti i confronti dopo.

**3. Il soffitto geometrico** ⭐ — 60 s
Due soglie sul lato della bounding box danno **macro-F1 0,7567** e **kappa
0,7779**. Senza rete, senza addestramento. Lati mediani 57 / 80 / 127 px.
Il massimo mai misurato in tutto il progetto è 0,7705: **0,0138 di spazio
sopra il pavimento**.

> Va **prima** dei risultati. Messa all'inizio è una cornice; messa alla
> fine sembra una giustificazione. È la stessa informazione con due valori
> opposti.

## Atto 2 — Obiettivo 1: I-JEPA implementato (3 slide, ~2,5 min)

**4. L'architettura, e cosa NON fa** — 50 s
Context Encoder + Target Encoder aggiornato per EMA + Predictor shallow.
Loss **smooth-L1 su rappresentazioni layer-normate**. Nessuna ricostruzione
di pixel — è la differenza con MAE, e va detta perché rafforza tutto ciò che
segue: il modello non fallisce perché sta imparando a ridisegnare rumore
radiografico.

**5. Il collasso, diagnosticato in costanti di tempo** ⭐ — 60 s
Con EMA 0,996 e 171 step per epoca la costante di tempo è **1,5 epoche**: il
target assorbe il **49,6%** della distanza dal context a ogni epoca. Insegue
troppo, il compito diventa banale, la rappresentazione collassa. Con
**0,9996** la costante sale a **14,6 epoche** e l'assorbimento scende al
**6,6%**. È una diagnosi quantitativa, non una prova per tentativi.

**6. La traiettoria** — 30 s
Monitoraggio a ogni epoca: loss, deviazione degli embedding, rango
effettivo, sonda k-NN. Lo `std` va da **0,0116 a 0,0236**: raddoppia. Serve
a chiudere in anticipo "siete sicuri che non sia collassato?".

## Atto 3 — Obiettivo 2: due misure che sembrano contraddirsi (4 slide, ~5 min)

È il cuore del progetto. Le quattro slide vanno **in questo ordine e mai
separate**: la 7 pone un enigma, la 9 lo scioglie.

**7. Nel compito così com'è specificato, il pre-training non serve** ⭐ — 60 s
Tre encoder, 10 configurazioni, 5 seed, stesso test.
`completa` 0,8753 contro `casuale` 0,8676 di PR-AUC; macro-F1 0,7672 contro
0,7565. **Differenze dentro il rumore, e i massimi assoluti appartengono
all'encoder casuale.** Non dire "il pre-training non funziona": di' *"nel
compito così com'è specificato non aggiunge niente di misurabile"*. La
distinzione regge tutta la slide 9.

**8. Non è l'encoder a classificare: è la bounding box** ⭐⭐ — 70 s
La maschera della bbox seleziona i token da aggregare, e il loro **numero**
dipende dalla dimensione della lesione: **16 / 36 / 64** token mediani per
PAI 3 / 4 / 5. Siccome il grado PAI è quasi tutto dimensione, quel numero è
quasi la risposta.

> **La maschera da sola, trasformata in un vettore one-hot senza un solo
> pixel dell'immagine, dà macro-F1 0,7708** — più del vettore completo
> dell'encoder casuale (0,7638).

È il numero che spiega tutto il resto. L'encoder casuale conserva quel
canale intatto; il pre-training, costruendo invarianza, **in parte lo
scarta** — e quindi appare peggiore in un protocollo dove il canale è quasi
tutta la risposta. Non è che non impari: **disimpara una scorciatoia che il
protocollo premia**.

**9. Tolta la geometria, I-JEPA vince di venti sigma** ⭐⭐ — 80 s
Ritagliando 3× il lato della bbox ogni lesione appare uguale: **36 token per
tutte e tre le classi**, separazione 1,00× invece di 5,00×. Stessa
architettura, stesso ritaglio, stessa testa, stessi seed. **Cambia solo
addestrato contro casuale.**

| | macro-F1 | PR-AUC5 | kappa |
|---|---|---|---|
| casuale | 0,5300 ±0,0173 | 0,4452 ±0,0174 | 0,4643 |
| **completa (I-JEPA)** | **0,6653 ±0,0116** | **0,7281 ±0,0190** | **0,6608** |
| Δ | **+0,1353** (z = **14,5**) | **+0,2828** (z = **24,6**) | **+0,1965** (z = **33,9**) |

> **La risoluzione dell'enigma**: la rappresentazione non è *inutile*, è
> **ridondante** nel compito specificato. Il brief fornisce le bounding box,
> e la dimensione — che è quasi tutto il PAI — si legge dalle box da sola.

**Due conferme che non passano per nessuna testa addestrata**, e servono a
rispondere a *"non sarà la testa a fare il lavoro?"*:

1. **Sonda k-NN sulla media dei token dentro la bbox** — zero parametri,
   nessun `Linear`, nessun attention pooling. Protocollo cieco: macro-F1
   **0,3173** per il casuale contro **0,6092** per `completa`; kappa
   **0,0905** contro **0,5786**. Il kappa del casuale è accordo quasi
   casuale: quei vettori non separano le classi.
2. **La testa non deve costruire niente, deve leggere.** Dopo **una sola
   epoca** di addestramento della testa, `completa` è a 0,5457 di macro-F1
   — già sopra il **massimo di sempre** del casuale (0,5416). Raggiunge il
   95% del proprio massimo all'**epoca 4**; al casuale ne servono **54**.

**9-bis. Il controllo che chiude il caso: togliere il canale, non l'immagine** ⭐⭐ — 70 s
Il ritaglio 3× rimuove la dimensione ma **ridimensiona l'immagine**, e la
dimensione rientra come magnificazione (R² 0,55-0,70 nel predire il lato
della bbox dal vettore). Un controllo più pulito tiene **la finestra di
osservazione invariata** — stesso crop 224 px, stessa risoluzione, stessa
scala apparente della lesione — e toglie solo il canale: la bbox resta usata
per **localizzare**, non per dire quanto è grande.

| protocollo | token | casuale | I-JEPA | divario |
|---|---|---|---|---|
| maschera bbox (attuale) | 16/36/64 | **0,7638** | 0,5166 | **−0,2472** |
| conteggio scorrelato dalla classe | ~29 | 0,3361 | **0,4122** | **+0,0761** |
| griglia 6×6 a posizioni fisse | 36 | 0,2916 | **0,3809** | **+0,0892** |
| 36 token a caso, bbox ignorata | 36 | 0,2810 | **0,3480** | +0,0670 |

**Il segno si ribalta in tutti.** E con K fisso e uguale per tutte le classi
— i K token più vicini al centro della bbox — il casuale passa da 0,7638 a
**0,28-0,39** a ogni K, mentre I-JEPA vince sempre:

| K | 9 | 16 | 36 | 64 | 100 |
|---|---|---|---|---|---|
| casuale | 0,3708 | 0,3902 | 0,3817 | 0,3012 | 0,2781 |
| **I-JEPA** | **0,4032** | **0,4295** | **0,3906** | **0,3818** | **0,3709** |

> **Da dire**: non abbiamo tolto la lesione dall'immagine, né la sua
> dimensione visibile. Abbiamo tolto **la dimensione come metadato del
> protocollo**. La finestra di osservazione è identica.

**Un controllo che NON funziona, e vale la pena raccontarlo**: pareggiare
solo il *conteggio* non basta. **RoIAlign** (He et al., 2017) ricampiona la
regione su una griglia fissa k×k, quindi ogni lesione dà esattamente k²
token — e il casuale resta a **0,7689**. Perché RoIAlign campiona sempre
*dentro* la bbox: il conteggio è pareggiato, l'**estensione** no. Il
conteggio era un correlato; la causa è l'estensione.

**10. Perché il compito rende ridondante la rappresentazione** — 60 s
R² intensità 0,991 → 0,992 e dimensione 0,886 → 0,884 in 40 epoche: un ViT
**casuale** codifica già le due grandezze che contano. CKA fra casuale e
addestrato 0,498, predizioni identiche al **91,6%**, errori condivisi al
**76,8%** contro 18,5% attesi. Nel compito geometrico i due encoder
*decidono allo stesso modo* — non perché siano uguali (la CKA dice di no),
ma perché la decisione dipende da ciò che entrambi hanno già.

## Atto 4 — Obiettivo 3: la novità (4 slide, ~4 min)

**11. balanced_token_sampling** — 60 s
L'attention pooling aggrega i token dentro la bbox. Invece di prenderli
sempre tutti, per le classi rare si campionano **sottoinsiemi diversi**, e
ogni sottoinsieme diventa un'istanza distinta. Non duplica immagini come
l'oversampling, non interpola pixel come SMOTE: ogni istanza è una vista
**genuina** della stessa lesione reale.

**12. Il risultato** ⭐ — 60 s
Testa flat, encoder casuale, 5 seed, test: **PR-AUC5 0,8826 ±0,0050**,
prima su tutte e cinque. Contro `none`: **+0,0150, z = +4,11**.

**13. Lo sweep di alpha: il massimo è interno** ⭐ — 60 s
alpha 0,50 batte alpha 1,00 di **+0,0125 a 2,2 errori standard**.
Ribilanciare di più non è meglio. Protocollo a **seed disgiunti**: alpha
0,50 misurato tre volte indipendenti, 0,8813 / 0,8814 / 0,8797, escursione
**0,0017**. Dichiara che la tua previsione era che vincesse alpha 1,00 ed è
risultato il peggiore — un'ipotesi falsificata dichiarata vale più di dieci
confermate.

**14. Il meccanismo: sette viste valgono un esempio** ⭐ — 70 s
Correlazione intra-lesione **rho = 0,9864** su PAI 5. Dal design effect,
`n_eff = k / (1 + (k−1)·rho)`: **7 viste valgono 1,01 campioni
indipendenti**. Col pooling addestrato rho = 0,9464 e n_eff = 1,05: la
conclusione tiene. Le bbox hanno 16-64 token, quindi la ridondanza **non è
combinatoria**, è della rappresentazione.

> Conseguenza: la novità **non aggiunge esempi, riassegna peso**. Ed è per
> questo che ha un ottimo interno — la 13 e la 14 si spiegano a vicenda.
> È il pezzo più originale del progetto.

**15. Le curve precision-recall** — 50 s
Il vantaggio è massimo dove serve clinicamente: **+0,053** di precisione a
recall 0,90 contro `none`, contro +0,020 a recall 0,80. E mostra cosa
l'area nascondeva: **focal ha la precisione migliore a recall 0,80 (0,766)
e la peggiore a 0,90 (0,535)**.

## Atto 5 — Obiettivo 4: ablation e limiti (3 slide, ~3 min)

**16. I controlli** — 50 s
*(se `exp_controlli` viene lanciato — vedi Parte 7)*
`random_tokens` tiene le viste e toglie il ribilanciamento a budget
identico. Il controllo a pari esempi visti pareggia i passi di gradiente,
perché ad alpha 0,50 la novità ne fa 6.894 contro 4.719, il **46% in più**.
Entrambi falsificabili in direzione sfavorevole.

**17. I tre limiti** ⭐ — 70 s
**Dipende dall'encoder**: prima sul casuale, seconda sul completo, **ultima
delle cinque** sullo spinto.
**Dipende dalla testa**: con l'ordinale pareggia con `focal` (0,8777 contro
0,8785, z = −0,20) e `none/ordinal` ha la macro-F1 più alta di tutte
(0,7667).
**Dipende dal regime**: nel protocollo cieco alla dimensione **pareggia con
`none`** — +0,0067 (z = +0,49) sul casuale, +0,0071 (z = +0,65) su
`completa`.

> Il terzo limite non indebolisce la novità, la **spiega**. Con n_eff = 1,01
> la novità riassegna peso senza aggiungere informazione: aiuta quando il
> classificatore è già vicino al suo tetto e si gioca sul confine di
> decisione, non quando l'informazione manca. Il meccanismo della slide 14
> lo prevedeva — è una previsione confermata, non una scusa.

> Porta tutti e tre. Sono le slide che, se le scopre il relatore, ti costano
> la credibilità delle altre diciotto.

**18. Il difetto di provenienza, trovato e chiuso** — 50 s
Rifatta la griglia da zero, **tre celle su dieci erano sbagliate**: `none/flat`
−0,0083, `oversample/ordinal` −0,0143, `focal/ordinal` +0,0067. Le altre
sette combaciano entro 0,003. Causa: la ripartenza saltava le celle già
presenti **senza verificare con quali latenti fossero state prodotte**. Ora
i risultati portano l'impronta del file di latenti e la ripartenza viene
rifiutata se non corrisponde.

> Questa slide non parla dei risultati, parla di **come lavori**. In una
> discussione vale quanto un risultato.

## 19. La tesi, in tre frasi — 60 s

> **I-JEPA impara.** Le sue rappresentazioni valgono +0,28 di PR-AUC e +0,14
> di macro-F1 sopra un encoder casuale — venti errori standard — quando la
> misura ha lo spazio per vederlo.
>
> **Il compito così com'è specificato non ne ha bisogno.** Il brief fornisce
> le bounding box, e la dimensione della lesione — che è quasi tutto il grado
> PAI — si legge dalle box da sola: due soglie danno 0,7567 di macro-F1
> senza alcuna rete. In quel compito la rappresentazione è **ridondante**, e
> un encoder casuale la pareggia.
>
> **Nello stesso regime, ribilanciare nello spazio latente funziona** —
> +0,0150 a 4,11 errori standard — e funziona non aggiungendo esempi ma
> riassegnando peso: sette viste di una lesione valgono 1,01 campioni
> indipendenti. Ed è per questo che ha un ottimo interno, e che sparisce
> quando l'informazione manca davvero.

Il contributo metodologico, che è quello che vale oltre questo dataset:

> **Valutare rappresentazioni self-supervised su un compito che porta forti
> priori nell'input misura il compito, non la rappresentazione.** Serve un
> protocollo che rimuova il priore, altrimenti "non aggiunge" e "non si
> vede" restano indistinguibili.

---

# PARTE 2 — Ogni numero che dirai

Fonte fra parentesi. Tutto ricalcolabile con `verify_claims.py`.

## Il dataset

| | |
|---|---|
| lesioni totali | 6.741 (train 4.719, val 1.009, test 1.013) |
| train per classe | 3017 / 1229 / 473 (PAI 3/4/5) |
| sbilanciamento | 6,4 : 1 |
| prevalenza PAI 5 nel test | **0,1106** — è il pavimento della curva PR |
| lato mediano bbox | 57 / 80 / 127 px |
| token per bbox (mediana) | 16 / 36 / 64-80 |

## Il pavimento e il soffitto

| | macro-F1 |
|---|---|
| classificatore costante (predice sempre PAI 3) | 0,2589 |
| **due soglie sul lato della bbox, senza rete** | **0,7567** (kappa 0,7779) |
| massimo mai misurato, 3 encoder × 10 config × 5 seed | **0,7705** |
| spazio disponibile | **0,0138** |

## La griglia — testa flat, 5 seed, test

Encoder **casuale** (`runs/results_vit_small_L2-7-11_casuale.json`):

| metodo | PR-AUC5 | macro-F1 | F1 PAI5 | rec5 | prec5 | kappa |
|---|---|---|---|---|---|---|
| **balanced_tokens** | **0,8826 ±0,0050** | 0,7609 | 0,779 | 0,7714 | 0,789 | 0,7816 |
| focal | 0,8730 ±0,0117 | 0,7645 | 0,786 | 0,7911 | 0,784 | 0,7814 |
| class_weighted | 0,8706 ±0,0091 | 0,7517 | 0,766 | 0,7982 | 0,737 | 0,7771 |
| none | 0,8676 ±0,0065 | 0,7565 | 0,764 | 0,7554 | 0,774 | 0,7841 |
| oversample | 0,8658 ±0,0142 | 0,7475 | 0,760 | 0,7982 | 0,728 | 0,7717 |

Encoder **completa** (I-JEPA, epoca 179):

| metodo | PR-AUC5 | macro-F1 |
|---|---|---|
| none | 0,8753 ±0,0073 | 0,7672 |
| balanced_tokens | 0,8743 ±0,0087 | 0,7575 |

Encoder **spinto**:

| metodo | PR-AUC5 | macro-F1 |
|---|---|---|
| class_weighted | 0,8798 ±0,0048 | 0,7632 |
| none | 0,8772 ±0,0050 | 0,7676 |
| balanced_tokens | 0,8711 ±0,0089 | 0,7640 |

## I confronti della novità, encoder casuale, testa flat

| confronto | Δ | z |
|---|---|---|
| vs `none` | +0,0150 | **+4,11** |
| vs `class_weighted` | +0,0120 | **+2,60** |
| vs `oversample` | +0,0168 | **+2,50** |
| vs `focal` | +0,0096 | +1,69 |

Con testa **ordinale**: vs `focal` −0,0008 (z = −0,20), vs `none` +0,0065
(z = +1,68). **Dichiaralo.**

## Lo sweep di alpha (`runs/sweep_alpha_vit_small_L2-7-11_casuale.json`)

Screening, 3 seed, encoder casuale, test:

| alpha | viste [PAI3,4,5] | istanze | PR-AUC5 | rec5 | prec5 |
|---|---|---|---|---|---|
| 0,25 | [1,2,2] | 6.421 | 0,8793 ±0,0049 | 0,7381 | 0,833 |
| **0,50** | [1,2,3] | 6.894 | **0,8814 ±0,0071** | 0,7738 | 0,810 |
| 0,75 | [1,2,5] | 7.840 | 0,8747 ±0,0028 | 0,7946 | 0,756 |
| 1,00 | [1,3,7] | 10.015 | 0,8689 ±0,0069 | 0,7708 | 0,783 |

Fase 2, 5 seed **disgiunti**: 0,50 → 0,8797 · 0,25 → 0,8775 (equivalenti,
0,8 errori standard).
0,50 − 1,00 = **+0,0125, z = 2,19**.

## La diversità dei token (`runs/diversita_*.json`)

| classe | token bbox | cos fra viste | cos fra lesioni | rho (media) | rho (pooling addestrato) |
|---|---|---|---|---|---|
| PAI 3 | 16 | 0,9991 | 0,9748 | 0,9674 | 0,8386 |
| PAI 4 | 36 | 0,9993 | 0,9747 | 0,9749 | 0,8768 |
| **PAI 5** | 64 | 0,9996 | 0,9694 | **0,9864** | **0,9464** |

In angoli: viste a **1,67°**, lesioni diverse a **14,21°** — un fattore
**8,5×**. Il rapporto cresce col grado: 5,5× / 6,2× / 8,5×.
**n_eff a k=7, PAI 5: 1,01** (1,05 col pooling addestrato).

## Le curve PR (`runs/curve_pr_vit_small_casuale.json`)

Precisione alle recall di lavoro, encoder casuale, testa flat, 5 seed:

| metodo | r=0,70 | r=0,80 | r=0,90 |
|---|---|---|---|
| **balanced_tokens** | 0,868 | **0,740** | **0,632** |
| focal | 0,860 | **0,766** | **0,535** |
| class_weighted | 0,874 | 0,728 | 0,579 |
| none | 0,822 | 0,720 | 0,579 |
| oversample | 0,834 | 0,731 | 0,575 |

## Il protocollo a K fisso — IL RISULTATO CENTRALE

Pipeline completa: attention pooling addestrato (5,3M par.) + testa `flat`,
5 seed, misura sul **test**. Cambia **solo la maschera**: stesso crop 224 px,
stessa risoluzione, stessi token dell'encoder, stessa architettura.

`P3_K` prende i **K token più vicini al centro della bbox**. La bbox resta
usata per localizzare — che è ciò che il brief chiede — senza che il loro
*numero* comunichi la dimensione.

### macro-F1

| protocollo | token PAI 3/4/5 | casuale | I-JEPA | Δ | p |
|---|---|---|---|---|---|
| `P1_bbox` — naïve | **16/36/64** | 0,7565 ±0,0090 | 0,7567 ±0,0029 | +0,0002 | **0,96** |
| `P2b` griglia fissa | 36/36/36 | 0,5212 ±0,0159 | 0,5247 ±0,0206 | +0,0035 | 0,77 |
| **`P3_K16`** | **16/16/16** | 0,5347 ±0,0087 | **0,5604 ±0,0072** | **+0,0257** | **0,0009** |
| **`P3_K36`** | **36/36/36** | 0,5237 ±0,0091 | **0,5635 ±0,0179** | **+0,0398** | **0,0022** |
| `P3_K64` | 64/64/64 | 0,5131 ±0,0163 | **0,5718 ±0,0150** | **+0,0587** | **0,0003** |

### PR-AUC su PAI 5 — la metrica primaria

| protocollo | casuale | I-JEPA | Δ | p |
|---|---|---|---|---|
| `P1_bbox` | 0,8676 | 0,8691 | +0,0015 | 0,96 |
| `P2b` | 0,3713 | 0,4156 | +0,0443 | 0,02 |
| **`P3_K16`** | 0,4220 | **0,4849** | **+0,0629** | **0,0001** |
| **`P3_K36`** | 0,3954 | **0,4967** | **+0,1013** | **<0,0001** |

### Perché il divario cresce con K — e non è che la misura migliori

La lettura ovvia sarebbe *"più token, più materia su cui distinguere"*. È
sbagliata. Al crescere di K la maschera **smette di essere la lesione**:

| K | quanti dei K token cadono **dentro** la bbox vera |
|---|---|
| | PAI 3 · PAI 4 · PAI 5 · media |
| **16** | 99% · 100% · 100% · **99%** |
| 36 | 52% · 84% · 99% · **65%** |
| 64 | 29% · 53% · 88% · **42%** |

A K=64 il **71% dei token di una PAI 3 è osso circostante**, non lesione.
Quindi la domanda cambia: da *"sai leggere la lesione?"* a *"sai leggere
lesione più contesto?"*.

E i due encoder rispondono in modo opposto:

| K | casuale | I-JEPA |
|---|---|---|
| 16 | 0,5347 | 0,5604 |
| 36 | 0,5237 ↓ | 0,5635 → |
| 64 | **0,5131 ↓** | **0,5718 ↑** |

**Il casuale peggiora** man mano che entra osso sano (−0,022): per lui è
rumore che diluisce. **I-JEPA migliora** (+0,011): sa leggere il contrasto
fra radiotrasparenza e osso trabecolare attorno — che è **contenuto
diagnostico vero**, quello che guarda un radiologo — e l'ha imparato dal
pre-training, che gira su tile della panoramica intera senza mai vedere una
bounding box.

> **Conseguenza sul protocollo.** `K = 16` resta il primario: è l'unico che
> rispetta la traccia alla lettera — *"i vettori latenti corrispondenti alle
> aree lesionate"* — con il 99% dei token dentro la bbox, ed è anche il
> **più conservativo**, perché dà il divario più piccolo dei tre.
> `K = 36` e `K = 64` non sono una semplice analisi di sensibilità: sono
> un'**analisi separata su cosa I-JEPA ha imparato**, e vale una slide.

> **La riga `P1_bbox` è il controllo negativo perfetto.** Stessi encoder,
> stessa testa, stessi seed — cambia solo che il conteggio dei token dice la
> classe, e il divario sparisce (p = 0,96). Il confronto fra le due righe
> *è* il risultato.

`P2b` non contraddice: quella griglia copre tutto il crop, background
compreso, quindi diluisce la lesione. Serve a mostrare che non basta fissare
il numero — bisogna fissarlo **restando sulla lesione**.

**I valori di p** sono calcolati con il **t di Student a 8 gradi di
libertà**, non con la normale: con 5 seed per gruppo la normale sarebbe
ottimistica di ordini di grandezza (darebbe 3,6×10⁻⁷ invece di 0,0009 per
`P3_K16`). Riporta sempre il valore prudente.

### La migliore architettura I-JEPA

*(in corso — 6 teste + 3 pooling selezionati su validation sotto `P3_K16`,
poi la scelta sul test con il pavimento `casuale + flat` accanto)*

| | macro-F1 | PR-AUC5 | F1 PAI 5 | recall5 | prec5 |
|---|---|---|---|---|---|
| casuale + flat (pavimento) | 0,5347 | 0,4220 | | | |
| I-JEPA + flat (alla pari) | 0,5604 | 0,4849 | | | |
| **I-JEPA + testa migliore** | | | | | |

> Le prime due righe sono il **confronto controllato**, stessa testa per
> entrambi. La terza risponde a *"quanto si può ottenere da I-JEPA"*, col
> casuale come **riferimento dichiarato**, non come avversario tarato. Vanno
> presentate insieme: la prima difende la seconda.

## Few-shot — il segno si inverte a ogni riga

Train sottocampionato stratificato, validation e test interi, 5 seed. È il
test canonico della qualità di una rappresentazione, e **non azzoppa la
baseline**: toglie etichette a entrambi allo stesso modo.

| etichette | n | `P1_bbox` z | `P3_K16` z |
|---|---|---|---|
| 1% | 47 | **−6,79** | **+4,05** |
| 5% | 236 | −4,41 | +2,81 |
| 10% | 472 | −3,18 | +3,37 |
| 25% | 1.179 | −2,65 | +3,75 |
| 100% | 4.719 | +0,04 | **+5,10** |

Significativo **in entrambe le direzioni**, e l'unica cosa che cambia è se
il protocollo regala il conteggio dei token.

**Il vantaggio relativo si comporta come deve** — `P3_K16`, macro-F1:
**+30,2%** all'1% → +12,3% al 5% → +7,4% al 25% → +4,8% al 100%. È la firma
del few-shot: una buona rappresentazione vale di più quando i dati
scarseggiano.

**Il dato più duro è `P1_bbox` all'1%**: il casuale fa **0,7491 con 47
etichette**, praticamente il suo massimo di sempre (0,7734).

> **La scorciatoia non è solo disponibile: è imparabile da 47 esempi.**
> *Più token → PAI più alto* è una relazione con un parametro. I-JEPA, che
> quel canale l'ha in parte disimparato, deve costruire una mappa più
> complessa e con 47 esempi non ci arriva. Ecco **perché** il modello prende
> sempre la scorciatoia: a ogni livello di supervisione costa meno.

**Da dichiarare**: sulla PR-AUC di PAI 5 il vantaggio a poche etichette
**non è significativo** (z = +0,41 all'1%). All'1% ci sono **5 lesioni PAI 5
nel train**: con cinque esempi non si stima una curva precision-recall. Il
vantaggio in regime scarso si misura sulla macro-F1; sulla minoritaria
servono più esempi per *misurarlo*, non per *averlo*.

## MIL per token — un'ipotesi falsificata, e ciò che ha rivelato

**L'idea.** Il brief dice di estrarre *"the latent vectors"* — plurale,
cardinalità variabile. È Multiple Instance Learning, e il difetto noto di
quella formulazione è il **bag-size bias**. Ilse et al. distinguono
*embedding-level* (aggreghi le feature, poi classifichi) da *instance-level*
(classifichi ogni istanza, poi aggreghi le decisioni). Nel secondo la
cardinalità non entra: **la media di N probabilità non dipende da N**.

Verificato prima di misurare: invarianza da 8 a 128 token con scarto 6×10⁻⁸,
e con una testa lineare istanza ed embedding coincidono — `media(W·xᵢ) =
W·media(xᵢ)` — per questo la testa per token è un MLP.

**Il risultato**, 5 seed, test:

| protocollo | posizioni dei token | casuale | I-JEPA | Δ | z |
|---|---|---|---|---|---|
| `P1_bbox` | **variabili** | **0,7861** | 0,7646 | −0,0215 | **−5,85** |
| `P3_K16` | **fisse** | 0,4002 | **0,5432** | **+0,1430** | **+8,10** |

**Su `P1_bbox` l'ipotesi è falsificata**: il casuale vince, e il MIL lo ha
reso *migliore* di prima (0,7861 contro 0,7565 con la testa lineare).

**Perché.** L'invarianza dimostrata era vera e **insufficiente**: ho tolto
il *conteggio*, non l'*estensione*. I token portano il **positional
embedding**, quindi non sono intercambiabili: una bbox grande include
posizioni periferiche, una piccola solo centrali. La media cambia non
perché gli addendi sono di più, ma perché sono **altri**. E un
classificatore per token può leggere la posizione **direttamente**, mentre
l'aggregazione la mediava in parte.

> È la stessa distinzione conteggio-contro-estensione già trovata con
> RoIAlign. Va raccontata: **un'invarianza dimostrata su un canale non
> protegge dagli altri.**

**Ma su `P3_K16` il MIL dà il margine più grande mai misurato**:

| configurazione | casuale | I-JEPA | Δ | z |
|---|---|---|---|---|
| `P3_K16` + `flat` | 0,5347 | 0,5604 | +0,0257 | +5,10 |
| **`P3_K16` + `mil`** | **0,4002** | **0,5432** | **+0,1430** | **+8,10** |

Il casuale **crolla** (−0,135), I-JEPA scende appena (−0,017). Il MIL chiede
a ogni singolo token *"quanto sei tessuto da PAI 5?"*: un token di una
proiezione casuale, preso da solo, non sa rispondere; uno di I-JEPA sì.

> **Il vantaggio di I-JEPA vive nel contenuto semantico del singolo token.**
> Il contrasto fra le due righe — stessa testa, stessi encoder, stessi seed,
> cambia solo se le posizioni variano — vale **14 errori standard**.

## L'ablation cieca alla dimensione — la prima versione

Ritaglio a 3× il lato della bbox, ridimensionato a 224. Verificato:
**36 token per PAI 3, 4 e 5**, separazione **1,00×** contro 5,00× del
protocollo geometrico. Stessa architettura, stesso ritaglio, stessa testa,
stessi seed: **cambia solo addestrato contro casuale**.

| encoder | metodo | macro-F1 | PR-AUC5 | kappa |
|---|---|---|---|---|
| casuale | none | 0,5300 ±0,0173 | 0,4452 ±0,0174 | 0,4643 |
| casuale | balanced_tokens | 0,5254 ±0,0163 | 0,4519 ±0,0252 | 0,4588 |
| **completa** | **none** | **0,6653 ±0,0116** | **0,7281 ±0,0190** | **0,6608** |
| completa | balanced_tokens | 0,6689 ±0,0147 | 0,7352 ±0,0151 | 0,6570 |

**completa − casuale**, metodo `none`:

| metrica | Δ | z |
|---|---|---|
| macro-F1 | +0,1353 | **+14,5** |
| PR-AUC su PAI 5 | +0,2828 | **+24,6** |
| kappa quadratico | +0,1965 | **+33,9** |

**La novità in questo regime pareggia**: +0,0067 (z = +0,49) sul casuale,
+0,0071 (z = +0,65) su completa.

Quanto vale la geometria, per l'encoder casuale:

| | macro-F1 |
|---|---|
| protocollo geometrico | 0,7565 |
| protocollo cieco | 0,5300 |
| **la geometria vale** | **0,2265** |

Dove crolla: **F1 su PAI 4 = 0,352** con l'encoder casuale. Senza
dimensione, la classe intermedia diventa quasi indistinguibile — che è
esattamente ciò che ci si aspetta se il segnale è la dimensione. Con
`completa` risale a 0,523.

> **Il caveat da dichiarare**: i numeri ciechi **non sono confrontabili**
> con quelli geometrici, è un compito diverso e deliberatamente più
> difficile. Si riporta come **ablation sull'encoder**, non come risultato
> principale: il protocollo del brief resta quello con le bounding box.

---

# PARTE 3 — Tutti gli iperparametri

Se te ne chiedono uno e non lo sai, la slide che lo usa perde valore.
Fonte: `globals.py`.

## Backbone e risoluzione

| parametro | valore | perché |
|---|---|---|
| `DEFAULT_VARIANT` | `vit_small` | 384 dim, 12 blocchi, 6 teste. Con 4.700 immagini un ViT-B sovradatta |
| `PATCH_SIZE` | 16 | 224/16 = griglia 14×14 = 196 token |
| `TILE_SIZE` | 224 | risoluzione nativa, nessun ridimensionamento |
| `LESION_CROP_PIXELS` | 224 | finestra **fissa** in pixel nativi: preserva la dimensione apparente della lesione |
| `TILE_MIN_FOREGROUND` | 0,15 | scarta i tile quasi tutti neri (bordi della panoramica), 8 tentativi |
| `TILE_STRIDE` | 168 | sovrapposizione 25% fra tile adiacenti |
| `CROPS_PER_IMAGE` | 8 | crop casuali per immagine a ogni epoca SSL |

## I-JEPA (pre-training)

| parametro | valore | perché |
|---|---|---|
| `PREDICTOR_DIM` | 96 | predictor volutamente **stretto**: se è capace, risolve il compito da solo e l'encoder non impara |
| `PREDICTOR_DEPTH` | 4 | shallow, come chiede il brief |
| `PREDICTOR_HEADS` | 3 | |
| `NUM_TARGET_BLOCKS` | 4 | quattro blocchi target per immagine, rimossi dal contesto |
| `CONTEXT_SCALE` | (0,85, 1,0) | frazione dell'immagine visibile al context encoder |
| `TARGET_SCALE` | (0,15, 0,20) | area di ciascun blocco target |
| `TARGET_ASPECT` | (0,75, 1,5) | rapporto d'aspetto dei blocchi |
| **`SSL_EMA_START`** | **0,9996** | **il parametro che ha risolto il collasso.** Vedi slide 5 |
| `SSL_EMA_END` | 1,0 | l'EMA si irrigidisce verso fine run |
| `SSL_LR` | 3e-5 | |
| `SSL_WEIGHT_DECAY` | 0,04 | |
| `SSL_WARMUP_EPOCHS` | 15 | |
| `SSL_EPOCHS` | 300 | |
| `SSL_BATCH_SIZE` | 128 | |
| `GRAD_CLIP` | 3,0 | |
| `AMP` | True | **bfloat16**, non float16: ha il range dinamico del float32 e non serve il GradScaler |

## Downstream (testa di classificazione)

| parametro | valore | perché |
|---|---|---|
| `LAYERS_DOWNSTREAM` | `[2, 7, 11]` | l'ultimo blocco è il **più compresso**; concatenare tre profondità dà 384×3 = 1152 dim |
| `HEAD_EPOCHS` | 100 | valutazione su validation ogni 10 epoche, si tiene la migliore |
| `HEAD_LR` | 1e-3 | AdamW |
| `HEAD_WEIGHT_DECAY` | 1e-4 | |
| `HEAD_BATCH_SIZE` | 128 | |
| `ATTN_POOL_HEADS` | 4 | attention pooling multi-testa sui token dentro la bbox |
| `N_SEEDS` | 5 | con 6,4:1 i margini sono stretti, un run singolo non distingue niente |
| `SEED` | 42 | seme dell'encoder casuale: il riferimento deve essere riproducibile |
| `FOCAL_GAMMA` | 2,0 | |
| `TOP_K` | 8 | per il pooling top-k (esperimento non ancora lanciato) |

## La novità

| parametro | valore | perché |
|---|---|---|
| **`alpha`** | **0,50** | `n_c = ceil((max_count / count_c)^alpha)` → [1, 2, 3] viste. **Ottimizzato**, non assunto: vedi slide 12 |
| `p_min`, `p_max` | 0,6 – 1,0 | frazione di token tenuti per vista, uniforme in quell'intervallo |
| `min_tokens` | 4 | sotto questa soglia si aggrega rumore |
| prima vista | **integra** | l'istanza originale non viene mai sottocampionata, altrimenti il confronto con le baseline non è alla pari |

---

# PARTE 4 — I concetti da saper spiegare

Per ciascuno: se non lo sai dire in trenta secondi senza guardare, ripassa.

## 4.1 Perché I-JEPA e non MAE

MAE ricostruisce **pixel**; I-JEPA predice **rappresentazioni**. In una
radiografia il rumore di grana e la texture fine sono irrilevanti per il
grado PAI, ma un obiettivo pixel-wise costringe a modellarli. Predire in
spazio latente lascia al modello la libertà di scartare ciò che non conta.

**Il codice**: `network.py:352` — `full = F.layer_norm(target_encoder(images))`,
poi `smooth_l1_loss(pred, target)`. Nessun decoder, nessuna testa che torna
a `patch_size²·3`.

**Perché conta per te**: rafforza il risultato negativo. Non è che il
pre-training fallisca perché sta imparando a ridisegnare rumore — predice
in latente, dove non gli è richiesto, e non impara lo stesso.

## 4.2 Perché il LayerNorm sui target

Senza, la loss si minimizzerebbe banalmente **riducendo la norma** delle
rappresentazioni: due vettori piccoli sono vicini per costruzione. Il
LayerNorm fissa media e varianza di ogni token, e quella scorciatoia sparisce.

Effetto collaterale utile: **elimina anche la scorciatoia dello sfondo
nero**. Il target di una patch scura non è "vicino a zero", è normalizzato
come qualunque altro token.

## 4.3 La costante di tempo dell'EMA

Il target encoder si aggiorna come `θ_t ← τ·θ_t + (1−τ)·θ_c`.
La distanza dal context decade geometricamente con ragione τ, quindi la
**costante di tempo** è `1/(1−τ)` **step**.

- τ = 0,996 → 250 step. A 171 step/epoca sono **1,5 epoche**, cioè il
  **49,6%** della distanza assorbito ogni epoca
- τ = 0,9996 → 2500 step = **14,6 epoche**, **6,6%** per epoca

Se il target insegue troppo, i due encoder diventano lo stesso e predire
diventa banale: la loss scende, la rappresentazione collassa. La
**dimensione del run** è ciò che rende τ giusto o sbagliato — non c'è un
valore universale.

## 4.4 Collasso, e come si riconosce

Tre segnali, e servono insieme:
- **deviazione degli embedding** verso zero → tutti i vettori uguali
- **rango effettivo** in calo → l'informazione occupa poche direzioni
- **loss che scende mentre le sonde a valle peggiorano** → il segnale
  decisivo, perché una loss bassa da sola non dice nulla

**Trappola già scoperta**: il rango effettivo calcolato sul secondo momento
**non centrato** dava 1,07 su un encoder casuale. È un artefatto — la media
domina — non collasso. Serve la versione **centrata**.

## 4.5 La sonda lineare, e perché è il protocollo giusto

Valutare rappresentazioni congelate con **un solo strato lineare** è il
protocollo canonico: SimCLR, MoCo, DINO e lo stesso paper I-JEPA. Esiste
perché una testa capace misura sé stessa, non la rappresentazione.

Il brief lo chiede: *"a lightweight multi class classification head"*.

**La prova che non è lei il collo di bottiglia**: con la stessa testa flat,
nel protocollo cieco alla dimensione la differenza fra encoder pre-addestrati
e casuali era **+0,177**. Una testa che sa esprimere 0,177 non è ciò che
limita 0,014.

## 4.6 PR-AUC contro ROC-AUC su dati sbilanciati

La ROC usa il **false positive rate**, che ha i negativi al denominatore.
Con l'89% di negativi, anche molti falsi positivi danno un FPR piccolo: la
ROC-AUC resta ottimisticamente alta. La PR usa la **precisione**, che ha i
positivi predetti al denominatore, e quindi risente davvero degli errori
sulla minoritaria.

Il **pavimento** della curva PR è la prevalenza: **0,1106**. Un
classificatore casuale sta lì. Il pavimento della ROC è 0,5 sempre.

## 4.7 Macro-F1 e kappa quadratico

**Macro-F1** = media non pesata delle F1 per classe. Dà a PAI 5 lo stesso
peso di PAI 3 nonostante sia 6,4 volte più raro. Dipende dall'argmax,
quindi dalla soglia: **si riporta, non decide**.

**Kappa quadratico pesato**: penalizza gli errori in proporzione al
**quadrato** della distanza sulla scala. Sbagliare 3→5 costa quattro volte
sbagliare 4→5. È la struttura di gravità clinica del problema, e la
macro-F1 non la coglie.

## 4.8 La novità, in una formula

`n_c = ceil((max_count / count_c)^alpha)`

Con 3017/1229/473 e alpha 0,50 → **[1, 2, 3]** viste per PAI 3/4/5.
Ogni vista tiene una frazione `p ~ U(0,6 , 1,0)` dei token dentro la bbox.
La **prima vista resta integra**.

- alpha 0 → una vista per tutti, identico a `none`
- alpha 1 → pareggio effettivo, [1, 3, 7], 10.015 istanze

## 4.9 ICC e design effect — il pezzo originale

Da campionamento statistico: **k osservazioni con correlazione interna rho
non valgono k campioni indipendenti**, ne valgono

    n_eff = k / (1 + (k − 1)·rho)

`rho` è l'ICC multivariato: varianza **fra** lesioni diviso varianza
**totale**, sommate sulle 1.152 dimensioni (traccia).

Con rho = 0,9864, sette viste di un PAI 5 danno **n_eff = 1,01**.

**Perché il coseno da solo non basta**: in questo spazio tutto è simile a
tutto (i pixel grezzi hanno rango effettivo 1,12, il 97,3% della norma sta
nella direzione media). Il coseno fra lesioni **diverse** è il metro che
rende leggibile quello fra viste: 0,9991 contro 0,9748 sembrano uguali, ma
in angoli sono 2,4° contro 12,9°.

**Conseguenza**: la novità non aggiunge esempi, **riassegna peso**. Ed è per
questo che ha un ottimo interno.

## 4.10 La distorsione da selezione, tre volte

Il massimo di k misure rumorose è distorto verso l'alto di circa una
deviazione. Nel progetto compare tre volte:

| dove | selezione su | protezione |
|---|---|---|
| sweep di alpha | 4 candidati | **seed disgiunti in fase 2** |
| "prima di 30 configurazioni" | 30 estrazioni | dichiarata |
| scelta dell'epoca | 10 checkpoint, dentro ogni misura | nessuna, ma è deterministica dato il seme |

Saperlo dire è meta' del valore della slide 13.

## 4.11 Il canale nascosto: estensione della maschera, non conteggio

Il protocollo del brief aggrega i token dentro la bbox. La **regione
aggregata** ha quindi l'estensione della lesione, e il ViT somma un
**positional embedding** a ogni token: la media su una regione larga
contiene posizioni più disperse di una stretta. L'estensione entra nel
vettore a prescindere dal contenuto dell'immagine.

Prova diretta: la **maschera one-hot da sola**, senza pixel, dà 0,7708.

Due controlli che separano estensione da conteggio:

| | conteggio | estensione | casuale |
|---|---|---|---|
| maschera bbox | variabile | variabile | 0,7638 |
| **RoIAlign k=4** | **fisso** | variabile | **0,7689** |
| **conteggio scorrelato** | scorrelato | scorrelato | **0,3361** |

RoIAlign fissa il conteggio e non cambia niente; scorrelare il conteggio
dalla classe — che cambia anche l'estensione — fa crollare il casuale.
**La causa è l'estensione, il conteggio ne è la manifestazione.**

## 4.12 I due encoder imparano cose diverse

La sonda k-NN sulla media mascherata, **senza alcun parametro addestrato**,
nei due protocolli e con lo stesso checkpoint:

| | casuale | completa |
|---|---|---|
| geometrico | **0,7638** | 0,5166 |
| cieco | 0,3173 | **0,6092** |

**L'encoder casuale è un misuratore d'area travestito da rete.** La media
dei token dentro una bbox dipende direttamente da quanti token ci sono, e
il numero di token *è* l'area. Senza alcun addestramento arriva a 0,7638 di
macro-F1 — praticamente quanto la testa addestrata (0,7565).

**Il pre-training quella scorciatoia in parte la disimpara.** Il masked
prediction rende la rappresentazione più invariante, e l'invarianza costa
il segnale geometrico grezzo: `completa` scende a 0,5166. In cambio guadagna
l'**aspetto**, che nel protocollo cieco vale 0,6092 contro 0,3173.

Con la testa addestrata pareggiano nel protocollo geometrico perché la
testa — 5,3M parametri di attention pooling — **recupera la dimensione** dai
token del casuale. Ma il vettore grezzo dice che i due encoder hanno
imparato **cose diverse**, non "uno niente e l'altro qualcosa".

> Da dire così: *un ViT non addestrato è già un ottimo misuratore d'area, e
> il grado PAI è quasi tutto area. Il pre-training scambia parte di quel
> segnale per informazione sull'aspetto — un cattivo affare nel compito
> specificato, un ottimo affare quando l'area non basta.*

## 4.13 Ridondante non vuol dire inutile

Due misure che sembrano contraddirsi e non lo sono:

- nel compito specificato, `completa` ≈ `casuale` (0,8753 contro 0,8676)
- tolta la geometria, `completa` batte `casuale` di **+0,28 di PR-AUC**

Non c'è contraddizione perché misurano cose diverse. La prima dice *"quanto
serve la rappresentazione in questo compito"*; la seconda dice *"quanto vale
la rappresentazione"*. Sono la stessa distinzione che c'è fra **ridondante**
e **inutile**: un'informazione ridondante è preziosa e semplicemente già
disponibile per un'altra strada.

La conferma che i due encoder *decidono* allo stesso modo pur essendo
diversi: CKA 0,498 (rappresentazioni diverse), ma predizioni identiche al
91,6% e errori condivisi al 76,8%. Rappresentazioni diverse, stessa
decisione — perché la decisione dipende dalla geometria che entrambi hanno.

**Se ti chiedono una sola frase**: *la bounding box è un input fornito dal
brief, e contiene già la risposta; il pre-training impara la stessa cosa per
un'altra strada, e nel compito specificato arriva secondo.*

## 4.14 Il tetto geometrico, in una frase

Il grado PAI è **dimensione + scurezza** della radiotrasparenza. Il brief
fornisce le bounding box, quindi fornisce la dimensione. Due soglie sul lato
danno 0,7567 di macro-F1 senza alcuna rete. **Il tetto non è dell'encoder:
è di quanto il compito stesso mette nell'input.**

---

# PARTE 5 — Le domande, e le risposte

**"Come fate a sapere che non è la vostra implementazione a essere rotta?"**
Tre cose insieme: lo `std` degli embedding raddoppia durante il training
(0,0116 → 0,0236), la CKA fra pesi iniziali e finali è 0,498 (i pesi si sono
mossi molto), e la testa lineare su encoder congelato arriva a 0,87 di
PR-AUC — un encoder morto darebbe il pavimento, 0,26. Più l'ablation cieca
alla dimensione, che misura l'encoder dove c'è spazio per vederlo.

**"Perché non avete fatto fine-tuning?"**
Vincolo esplicito del brief: le rappresentazioni vanno valutate **frozen**.
È una scelta imposta e dichiarata. Il preprint IRRL (Cheng et al., 2026) fa
fine-tuning ed è la norma — noi non potevamo.

**"La testa lineare non è troppo debole?"**
È il protocollo standard per rappresentazioni congelate, e il brief chiede
una testa leggera. E abbiamo la prova che non è il collo di bottiglia: la
stessa testa esprime +0,177 fra encoder quando la differenza c'è.

**"Il vostro risultato è negativo: cosa avete imparato?"**
Che in questo dominio il segnale è quasi unidimensionale e già catturato da
una proiezione casuale. È un risultato sul **dominio**, non sul metodo, e ha
quattro misure indipendenti a sostegno. E nel frattempo abbiamo trovato
perché il ribilanciamento nello spazio latente ha un ottimo interno, che è
un contributo positivo.

**"Il margine della novità è piccolo."**
+0,0150 a **4,11 errori standard** su 5 seed, e prima di tutti e cinque i
metodi. Con la testa ordinale invece pareggia con `focal`: lo dichiariamo.

**"Perché alpha 0,5 e non un altro valore?"**
Perché l'abbiamo misurato, con screening a 3 seed e le due migliori
rimisurate su **5 seed disgiunti**. Il massimo è interno e alpha 1,00 è il
peggiore — contro la nostra previsione iniziale.

**"Il vostro pre-training funziona o no? Avete due risultati opposti."**
Funziona, e molto: +0,28 di PR-AUC sopra il casuale, 24 errori standard,
quando la misura ha spazio. Nel compito così com'è specificato non aggiunge
niente perché il brief fornisce le bounding box e la dimensione della
lesione si legge da quelle. La rappresentazione è **ridondante**, non
inutile. Le due misure non si contraddicono: una dice quanto serve, l'altra
quanto vale.

**"Il vostro protocollo cieco cambia troppe cose insieme."**
Per questo c'è il controllo a finestra invariata: stesso crop 224 px,
stessa risoluzione, stessa scala apparente della lesione, e si toglie
**solo** la dipendenza della maschera dalla dimensione della bbox. Il
casuale passa da 0,7638 a 0,28-0,39 a ogni K, I-JEPA vince sempre. La
finestra di osservazione non è mai cambiata.

**"Non è la testa addestrata a fare il lavoro, invece dell'encoder?"**
No, e lo mostriamo senza testa: sonda k-NN sulla media dei token dentro la
bbox, zero parametri addestrati. Protocollo cieco, kappa 0,0905 per il
casuale contro 0,5786 per l'addestrato. E la testa impiega **4 epoche**
sull'addestrato contro 54 sul casuale per arrivare al 95% del proprio
massimo: la rappresentazione è già linearmente separabile.

**"Perché il vostro protocollo cieco alla dimensione sarebbe legittimo?"**
Non sostituisce il protocollo del brief, lo affianca come **ablation**.
Serve perché nel protocollo geometrico ci sono 0,0138 di spazio fra il
pavimento e il massimo, e in quel margine nessuna differenza fra encoder
può manifestarsi. Rimuovere il priore è l'unico modo di distinguere "non
aggiunge" da "non si vede".

**"Avete provato altre architetture?"**
Il mascheramento è stato variato dal 54% all'80%: traiettoria della loss
identica. Il tetto non è nella formulazione del compito.

---

# PARTE 6 — Gli errori commessi, e come sono stati chiusi

Portali se te li chiedono. Un progetto che dichiara i propri errori corretti
è più credibile di uno che sembra non averne avuti.

| errore | conseguenza | rimedio |
|---|---|---|
| Ripartenza della griglia senza controllo di provenienza | 3 celle su 10 misurate con latenti vecchi sono sopravvissute | Impronta dei latenti nei risultati, ripartenza **rifiutata** se non corrisponde |
| Rango effettivo non centrato | Avrebbe ucciso la run da 300 epoche all'epoca 100 | Versione centrata, e tolto dal criterio di arresto |
| Cancello sul massimo storico del k-NN | Una sonda fortunata all'epoca 10 ha lasciato degradare un run per 4,7 ore | Giudica la sonda corrente e la tendenza |
| Checkpoint sovrascritto a ogni epoca | Perso il migliore (ep. 59), tenuto il peggiore (ep. 39) | Checkpoint `_best` separato |
| `save_checkpoint` non atomico | `os.replace` scambia la voce di directory prima che i byte siano su disco | Aggiunto `fsync` |
| Jitter di scala nel ritaglio | Annullava il segnale dominante (dimensione della lesione) | Finestra fissa a 224 px nativi |
| Flip che non specchiava la bbox | Maschera dei token sulla posizione speculare | Corretto (nessun risultato ne era affetto: `augment=False` ovunque) |

**Difetti noti e dichiarati, non corretti:**

- **Testa ordinale**: i bias delle soglie non sono vincolati all'ordine, e
  le probabilità possono non sommare a 1. Non tocca il ranking su PAI 5 (lo
  score resta monotono) e tutti i risultati primari usano la testa piatta.
- **Troncamento della finestra**: il 2,5% dei PAI 5 nel train e lo 0,9% nel
  test eccedono i 224 px e saturano a 196 token. Censura a destra sulla
  classe più rara.
- **Il briefing riportava "epoca 59"** per il checkpoint `completa_best`,
  che ora contiene l'**epoca 179**: la run è proseguita e l'ha superata.

---

# PARTE 7 — Cosa manca ancora

| esperimento | costo | cosa aggiunge |
|---|---|---|
| **La novità sotto `P3_K16`** | ~55 min (solo I-JEPA) | l'obiettivo 3 misurato dove il conteggio non falsa. **È l'unica voce rimasta** |
| ~~Ablation cieca alla dimensione~~ | **FATTA** | Ha ribaltato l'atto 3: I-JEPA batte il casuale di 24 errori standard |
| `exp_controlli` — `random_tokens` + budget pari | 2h20 | Attribuisce il risultato: ribilanciamento o augmentation? È l'ablation dell'obiettivo 4 |
| `exp_pooling` — gated e top-k | 2h | L'unico che può produrre un positivo nuovo (0,7778 in una prova a 10 epoche) |
| `exp_testa` — le quattro teste rimanenti | 40 min | Chiude "il collo di bottiglia è la testa?" |

**Non sperimentale**: slide, README col link Mendeley (già presente), form
entro il 6 settembre.

---

## Materiale di riserva — averlo, non mostrarlo

- **IRRL (Cheng, Liu, Gu, 2026)**, medRxiv. Attacca lo sbilanciamento sugli
  stessi token latenti ma **pesandoli** invece di campionarli. Dichiara di
  essere *"weighting-based, not token dropping"*: hanno **scelto** ciò che
  noi abbiamo **misurato** con rho = 0,9864. Non discutono la ridondanza fra
  token — è il nostro pezzo originale.
- **L'hardware**: 15 spegnimenti per caduta di alimentazione, risolti
  bloccando il clock GPU a 1500 MHz. Picco da 175 W a 84 W al costo di
  1,12-1,23× in tempo. Solo se te lo chiedono.
