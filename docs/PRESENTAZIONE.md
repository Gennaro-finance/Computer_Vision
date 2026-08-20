# Presentazione — bozza slide per slide

Progetto 8 · Computer Vision A.A. 2025-2026 · Prof. Irene Amerini

**Vincoli dalle guidelines**, da rispettare alla lettera:
sfondo bianco · alto contrasto in grafici e immagini · numero di pagina su
ogni slide nel formato `n / totale` · **10 minuti** · niente salti avanti e
indietro durante l'esposizione · PDF o PowerPoint, caricata su GitHub e
salvata su chiavetta.

**Schema richiesto:** Titolo, Outline, Problem Statement, State of the Art,
Proposed Method, Dataset, Experimental Setup, Model Evaluation, Conclusions,
References. Le 14 slide sotto lo seguono.

**Budget di tempo.** 10 minuti sono pochi: circa 40 secondi a slide. Le
uniche su cui vale la pena spendere di piu' sono la 8 (la novita') e la 11
(il risultato scomodo). Tagliate altrove, non li'.

---

## 1 — Titolo · 20 s

**Self-Supervised Latent Representations for Imbalanced Apical
Periodontitis Grading**

Nomi, cognomi e matricole dei tre · Computer Vision A.A. 2025-2026 ·
Prof. Irene Amerini · Sapienza

---

## 2 — Outline · 20 s

Problema · Stato dell'arte · Metodo · Dataset · Setup · Valutazione ·
Conclusioni

> Una riga, letta in fretta. Non e' la slide su cui costruire.

---

## 3 — Problem Statement · 45 s

Classificare la gravita' di lesioni periapicali su radiografie panoramiche
secondo il **PAI** (Periapical Index), gradi 3, 4, 5.

Due difficolta', entrambe misurate sui nostri dati:

- **Sbilanciamento 6,4 : 1** — nel training 3017 PAI 3 contro 473 PAI 5
- **Scala** — la bbox mediana di una lesione e' **64 px** su panoramiche
  da 2444×1292

> Da dire ad alta voce: "un modello che predice sempre PAI 3 ottiene 63%
> di accuracy. E' il motivo per cui non useremo mai l'accuracy globale."

---

## 4 — La prima decisione: la scala · 50 s

Il brief prescrive *"resized panoramic images"*. Abbiamo verificato cosa
comporta:

```
panoramica 2444×1292  →  ridimensionata a 224×224
fattore di scala ≈ 0,092
lesione da 64 px      →  ~6 px
patch token del ViT   →  16 px
```

**La lesione finisce sotto la dimensione di un singolo token.** Il vettore
latente estratto alla bbox descriverebbe mandibola generica, non patologia.

→ Abbiamo lavorato a **tile su risoluzione nativa**: 21.968 tile da 2.746
immagini.

> È la slide che dimostra critical thinking. Non correte: il punto è che
> avete *misurato* prima di implementare.

---

## 5 — State of the Art · 45 s

- **I-JEPA** (Assran et al., CVPR 2023) — context encoder, target encoder
  EMA, predictor shallow. È l'architettura che l'Obiettivo 1 richiede.
- **LeJEPA** (Balestriero & LeCun, nov 2025) — rimuove EMA e stop-gradient,
  li sostituisce con **SIGReg**, che vincola gli embedding a una gaussiana
  isotropa.

Su **Galaxy10**, dataset specialistico di scala comparabile alla nostra,
LeJEPA riporta che il pre-training in-domain **batte** il transfer da
DINOv3. È la premessa che abbiamo messo alla prova.

> Citare che LeJEPA è di novembre 2025 e che è la ref [2] del brief stesso.

---

## 6 — Metodo proposto: la pipeline · 45 s

```
Stadio 1 — SSL      tile → context encoder + target EMA + predictor
                    (+ SIGReg come termine ausiliario)
                          ↓  encoder CONGELATO
Stadio 2 — downstream  bbox → attention pooling sui token interni
                             → testa piatta / ordinale → PAI 3/4/5
```

Obiettivo 1 soddisfatto alla lettera: i tre componenti ci sono tutti.
L'encoder è congelato nel downstream, come prescritto.

---

## 7 — Perché una testa ordinale · 35 s

Il PAI è una **scala ordinale**: 3 < 4 < 5. Confondere PAI 3 con PAI 5 è
clinicamente peggio che confondere 4 con 5, ma la Macro-F1 li pesa uguale.

→ Testa **CORAL** con soglie cumulative, e **kappa quadratico pesato** fra
le metriche, che penalizza l'errore a due gradi quattro volte.

Teniamo la softmax piatta come braccio di confronto: la scelta è
argomentata con i numeri, non per principio.

---

## 8 — La novità: balanced token sampling · 70 s ⭐

Il brief nomina esplicitamente le *"balanced token-sampling strategies"*.

**L'idea.** L'attention pooling aggrega i token dentro la bbox. Invece di
usarli sempre tutti, per le classi rare si campionano **sottoinsiemi
diversi** della stessa lesione: ogni sottoinsieme è un'istanza di training.

Numero di viste per classe: `n_c = ⌈(n_max / n_c)^α⌉`

| α | viste PAI 3 / 4 / 5 |
|---|---|
| 0.00 | 1 / 1 / 1 |
| 0.75 | 1 / 2 / 5 |
| 1.00 | 1 / 3 / 7 |

**Perché non è oversampling.** L'oversampling ripresenta lo stesso vettore
identico e invita all'overfitting; SMOTE interpola in un latente dove
l'interpolazione può non avere senso anatomico. Qui **ogni vista è una
vista genuina di una lesione reale**.

> È la slide della vostra idea. Prendetevi il tempo.

---

## 9 — Dataset · 40 s

Do, H.V. et al. (2024) · Mendeley DOI 10.17632/kx52tk2ddj.3

| | misurato |
|---|---|
| immagini originali | 3.924 |
| lesioni etichettate | 6.741 |
| PAI 3 / 4 / 5 (train) | 3017 / 1229 / 473 |

**Split a livello di paziente**, come impone il brief. Gli identificativi
sembravano non esistere: li abbiamo trovati nel campo `<filename>` degli
XML, nella forma `PN######`, sopravvissuti alla rinomina dei file
pubblicati. **3.924 identificativi distinti su 3.924 immagini** → una
panoramica per paziente, quindi lo split per immagine *è* per paziente.

Verificato dal codice a ogni esecuzione, non assunto.

> Seconda slide che dimostra critical thinking. Una frase secca: "il
> vincolo del brief non era verificabile, l'abbiamo reso verificabile."

---

## 10 — Experimental Setup · 40 s

- ViT-Small, 22,11 M parametri · tile 224² · patch 16 → griglia 14×14
- Pre-training 100 epoche, bfloat16, monitoraggio del collasso a ogni epoca
- **Tre bracci a encoder congelato**: JEPA in-domain, ImageNet ViT-B/16,
  ViT casuale. Stessi split, stessa testa, stesse metriche: **cambia solo
  l'encoder**
- **46 configurazioni × 5 seed = 230 teste addestrate**
- Metriche verificate contro scikit-learn (differenze ≤ 5,5×10⁻¹⁷)

> Il braccio "ViT casuale" va nominato qui: serve dopo.

---

## 11 — Il risultato scomodo · 70 s ⭐

**FIGURA: `fig1_bracci.png`**

| encoder congelato | Macro-F1 |
|---|---|
| ImageNet ViT-B/16 | **0.6914 ± 0.0175** |
| ViT casuale | 0.5356 ± 0.0171 |
| JEPA in-domain | 0.5069 ± 0.0116 |
| *pavimento (predice sempre PAI 3)* | *0.2589* |

**Il nostro pre-training non raggiunge le proiezioni casuali** (z ≈ 2,5,
significativo).

Perché lo diciamo noi per primi: senza il braccio casuale avremmo
dichiarato 0.507 senza sapere che non fare nulla ne fa 0.536.

**La spiegazione.** 2.746 immagini di training sono lo **0,3%** dei dati di
I-JEPA originale, su un dominio a bassissima diversità visiva e da un solo
centro clinico. LeJEPA riesce su Galaxy10; noi no, e sappiamo dire perché.

> Non difendetevi. Presentatelo come misura. È la slide che vi fa fare
> bella figura *proprio perché* è negativa.

---

## 12 — Il collasso, monitorato · 45 s

**FIGURA: `fig4_pretraining.png`**

Il modo in cui questo progetto poteva fallire in silenzio: la loss scende
mentre gli embedding collassano.

Abbiamo monitorato **rango effettivo** e **k-NN probe** a ogni epoca. Due
risultati:

- il rango sale da 1 a 13 su 280 — **il 4,6%**: le feature restano povere
- **la deviazione standard è quasi cieca**: un sottospazio di rango 6 dà
  std 0.0488 contro 0.0510 di uno isotropo, il **96% del valore sano**

> Chi monitora solo la varianza crede che vada tutto bene mentre il modello
> usa 6 direzioni su 384. È un contributo metodologico, ditelo.

---

## 13 — Ablation della novità · 60 s ⭐

**FIGURA: `fig2_alpha.png`**

Sul braccio JEPA, α regola le viste e la recall sulla minoritaria lo segue
in modo **monotono**: da 0.3696 (α=0) a **0.5964** (α=0.75), **+61%**.

Controllo di correttezza: a α=0 la novità riproduce `none` cifra per cifra.

| metodo (braccio JEPA) | Macro-F1 |
|---|---|
| **balanced_tokens / ordinale** | **0.5372 ± 0.0105** |
| latent_smote / piatta | 0.5339 ± 0.0038 |
| none / piatta | 0.5069 ± 0.0116 |

Batte la CE semplice (z ≈ 2,05) ed **eguaglia SMOTE latente** — la baseline
scomoda. Differenza 0.0033, z ≈ 0,66: sono pari, non la battiamo.

**FIGURA: `fig3_confusioni.png`**

> Dire "eguaglia" e non "batte" vi protegge dalla domanda successiva.

---

## 14 — Conclusioni · 50 s

**Cosa abbiamo trovato**

1. Il problema di scala è reale e misurabile: correggendo l'allineamento fra
   pre-training e downstream, macro-F1 da 0.4707 a **0.5069**
2. Su ~2.700 immagini il pre-training in-domain **non batte** né ImageNet né
   un encoder casuale
3. La novità funziona e mostra una **relazione dose-risposta monotona**;
   eguaglia SMOTE latente senza superarlo
4. Il rango effettivo rileva il collasso, la varianza no

**Limiti, detti da noi**

- 5 seed: i guadagni su PAI 5 valgono 0.8–1.7 errori standard, sotto la
  soglia di 2. Servirebbero più seed
- Tre leve mai esplorate: intensità di SIGReg, capacità del predictor,
  difficoltà del pretext task

**Lavoro futuro**: SIGReg più forte, contesto ridotto, braccio supervisionato
end-to-end per testare la premessa dell'abstract.

---

## 15 — References

1. Assran, M. et al. *Self-Supervised Learning from Images with a
   Joint-Embedding Predictive Architecture.* CVPR 2023.
2. Balestriero, R., LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised
   Learning Without the Heuristics.* arXiv:2511.08544, 2025.
3. Do, H.V. et al. *A Dataset of apical periodontitis lesions in panoramic
   radiographs.* Data in Brief 54:110486, 2024.
4. Cao, K. et al. *Learning Imbalanced Datasets with Label-Distribution-Aware
   Margin Loss.* NeurIPS 2019. *(LDAM, confronto sui margini)*
5. Cao, W. et al. *Rank consistent ordinal regression for neural networks.*
   Pattern Recognition Letters, 2020. *(CORAL, testa ordinale)*

---

## Domande che arriveranno, e la risposta

**"Perché il vostro JEPA perde contro un encoder casuale?"**
Perché 2.746 immagini sono lo 0,3% dei dati di I-JEPA e le proiezioni
casuali di alta dimensione sono una baseline notoriamente forte. Lo abbiamo
misurato invece di scoprirlo dopo.

**"La vostra novità batte SMOTE latente?"**
No, lo eguaglia: 0.5372 contro 0.5339, differenza dentro il rumore. Batte
però la CE semplice con z ≈ 2,05, e ha una curva dose-risposta monotona che
SMOTE non ha.

**"Perché non avete usato l'accuracy?"**
Perché il 63,5% del test è PAI 3: un modello costante fa 0.635 di accuracy
e 0.259 di macro-F1. Il brief la vieta, e ha ragione.

**"Il vostro split è davvero a livello di paziente?"**
Sì, ed è verificato: gli identificativi `PN######` sono nel campo
`<filename>` degli XML, 3.924 distinti su 3.924 immagini.

**"Avete provato un supervisionato da zero?"**
È fra il lavoro futuro. In una prova preliminare a 10 epoche e un seed
otteneva 0.5709, sopra il nostro JEPA: è il primo esperimento da completare.

---

## Checklist prima di consegnare

- [ ] Slide in PDF **e** PowerPoint, caricate su GitHub
- [ ] Sfondo bianco, numeri di pagina `n / 15`
- [ ] Figure ad alto contrasto (già generate da `make_figures.py`)
- [ ] Presentazione salvata anche su chiavetta USB
- [ ] Form inviato **entro il 6 settembre** (7 giorni prima dell'11)
- [ ] Repo modificabile fino a 2 giorni prima
- [ ] **Tre prove cronometrate**: se sforate i 10 minuti, tagliate le slide
      2 e 7, non la 8 e la 11
