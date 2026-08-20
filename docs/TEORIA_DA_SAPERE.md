# Cosa dovete sapere a livello teorico

Progetto 8 — Self-Supervised Latent Representations for Imbalanced Apical
Periodontitis Grading

---

## Come usare questa guida

Non è un corso di deep learning. È l'elenco ristretto di concetti che vi
servono **per scrivere la novità metodologica da soli** e **per rispondere
alle domande dopo i 10 minuti di presentazione**.

Le guidelines del corso penalizzano esplicitamente il contenuto generato da
modelli generativi. Il filtro pratico è la sessione di domande: se non sapete
spiegare perché il target encoder si aggiorna con l'EMA, si vede in trenta
secondi. Questa lista serve esattamente a quello.

Tre livelli di priorità:

- **[ESSENZIALE]** — ve lo chiederanno, o vi serve per scrivere il codice
- **[UTILE]** — vi fa fare bella figura e rende l'ablation sensato
- **[CONTESTO]** — leggetelo se avanza tempo

Budget realistico: **due giorni di studio distribuiti**, non bloccanti. La
maggior parte si legge mentre il pre-training gira.

Alla fine c'è un **autotest**: se sapete rispondere a quelle domande, siete
pronti.

---

## 1. Vision Transformer

### [ESSENZIALE] Da immagine a sequenza di token

Un ViT taglia l'immagine in patch fisse (16×16 nel vostro caso), le proietta
linearmente in vettori, e da lì in poi tratta l'immagine come una sequenza —
esattamente come un transformer tratta le parole.

**Perché conta per voi:** è tutta l'origine del problema di scala che avete
misurato. Un tile 224×224 con patch 16 dà una griglia 14×14 = 196 token. Se
la lesione è più piccola di un patch, semplicemente *non esiste* per il
modello. I vostri numeri: lesione 58×60 px su panoramica 2473×1252; a 224²
diventa 0,11 token di area, a risoluzione nativa 13,14.

### [ESSENZIALE] Self-attention

Ogni token produce query, key e value; l'attenzione pesa quanto ogni token
guarda ogni altro. Costo quadratico nel numero di token — motivo per cui la
risoluzione non si alza a piacere.

Vi basta saperlo a questo livello. Non dovete saper derivare i gradienti.

### [ESSENZIALE] Positional embedding

L'attenzione è invariante all'ordine: senza informazione di posizione, il
modello non sa dove sta ogni patch. Nel vostro codice sono sinusoidali 2D e
**fissi, non appresi**.

*Domanda probabile:* perché fissi? Perché con 4.000 immagini ogni parametro
appreso in più è un parametro che può overfittare, e le sinusoidali
generalizzano a risoluzioni diverse senza reinterpolazione.

### [ESSENZIALE] Perché ViT-Small e non ViT-Base

I ViT non hanno il bias induttivo delle CNN (località, invarianza per
traslazione): se lo devono costruire dai dati, e quindi ne vogliono tanti.
Su ~4.000 immagini un ViT-Base impara l'anatomia globale e overfitta.

Il vostro argomento non è "abbiamo poca GPU", è: **LeJEPA ottiene i suoi
risultati in-domain su Galaxy10 con modelli da 15-22M parametri**
(ConvNeXt-V2 Nano, ResNet-34), e ViT-Small con 22,1M è in quella fascia.

### [UTILE] Come si aggregano i token

Tre modi: token CLS, media dei token, **attention pooling** (una query
appresa che pesa i token). Voi usate il terzo, ristretto ai token dentro la
bbox della lesione.

*Perché:* la media diluirebbe la lesione nell'osso circostante; l'attention
pooling impara quali token contano. Ed è il punto in cui si innesta la
vostra novità.

---

## 2. Self-supervised learning

### [ESSENZIALE] La tassonomia, in quattro famiglie

| Famiglia | Idea | Esempi |
|---|---|---|
| Contrastive | avvicina viste della stessa immagine, allontana le altre | SimCLR, MoCo |
| Self-distillation | una rete predice l'output di una copia di sé stessa | BYOL, DINO |
| Masked modeling | ricostruisci la parte nascosta **nei pixel** | MAE, BEiT |
| **Joint-embedding predictive** | predici la parte nascosta **nel latente** | **I-JEPA** |

Sapete collocare I-JEPA in questa mappa: è la prima domanda naturale.

### [ESSENZIALE] Perché JEPA e non MAE — il punto concettuale del progetto

MAE ricostruisce i **pixel** mancanti. Per farlo bene deve modellare anche
grana del sensore, rumore, texture irrilevanti: capacità spesa su dettagli
che non servono a nessun compito a valle.

I-JEPA predice la **rappresentazione** della regione mancante, prodotta dal
target encoder. Il target è già astratto, quindi il modello non è costretto
a inventare dettagli di basso livello.

Su radiografie questo argomento è particolarmente forte: il rumore
radiografico è alto e completamente privo di informazione clinica. Ditelo in
presentazione.

### [ESSENZIALE] Collasso della rappresentazione

Il fallimento tipico del self-supervised: la rete scopre che mappare **tutto
a uno stesso vettore costante** rende la predizione banale. La loss scende
regolarmente e le rappresentazioni non valgono nulla.

Come lo si evita, per famiglia:

| Meccanismo | Chi lo usa |
|---|---|
| Negativi espliciti | SimCLR, MoCo |
| Stop-gradient + target EMA | BYOL, **I-JEPA** |
| Decorrelazione delle feature | Barlow Twins, VICReg |
| Regolarizzazione della distribuzione | **SIGReg / LeJEPA** |

Come lo **rilevate** — e questo è vostro codice, dovete saperlo spiegare:

- deviazione standard degli embedding → 0
- **rango effettivo** della matrice dei momenti secondi → 1
- k-NN probe fermo al livello della classe maggioritaria (0,612)

*Dettaglio da conoscere:* il rango effettivo va calcolato sui momenti
secondi di embedding normalizzati, **non** sulla covarianza centrata. Con la
covarianza centrata, embedding tutti identici più un epsilon di rumore
danno rango ≈ d invece di 1, perché dopo il centraggio resta solo il rumore,
che è isotropo. Il collasso costante — il caso più classico — passerebbe
inosservato.

### [ESSENZIALE] Il target encoder EMA

θ_target ← m · θ_target + (1−m) · θ_context, con m tipicamente 0,996 → 1,0.

Perché funziona: il bersaglio si muove **lentamente**. Se context e target
fossero la stessa rete aggiornata insieme, entrambi correrebbero verso la
soluzione costante. L'inerzia dell'EMA rende quella scorciatoia inaccessibile.

*Domanda probabile:* perché il momentum cresce verso 1 durante il training?
All'inizio il target è casuale e deve adeguarsi in fretta; alla fine deve
essere stabile per non introdurre rumore.

### [ESSENZIALE] Il masking di I-JEPA

Un **blocco di contesto** ampio (85-100% dell'area) e **quattro blocchi
target** piccoli (15-20%), con i target **rimossi dal contesto**.

Se non li rimuovete, il compito è banale: la risposta è nell'input. È un bug
classico e silenzioso — la loss crolla e sembra tutto perfetto.

Il predictor riceve i token di contesto codificati più dei *mask token*
posizionati dove stanno i target, e deve produrre le rappresentazioni target.
Va tenuto **stretto e poco profondo**: se ha troppa capacità risolve il
compito da solo e l'encoder non impara niente.

### [UTILE] LeJEPA e SIGReg

LeJEPA (Balestriero & LeCun, 2025) rimuove EMA, stop-gradient e
teacher-student, sostituendoli con **SIGReg**: si vincolano gli embedding a
distribuirsi come una **gaussiana isotropa**, usando proiezioni casuali e
test statistici 1-D. Un solo iperparametro di trade-off, costo lineare,
~50 righe.

Perché vi riguarda: è la ref [2] del vostro brief, ed è il vostro braccio di
confronto e l'assicurazione sul collasso. E il risultato Galaxy10 — SSL
in-domain su dataset specialistico che batte il transfer da DINOv3 — è la
giustificazione teorica della tesi del progetto.

*Tensione da conoscere:* l'obiettivo 1 del brief impone l'EMA, che LeJEPA
elimina. Per questo implementate I-JEPA come primario e SIGReg come
confronto.

---

## 3. Sbilanciamento di classe

### [ESSENZIALE] Perché l'accuracy è la metrica sbagliata

Con PAI 3 al 61%, un modello che predice sempre PAI 3 ottiene 61% di
accuracy e utilità clinica zero. Il brief lo vieta implicitamente.

Cosa si usa invece: **Macro-F1** (media delle F1 per classe, quindi ogni
classe pesa uguale), **PR-AUC** per la minoritaria, **balanced accuracy**,
**confusion matrix**.

*Perché PR-AUC e non ROC-AUC sulla minoritaria:* la ROC-AUC è ottimistica
quando i negativi sono tantissimi, perché il false positive rate resta basso
anche con molti falsi positivi in valore assoluto. La PR curve guarda la
precision, che quei falsi positivi li sente.

### [ESSENZIALE] Le tre famiglie di rimedi

| Famiglia | Come | Esempi |
|---|---|---|
| Re-weighting | pesi nella loss | inverse frequency, effective number |
| Re-sampling | cambi la distribuzione dei batch | oversampling, SMOTE |
| Loss shaping | cambi la forma della loss | focal loss, LDAM |

**Focal loss:** moltiplica la CE per (1−p_t)^γ, quindi gli esempi già facili
contribuiscono poco e il modello si concentra su quelli difficili.

**LDAM:** margini per classe proporzionali a n_c^(−1/4). Le classi rare
ottengono un margine più largo, cioè devono essere classificate con più
confidenza. La derivazione viene da un bound sull'errore di generalizzazione.

**SMOTE:** genera campioni sintetici interpolando tra vicini della stessa
classe minoritaria. Con encoder congelato i vostri campioni **sono già
vettori di feature**, quindi SMOTE si applica direttamente nel latente,
senza generare pixel. È la vostra baseline più scomoda: se la novità non la
batte, dovete saperlo prima della presentazione.

### [UTILE] Decoupling — perché la vostra architettura è già quella giusta

Kang et al. hanno mostrato che apprendimento della rappresentazione e
apprendimento del classificatore vanno **disaccoppiati**: la rappresentazione
si impara bene sulla distribuzione naturale sbilanciata, e solo la testa va
ribilanciata.

È esattamente quello che fate: encoder congelato + testa leggera. Citatelo,
perché trasforma una scelta pratica in una scelta motivata dalla letteratura.

### [ESSENZIALE] La vostra novità, in una frase

**Balanced token sampling:** per le classi rare si campionano sottoinsiemi
diversi dei token dentro la stessa bbox, e ogni sottoinsieme diventa
un'istanza di training distinta.

Perché è diversa dalle alternative:

- l'oversampling classico ripresenta **lo stesso identico vettore** e invita
  all'overfitting;
- SMOTE **interpola** in un latente dove l'interpolazione può non avere
  senso anatomico;
- questa produce viste **genuine** della stessa lesione reale.

Dovete saper difendere: la scelta di α, l'intervallo di frazione di token
tenuti, campionamento uniforme o pesato per attenzione, e il fatto che si
applica **solo in training** — mai in valutazione, altrimenti confrontate
cose diverse.

---

## 4. Regressione ordinale

### [ESSENZIALE] Il PAI è una scala ordinata

PAI 3 < PAI 4 < PAI 5. Trattarlo come tre categorie senza relazione butta
via informazione **e usa la metrica sbagliata**: confondere PAI 3 con PAI 5 è
clinicamente più grave che confondere PAI 4 con PAI 5, ma la Macro-F1 li
pesa uguale.

Verifica concreta fatta nel vostro codice: a parità di numero di errori, la
Macro-F1 valuta l'errore a due gradi **meglio** di quello a un grado (0,641
contro 0,619), mentre il kappa quadratico li ordina correttamente (0,429
contro 0,857).

### [ESSENZIALE] Kappa di Cohen quadratico pesato

Penalizza gli errori in proporzione al **quadrato** della distanza sulla
scala: sbagliare di due gradi costa quattro volte sbagliare di uno. È lo
standard per il grading ordinale in ambito medico.

1,0 = accordo perfetto, 0,0 = accordo casuale, negativo = peggio del caso.

### [UTILE] Teste ordinali (CORAL)

Invece di una softmax a 3 vie, si predicono K−1 soglie cumulative:
P(y > 3), P(y > 4). Con un peso condiviso e bias separati si garantisce la
**monotonicità** — non può risultare P(y>4) > P(y>3), che sarebbe
incoerente.

Tenete comunque la testa piatta come confronto: la scelta va argomentata con
i numeri, non per principio.

---

## 5. Metodologia sperimentale

### [ESSENZIALE] Le tre forme di leakage nel vostro dataset

1. **Livello paziente** — il brief lo chiede, ma il dataset non documenta ID
   paziente. Voi splittate per immagine e **dichiarate l'assunzione**.
2. **Immagini aumentate** — 13.071 derivate dalle originali. Splittarle a
   caso metterebbe varianti della stessa radiografia in train e test. Voi le
   escludete del tutto.
3. **Più lesioni per immagine** (1,44 misurato) — splittare per *lesione*
   metterebbe la stessa radiografia nei due lati. Si splitta per **immagine**
   e le lesioni seguono.

La terza è la più sottile ed è quella che quasi nessuno nota.

### [ESSENZIALE] Come si valuta un encoder self-supervised

Non si guarda la loss: si congela l'encoder e si misura quanto le feature
sono utili.

- **k-NN probe** — nessun parametro da addestrare, segnale immediato
- **linear probe** — un layer lineare sulle feature congelate, lo standard
  in letteratura
- **fine-tuning** — misura altro, non la qualità della rappresentazione

Voi usate il k-NN ogni 20 epoche come allarme precoce.

### [ESSENZIALE] Perché servono più seed

Lo sbilanciamento è 7:1, non 100:1. I margini tra i metodi saranno
**stretti**, e una singola run non distingue un miglioramento vero dal
rumore di inizializzazione. Cinque seed, media e deviazione standard.
Grazie ai latenti cachati costa secondi.

### [UTILE] Come si progetta un ablation

Si cambia **una cosa per volta** rispetto a una configurazione di
riferimento, e si riporta anche ciò che non ha funzionato. Un ablation che
mostra solo miglioramenti è sospetto.

---

## 6. Il dominio clinico

### [ESSENZIALE] Cos'è una lesione periapicale

L'infiammazione cronica all'apice della radice, conseguenza di necrosi
pulpare o infezione. Radiograficamente appare come **radiotrasparenza**
(area più scura) attorno all'apice, dovuta a riassorbimento osseo.

### [ESSENZIALE] Cos'è il PAI

Il **Periapical Index** di Ørstavik, Kerekes ed Eriksen (1986): una scala
ordinale a 5 punti per la valutazione radiografica della periodontite
apicale, costruita e calibrata su radiografie di riferimento. Va da strutture
periapicali normali fino a periodontite severa con segni di esacerbazione.

**Perché il vostro dataset ha solo 3, 4 e 5:** i punteggi bassi
corrispondono a periapice sano o a variazioni minime, e nella pratica si
considera malattia dal 3 in su. Il dataset annota solo lesioni, quindi solo
i gradi patologici.

Leggete l'abstract dell'articolo originale per le definizioni precise dei
singoli gradi: è una risposta che potrebbero chiedervi e va data con le
parole giuste.

### [UTILE] Perché le panoramiche sono difficili

Sono una **proiezione 2D di una struttura 3D curva**, ottenuta con un
movimento sincronizzato di sorgente e sensore. Conseguenze: sovrapposizione
di strutture, distorsione geometrica variabile lungo l'arcata, ingrandimento
non uniforme, zone di sfocatura fuori dallo strato focale.

Rispetto alle radiografie endorali, la panoramica è **meno accurata** per la
valutazione periapicale. È un limite onesto da citare nelle conclusioni.

---

## 7. Cosa leggere, in ordine

### Da leggere davvero

1. **Assran et al., I-JEPA (CVPR 2023)** — sezioni 1, 3 e 4. È
   l'architettura che state implementando.
2. **Balestriero & LeCun, LeJEPA (2025)** — abstract, introduzione e la
   figura dei risultati in-domain. Serve per lo State of the Art e per il
   braccio di confronto.
3. **Ørstavik et al. (1986)** — l'abstract basta, ma leggetelo.

### Da scorrere

4. **Kang et al., Decoupling representation and classifier** — l'argomento
   per cui la vostra architettura congelata è la scelta giusta.
5. **Cao et al., LDAM** — la baseline di loss shaping.
6. **Chawla et al., SMOTE** — sono poche pagine e vi serve per implementarlo.
7. **Cao & Chen, CORAL** — se implementate la testa ordinale.

### Per contesto

8. **He et al., MAE** — utile per l'argomento "pixel contro latente".
9. **Grill et al., BYOL** — l'origine del target EMA.
10. **Caron et al., DINO** — per collocare la self-distillation.

---

## 8. Autotest

Se sapete rispondere a queste, siete pronti per le domande.

**Architettura**
1. Perché I-JEPA predice nel latente invece che nei pixel, e perché su
   radiografie l'argomento è più forte?
2. Cosa succede se **non** rimuovete i blocchi target dal contesto?
3. Perché il predictor deve restare piccolo?
4. Perché il target encoder si aggiorna con l'EMA e non con i gradienti?
5. Perché il momentum EMA cresce verso 1 durante il training?

**Il vostro problema di scala**
6. Perché non ridimensionate la panoramica intera a 224×224? Date il numero.
7. Quanti token copre una lesione a risoluzione nativa, e perché basta?

**Collasso**
8. Cos'è il collasso della rappresentazione e perché la loss non lo rivela?
9. Come lo rilevate? Perché il rango effettivo va calcolato sui momenti
   secondi e non sulla covarianza centrata?

**Sbilanciamento**
10. Perché non riportate l'accuracy globale?
11. Perché PR-AUC e non ROC-AUC per PAI 5?
12. In cosa la vostra novità differisce da oversampling e da SMOTE latente?
13. La vostra novità batte SMOTE latente? Se no, cosa ne concludete?

**Ordinalità**
14. Perché il kappa quadratico e non solo la Macro-F1?
15. Cosa garantisce la monotonicità in una testa CORAL?

**Metodo**
16. Quali sono le tre forme di leakage in questo dataset, e come le gestite?
17. Perché cinque seed?
18. Il JEPA in-domain batte le feature ImageNet congelate? Se no, perché il
    risultato Galaxy10 di LeJEPA non si riproduce da voi?

La 18 è quella che vi faranno con più probabilità, perché è la domanda di
ricerca del progetto. Preparate la risposta in entrambe le direzioni.

---

## 9. Come dividerselo in tre

Coerente con la divisione dei compiti sul codice:

| Persona | Ruolo | Sezioni da padroneggiare | Sezioni da conoscere |
|---|---|---|---|
| **A** — Dati e SSL | ViT, I-JEPA, collasso | 1, 2, 5.1 | 3, 4 |
| **B** — Downstream e novità | sbilanciamento, ordinalità | 3, 4 | 1, 2 |
| **C** — Confronti e valutazione | metriche, metodo, dominio | 5, 6 | 2, 3 |

**Ma l'autotest lo fate tutti e tre.** Alla presentazione le domande non
arrivano divise per competenza, e rispondere "quella parte l'ha fatta il mio
collega" è la peggiore risposta possibile.
