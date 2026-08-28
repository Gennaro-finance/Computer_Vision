# Provenienza di ogni misura

Documento di verifica: da quale encoder viene ogni file di latenti, e quale
protocollo di ritaglio è stato usato. Esiste perché il 27 agosto abbiamo
scoperto **due** guasti di provenienza, e senza questa tabella non erano
individuabili:

1. `run_grid` riprendeva da file saltando le celle già presenti **senza
   verificare con quali latenti fossero state prodotte**. Tre celle su
   dieci della griglia `_casuale` erano sopravvissute da uno stato
   precedente.
2. Il braccio `completa` della tabella dei tre encoder era misurato su
   latenti del **23 agosto**, mentre il checkpoint che dicevamo di usare
   (`completa_best`, epoca 179) è stato scritto il **24 agosto alle 20:20**.
   Quei latenti precedono di un giorno il modello dichiarato.

Da oggi `run_grid` salva l'impronta dei latenti nei risultati e **rifiuta**
la ripartenza quando non corrisponde. Questa tabella copre ciò che è stato
prodotto prima.

---

## I due protocolli di ritaglio

| protocollo | finestra | segnale geometrico | uso |
|---|---|---|---|
| **geometrico** | fissa, 224 px nativi, centrata sulla lesione | preservato: 16/36/64 token per PAI 3/4/5 | **è il protocollo del brief** |
| **cieco alla dimensione** | 3× il lato della bbox, ridimensionata a 224 | azzerato nel conteggio: 36 token per tutti | ablation sull'encoder |

Il cieco **non sostituisce** il protocollo del brief: lo affianca come
strumento di misura. E non è un isolamento perfetto — la dimensione rientra
in parte come magnificazione (R² 0,55–0,70 nel predire il lato della bbox
dal vettore).

---

## Latenti

| file | encoder | epoca | protocollo | verificato |
|---|---|---|---|---|
| `latents_..._casuale.pt` | `build_ijepa`, seme 42, **nessun peso addestrato** | — | geometrico | per costruzione |
| `latents_..._geo_completa.pt` | `ijepa_vit_small_completa_best` | **179** | geometrico | estratto il 27 ago |
| `latents_..._spinto.pt` | `ijepa_vit_small_spinto_best` | **29** | geometrico | **confronto numerico** (errore 2,3e-4 contro 2,8e-2 dell'altro candidato) |
| `latents_..._imagenet.pt` | torchvision ViT-B/16, `IMAGENET1K_V1` | — | geometrico | estratto il 27 ago |
| `latents_..._cieco_casuale.pt` | `build_ijepa`, seme 42 | — | cieco | estratto il 27 ago |
| `latents_..._cieco_completa.pt` | `ijepa_vit_small_completa_best` | **179** | cieco | estratto il 27 ago |
| `latents_..._cieco_notte.pt` | `ijepa_vit_small_notte` | **40** | cieco | estratto il 27 ago |
| `latents_..._cieco_mask80.pt` | `ijepa_vit_small_mask80` | **208** | cieco | estratto il 27 ago |
| `latents_..._cieco_imagenet.pt` | torchvision ViT-B/16 | — | cieco | estratto il 27 ago |
| `latents_..._geo_finale.pt` | `ijepa_vit_small_finale_best` | **69** | geometrico | log di estrazione, 28 ago |
| `latents_..._cieco_finale.pt` | `ijepa_vit_small_finale_best` | **69** | cieco | log di estrazione, 28 ago |
| `latents_..._geo_ultima.pt` | `ijepa_vit_small_finale` (ULTIMA epoca, non la migliore) | **288** | geometrico | atteso dalla catena |
| ~~`latents_vit_small_L2-7-11.pt`~~ | **IGNOTO**, 23 ago | ? | geometrico | ❌ **non identificabile** |

---

## Risultati

### Validi, provenienza nota

| file | encoder | note |
|---|---|---|
| `results_..._casuale.json` | casuale | griglia rifatta da zero il 27 ago con impronta |
| `results_..._cieco_casuale.json` | casuale, cieco | |
| `results_..._cieco_completa.json` | completa ep. 179, cieco | |
| `results_..._imagenet.json` | ViT-B/16 ImageNet | griglia ridotta, `none` + `balanced_tokens`, testa flat |
| `sweep_alpha_..._casuale.json` | casuale | |
| `curve_pr_..._casuale.json` | casuale | |
| `diversita_..._casuale*.json` | casuale | |
| `rumore_..._casuale.json` | casuale | |
| `sonde_vit_small.json` | tutti, entrambi i protocolli | sonde k-NN senza parametri |
| `mascheramento_vit_small.json` | casuale + completa ep. 179 | protocolli di mascheramento |
| `stratificata_vit_small.json` | casuale + completa ep. 179 | |
| `traiettoria_testa_cieco.json` | casuale + completa ep. 179, cieco | |

### Da sostituire

| file | problema |
|---|---|
| `results_vit_small_L2-7-11.json` | latenti del 23 ago da checkpoint **ignoto**. È il braccio `completa` della tabella principale. **Va rifatto su `_geo_completa`** |

### Archiviati — protocollo superato

Tutti del 20 agosto, con **ultimo blocco soltanto** (non `L2-7-11`) e
**ritaglio relativo alla bbox** (rimosso il 22 agosto perché annullava il
segnale dominante). Non sono confrontabili con nulla di attuale, e restano
solo come storia.

- `results_ijepa_vit_small.json` — I-JEPA di allora, un checkpoint collassato
- `results_imagenet_vit_small.json` — ImageNet
- `results_random_vit_small.json` — casuale
- `sweep_alpha_ijepa_vit_small.json` — primo sweep di alpha

---

## Un limite dichiarato del checkpoint `completa`

`ijepa_vit_small_completa_best.pt` è l'**epoca 179 su 300 configurate**, e
la run si è fermata a **230** per uno spegnimento hardware, non per
convergenza. Lo stato a 230 non è stato salvato: solo `_best` viene scritto.

E il `_best` è stato scelto massimizzando la macro-F1 downstream **nel
protocollo geometrico** — che il 27 agosto abbiamo dimostrato essere
dominato dal canale della maschera (la sola maschera one-hot, senza pixel,
dà macro-F1 0,7708). **Il criterio di selezione era cieco alla qualità
della rappresentazione.**

Entrambe le cose vanno dichiarate in presentazione.

**Correzione del 28 agosto.** Qui c'era scritto che la curva di apprendimento
— 0 epoche 0,3173, 40 epoche 0,4668, 179 epoche 0,6092, sonda k-NN a K fisso
— è *«monotona e non satura, il che suggerisce che più pre-training
aiuterebbe»*. È stata **la motivazione con cui è stato deciso il run
`finale`**, e il run stesso l'ha smentita: con una testa addestrata su
`P3_K16` la qualità satura verso l'epoca 70 e poi non si muove più
(+0,0023, z = +0,31 fra le prime e le ultime dieci sonde).

Le due misure non si contraddicono formalmente — una è k-NN a zero
parametri, l'altra una testa addestrata, e una testa capace raggiunge prima
il tetto — ma la conclusione operativa che se n'era tratta era sbagliata.
Più epoche **non** comprano più qualità.

Comprano un'altra cosa, che vale di più: il **disimparamento della bounding
box** (−0,0190, z = −6,73). Vedi la sezione sul run `finale`.

---

## Il run `finale` (28 agosto)

Rifatto da zero con la sonda di selezione corretta: sceglie su `P3_K16`
(16 token per tutti) invece che sul protocollo del brief, che il 27 agosto
avevamo dimostrato dominato dal canale della maschera.

| | |
|---|---|
| epoche | **289 su 300** — fermato da `sorveglia` per sforamento di potenza (101 W su 95), non da un errore |
| lr | 3e-5, confermato da misura contro 3e-4 (divario −0,0304, z = −2,98 su validation) |
| EMA | 0,9996 → 1,0, invariata |
| checkpoint migliore | **epoca 69**, macro-F1 K16 = 0,5591 su validation |
| ultimo checkpoint | epoca 288, usato come braccio "massimo disimparamento" |

**Il risultato della traiettoria.** Fra le prime 10 e le ultime 10 sonde:

| serie | prime 10 | ultime 10 | Δ | z |
|---|---|---|---|---|
| K16 (qualità) | 0,5406 | 0,5430 | +0,0023 | **+0,31** |
| P1_bbox (scorciatoia) | 0,7539 | 0,7349 | **−0,0190** | **−6,73** |

La qualità satura verso l'epoca 70; le duecento epoche successive servono a
**disimparare la bounding box**. È la tesi centrale del progetto misurata
come traiettoria invece che dedotta da due estremi.

**Da dichiarare**: 289/300 epoche, e il confronto sull'lr è stato prima
misurato per errore su torch 2.12 (numeri diversi: −0,0440, z = −4,88).
I valori citabili sono quelli su torch 2.13 del `.venv`.
