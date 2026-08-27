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

Entrambe le cose vanno dichiarate in presentazione. La curva di
apprendimento misurata con una sonda che vede — 0 epoche 0,3173, 40 epoche
0,4668, 179 epoche 0,6092 — è **monotona e non satura**, il che suggerisce
che più pre-training aiuterebbe.
