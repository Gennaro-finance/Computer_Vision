# Esito della catena

Avviata e conclusa il 29/08/2026, fine alle 00:32.

**lr scelto: 3e-05** — spinto e' PEGGIORE in modo separabile -> lr 3e-5 confermato dalla misura, non per prudenza.  completa(3e-5)=0.5763+-0.0182  spinto(3e-4)=0.5322+-0.0087  divario=-0.0440  z=-4.88   [ATTENZIONE: misurato su torch 2.12 del python di sistema, non sul .venv del progetto. Il divario e' interno e regge; i valori assoluti li rimisura lo stadio verifica_lr]

| stadio | esito | cosa produceva |
|---|---|---|
| `pretrain` | fatto | pre-training I-JEPA, 300 epoche, sonda di selezione su P3_K16 |
| `estrai_geo` | fatto | latenti nel protocollo del brief (finestra fissa 224 px) |
| `estrai_cieco` | fatto | latenti ciechi alla dimensione (finestra 3x la bbox) |
| `fixedk_flat` | fatto | i cinque protocolli di mascheramento, testa flat, sul test |
| `fixedk_mil` | fatto | MIL per token: il margine piu' grande del progetto |
| `fewshot` | fatto | few-shot 1/5/10/25/100% sui due protocolli |
| `griglia_geo` | fatto | griglia principale: 5 metodi x 2 teste, protocollo del brief |
| `griglia_cieca` | fatto | stessa griglia nel protocollo cieco, solo metodo none |
| `novita_K` | fatto | obiettivo 3, la novita', misurata DOVE si vede (P3_K16) |
| `mascheramento` | fatto | protocolli di mascheramento, sonda k-NN senza parametri |
| `stratificata` | fatto | prestazione stratificata per dimensione della lesione |
| `testa_pooling` | fatto | confronto fra i pooling, protocollo del brief |
| `testa_pooling_K16` | fatto | stesso confronto a conteggio fisso, dove il metro vede |
| `traiettoria` | fatto | traiettoria della testa nel protocollo cieco |
| `sonde` | fatto | sonde k-NN su tutti gli encoder, curva di apprendimento |
| `estrai_ultima` | fatto | latenti dall'ULTIMA epoca (non dal best): il piu' disimparato |
| `fixedk_ultima` | fatto | stesso K16 ma P1 piu' basso? la previsione sulla pipeline |
| `scarto` | fatto | da dove viene lo scarto di 0,0140 sul casuale |
| `verifica_lr` | fatto | rimisura del confronto lr sullo stack giusto (provenienza) |

Tutto a posto.

Il checkpoint nuovo e' `runs/checkpoints/ijepa_vit_small_finale_best.pt`. I bracci gia'
misurati non sono stati toccati: la catena scrive solo su tag
`finale`.
