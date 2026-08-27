"""
La curva di validation epoca per epoca, per due encoder a confronto.

LA DOMANDA CHE LA MEDIA NASCONDE. La griglia riporta un numero per
configurazione: la prestazione dell'epoca migliore, mediata su 5 seed. Due
encoder che arrivano allo stesso punto danno lo stesso numero anche se ci
arrivano in modo molto diverso - uno subito e poi fermo, l'altro lentamente
e ancora in salita alla fine. Sono situazioni diverse e la media le
confonde.

Serve soprattutto nel protocollo CIECO ALLA DIMENSIONE, dove la domanda e'
se il nostro I-JEPA abbia imparato qualcosa che il protocollo geometrico
mascherava. Se `completa` sale piu' in fretta del casuale anche arrivando
allo stesso posto, quella e' informazione: vuol dire che la
rappresentazione e' piu' facile da leggere, che e' esattamente cio' che il
pre-training dovrebbe produrre.

COSA REGISTRA. La macro-F1 e la PR-AUC su PAI 5 calcolate sul VALIDATION
ogni `--ogni` epoche, mediate su piu' seed. Il validation e non il test:
qui si guarda una traiettoria, e guardarla sul test significherebbe
consumarlo per un'analisi esplorativa.

NOTA SULLA SELEZIONE. `traccia_ogni` in train_head e' solo registrazione:
la scelta dell'epoca migliore continua a usare la griglia ogni 10 epoche.
Tenerle separate e' necessario - se la selezione vedesse piu' candidati
sceglierebbe il massimo di piu' estrazioni, e la distorsione cambierebbe il
numero misurato invece di limitarsi a descriverlo.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- \\
        python exp_traiettoria_testa.py --tag _cieco_casuale _cieco_completa
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from globals import OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CARICO = 100
OGNI = 5
SEEDS = [0, 1, 2]


def traiettoria(cached, metodo, testa, seeds, ogni, freno):
    """Media delle curve sui seed, su una griglia comune di epoche."""
    curve = []
    for s in seeds:
        t0 = time.perf_counter()
        _, best = train_head(cached, metodo, testa, seed=s, traccia_ogni=ogni)
        curve.append(best["traiettoria"])
        torch.cuda.empty_cache()
        freno.pausa(t0)

    epoche = [p["epoca"] for p in curve[0]]
    fuori = {"epoche": epoche}
    for k in ("val_macro_f1", "val_pr_auc_pai5"):
        m = np.array([[p[k] for p in c] for c in curve])
        fuori[k] = m.mean(0).tolist()
        fuori[k + "_std"] = m.std(0).tolist()
    return fuori


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["_cieco_casuale", "_cieco_completa"])
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--metodo", default="none")
    ap.add_argument("--testa", default="flat")
    ap.add_argument("--ogni", type=int, default=OGNI)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--carico", type=int, default=CARICO)
    ap.add_argument("--nome", default="cieco",
                    help="suffisso del file di uscita")
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    print(f"metodo {a.metodo}, testa {a.testa}, {len(a.seeds)} seed, "
          f"validation ogni {a.ogni} epoche\n")

    fuori = {"metodo": a.metodo, "testa": a.testa, "seeds": a.seeds,
             "ogni": a.ogni, "curve": {}}
    percorso = os.path.join(OUT_DIR, f"traiettoria_testa_{a.nome}.json")

    for tag in a.tag:
        cached = load_latents(a.variant, layers=a.layers, tag=tag)
        c = traiettoria(cached, a.metodo, a.testa, a.seeds, a.ogni, freno)
        fuori["curve"][tag] = c
        save_json(fuori, percorso)
        n = len(c["epoche"])
        print(f"{tag}")
        print(f"  {'epoca':>6s} {'macro-F1':>18s} {'PR-AUC5':>18s}")
        for i in range(0, n, max(1, n // 10)):
            print(f"  {c['epoche'][i]:6d} "
                  f"{c['val_macro_f1'][i]:9.4f}+-{c['val_macro_f1_std'][i]:.4f} "
                  f"{c['val_pr_auc_pai5'][i]:9.4f}+-{c['val_pr_auc_pai5_std'][i]:.4f}")
        print(f"  finale {c['val_macro_f1'][-1]:9.4f}"
              f"          {c['val_pr_auc_pai5'][-1]:9.4f}\n", flush=True)
        del cached

    # Confronto: quanto prima uno raggiunge il massimo dell'altro?
    if len(a.tag) == 2:
        x, y = (fuori["curve"][t] for t in a.tag)
        mx, my = max(x["val_macro_f1"]), max(y["val_macro_f1"])
        print(f"{'':22s} {a.tag[0]:>18s} {a.tag[1]:>18s}")
        print(f"{'massimo macro-F1':22s} {mx:18.4f} {my:18.4f}")
        # epoca in cui ciascuno raggiunge il 95% del proprio massimo
        for nome, c in zip(a.tag, (x, y)):
            soglia = 0.95 * max(c["val_macro_f1"])
            e = next(ep for ep, v in zip(c["epoche"], c["val_macro_f1"])
                     if v >= soglia)
            print(f"  {nome:20s} raggiunge il 95% del suo massimo all'epoca {e}")

    print(f"\nRisultati in {percorso}")
