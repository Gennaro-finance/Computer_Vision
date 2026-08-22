"""
Evaluation - quanto pesa la scala della lesione, e quanto la sola geometria.

Sezione "Evaluation" della struttura richiesta dal corso.

DA DOVE NASCE. Il crop del downstream era proporzionale alla bbox e poi
ridimensionato a TILE_SIZE: il fattore di scala variava da lesione a lesione
esattamente in modo da ANNULLARE la differenza di dimensione. Misurato il
21 ago, i lati mediani della bbox per grado sono 57 / 81 / 126 px, e due sole
soglie su quel numero - senza rete, senza immagine - danno macro-F1 0.7567 e
kappa 0.7779 sul test, piu' di qualunque encoder provato.

TRE CONFIGURAZIONI, per separare gli effetti:
  relative        crop proporzionale (comportamento originale)
  fixed           crop a lato costante: la dimensione sopravvive nei pixel
  fixed + geom    in piu', la geometria della bbox data alla testa

L'ultima e' uno STIRAMENTO della traccia e va letta come ablation, non come
metodo: il Task usa le coordinate per ESTRARRE i latenti, e la testa deve
classificare dai latenti. Se la geometria domina, le differenze fra encoder
si schiacciano e l'ablation dell'obiettivo 4 perde significato. Serve a
rispondere a "quanta parte del segnale e' puramente geometrica", che e' una
domanda onesta e va riportata.

Uso:
    python exp_scala.py --cache     # rigenera le cache nella modalita' attuale
    python exp_scala.py             # misura
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation import evaluate_split
from globals import LESION_CROP_MODE, N_SEEDS, OUT_DIR
from train_downstream import cache_latents, load_latents, train_head

BRACCI = [("ijepa", "pred48"), ("random", ""), ("imagenet", "")]
LAYERS = [2, 7, 11]
SOLO_SENZA_GEOM = False


def rigenera_cache():
    for arm, tag in BRACCI:
        print(f"\n=== cache [{arm}] modalita' crop = {LESION_CROP_MODE}")
        cache_latents("vit_small", arm=arm, layers=LAYERS, ckpt_tag=tag)


def misura(seeds=N_SEEDS):
    righe = []
    print(f"\nmodalita' crop = {LESION_CROP_MODE}   layer = {LAYERS}   seed = {seeds}")
    # L'ordinamento primario e' la PR-AUC su PAI 5, non la macro-F1.
    # Il Task lo chiede esplicitamente: "Performance on the MINORITY CLASS
    # should be reported using THRESHOLD-AGNOSTIC metrics". La macro-F1 non
    # e' threshold-agnostic (dipende dall'argmax) e media le tre classi con
    # lo stesso peso, che non e' cio' che la traccia prescrive. La PR-AUC su
    # PAI 5 lo e', ed e' la seconda metrica nominata dal brief.
    print(f"{'braccio':10s} {'geom':>5s} {'PR-AUC PAI5':>16s} {'macro-F1':>17s} "
          f"{'rec5':>6s} {'prec5':>6s} {'F1 3/4/5':>18s} {'kappa':>7s}")
    print("-" * 92)

    for arm, _ in BRACCI:
        try:
            cached = load_latents("vit_small", arm, layers=LAYERS)
        except FileNotFoundError:
            print(f"{arm:10s} (cache mancante, lanciate --cache)")
            continue

        for use_geom in ((False,) if SOLO_SENZA_GEOM else (False, True)):
            per = []
            for s in range(seeds):
                clf, _ = train_head(cached, "none", "flat", seed=s, use_geom=use_geom)
                per.append(evaluate_split(clf, cached["data"]["test"], "flat",
                                          use_geom=use_geom))
                del clf
                torch.cuda.empty_cache()

            def m(k):
                return float(np.mean([p[k] for p in per]))

            def sd(k):
                return float(np.std([p[k] for p in per]))

            righe.append({"arm": arm, "crop_mode": LESION_CROP_MODE,
                          "geom": use_geom, "layers": LAYERS, "n_seeds": seeds,
                          **{f"{k}_mean": m(k) for k in
                             ("macro_f1", "f1_pai3", "f1_pai4", "f1_pai5",
                              "recall_pai5", "precision_pai5", "pr_auc_pai5",
                              "quadratic_kappa")},
                          **{f"{k}_std": sd(k) for k in
                             ("macro_f1", "pr_auc_pai5", "recall_pai5")}})
            print(f"{arm:10s} {'si' if use_geom else 'no':>5s} "
                  f"{m('pr_auc_pai5'):8.4f}+-{sd('pr_auc_pai5'):.4f} "
                  f"{m('macro_f1'):9.4f}+-{sd('macro_f1'):.4f} "
                  f"{m('recall_pai5'):6.3f} {m('precision_pai5'):6.3f} "
                  f"{m('f1_pai3'):5.3f}/{m('f1_pai4'):.3f}/{m('f1_pai5'):.3f} "
                  f"{m('quadratic_kappa'):7.4f}", flush=True)
        del cached

    path = os.path.join(OUT_DIR, f"exp_scala_{LESION_CROP_MODE}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(righe, f, indent=2)
    print(f"\nSalvato in {path}")
    print("\nRiferimenti misurati (macro-F1):")
    print("  due soglie sul solo lato della bbox : 0.7567   (nessuna rete)")
    print("  pavimento, sempre PAI 3             : 0.2589")
    if righe:
        b = max(righe, key=lambda r: r["pr_auc_pai5_mean"])
        print(f"\nMigliore sul CRITERIO DEL TASK (PR-AUC su PAI 5): "
              f"{b['arm']}, geom={'si' if b['geom'] else 'no'} -> "
              f"{b['pr_auc_pai5_mean']:.4f}+-{b['pr_auc_pai5_std']:.4f}")
        bm = max(righe, key=lambda r: r["macro_f1_mean"])
        if bm["arm"] != b["arm"]:
            print(f"  (per macro-F1 vincerebbe invece {bm['arm']}: "
                  f"{bm['macro_f1_mean']:.4f} - le due metriche non concordano, "
                  f"ed e' un risultato da riportare)")
    return righe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--no-geom-only", action="store_true",
                    help="salta le righe con geometria: misurato che aggiunge "
                         "+0.001, cioe' nulla, su entrambi i bracci provati")
    a = ap.parse_args()
    if a.no_geom_only:
        globals()["SOLO_SENZA_GEOM"] = True
    if a.cache:
        rigenera_cache()
    misura(a.seeds)
