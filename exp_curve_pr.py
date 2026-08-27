"""
Curve precision-recall su PAI 5 - la metrica primaria, disegnata invece che
riassunta.

PERCHE' SERVE, oltre alla PR-AUC che c'e' gia'. L'area e' un integrale, e
integrali uguali possono venire da curve molto diverse: un metodo preciso
finche' si pretende poca recall e inutile appena la si alza, e un altro che
tiene botta fino in fondo, possono avere la stessa PR-AUC. Su una
minoritaria clinica non interessa "tutta la curva": interessa la precisione
che resta quando si pretende di trovare, mettiamo, l'80% dei PAI 5. Quella
la si legge sulla curva e non nell'area.

E' anche l'unico modo di mostrare la metrica PRIMARIA del brief senza
comprimerla in un numero, cosa che in una presentazione conta.

COSA VIENE SALVATO. Le curve di tutti i seed, mediate in verticale su una
griglia comune di recall (vedi `curva_media` in evaluation.py): mediare
punto a punto sarebbe impossibile, ogni seed produce un numero diverso di
punti a valori di recall diversi. Insieme alle curve si salva la precisione
a tre recall di lavoro - 0.70, 0.80, 0.90 - perche' e' quello che si cita a
voce mentre la curva sta sullo schermo.

Uso:
    python exp_curve_pr.py --tag _casuale
    python figure_finali.py            # disegna fin6_curve_pr.png
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluation import curva_media, evaluate_split, pr_curve
from globals import OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CARICO = 70
METODI = ["none", "class_weighted", "focal", "oversample", "balanced_tokens"]
RECALL_LAVORO = (0.70, 0.80, 0.90)
CLASSE_PAI5 = 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_casuale")
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--head", default="flat")
    ap.add_argument("--metodi", nargs="+", default=METODI)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=CARICO)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)
    test = cached["data"]["test"]
    prevalenza = float((test["labels"] == CLASSE_PAI5).float().mean())
    print(f"\nencoder{a.tag}, testa {a.head}, {len(a.seeds)} seed, TEST")
    print(f"prevalenza di PAI 5 nel test: {prevalenza:.4f} "
          f"(e' il pavimento della curva: un classificatore casuale sta li')")

    fuori = {"tag": a.tag, "head": a.head, "seeds": a.seeds, "alpha": a.alpha,
             "prevalenza_pai5": prevalenza, "recall_lavoro": RECALL_LAVORO,
             "curve": {}, "precisione_a_recall": {}, "pr_auc": {}}
    percorso = os.path.join(OUT_DIR, f"curve_pr_{a.variant}{a.tag}.json")

    # RIPARTENZA, e salvataggio dopo OGNI metodo invece che alla fine. Sono
    # cinque metodi per cinque seed, circa un'ora: salvare solo in fondo
    # significa che un'interruzione al minuto cinquanta butta via tutto. E
    # le interruzioni qui sono previste - sorveglia.py ferma il comando
    # apposta se la GPU sfora.
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            vecchio = json.load(f)
        if all(vecchio.get(k) == v for k, v in
               (("seeds", a.seeds), ("head", a.head), ("alpha", a.alpha))):
            fuori = vecchio
            print(f"Riprendo: {len(fuori['curve'])} metodi gia' su disco")
        else:
            print(f"{os.path.basename(percorso)} esiste ma con un altro "
                  f"protocollo: riparto da capo")

    print(f"\n  {'metodo':18s} {'PR-AUC5':>9s}   "
          + "  ".join(f"prec@r{r:.2f}" for r in RECALL_LAVORO))
    print("  " + "-" * 60)
    for m in a.metodi:
        if m in fuori["curve"]:
            pr = fuori["precisione_a_recall"][m]
            print(f"  {m:18s} {fuori['pr_auc'][m][0]:.4f}    "
                  + "     ".join(f"{pr[f'{r:.2f}']:.3f}" for r in RECALL_LAVORO)
                  + "   [da disco]")
            continue
        curve, aree = [], []
        for s in a.seeds:
            t0 = time.perf_counter()
            clf, _ = train_head(cached, m, a.head, seed=s, bts_alpha=a.alpha)
            r = evaluate_split(clf, test, a.head, con_punteggi=True)
            y5 = (r["y_true"] == CLASSE_PAI5).astype(int)
            curve.append(pr_curve(r["scores"][:, CLASSE_PAI5], y5))
            aree.append(r["pr_auc_pai5"])
            del clf
            torch.cuda.empty_cache()
            freno.pausa(t0)

        griglia, media, dev = curva_media(curve)
        fuori["curve"][m] = {"recall": griglia.tolist(),
                             "precision": media.tolist(),
                             "dev": dev.tolist()}
        fuori["pr_auc"][m] = [float(np.mean(aree)), float(np.std(aree))]
        # `np.interp` sulla curva media, non media delle interpolazioni per
        # seed: sono la stessa cosa, la griglia e' gia' comune.
        prec = {f"{r:.2f}": float(np.interp(r, griglia, media))
                for r in RECALL_LAVORO}
        fuori["precisione_a_recall"][m] = prec
        print(f"  {m:18s} {fuori['pr_auc'][m][0]:.4f}    "
              + "     ".join(f"{prec[f'{r:.2f}']:.3f}" for r in RECALL_LAVORO),
              flush=True)
        save_json(fuori, percorso)

    save_json(fuori, percorso)
    print(f"\nRisultati in {percorso}")
    print("Ora: python figure_finali.py  ->  fin6_curve_pr.png")
