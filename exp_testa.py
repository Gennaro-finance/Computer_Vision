"""
La testa e' il solo pezzo addestrabile: e' lei il collo di bottiglia?

L'encoder e' congelato per vincolo del brief, quindi tutta la capacita'
disponibile sta nel pooling piu' la testa. La testa attuale e' un solo
nn.Linear(1152, 3): vale la pena chiedersi quanto costa quella semplicita'.

DUE LEVE, misurate separatamente per poterle attribuire:
  norm  LayerNorm in ingresso. Le feature sono i blocchi 2, 7 e 11
        concatenati, con scale diverse: senza normalizzare, quello con la
        scala maggiore domina il gradiente.
  mlp   uno strato nascosto con GELU. Il grado PAI dipende dalla
        congiunzione di dimensione e scurezza, che non e' lineare nelle due
        separate.

Valutazione su VALIDATION: qui si sceglie il protocollo, e sceglierlo
guardando il test sarebbe barare. Il test si tocca solo alla fine.
"""

import time

import numpy as np
import torch

from evaluation import evaluate_split
from globals import N_SEEDS
from train_downstream import load_latents, train_head
from utils import Freno

CARICO = 80

TESTE = [
    ("flat",     "lineare (attuale)"),
    ("norm",     "LayerNorm + lineare"),
    ("mlp",      "LayerNorm + nascosto 256 + GELU"),
    ("ordinal",  "ordinale (attuale)"),
    ("norm_ord", "LayerNorm + ordinale"),
    ("mlp_ord",  "LayerNorm + nascosto + ordinale"),
]


def misura(cached, head, freno, seeds):
    f1, pr, rec = [], [], []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, "none", head, seed=s)
        r = evaluate_split(clf, cached["data"]["val"], head)
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
        f1.append(r["macro_f1"])
        pr.append(r.get("pr_auc_pai5", np.nan))
        rec.append(r["recall_per_class"][2] if "recall_per_class" in r
                   else r.get("recall_pai5", np.nan))
    return np.array(f1), np.array(pr), np.array(rec)


if __name__ == "__main__":
    freno = Freno(CARICO)
    print(f"[freno] {freno}\n")

    t0 = time.time()
    cached = load_latents("vit_small", layers=[2, 7, 11])
    print(f"latenti caricati in {time.time()-t0:.0f}s, embed {cached['embed_dim']}\n")

    seeds = list(range(N_SEEDS))
    print(f"{'testa':34s} {'macro-F1':>17s} {'PR-AUC PAI5':>17s} {'recall PAI5':>13s}")
    print("-" * 86)
    esiti = {}
    for head, nome in TESTE:
        f1, pr, rec = misura(cached, head, freno, seeds)
        esiti[head] = (f1, pr)
        print(f"  {nome:32s} {f1.mean():.4f}+-{f1.std():.4f}  "
              f"{pr.mean():.4f}+-{pr.std():.4f}  {np.nanmean(rec):11.4f}")

    print(f"\n{'confronto':34s} {'d macro-F1':>12s} {'err.std':>9s} {'d PR-AUC5':>11s}")
    print("-" * 70)
    for base, var, etichetta in [("flat", "norm", "norm vs lineare"),
                                 ("flat", "mlp", "mlp vs lineare"),
                                 ("norm", "mlp", "mlp vs norm (solo il nascosto)"),
                                 ("ordinal", "norm_ord", "norm_ord vs ordinale"),
                                 ("ordinal", "mlp_ord", "mlp_ord vs ordinale")]:
        (fb, pb), (fv, pv) = esiti[base], esiti[var]
        d = fv.mean() - fb.mean()
        # errore standard della differenza fra due medie su n seed ciascuna
        se = float(np.sqrt(fv.var(ddof=1)/len(fv) + fb.var(ddof=1)/len(fb)))
        z = d / se if se > 0 else float("inf")
        print(f"  {etichetta:32s} {d:+12.4f} {z:8.1f}x {pv.mean()-pb.mean():+11.4f}")

    print(f"\nPausa accumulata dal freno: {freno.pausa_totale/60:.1f} min")
    print("Nessuna differenza sotto ~2 errori standard e' dichiarabile.")
