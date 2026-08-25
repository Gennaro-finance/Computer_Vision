"""
Spostarsi di piu' dai pesi casuali aiuta o peggiora?

L'ipotesi da testare: esiste una configurazione che si allontana molto
dall'inizializzazione E rende meglio. I due punti che abbiamo suggeriscono
il contrario, ma due punti non sono una curva.

Qui si misurano TUTTI i checkpoint sul disco - regimi diversi, epoche
diverse, iperparametri diversi - e si mette in relazione:

    quanto si e' spostato       ->    quanto rende
    (pesi, e CKA rispetto al          (downstream vero, su validation)
     casuale)

Se la relazione e' piatta, lo spostamento non conta.
Se e' decrescente, allontanarsi PEGGIORA e l'ipotesi cade.
Se e' crescente, vale la pena cercare configurazioni che si spostano di piu'.
"""
import glob
import math
import os
import time

import numpy as np
import torch

from globals import DEVICE, SEED
from network import bbox_to_token_mask, build_ijepa
from utils import Freno, load_checkpoint, set_seed

CARICO = 80
SEEDS = [0, 1, 2]
ESCLUSI = ("lejepa",)          # architettura diversa, non confrontabile


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).pow(2).sum()) / math.sqrt(
        float((X.T @ X).pow(2).sum()) * float((Y.T @ Y).pow(2).sum()))


def spostamento(sd, sd0):
    num = math.sqrt(sum(float(((sd[k].float() - sd0[k].float())**2).sum())
                        for k in sd0 if k.startswith("context_encoder")
                        and sd0[k].dtype.is_floating_point and k in sd))
    den = math.sqrt(sum(float((sd0[k].float()**2).sum())
                        for k in sd0 if k.startswith("context_encoder")
                        and sd0[k].dtype.is_floating_point))
    return num / den


if __name__ == "__main__":
    import train_ssl
    from data import (CropCacheDataset, cache_crop, load_splits, make_loader,
                      parse_annotations)

    freno = Freno(CARICO)
    print(f"[freno] {freno}\n")
    recs = parse_annotations(verbose=False)
    sp = load_splits()
    ld = make_loader(CropCacheDataset(cache_crop(recs, sp["val"], "val")),
                     shuffle=False, batch_size=64, num_workers=0)

    def feats(m, cap=10):
        F = []
        with torch.no_grad():
            for i, b in enumerate(ld):
                if i >= cap: break
                t = m.encode(b["image"].to(DEVICE))
                msk = bbox_to_token_mask(b["bbox"].to(DEVICE), m.grid).float().unsqueeze(-1)
                F.append(((t * msk).sum(1) / msk.sum(1).clamp(min=1)).float().cpu())
        return torch.cat(F)

    set_seed(SEED)
    casuale = build_ijepa("vit_small").to(DEVICE).eval()
    sd0 = {k: v.clone() for k, v in casuale.state_dict().items()}
    F0 = feats(casuale)

    t0 = time.perf_counter()
    d0 = np.mean([train_ssl.sonda_downstream(casuale, recs, sp, seed=s)[0] for s in SEEDS])
    freno.pausa(t0)
    print(f"riferimento CASUALE: downstream {d0:.4f}  (media su {len(SEEDS)} seed)\n")

    righe = []
    for p in sorted(glob.glob("runs/checkpoints/*.pt")):
        nome = os.path.basename(p)[:-3]
        if any(e in nome for e in ESCLUSI): continue
        try:
            ck = load_checkpoint(nome, map_location=DEVICE)
            m = build_ijepa("vit_small").to(DEVICE)
            m.load_state_dict(ck["model"]); m.eval()
        except Exception as e:
            print(f"  salto {nome}: {type(e).__name__}"); continue

        t1 = time.perf_counter()
        sp_w = spostamento(ck["model"], sd0)
        c = cka(F0, feats(m))
        dw = np.mean([train_ssl.sonda_downstream(m, recs, sp, seed=s)[0] for s in SEEDS])
        del m; torch.cuda.empty_cache()
        freno.pausa(t1)

        righe.append((nome.replace("ijepa_vit_small", "").strip("_") or "base",
                      ck["epoch"], sp_w, c, dw))
        print(f"  {righe[-1][0]:16s} ep{ck['epoch']:4d}  spostamento {sp_w:6.3f}  "
              f"CKA {c:.3f}  downstream {dw:.4f} ({dw-d0:+.4f})", flush=True)

    print("\n" + "=" * 84)
    print(f"{'checkpoint':16s} {'ep':>5s} {'spostam.':>9s} {'CKA':>7s} "
          f"{'downstream':>11s} {'vs casuale':>11s}")
    print("-" * 84)
    for n, e, s, c, d in sorted(righe, key=lambda r: r[2]):
        print(f"{n:16s} {e:5d} {s:9.3f} {c:7.3f} {d:11.4f} {d-d0:+11.4f}")

    S = np.array([r[2] for r in righe]); C = np.array([r[3] for r in righe])
    D = np.array([r[4] for r in righe])
    for nome, x in (("spostamento dei pesi", S), ("CKA (1 = come il casuale)", C)):
        r = np.corrcoef(x, D)[0, 1]
        n = len(x)
        t = r * math.sqrt((n - 2) / max(1 - r**2, 1e-9))
        print(f"\ncorrelazione {nome:26s} vs downstream: r = {r:+.3f}  "
              f"({n} punti, t = {t:+.1f})")
    print("\nr negativo su 'spostamento' = allontanarsi dal casuale PEGGIORA.")
