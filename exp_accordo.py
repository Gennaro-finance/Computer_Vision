"""
I due encoder sbagliano sugli STESSI casi?

Il CKA dice che condividono solo il 50% della struttura, ma ottengono lo
stesso punteggio. Due esiti possibili, con conseguenze opposte:

  errori CORRELATI    -> sono funzionalmente lo stesso modello, e non c'e'
                         niente da guadagnare combinandoli
  errori DECORRELATI  -> sono due modelli diversi che per caso valgono
                         uguale, e un ensemble puo' superare entrambi

Si misura sul TEST, 5 seed, con le teste addestrate separatamente su
ciascun braccio. L'ensemble media le PROBABILITA', non le predizioni: le
predizioni sono discrete e mediarle butta via la confidenza.
"""
import time

import numpy as np
import torch

from evaluation import confusion_matrix, macro_f1, pr_auc
from train_downstream import load_latents, train_head
from utils import Freno, set_seed

CARICO = 80
SEEDS = [0, 1, 2, 3, 4]


@torch.no_grad()
def probabilita(clf, split, batch=256):
    """Probabilita' per classe su tutto lo split, dalla testa addestrata."""
    from globals import DEVICE
    clf.eval()
    tok, msk = split["tokens"], split["mask"]
    out = []
    for i in range(0, tok.shape[0], batch):
        t = tok[i:i+batch].to(DEVICE).float()
        m = msk[i:i+batch].to(DEVICE)
        logits, _, _ = clf(t, token_mask=m)
        out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    freno = Freno(CARICO)
    print(f"[freno] {freno}\n")

    P = {}
    for tag, nome in (("", "JEPA"), ("_casuale", "casuale")):
        t0 = time.time()
        cached = load_latents("vit_small", layers=[2, 7, 11], tag=tag)
        y = cached["data"]["test"]["labels"].numpy()
        print(f"{nome}: latenti caricati in {time.time()-t0:.0f}s")
        pp = []
        for s in SEEDS:
            t1 = time.perf_counter()
            clf, _ = train_head(cached, "none", "flat", seed=s)
            pp.append(probabilita(clf, cached["data"]["test"]))
            del clf
            torch.cuda.empty_cache()
            freno.pausa(t1)
        P[nome] = np.stack(pp)          # (seed, N, 3)
        del cached
        print(f"  {len(SEEDS)} teste addestrate")

    N = len(y)
    print(f"\n{N} lesioni di test\n")

    # --- accordo fra i due bracci, seed per seed appaiato
    acc_j, acc_c, acc_e, conc, err_comuni, pr_j, pr_c, pr_e = [], [], [], [], [], [], [], []
    for s in range(len(SEEDS)):
        pj, pc = P["JEPA"][s], P["casuale"][s]
        dj, dc = pj.argmax(1), pc.argmax(1)
        de = ((pj + pc) / 2).argmax(1)          # ensemble: media delle probabilita'
        acc_j.append(macro_f1(confusion_matrix(y, dj)))
        acc_c.append(macro_f1(confusion_matrix(y, dc)))
        acc_e.append(macro_f1(confusion_matrix(y, de)))
        pr_j.append(pr_auc(pj[:, 2], (y == 2).astype(int)))
        pr_c.append(pr_auc(pc[:, 2], (y == 2).astype(int)))
        pr_e.append(pr_auc(((pj + pc) / 2)[:, 2], (y == 2).astype(int)))
        conc.append((dj == dc).mean())
        ej, ec = dj != y, dc != y
        # quota di errori del JEPA che sono anche errori del casuale
        err_comuni.append((ej & ec).sum() / max(ej.sum(), 1))

    def r(v): return np.array(v)
    print(f"{'':22s} {'macro-F1':>18s} {'PR-AUC PAI5':>18s}")
    print("-" * 62)
    for nome, a, p in (("JEPA", acc_j, pr_j), ("casuale", acc_c, pr_c),
                       ("ENSEMBLE dei due", acc_e, pr_e)):
        print(f"{nome:22s} {r(a).mean():10.4f}+-{r(a).std():.4f} "
              f"{r(p).mean():10.4f}+-{r(p).std():.4f}")

    mig = max(r(acc_j).mean(), r(acc_c).mean())
    d = r(acc_e).mean() - mig
    se = float(np.sqrt(r(acc_e).var(ddof=1)/5 + max(r(acc_j).var(ddof=1), r(acc_c).var(ddof=1))/5))
    print(f"\nensemble vs il migliore dei due: {d:+.4f}  ({abs(d)/se:.1f} err.std)")

    print(f"\n--- accordo fra i due modelli ---")
    print(f"  predizioni identiche      : {r(conc).mean():.1%}")
    print(f"  errori del JEPA che sono  : {r(err_comuni).mean():.1%}")
    print(f"  anche errori del casuale")
    # quota attesa se gli errori fossero indipendenti
    tj = 1 - np.mean([( P['JEPA'][s].argmax(1)==y).mean() for s in range(5)])
    tc = 1 - np.mean([(P['casuale'][s].argmax(1)==y).mean() for s in range(5)])
    print(f"  attesa se INDIPENDENTI    : {tc:.1%}   (tasso d'errore del casuale)")
    print(f"\n  Se la quota misurata e' molto sopra l'attesa, gli errori sono")
    print(f"  CORRELATI: i due modelli sbagliano sugli stessi casi difficili.")
