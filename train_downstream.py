"""
Train (stadio 2) - caching dei latenti + testa di classificazione PAI.

Sezione "Train" della struttura richiesta dal corso.

IL PUNTO CHIAVE DI QUESTO FILE: si estraggono i latenti dall'encoder
congelato UNA VOLTA e si salvano su disco. Da quel momento ogni esperimento
sullo sbilanciamento - novita', cinque baseline, sweep, cinque seed -
gira in secondi, anche su CPU. L'ablation dell'obiettivo 4 diventa
praticamente gratuito, ed e' la ragione per cui la seconda meta' del progetto
e' molto piu' tranquilla della prima.

Uso:
    python train_downstream.py --cache                  # una volta sola
    python train_downstream.py --method balanced_tokens --head ordinal
    python train_downstream.py --grid                   # tutti i metodi x teste x seed
"""

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import LesionCropDataset, load_splits, make_loader, parse_annotations
from globals import (
    CACHE_DIR, DEFAULT_VARIANT, DEVICE, HEAD_BATCH_SIZE, HEAD_EPOCHS, HEAD_LR,
    HEAD_TYPES, HEAD_WEIGHT_DECAY, IMBALANCE_METHODS, N_SEEDS, NUM_CLASSES,
    OUT_DIR,
)
from imbalance import (
    balanced_sampler_weights, balanced_token_sampling, class_counts,
    compute_loss, latent_smote_tokens,
)
from network import (
    FrozenImageNetEncoder, LesionClassifier, bbox_to_token_mask, build_ijepa,
)
from utils import load_checkpoint, save_json, set_seed


# ==========================================================================
# 1. Caching dei latenti - si fa una volta
# ==========================================================================
@torch.no_grad()
def cache_latents(variant=DEFAULT_VARIANT, batch_size=64, arm="ijepa"):
    """
    Estrae e salva i token dell'encoder congelato per tutte le lesioni.

    `arm` seleziona il braccio di confronto (sez.9 dell'analisi):
      'ijepa'    -> il vostro encoder pre-addestrato in-domain
      'imagenet' -> ViT/ResNet pre-addestrato su ImageNet  [DA IMPLEMENTARE]
      'random'   -> pesi casuali: il pavimento assoluto

    Il braccio 'imagenet' non e' opzionale: e' il confronto che fa o rompe
    la storia del progetto, ed e' il piu' economico dei tre.
    """
    records = parse_annotations(verbose=False)
    splits = load_splits()

    if arm == "ijepa":
        ckpt = load_checkpoint(f"ijepa_{variant}", map_location=DEVICE)
        if ckpt is None:
            raise FileNotFoundError("Nessun checkpoint SSL. Lanciate train_ssl.py")
        model = build_ijepa(ckpt.get("variant", variant)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
    elif arm == "random":
        model = build_ijepa(variant).to(DEVICE)
    elif arm == "imagenet":
        # Braccio critico della sez.9: ViT-B/16 ImageNet congelato, stessa
        # griglia 14x14 del nostro ViT, quindi il confronto e' alla pari.
        model = FrozenImageNetEncoder().to(DEVICE)
    else:
        raise ValueError(arm)

    model.eval()
    out = {}

    for split, ids in splits.items():
        ds = LesionCropDataset(records, ids)
        loader = make_loader(ds, batch_size=batch_size)
        toks, masks, labels = [], [], []

        for batch in loader:
            t = model.encode(batch["image"].to(DEVICE))
            m = bbox_to_token_mask(batch["bbox"].to(DEVICE), model.grid)
            toks.append(t.half().cpu())
            masks.append(m.cpu())
            labels.append(batch["label"])

        out[split] = {
            "tokens": torch.cat(toks),
            "mask": torch.cat(masks),
            "labels": torch.cat(labels),
        }
        c = class_counts(out[split]["labels"])
        print(f"  {split:6s}: {len(ds):5d} lesioni  token={tuple(out[split]['tokens'].shape)}  "
              f"PAI3/4/5 = {c.int().tolist()}")

    path = os.path.join(CACHE_DIR, f"latents_{arm}_{variant}.pt")
    torch.save({"data": out, "embed_dim": model.embed_dim, "grid": model.grid}, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nLatenti salvati: {path} ({size_mb:.0f} MB)")
    print("Da qui in poi ogni esperimento sullo sbilanciamento gira in secondi.")
    return path


def load_latents(variant=DEFAULT_VARIANT, arm="ijepa"):
    path = os.path.join(CACHE_DIR, f"latents_{arm}_{variant}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} mancante. Lanciate --cache")
    return torch.load(path, map_location="cpu", weights_only=False)


# ==========================================================================
# 2. Training della testa sui latenti cachati
# ==========================================================================
def train_head(cached, method="none", head_type="flat", seed=0,
               epochs=HEAD_EPOCHS, verbose=False, bts_alpha=0.5):
    """
    Addestra attention pooling + testa sui latenti congelati.

    Gira in secondi: e' il motivo per cui potete permettervi N_SEEDS seed e
    intervalli di confidenza. Con sbilanciamento 7:1 i margini tra i metodi
    sono stretti e un singolo run non distingue niente.
    """
    set_seed(seed)
    data, dim, grid = cached["data"], cached["embed_dim"], cached["grid"]

    tr, va = data["train"], data["val"]
    train_labels = tr["labels"]

    clf = LesionClassifier(dim, grid, head_type).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)

    n = len(train_labels)
    if method == "oversample":
        w = balanced_sampler_weights(train_labels).double()
        sampler = torch.utils.data.WeightedRandomSampler(w, n, replacement=True)
        order_fn = lambda: torch.tensor(list(sampler))
    else:
        order_fn = lambda: torch.randperm(n)

    # SMOTE latente: si sintetizza UNA VOLTA prima del ciclo, e solo dal
    # train split. Interpolare campioni di validation nel train falsifica
    # tutto, ed e' un errore che non si vede nelle metriche.
    tr_tokens, tr_mask, tr_labels_ep = tr["tokens"], tr["mask"], train_labels
    if method == "latent_smote":
        tr_tokens, tr_mask, tr_labels_ep = latent_smote_tokens(
            tr["tokens"], tr["mask"], train_labels, seed=seed
        )
        n = len(tr_labels_ep)
        order_fn = lambda: torch.randperm(n)

    # I token dello split stanno sulla GPU UNA VOLTA SOLA, non batch per
    # batch. La griglia dell'obiettivo 4 sono 6 metodi x 2 teste x N_SEEDS =
    # decine di addestramenti da migliaia di step ciascuno, e ricopiare ogni
    # batch domina il tempo totale. Con ~1.4 GB per lo split piu' grosso ci
    # sta; se non ci sta si continua dalla CPU senza cambiare i risultati.
    try:
        tr_tokens = tr_tokens.to(DEVICE)
        tr_mask = tr_mask.to(DEVICE)
        tr_labels_ep = tr_labels_ep.to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # La novita' invece si applica PER BATCH: le viste devono cambiare a ogni
    # epoca, altrimenti sono duplicati e non aggiungono informazione.
    bts_counts = class_counts(train_labels).to(tr_labels_ep.device)
    bts_gen = torch.Generator(device=tr_tokens.device).manual_seed(seed)

    best = {"val_f1": -1.0}
    for epoch in range(epochs):
        clf.train()
        order = order_fn().to(tr_tokens.device)
        for i in range(0, n, HEAD_BATCH_SIZE):
            idx = order[i:i + HEAD_BATCH_SIZE]
            tok = tr_tokens[idx].float()
            msk = tr_mask[idx]
            y = tr_labels_ep[idx]

            if method == "balanced_tokens":
                tok, msk, y = balanced_token_sampling(
                    tok, msk, y, bts_counts, generator=bts_gen, alpha=bts_alpha
                )

            tok, msk, y = tok.to(DEVICE), msk.to(DEVICE), y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits, _, _ = clf(tok, token_mask=msk)
            loss = compute_loss(logits, y, method, head_type, train_labels)
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            from evaluation import evaluate_split
            m = evaluate_split(clf, va, head_type)
            if m["macro_f1"] > best["val_f1"]:
                best = {"val_f1": m["macro_f1"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in clf.state_dict().items()}}
            if verbose:
                print(f"    ep{epoch:03d} loss={loss.item():.4f} "
                      f"val_macroF1={m['macro_f1']:.4f}")

    if "state" in best:
        clf.load_state_dict(best["state"])
    return clf, best


def run_grid(variant=DEFAULT_VARIANT, arm="ijepa", methods=None, heads=None,
             seeds=None):
    """
    Griglia completa: metodo x tipo di testa x seed.

    Copre l'obiettivo 4 (ablation) e produce gli intervalli di confidenza
    senza cui, con 7:1 di sbilanciamento, i confronti non significano nulla.
    """
    from evaluation import evaluate_split

    cached = load_latents(variant, arm)
    methods = methods or IMBALANCE_METHODS
    heads = heads or HEAD_TYPES
    seeds = seeds or list(range(N_SEEDS))

    rows = []
    for method, head in itertools.product(methods, heads):
        per_seed = []
        for s in seeds:
            try:
                clf, _ = train_head(cached, method, head, seed=s)
                per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
            except NotImplementedError as exc:
                print(f"  salto {method}/{head}: {exc}")
                per_seed = []
                break
        if not per_seed:
            continue

        agg = {"method": method, "head": head, "n_seeds": len(per_seed)}
        # Le F1 per classe servono per il criterio operativo: si accetta un
        # metodo se alza PAI 5 SENZA erodere PAI 3 e 4. Con la sola macro-F1
        # un guadagno sulla minoritaria pagato dalle altre due sembrerebbe
        # un miglioramento.
        for k in ("macro_f1", "balanced_acc", "recall_pai5", "pr_auc_pai5",
                  "quadratic_kappa", "f1_pai3", "f1_pai4", "f1_pai5",
                  "precision_pai5"):
            vals = [m[k] for m in per_seed]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        rows.append(agg)
        print(f"  {method:16s} {head:8s} macroF1={agg['macro_f1_mean']:.4f}"
              f"+-{agg['macro_f1_std']:.4f}  F1(3/4/5)="
              f"{agg['f1_pai3_mean']:.3f}/{agg['f1_pai4_mean']:.3f}/{agg['f1_pai5_mean']:.3f}"
              f"  recall5={agg['recall_pai5_mean']:.4f}"
              f"  prec5={agg['precision_pai5_mean']:.3f}"
              f"  kappa={agg['quadratic_kappa_mean']:.4f}")

    if rows:
        path = os.path.join(OUT_DIR, f"results_{arm}_{variant}.json")
        save_json(rows, path)
        print(f"\nRisultati in {path}")
    return rows


def sweep_alpha(variant=DEFAULT_VARIANT, arm="ijepa", alphas=None, heads=None,
                seeds=None):
    """
    Ablation su alpha della novita' - richiesto dall'obiettivo 4.

    alpha regola quante viste riceve ogni classe: n_c = ceil((max/n_c)^alpha).
    Con lo sbilanciamento reale (3017/1229/473 nel train) alpha=0.5 da'
    [1,2,3] viste, cioe' PAI 5 resta sotto-rappresentata di circa un fattore
    2; alpha=1.0 da' [1,2,6], cioe' il pareggio effettivo. E' la ragione per
    cui la novita' perdeva contro `oversample`, che invece pareggia davvero.

    Il confronto interessante e' proprio con `oversample`: stesso numero di
    istanze per classe, ma li' sono duplicati identici, qui sono
    sottoinsiemi di token diversi della stessa lesione.
    """
    from evaluation import evaluate_split

    cached = load_latents(variant, arm)
    alphas = alphas or [0.0, 0.25, 0.5, 0.75, 1.0]
    heads = heads or HEAD_TYPES
    seeds = seeds or list(range(N_SEEDS))

    counts = class_counts(cached["data"]["train"]["labels"])
    print(f"conteggi train PAI3/4/5: {counts.int().tolist()}")
    from imbalance import n_views_per_class
    for al in alphas:
        print(f"  alpha={al:.2f} -> viste per classe {n_views_per_class(counts, al).tolist()}")

    rows = []
    for al, head in itertools.product(alphas, heads):
        per_seed = []
        for s in seeds:
            clf, _ = train_head(cached, "balanced_tokens", head, seed=s, bts_alpha=al)
            per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
        agg = {"method": f"balanced_tokens", "alpha": al, "head": head,
               "n_seeds": len(per_seed)}
        for k in ("macro_f1", "balanced_acc", "recall_pai5", "pr_auc_pai5",
                  "quadratic_kappa", "f1_pai3", "f1_pai4", "f1_pai5",
                  "precision_pai5"):
            vals = [m[k] for m in per_seed]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        rows.append(agg)
        print(f"  alpha={al:.2f} {head:8s} macroF1={agg['macro_f1_mean']:.4f}"
              f"+-{agg['macro_f1_std']:.4f}  F1(3/4/5)="
              f"{agg['f1_pai3_mean']:.3f}/{agg['f1_pai4_mean']:.3f}/{agg['f1_pai5_mean']:.3f}"
              f"  recall5={agg['recall_pai5_mean']:.4f}+-{agg['recall_pai5_std']:.4f}"
              f"  prec5={agg['precision_pai5_mean']:.3f}")

    path = os.path.join(OUT_DIR, f"sweep_alpha_{arm}_{variant}.json")
    save_json(rows, path)
    print(f"\nRisultati in {path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--sweep-alpha", action="store_true")
    ap.add_argument("--arm", default="ijepa", choices=["ijepa", "imagenet", "random"])
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--method", default="none", choices=IMBALANCE_METHODS)
    ap.add_argument("--head", default="flat", choices=HEAD_TYPES)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()

    if a.cache:
        cache_latents(a.variant, arm=a.arm)
    elif a.sweep_alpha:
        sweep_alpha(a.variant, a.arm)
    elif a.grid:
        run_grid(a.variant, a.arm)
    else:
        from evaluation import evaluate_split, print_report
        cached = load_latents(a.variant, a.arm)
        clf, best = train_head(cached, a.method, a.head, verbose=True)
        print_report(evaluate_split(clf, cached["data"]["test"], a.head),
                     f"{a.method} / {a.head}")
