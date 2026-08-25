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
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import LesionCropDataset, load_splits, make_loader, parse_annotations
from globals import (
    CACHE_DIR, DEFAULT_VARIANT, DEVICE, HEAD_BATCH_SIZE, HEAD_EPOCHS, HEAD_LR,
    HEAD_TYPES, HEAD_WEIGHT_DECAY, IMBALANCE_METHODS, N_SEEDS, NUM_CLASSES,
    OUT_DIR, SEED,
)
from imbalance import (
    balanced_sampler_weights, balanced_token_sampling, class_counts,
    compute_loss,
)
from network import LesionClassifier, bbox_to_token_mask, build_ijepa
from utils import load_checkpoint, save_json, set_seed


# ==========================================================================
# 1. Caching dei latenti - si fa una volta
# ==========================================================================
@torch.no_grad()
def cache_latents(variant=DEFAULT_VARIANT, batch_size=64, layers=None,
                  ckpt_tag="", casuale=False, tag=""):
    """
    Estrae e salva i token dell'encoder congelato per tutte le lesioni.

    L'encoder e' quello pre-addestrato da train_ssl.py e resta CONGELATO:
    l'obiettivo 2 del brief chiede esplicitamente di valutare le
    rappresentazioni "frozen", quindi qui non si aggiorna nessun peso del
    backbone - si estraggono e basta.
    """
    # `layers` concatena piu' profondita' del ViT invece del solo ultimo
    # blocco. Misurato il 21 ago: le feature dell'ultimo blocco sono le piu'
    # COMPRESSE, e con una sonda lineare - che e' cio' che fa la testa - un
    # blocco intermedio rende molto di piu'. Il protocollo va tenuto IDENTICO
    # su tutte le configurazioni dell'ablation, altrimenti si confrontano i
    # protocolli di estrazione invece dei metodi.
    records = parse_annotations(verbose=False)
    splits = load_splits()

    # ckpt_tag sceglie QUALE run di pre-training usare: gli esperimenti
    # producono checkpoint distinti e il downstream deve puntare a quello
    # voluto, non al primo che c'e'.
    if casuale:
        # ENCODER CASUALE: stessa architettura, pesi non addestrati.
        # Non e' un "braccio di confronto" opzionale, e' il RIFERIMENTO
        # senza cui i numeri del pre-training non significano niente: dice
        # quanta della prestazione viene dall'addestramento e quanta e'
        # gia' data dall'architettura piu' le bbox. Il seme e' fisso, cosi'
        # il riferimento e' riproducibile e non cambia a ogni misura.
        set_seed(SEED)
        model = build_ijepa(variant).to(DEVICE)
        print(f"Encoder CASUALE (seme {SEED}), nessun peso addestrato")
    else:
        nome = f"ijepa_{variant}{('_' + ckpt_tag) if ckpt_tag else ''}"
        ckpt = load_checkpoint(nome, map_location=DEVICE)
        if ckpt is None:
            raise FileNotFoundError(f"Nessun checkpoint {nome}. Lanciate train_ssl.py")
        model = build_ijepa(ckpt.get("variant", variant)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"Encoder da {nome}, epoca {ckpt['epoch']}")

    model.eval()
    out = {}

    for split, ids in splits.items():
        ds = LesionCropDataset(records, ids)
        loader = make_loader(ds, batch_size=batch_size)
        toks, masks, labels, geoms = [], [], [], []

        for batch in loader:
            t = model.encode(batch["image"].to(DEVICE), return_layers=layers)
            m = bbox_to_token_mask(batch["bbox"].to(DEVICE), model.grid)
            toks.append(t.half().cpu())
            masks.append(m.cpu())
            labels.append(batch["label"])
            geoms.append(batch["geom"])

        out[split] = {
            "tokens": torch.cat(toks),
            "mask": torch.cat(masks),
            "labels": torch.cat(labels),
            "geom": torch.cat(geoms),
        }
        c = class_counts(out[split]["labels"])
        print(f"  {split:6s}: {len(ds):5d} lesioni  token={tuple(out[split]['tokens'].shape)}  "
              f"PAI3/4/5 = {c.int().tolist()}")

    dim = out["train"]["tokens"].shape[-1]
    suffisso = "" if layers is None else "_L" + "-".join(map(str, layers))
    path = os.path.join(CACHE_DIR, f"latents_{variant}{suffisso}{tag}.pt")
    torch.save({"data": out, "embed_dim": dim, "grid": model.grid,
                "layers": layers}, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nLatenti salvati: {path} ({size_mb:.0f} MB)")
    print("Da qui in poi ogni esperimento sullo sbilanciamento gira in secondi.")
    return path


def load_latents(variant=DEFAULT_VARIANT, layers=None, tag=""):
    suffisso = "" if layers is None else "_L" + "-".join(map(str, layers))
    path = os.path.join(CACHE_DIR, f"latents_{variant}{suffisso}{tag}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} mancante. Lanciate --cache")
    return torch.load(path, map_location="cpu", weights_only=False)


# ==========================================================================
# 2. Training della testa sui latenti cachati
# ==========================================================================
def train_head(cached, method="none", head_type="flat", seed=0,
               epochs=HEAD_EPOCHS, verbose=False, bts_alpha=0.5, use_geom=False):
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

    gdim = tr["geom"].shape[1] if (use_geom and "geom" in tr) else 0
    clf = LesionClassifier(dim, grid, head_type, geom_dim=gdim).to(DEVICE)
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
    tr_geom = tr.get("geom")

    # I token dello split stanno sulla GPU UNA VOLTA SOLA, non batch per
    # batch. La griglia dell'obiettivo 4 sono 6 metodi x 2 teste x N_SEEDS =
    # decine di addestramenti da migliaia di step ciascuno, e ricopiare ogni
    # batch domina il tempo totale. Con ~1.4 GB per lo split piu' grosso ci
    # sta; se non ci sta si continua dalla CPU senza cambiare i risultati.
    try:
        tr_tokens = tr_tokens.to(DEVICE)
        tr_mask = tr_mask.to(DEVICE)
        tr_labels_ep = tr_labels_ep.to(DEVICE)
        if gdim:
            tr_geom = tr_geom.to(DEVICE)
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
            gm = tr_geom[idx] if gdim else None

            if method == "balanced_tokens":
                # La geometria segue le viste: ogni vista e' la STESSA
                # lesione, quindi eredita la sua bbox.
                n0 = tok.shape[0]
                tok, msk, y = balanced_token_sampling(
                    tok, msk, y, bts_counts, generator=bts_gen, alpha=bts_alpha
                )
                if gdim:
                    rip = tok.shape[0] // n0 if tok.shape[0] % n0 == 0 else None
                    gm = (gm.repeat_interleave(rip, 0) if rip
                          else gm[:tok.shape[0]])

            tok, msk, y = tok.to(DEVICE), msk.to(DEVICE), y.to(DEVICE)
            if gdim:
                gm = gm.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits, _, _ = clf(tok, token_mask=msk, geom=gm)
            loss = compute_loss(logits, y, method, head_type, train_labels)
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            from evaluation import evaluate_split
            m = evaluate_split(clf, va, head_type, use_geom=bool(gdim))
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


def run_grid(variant=DEFAULT_VARIANT, methods=None, heads=None,
             seeds=None, layers=None, tag=""):
    """
    Griglia completa: metodo x tipo di testa x seed.

    Copre l'obiettivo 4 (ablation) e produce gli intervalli di confidenza
    senza cui, con 7:1 di sbilanciamento, i confronti non significano nulla.
    """
    from evaluation import evaluate_split

    cached = load_latents(variant, layers=layers, tag=tag)
    methods = methods or IMBALANCE_METHODS
    heads = heads or HEAD_TYPES
    seeds = seeds or list(range(N_SEEDS))

    # RIPARTENZA. La griglia dura ~40 minuti e questa macchina si e' spenta
    # da sola sette volte in cinque giorni. Senza ripartenza un crash a
    # meta' butta via tutto: e' gia' successo, 30 minuti persi perche' i
    # risultati stavano solo in memoria.
    suff = "" if layers is None else "_L" + "-".join(map(str, layers))
    percorso = os.path.join(OUT_DIR, f"results_{variant}{suff}{tag}.json")
    rows = []
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            rows = json.load(f)
        if rows:
            print(f"Riprendo: {len(rows)} righe gia' su disco in "
                  f"{os.path.basename(percorso)}")
    fatte = {(r["method"], r["head"]) for r in rows}

    for method, head in itertools.product(methods, heads):
        if (method, head) in fatte:
            print(f"  {method:16s} {head:8s} gia' fatta, salto")
            continue
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
        # Salvataggio incrementale: il costo di un crash scende da "tutta la
        # griglia" a "la riga in corso".
        save_json(rows, percorso)
        print(f"  {method:16s} {head:8s} macroF1={agg['macro_f1_mean']:.4f}"
              f"+-{agg['macro_f1_std']:.4f}  F1(3/4/5)="
              f"{agg['f1_pai3_mean']:.3f}/{agg['f1_pai4_mean']:.3f}/{agg['f1_pai5_mean']:.3f}"
              f"  recall5={agg['recall_pai5_mean']:.4f}"
              f"  prec5={agg['precision_pai5_mean']:.3f}"
              f"  kappa={agg['quadratic_kappa_mean']:.4f}")

    if rows:
        path = percorso
        save_json(rows, path)
        print(f"\nRisultati in {path}")
    return rows


def sweep_alpha(variant=DEFAULT_VARIANT, alphas=None, heads=None,
                seeds=None, layers=None):
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

    cached = load_latents(variant, layers=layers, tag=tag)
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

    suff = "" if layers is None else "_L" + "-".join(map(str, layers))
    path = os.path.join(OUT_DIR, f"sweep_alpha_{variant}{suff}.json")
    save_json(rows, path)
    print(f"\nRisultati in {path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--sweep-alpha", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="blocchi da concatenare, es. --layers 2 7 11")
    ap.add_argument("--random", action="store_true",
                    help="encoder con pesi CASUALI: il riferimento senza cui "
                         "i numeri del pre-training non significano nulla")
    ap.add_argument("--tag", default="",
                    help="suffisso dei file, per tenere i bracci separati")
    ap.add_argument("--ckpt-tag", default="",
                    help="quale run SSL usare per il braccio ijepa (es. pred48)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--method", default="none", choices=IMBALANCE_METHODS)
    ap.add_argument("--head", default="flat", choices=HEAD_TYPES)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()

    if a.cache:
        cache_latents(a.variant, layers=a.layers, ckpt_tag=a.ckpt_tag,
                      casuale=a.random, tag=a.tag)
    elif a.sweep_alpha:
        sweep_alpha(a.variant, layers=a.layers)
    elif a.grid:
        run_grid(a.variant, layers=a.layers, tag=a.tag)
    else:
        from evaluation import evaluate_split, print_report
        cached = load_latents(a.variant, layers=a.layers)
        clf, best = train_head(cached, a.method, a.head, verbose=True)
        print_report(evaluate_split(clf, cached["data"]["test"], a.head),
                     f"{a.method} / {a.head}")
