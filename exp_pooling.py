"""
CONFRONTO — PR-AUC PAI 5 fra tipi di attention pooling.

Pooling: gated e top-k contro l'attention a query fissa - obiettivi 2 e 4.

PERCHE' IL POOLING E NON LA TESTA. L'encoder e' congelato per vincolo del
brief, quindi tutta la capacita' addestrabile sta in due pezzi: come i
token della bbox diventano un vettore (pooling) e come quel vettore diventa
tre logit (testa). `exp_testa.py` misura il secondo. Questo misura il
primo, e c'e' una ragione per aspettarsi che sia li' il collo di bottiglia:
la testa vede gia' un vettore aggregato, e se l'aggregazione ha buttato via
l'informazione nessuna testa la recupera.

LE TRE IPOTESI, che sono in disaccordo fra loro. E' il motivo per cui vale
la pena misurarle: qualunque cosa vinca, si impara qualcosa.

  attn   (attuale) query APPRESA MA FISSA, softmax denso su tutti i token
         della bbox. Il peso di un token dipende da lui solo attraverso un
         prodotto scalare con un'unica direzione.

  gated  punteggio non lineare per token, stile MIL (Ilse et al. 2018):
             a_i ~ w^T ( tanh(V h_i) * sigmoid(U h_i) )
         Il prodotto fra i due rami esprime "questo token conta SE anche
         quest'altra caratteristica c'e'". Sul PAI e' la struttura del
         problema: una regione conta se e' insieme grande E scura. Con
         nascosto 128 sono 0.3M parametri contro i 5.3M dell'attention che
         sostituisce - e' piu' LEGGERO, come il brief chiede.

  topk   solo i k token piu' forti, il resto scartato. Le bbox mediane
         hanno 16 / 36 / 64 token per PAI 3 / 4 / 5, e su quelle grandi la
         radiotrasparenza occupa una frazione del rettangolo: il resto e'
         osso sano che il softmax denso media dentro comunque.

DISACCORDO ESPLICITO CON LA NOVITA'. `balanced_token_sampling` si difende
dicendo "non appoggiarti a pochi token, guarda sottoinsiemi diversi". Il
top-k dice l'opposto: "appoggiati solo ai migliori". Se il top-k vince,
l'argomento della novita' si indebolisce, e va saputo PRIMA della
presentazione invece che durante. Misurarlo e' quindi un atto di onesta',
non un'aggiunta opzionale.

SELEZIONE SU VALIDATION. Qui si sceglie un componente dell'architettura, e
sceglierlo guardando il test e' barare. Il test si riporta solo per la
configurazione scelta, e alla fine.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from globals import OUT_DIR, POOL_TYPES, TOP_K
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CARICO = 70
CHIAVI = ("macro_f1", "pr_auc_pai5", "recall_pai5", "precision_pai5",
          "f1_pai5", "quadratic_kappa")


def misura(cached, pool, method, head, seeds, freno, split="val"):
    per_seed = []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, method, head, seed=s, pool_type=pool)
        per_seed.append(evaluate_split(clf, cached["data"][split], head))
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return {k: (float(np.mean([m[k] for m in per_seed])),
                float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}


def riga(nome, m):
    return (f"  {nome:28s} {m['macro_f1'][0]:.4f}+-{m['macro_f1'][1]:.4f}"
            f"  {m['pr_auc_pai5'][0]:.4f}+-{m['pr_auc_pai5'][1]:.4f}"
            f"  {m['recall_pai5'][0]:.4f}  {m['precision_pai5'][0]:.3f}")


def intestazione():
    return (f"  {'pooling x metodo':28s} {'macro-F1':>13s}  {'PR-AUC5':>13s}"
            f"  {'rec5':>6s}  {'prec5':>5s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_casuale")
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--head", default="flat")
    ap.add_argument("--metodi", nargs="+", default=["none", "balanced_tokens"])
    ap.add_argument("--pool", nargs="+", default=POOL_TYPES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=CARICO)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)

    msk = cached["data"]["train"]["mask"]
    lab = cached["data"]["train"]["labels"]
    print(f"\nencoder{a.tag}, testa {a.head}, {len(a.seeds)} seed")
    print(f"token per bbox (mediana per classe): "
          + ", ".join(f"PAI{g}={int(msk[lab == c].sum(1).float().median())}"
                      for c, g in enumerate((3, 4, 5))))
    print(f"top-k usa k={TOP_K}: sotto la mediana di PAI 4 e 5, sopra quella "
          f"di PAI 3 - dove degenera nel softmax denso, che e' il controllo.")

    percorso = os.path.join(OUT_DIR, f"pooling_{a.variant}{a.tag}.json")
    fuori = {"tag": a.tag, "head": a.head, "seeds": a.seeds, "top_k": TOP_K,
             "val": {}, "test": {}}
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            vecchio = json.load(f)
        if vecchio.get("seeds") == a.seeds and vecchio.get("head") == a.head:
            fuori = vecchio
            print(f"Riprendo: {len(fuori['val'])} celle gia' su disco")

    print(f"\n{'=' * 84}\nSELEZIONE SU VALIDATION - il test non si guarda\n{'=' * 84}")
    print(intestazione())
    for pool, method in itertools.product(a.pool, a.metodi):
        chiave = f"{pool}|{method}"
        if chiave in fuori["val"]:
            print(f"  {chiave:28s} gia' fatta, salto")
            continue
        fuori["val"][chiave] = misura(cached, pool, method, a.head, a.seeds,
                                      freno, "val")
        print(riga(chiave, fuori["val"][chiave]), flush=True)
        save_json(fuori, percorso)

    # Si sceglie sulla macro-F1 di validation, che e' il criterio usato da
    # train_head per fermarsi: usarne uno diverso qui selezionerebbe una
    # configurazione su un criterio e la addestrerebbe su un altro.
    vincente = max(fuori["val"], key=lambda k: fuori["val"][k]["macro_f1"][0])
    pool_v, met_v = vincente.split("|")
    print(f"\nScelta su validation: pooling {pool_v}, metodo {met_v} "
          f"(macro-F1 {fuori['val'][vincente]['macro_f1'][0]:.4f})")

    # Sul TEST vanno solo la scelta e il suo riferimento: riportare tutta la
    # griglia di test dopo aver selezionato su validation reintrodurrebbe
    # dalla finestra il massimo su molte estrazioni che la selezione su
    # validation serviva a tenere fuori dalla porta.
    print(f"\n{'=' * 84}\nTEST - solo la scelta e il riferimento\n{'=' * 84}")
    print(intestazione())
    for chiave in dict.fromkeys([vincente, f"attn|{met_v}", "attn|none"]):
        p, m = chiave.split("|")
        if chiave not in fuori["test"]:
            fuori["test"][chiave] = misura(cached, p, m, a.head, a.seeds,
                                           freno, "test")
            save_json(fuori, percorso)
        print(riga(chiave, fuori["test"][chiave]), flush=True)

    n = len(a.seeds)
    if vincente != f"attn|{met_v}":
        x = fuori["test"][vincente]["macro_f1"]
        y = fuori["test"][f"attn|{met_v}"]["macro_f1"]
        se = float(np.sqrt(x[1] ** 2 + y[1] ** 2) / np.sqrt(n))
        print(f"\n{pool_v} - attn, a parita' di metodo, macro-F1 sul test: "
              f"{x[0] - y[0]:+.4f}  ({(x[0] - y[0]) / se:+.2f} err.std)")

    print(f"\nRisultati in {percorso}")
