"""
Il protocollo definitivo, in tre stadi con la selezione neutralizzata.

PERCHE' TRE STADI E NON UNA GRIGLIA SOLA. Provando 6 teste per 3 pooling e
riportando la migliore, il numero ottenuto e' il massimo di 18 estrazioni
rumorose - la stessa distorsione contro cui lo sweep di alpha ha usato i
seed disgiunti. Un disegno serio la neutralizza prima, non la giustifica
dopo.

--------------------------------------------------------------------------
STADIO 0 - il protocollo si fissa A PRIORI

K = 16, e la ragione non guarda i risultati: e' la MEDIANA DEI TOKEN DI
PAI 3, la classe con le lesioni piu' piccole. Sotto quel valore si
butterebbe informazione proprio dove ce n'e' meno.

La robustezza si mostra col sweep di exp_fixedk.py su K = 16, 36, 64, che
va riportato come ANALISI DI SENSIBILITA' e non come selezione: se il
vantaggio regge a tutti e tre, K non e' un grado di liberta' sfruttato.

I K token piu' vicini al centro della bbox. La bbox resta usata per
LOCALIZZARE la lesione - che e' cio' che il brief chiede, "usare le
bounding box per estrarre i vettori latenti corrispondenti alle aree
lesionate" - senza che il loro NUMERO comunichi la dimensione. Nel
protocollo naif quel numero e' 16 / 36 / 64 per PAI 3 / 4 / 5, e la
maschera da sola, one-hot e senza un solo pixel, da' macro-F1 0.7708:
piu' del vettore completo dell'encoder casuale.

--------------------------------------------------------------------------
STADIO 1 - selezione, su VALIDATION e sull'encoder CASUALE, seed 0-2

Sei teste col pooling attuale, poi tre pooling con la testa vincente.

LA REGOLA CHE RENDE IL DISEGNO SOLIDO: si sceglie la testa che massimizza
la BASELINE, non la nostra. Scegliere la testa migliore per I-JEPA
significherebbe tarare il protocollo sul braccio che si vuole far vincere;
sceglierla sul casuale da' alla baseline la miglior testa possibile, e
qualunque vantaggio di I-JEPA che sopravvive e' CONSERVATIVO.

--------------------------------------------------------------------------
STADIO 2 - conferma, su TEST e con seed DISGIUNTI (10-14)

Protocollo, testa e pooling congelati. Si misura una volta sola, su
entrambi gli encoder, in entrambi i protocolli: K fisso e naif. Il
contrasto fra le due colonne E' la scoperta.

I seed disgiunti da quelli della selezione ELIMINANO la distorsione invece
di limitarsi a dichiararla.

--------------------------------------------------------------------------
STADIO 3 - la novita' sotto il protocollo nuovo, entrambi gli encoder

Sotto K fisso `balanced_token_sampling` cambia natura, e in meglio: non
campiona piu' sottoinsiemi di un insieme di dimensione variabile, ma
sottoinsiemi di 16 token uguali per tutti. Il numero di VISTE resta
l'unica cosa che varia fra le classi, che e' esattamente cio' che la
novita' rivendica.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_protocollo.py
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from exp_fixedk import con_maschera
from globals import IMBALANCE_METHODS, OUT_DIR, POOL_TYPES
from train_downstream import load_latents, train_head
from utils import Freno, save_json

TESTE = ["flat", "norm", "mlp", "ordinal", "norm_ord", "mlp_ord"]
CHIAVI = ("macro_f1", "pr_auc_pai5", "recall_pai5", "precision_pai5",
          "f1_pai5", "quadratic_kappa")


def misura(cached, metodo, head, pool, seeds, split, freno):
    per_seed = []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, metodo, head, seed=s, pool_type=pool)
        per_seed.append(evaluate_split(clf, cached["data"][split], head))
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return {k: (float(np.mean([m[k] for m in per_seed])),
                float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}


def scarto(a, b, n):
    d = a[0] - b[0]
    se = math.sqrt(a[1] ** 2 + b[1] ** 2) / math.sqrt(n)
    return d, (d / se if se > 0 else float("nan"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--casuale", default="_casuale")
    ap.add_argument("--ijepa", default="_geo_completa")
    ap.add_argument("--K", type=int, default=16,
                    help="mediana dei token di PAI 3: scelto a priori")
    ap.add_argument("--seeds-sel", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds-test", type=int, nargs="+",
                    default=[10, 11, 12, 13, 14])
    ap.add_argument("--carico", type=int, default=100)
    a = ap.parse_args()

    prot_K = f"P3_K{a.K}"
    freno = Freno(a.carico)
    print(f"[freno] {freno}")
    print(f"protocollo {prot_K} (K = mediana dei token di PAI 3, scelto a priori)")
    print(f"selezione: validation, encoder CASUALE, seed {a.seeds_sel}")
    print(f"conferma : test, seed {a.seeds_test} DISGIUNTI\n", flush=True)

    percorso = os.path.join(OUT_DIR, f"protocollo_{a.variant}_K{a.K}.json")
    F = {"K": a.K, "seeds_sel": a.seeds_sel, "seeds_test": a.seeds_test,
         "stadio1_teste": {}, "stadio1_pooling": {},
         "stadio2": {}, "stadio3": {}}
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            v = json.load(f)
        if v.get("seeds_sel") == a.seeds_sel and v.get("K") == a.K:
            F = v
            print("Riprendo da quanto gia' su disco\n")

    lat = {}
    def prendi(tag, prot):
        if (tag, prot) not in lat:
            c = load_latents(a.variant, layers=a.layers, tag=tag)
            lat[(tag, prot)] = con_maschera(c, prot)
            del c
        return lat[(tag, prot)]

    # ============ STADIO 1 ============
    print("=" * 78)
    print("STADIO 1 - sei teste, encoder CASUALE, VALIDATION")
    print("=" * 78)
    print(f"{'testa':12s} {'macro-F1':>18s} {'PR-AUC su PAI 5':>18s}")
    print("-" * 52)
    cas_K = prendi(a.casuale, prot_K)
    for h in TESTE:
        if h not in F["stadio1_teste"]:
            F["stadio1_teste"][h] = misura(cas_K, "none", h, "attn",
                                           a.seeds_sel, "val", freno)
            save_json(F, percorso)
        r = F["stadio1_teste"][h]
        print(f"{h:12s} {r['macro_f1'][0]:9.4f}+-{r['macro_f1'][1]:.4f} "
              f"{r['pr_auc_pai5'][0]:9.4f}+-{r['pr_auc_pai5'][1]:.4f}", flush=True)
    testa = max(F["stadio1_teste"], key=lambda h: F["stadio1_teste"][h]["macro_f1"][0])
    print(f"\n-> testa scelta (massimizza la BASELINE, quindi conservativa "
          f"verso di noi): {testa}")

    print(f"\n{'pooling':12s} {'macro-F1':>18s} {'PR-AUC su PAI 5':>18s}")
    print("-" * 52)
    for p in POOL_TYPES:
        if p not in F["stadio1_pooling"]:
            F["stadio1_pooling"][p] = (F["stadio1_teste"][testa] if p == "attn"
                                       else misura(cas_K, "none", testa, p,
                                                   a.seeds_sel, "val", freno))
            save_json(F, percorso)
        r = F["stadio1_pooling"][p]
        print(f"{p:12s} {r['macro_f1'][0]:9.4f}+-{r['macro_f1'][1]:.4f} "
              f"{r['pr_auc_pai5'][0]:9.4f}+-{r['pr_auc_pai5'][1]:.4f}", flush=True)
    pool = max(F["stadio1_pooling"], key=lambda p: F["stadio1_pooling"][p]["macro_f1"][0])
    F["scelta"] = {"testa": testa, "pooling": pool}
    save_json(F, percorso)
    print(f"\n-> CONGELATO: testa {testa}, pooling {pool}, protocollo {prot_K}")

    # ============ STADIO 2 ============
    print(f"\n{'=' * 78}")
    print(f"STADIO 2 - conferma sul TEST, seed {a.seeds_test} disgiunti")
    print("=" * 78)
    print(f"{'protocollo':16s} {'encoder':12s} {'macro-F1':>18s} {'PR-AUC su PAI 5':>18s}")
    print("-" * 70)
    for prot in (prot_K, "P1_bbox"):
        for nome, tag in (("casuale", a.casuale), ("I-JEPA", a.ijepa)):
            ch = f"{prot}|{tag}"
            if ch not in F["stadio2"]:
                F["stadio2"][ch] = misura(prendi(tag, prot), "none", testa,
                                          pool, a.seeds_test, "test", freno)
                save_json(F, percorso)
            r = F["stadio2"][ch]
            print(f"{prot:16s} {nome:12s} {r['macro_f1'][0]:9.4f}+-{r['macro_f1'][1]:.4f} "
                  f"{r['pr_auc_pai5'][0]:9.4f}+-{r['pr_auc_pai5'][1]:.4f}", flush=True)
        d, z = scarto(F["stadio2"][f"{prot}|{a.ijepa}"]["macro_f1"],
                      F["stadio2"][f"{prot}|{a.casuale}"]["macro_f1"], len(a.seeds_test))
        dr, zr = scarto(F["stadio2"][f"{prot}|{a.ijepa}"]["pr_auc_pai5"],
                        F["stadio2"][f"{prot}|{a.casuale}"]["pr_auc_pai5"], len(a.seeds_test))
        print(f"{'':16s} {'I-JEPA - cas.':12s} {d:+9.4f} (z={z:+5.2f}) "
              f"{dr:+9.4f} (z={zr:+5.2f})\n", flush=True)

    # ============ STADIO 3 ============
    print("=" * 78)
    print(f"STADIO 3 - i metodi di sbilanciamento sotto {prot_K}, TEST")
    print("=" * 78)
    print(f"{'metodo':18s} {'casuale':>26s} {'I-JEPA':>26s}")
    print("-" * 74)
    for m in IMBALANCE_METHODS:
        riga = {}
        for nome, tag in (("casuale", a.casuale), ("I-JEPA", a.ijepa)):
            ch = f"{m}|{tag}"
            if ch not in F["stadio3"]:
                F["stadio3"][ch] = misura(prendi(tag, prot_K), m, testa, pool,
                                          a.seeds_test, "test", freno)
                save_json(F, percorso)
            riga[nome] = F["stadio3"][ch]
        print(f"{m:18s} " + "  ".join(
            f"{riga[n]['macro_f1'][0]:.4f}+-{riga[n]['macro_f1'][1]:.4f} "
            f"({riga[n]['pr_auc_pai5'][0]:.4f})" for n in ("casuale", "I-JEPA")),
            flush=True)

    print(f"\nLA NOVITA' contro `none`, sotto {prot_K}:")
    for nome, tag in (("casuale", a.casuale), ("I-JEPA", a.ijepa)):
        for k in ("macro_f1", "pr_auc_pai5"):
            d, z = scarto(F["stadio3"][f"balanced_tokens|{tag}"][k],
                          F["stadio3"][f"none|{tag}"][k], len(a.seeds_test))
            print(f"  {nome:10s} {k:14s} {d:+.4f}   z = {z:+.2f}")

    save_json(F, percorso)
    print(f"\nRisultati in {percorso}")
