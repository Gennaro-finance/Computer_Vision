"""
MISURAZIONE — riproducibilita' della stessa misura fra due esecuzioni.

Pavimento di rumore: quanto varia la STESSA misura fra due esecuzioni.

PERCHE' ESISTE QUESTO FILE. Il 27 agosto `none` sull'encoder casuale ha dato
0.8676 di PR-AUC su PAI 5. La griglia del 25 agosto, stesso encoder, stessa
testa, stessi seed 0-4, stesso test, aveva dato 0.8758. Sono 0.0082 di
scarto fra due misure che dovrebbero essere identiche - e 0.0082 e' piu'
grande della meta' delle differenze su cui questo progetto costruisce le
sue affermazioni.

Nella stessa esecuzione `class_weighted` ha dato 0.8706 contro lo 0.8706
della griglia: identico alla quarta cifra. Quindi non e' deriva numerica
uniforme, e' instabilita' che colpisce alcuni metodi e non altri.

DUE CAUSE, ENTRAMBE PRESENTI
  1. `set_seed` non imposta il determinismo di cuDNN. Nessuno lo imposta:
     le run su GPU non sono riproducibili bit per bit, perche' le riduzioni
     usano atomiche e l'ordine di somma cambia.
  2. `train_head` SELEZIONA l'epoca migliore sulla macro-F1 di validation,
     controllata ogni 10 epoche. Quando la curva di validation e' piatta
     vicino al massimo, una differenza numerica minima fa vincere un'altra
     epoca - e un'altra epoca da' un test diverso. E' un amplificatore di
     rumore piazzato dentro il protocollo di misura.

La seconda spiega perche' l'instabilita' e' per metodo: dipende da quanto
e' piatta la curva di validation di quel metodo.

CHE COSA MISURA QUESTO FILE, ESATTAMENTE
Non la dispersione FRA SEED - quella e' gia' nei risultati ed e' quella che
finisce negli errori standard. Misura la dispersione FRA ESECUZIONI della
media a 5 seed: cioe' l'incertezza sul numero che si riporta.

E' la grandezza giusta perche' ogni confronto del progetto ha la forma
"media a 5 seed del metodo A contro media a 5 seed del metodo B". Se due
esecuzioni della stessa media differiscono di 0.008, un confronto che vale
0.005 non significa niente, per quanto piccolo sia l'errore standard fra
seed.

Registra anche QUALE EPOCA e' stata scelta a ogni esecuzione. Se cambia, la
causa 2 e' confermata e non e' piu' un'ipotesi.

QUESTO FILE NON AGGIUSTA NIENTE, MISURA E BASTA. Le correzioni possibili -
fissare il determinismo di cuDNN, togliere la selezione sull'epoca,
valutare a ogni epoca - vanno decise DOPO aver visto quanto vale il
problema, e sapendo che cambiare protocollo dopo aver visto i risultati e'
esattamente il modo di fabbricare la conclusione che si preferisce.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_rumore.py
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from globals import OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CARICO = 100
METODI = ["none", "class_weighted", "balanced_tokens"]
RIPETIZIONI = 2
CHIAVI = ("pr_auc_pai5", "macro_f1", "f1_pai5", "recall_pai5")

# Misure gia' esistenti degli stessi metodi, dalla griglia e da
# exp_curve_pr.py: valgono come esecuzioni indipendenti in piu' e vanno
# nel conto della dispersione.
STORICHE = {
    "none":            {"griglia_25ago": 0.8758, "curve_pr_27ago": 0.8676},
    "class_weighted":  {"griglia_25ago": 0.8706, "curve_pr_27ago": 0.8706},
    "balanced_tokens": {"griglia_25ago": 0.8813},
}


def una_esecuzione(cached, method, seeds, head, freno):
    """Una misura completa a 5 seed: quello che si riporta come 'il numero'."""
    per_seed, epoche = [], []
    for s in seeds:
        t0 = time.perf_counter()
        clf, best = train_head(cached, method, head, seed=s)
        per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
        # L'epoca scelta e' la diagnosi: se cambia fra esecuzioni con lo
        # stesso seed, l'instabilita' viene dalla selezione e non dal
        # rumore numerico grezzo.
        epoche.append(int(best.get("epoch", -1)))
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return ({k: float(np.mean([m[k] for m in per_seed])) for k in CHIAVI},
            {k: float(np.std([m[k] for m in per_seed])) for k in CHIAVI},
            epoche,
            [float(m["pr_auc_pai5"]) for m in per_seed])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_casuale")
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--head", default="flat")
    ap.add_argument("--metodi", nargs="+", default=METODI)
    ap.add_argument("--ripetizioni", type=int, default=RIPETIZIONI)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=CARICO)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)

    print(f"\nencoder{a.tag}, testa {a.head}, seed {a.seeds}")
    print(f"{a.ripetizioni} esecuzioni indipendenti per metodo, "
          f"SEED IDENTICI a ogni esecuzione")
    print("Se i seed sono gli stessi, ogni differenza fra esecuzioni e'")
    print("rumore: non c'e' altra sorgente.\n")

    percorso = os.path.join(OUT_DIR, f"rumore_{a.variant}{a.tag}.json")
    fuori = {"tag": a.tag, "head": a.head, "seeds": a.seeds,
             "storiche": STORICHE, "esecuzioni": {}}
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            vecchio = json.load(f)
        if vecchio.get("seeds") == a.seeds and vecchio.get("head") == a.head:
            fuori = vecchio
            print(f"Riprendo: {sum(len(v) for v in fuori['esecuzioni'].values())} "
                  f"esecuzioni gia' su disco\n")

    for m in a.metodi:
        fuori["esecuzioni"].setdefault(m, {})
        for r in range(a.ripetizioni):
            chiave = f"run{r}"
            if chiave in fuori["esecuzioni"][m]:
                print(f"  {m:16s} {chiave}  gia' fatta, salto")
                continue
            medie, dev, epoche, per_seed = una_esecuzione(
                cached, m, a.seeds, a.head, freno)
            fuori["esecuzioni"][m][chiave] = {
                "medie": medie, "dev_fra_seed": dev,
                "epoca_scelta": epoche, "pr_auc_per_seed": per_seed}
            print(f"  {m:16s} {chiave}  PR-AUC5 {medie['pr_auc_pai5']:.4f} "
                  f"+-{dev['pr_auc_pai5']:.4f}   macroF1 {medie['macro_f1']:.4f}"
                  f"   epoche scelte {epoche}", flush=True)
            save_json(fuori, percorso)

    # ---------------------------------------------------------------- lettura
    print(f"\n{'=' * 78}\nPAVIMENTO DI RUMORE\n{'=' * 78}")
    print(f"{'metodo':18s} {'esecuzioni':>10s} {'min':>8s} {'max':>8s} "
          f"{'escursione':>11s} {'err.std fra seed':>17s}")
    print("-" * 78)
    riepilogo = {}
    for m in a.metodi:
        val = [v["medie"]["pr_auc_pai5"] for v in fuori["esecuzioni"][m].values()]
        val += list(STORICHE.get(m, {}).values())
        if len(val) < 2:
            continue
        # L'errore standard fra seed e' quello che i confronti usano oggi.
        # L'escursione fra esecuzioni e' quello che dovrebbero usare.
        es = np.mean([v["dev_fra_seed"]["pr_auc_pai5"]
                      for v in fuori["esecuzioni"][m].values()]) / np.sqrt(len(a.seeds))
        riepilogo[m] = {"n": len(val), "min": min(val), "max": max(val),
                        "escursione": max(val) - min(val), "err_std_seed": float(es)}
        print(f"{m:18s} {len(val):10d} {min(val):8.4f} {max(val):8.4f} "
              f"{max(val) - min(val):11.4f} {es:17.4f}")

    if riepilogo:
        peggio = max(riepilogo.values(), key=lambda x: x["escursione"])
        print(f"\nEscursione peggiore fra esecuzioni: {peggio['escursione']:.4f}")
        print("Confronti del progetto, e se sopravvivono a questo pavimento:")
        for nome, d in (("alpha 0.50 - alpha 1.00", 0.0125),
                        ("balanced - oversample", 0.0177),
                        ("balanced - class_weighted", 0.0107),
                        ("balanced - focal", 0.0078),
                        ("balanced - none", 0.0054)):
            verdetto = ("REGGE" if d > 2 * peggio["escursione"] else
                        "AL LIMITE" if d > peggio["escursione"] else
                        "DENTRO IL RUMORE")
            print(f"  {nome:28s} {d:+.4f}   {verdetto}")

    # Le epoche scelte sono la diagnosi della causa.
    print(f"\n{'=' * 78}\nEPOCA SCELTA, per seed e per esecuzione\n{'=' * 78}")
    for m in a.metodi:
        runs = fuori["esecuzioni"][m]
        if len(runs) < 2:
            continue
        chiavi = sorted(runs)
        print(f"\n{m}")
        for k in chiavi:
            print(f"  {k}: {runs[k]['epoca_scelta']}")
        cambiate = sum(len({runs[k]["epoca_scelta"][i] for k in chiavi}) > 1
                       for i in range(len(a.seeds)))
        print(f"  -> l'epoca scelta cambia su {cambiate}/{len(a.seeds)} seed")
        if cambiate:
            print("     la selezione sull'epoca amplifica il rumore: confermato")

    fuori["riepilogo"] = riepilogo
    save_json(fuori, percorso)
    print(f"\nRisultati in {percorso}")
