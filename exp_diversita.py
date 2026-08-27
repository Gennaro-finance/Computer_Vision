"""
Diversita' dei token - il meccanismo dietro il risultato dello sweep di alpha.

LA DOMANDA. Lo sweep ha trovato il massimo a alpha 0.50, non a 1.00: dare
sette viste a un PAI 5 rende MENO che dargliene tre. L'interpretazione
proposta e' che le viste siano ridondanti - sottoinsiemi di token della
stessa lesione, quindi molto correlati fra loro - e che sette viste
correlate non valgano sette esempi. Finche' resta un'interpretazione e'
una storia; qui diventa un numero.

TRE MISURE, in ordine di forza.

1. QUANTI TOKEN CI SONO. Se le bbox contenessero pochissimi token, i
   sottoinsiemi possibili sarebbero pochi e la ridondanza sarebbe una
   banalita' combinatoria. Va escluso prima di dire qualunque altra cosa.

2. QUANTO SI SOMIGLIANO LE VISTE. Coseno medio fra viste della STESSA
   lesione, confrontato col coseno fra lesioni DIVERSE della stessa classe.
   Il secondo e' il riferimento: se due lesioni diverse si somigliano gia'
   tanto quanto due viste della stessa, la ridondanza non e' delle viste,
   e' della rappresentazione.

3. QUANTE VISTE INDIPENDENTI VALGONO. E' la misura che risponde davvero.
   Da campionamento statistico: k osservazioni con correlazione interna rho
   non valgono k campioni indipendenti, ne valgono

        n_eff = k / (1 + (k - 1) * rho)

   che e' l'inverso del design effect. Con rho vicino a 1, n_eff resta
   vicino a 1 per qualunque k: aggiungere viste non aggiunge informazione,
   sposta solo il peso nella loss. E' esattamente la forma del risultato
   osservato - un massimo interno invece che a alpha 1.00.

   rho e' l'ICC multivariato: varianza fra lesioni diviso varianza totale,
   sommate su tutte le dimensioni (traccia). Con rappresentazioni a 1152
   dimensioni la versione per singola dimensione non ha senso da riportare.

POOLING USATO: media semplice sui token in maschera. E' senza parametri,
quindi la misura non dipende da un pooling addestrato con un seed
particolare - che sarebbe una scelta arbitraria dentro una misura che deve
descrivere i DATI, non un modello. La stessa misura si rifa' con il pooling
addestrato passando --addestrato, e le due vanno riportate insieme se
differiscono: se rho resta alto anche con un aggregatore che ha imparato a
guardare i token, la ridondanza sta nei dati e non nel modo di aggregarli.

Uso:
    python exp_diversita.py --tag _casuale
    python exp_diversita.py --tag _casuale --classe 2 --viste 7
"""

import argparse
import json
import os

import numpy as np
import torch

from globals import CACHE_DIR, DEVICE, NUM_CLASSES, OUT_DIR, PAI_GRADES, SEED
from imbalance import _espandi, class_counts, n_views_per_class
from train_downstream import load_latents
from utils import save_json

VISTE = 8          # abbastanza da vedere la saturazione, poche da stare in RAM
LESIONI = 600      # per classe; oltre non cambia le medie di terza cifra


def _media_mascherata(tok, msk):
    """Media dei token dentro la maschera. Senza parametri, per costruzione."""
    w = msk.float()
    return (tok * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)


def _coseno_medio(x):
    """Coseno medio fra tutte le coppie distinte delle righe di x."""
    if x.shape[0] < 2:
        return float("nan")
    z = torch.nn.functional.normalize(x.float(), dim=-1)
    g = z @ z.T
    n = g.shape[0]
    fuori = ~torch.eye(n, dtype=torch.bool, device=g.device)
    return float(g[fuori].mean())


def icc_multivariato(viste_per_lesione):
    """
    Correlazione intra-lesione rho, versione multivariata a traccia.

    viste_per_lesione: (L, K, D) - L lesioni, K viste ciascuna, D dimensioni.

    rho = tr(Sigma_fra) / (tr(Sigma_fra) + tr(Sigma_dentro))

    Le due varianze si sommano sulle dimensioni prima di dividere, non si
    mediano i rapporti per dimensione: un rapporto calcolato su una
    dimensione a varianza quasi nulla e' dominato dal rumore numerico, e
    mediarlo con le altre gli darebbe lo stesso peso di una dimensione
    informativa.
    """
    x = viste_per_lesione.double()
    L, K, D = x.shape
    mu_les = x.mean(dim=1)                       # (L, D) media per lesione
    mu = mu_les.mean(dim=0)                      # (D,)   media generale

    # varianza DENTRO: quanto si spostano le viste attorno alla loro lesione
    dentro = float(((x - mu_les[:, None, :]) ** 2).sum() / (L * (K - 1)))
    # varianza FRA: quanto si spostano le lesioni fra loro
    fra_grezza = float(((mu_les - mu) ** 2).sum() / (L - 1))
    # la media di K viste ha gia' dentro una quota di varianza interna: la si
    # toglie, altrimenti rho risulta piu' basso di quanto e' e la conclusione
    # sarebbe conservativa nella direzione sbagliata
    fra = max(fra_grezza - dentro / K, 0.0)
    tot = fra + dentro
    return (fra / tot if tot > 0 else float("nan")), fra, dentro


def n_efficace(k, rho):
    """k viste correlate rho valgono questi campioni indipendenti."""
    return k / (1.0 + (k - 1) * rho)


def analizza(cached, classe, n_viste=VISTE, n_lesioni=LESIONI, seed=SEED,
             addestrato=None):
    tr = cached["data"]["train"]
    sel = (tr["labels"] == classe).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    if len(sel) > n_lesioni:
        sel = sel[torch.randperm(len(sel), generator=g)[:n_lesioni]]

    tok = tr["tokens"][sel].float()
    msk = tr["mask"][sel]
    lab = tr["labels"][sel]

    n_tok = msk.sum(1)
    # 1. QUANTI TOKEN. Se sono pochi, la ridondanza e' combinatoria e non
    #    dice niente sulla rappresentazione.
    quanti = {
        "mediana": float(n_tok.float().median()),
        "q1": float(n_tok.float().quantile(0.25)),
        "q3": float(n_tok.float().quantile(0.75)),
        "min": int(n_tok.min()), "max": int(n_tok.max()),
    }

    # k viste per OGNI lesione, generate dalla stessa procedura della novita'
    per_sample = torch.full((len(sel),), n_viste, dtype=torch.long)
    gv = torch.Generator().manual_seed(seed + 1)
    tv, mv, _, orig = _espandi(tok, msk, lab, per_sample, generator=gv)

    pooled = (addestrato(tv, mv) if addestrato is not None
              else _media_mascherata(tv, mv))
    pooled = pooled.reshape(len(sel), n_viste, -1)

    # 2. COSENI. Dentro la lesione contro fra lesioni diverse.
    dentro = float(np.mean([_coseno_medio(pooled[i]) for i in range(len(sel))]))
    fra = _coseno_medio(pooled[:, 0, :])          # prima vista di ogni lesione

    # 3. rho e viste efficaci.
    rho, v_fra, v_dentro = icc_multivariato(pooled)

    return {
        "classe": int(classe), "pai": PAI_GRADES[classe],
        "n_lesioni": int(len(sel)), "n_viste": int(n_viste),
        "token": quanti,
        "coseno_dentro_lesione": dentro,
        "coseno_fra_lesioni": fra,
        "rho": rho,
        "var_fra": v_fra, "var_dentro": v_dentro,
    }


def stampa(res, counts):
    print(f"\n{'':10s} {'token bbox':>18s} {'cos dentro':>11s} {'cos fra':>9s} "
          f"{'rho':>7s}")
    print("-" * 60)
    for r in res:
        t = r["token"]
        print(f"PAI {r['pai']:<6d} {t['mediana']:6.0f} [{t['q1']:.0f}-{t['q3']:.0f}]"
              f"{'':6s} {r['coseno_dentro_lesione']:11.4f} "
              f"{r['coseno_fra_lesioni']:9.4f} {r['rho']:7.4f}")

    print("\nQuante viste INDIPENDENTI valgono quelle assegnate dalla novita':")
    print(f"{'alpha':>6s}  " + "  ".join(f"PAI {g}" for g in PAI_GRADES)
          + "      (viste assegnate -> viste efficaci)")
    print("-" * 78)
    for al in (0.25, 0.50, 0.75, 1.00):
        v = n_views_per_class(counts, al).tolist()
        celle = []
        for c, r in enumerate(res):
            k = v[c]
            celle.append(f"{k} -> {n_efficace(k, r['rho']):.2f}")
        print(f"{al:6.2f}  " + "  ".join(f"{c:>9s}" for c in celle))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_casuale")
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--viste", type=int, default=VISTE)
    ap.add_argument("--lesioni", type=int, default=LESIONI)
    ap.add_argument("--classe", type=int, default=None,
                    help="solo questa classe (0/1/2); per difetto tutte")
    ap.add_argument("--addestrato", action="store_true",
                    help="usa il pooling ADDESTRATO invece della media "
                         "mascherata: e' la prima obiezione che riceve questa "
                         "misura, e va risposta con un numero")
    ap.add_argument("--pool", default="attn",
                    help="quale pooling addestrare, con --addestrato")
    a = ap.parse_args()

    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)
    counts = class_counts(cached["data"]["train"]["labels"])
    print(f"encoder{a.tag}, train PAI3/4/5 = {counts.int().tolist()}")
    print(f"{a.viste} viste per lesione, max {a.lesioni} lesioni per classe, "
          f"pooling = media mascherata")

    pooling = None
    if a.addestrato:
        # Si addestra la testa una volta col metodo di riferimento e si
        # riusa il SUO pooling. Non si misura la testa: si misura se un
        # pooling addestrato separa le viste piu' di quanto le separi una
        # media. Se rho resta alto anche cosi', la ridondanza e' nei dati e
        # nessun aggregatore la puo' inventare.
        from train_downstream import train_head
        clf, _ = train_head(cached, "none", "flat", seed=SEED,
                            pool_type=a.pool)
        clf.eval()

        @torch.no_grad()
        def pooling(tv, mv, _clf=clf):
            fuori = []
            for i in range(0, tv.shape[0], 512):
                p_, _w = _clf.pool(tv[i:i + 512].to(DEVICE),
                                   mv[i:i + 512].to(DEVICE))
                fuori.append(p_.cpu())
            return torch.cat(fuori)

        print(f"pooling ADDESTRATO ({a.pool}) invece della media mascherata")

    classi = [a.classe] if a.classe is not None else list(range(NUM_CLASSES))
    res = [analizza(cached, c, a.viste, a.lesioni, addestrato=pooling)
           for c in classi]
    stampa(res, counts)

    suff = f"_{a.pool}addestrato" if a.addestrato else ""
    path = os.path.join(OUT_DIR, f"diversita_{a.variant}{a.tag}{suff}.json")
    save_json({"tag": a.tag, "viste": a.viste, "classi": res,
               "pooling": (f"{a.pool} addestrato" if a.addestrato
                           else "media mascherata"),
               "counts": counts.int().tolist()}, path)
    print(f"\nRisultati in {path}")
