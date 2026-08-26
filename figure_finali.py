"""
Le figure della presentazione, dai file di risultati.

Nessuna GPU, nessun latente: legge i JSON in runs/ e i log delle traiettorie.
Gira ovunque in pochi secondi, anche su una macchina senza scheda video.

    python figure_finali.py

Ogni figura che non trova i suoi dati viene saltata con un avviso, invece di
far fallire l'intero script: cosi' si possono generare le figure disponibili
mentre un esperimento e' ancora in corso.
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from globals import FIG_DIR, OUT_DIR

# --------------------------------------------------------------------------
# Aspetto
# --------------------------------------------------------------------------
# Palette categorica verificata con lo strumento della guida dataviz:
# banda di luminosita', soglia di croma, separazione per daltonismo (peggior
# coppia dE 9.2 deutan, 27.6 a visione normale) e contrasto. L'unico avviso
# riguardava il contrasto dell'acqua sul fondo chiaro, ed e' coperto dalle
# etichette scritte accanto a ogni marca: l'identita' non e' mai affidata al
# solo colore.
BLU, ARANCIO, ACQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MID, INK_SOFT = "#151b21", "#4a5560", "#6e7a85"
GRIGLIA = "#e2e7eb"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.edgecolor": GRIGLIA,
    "axes.labelcolor": INK_MID,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "grid.color": GRIGLIA,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})

PAVIMENTO = 0.2589
SOGLIE_BBOX = 0.7567     # due tagli sul lato della bbox, senza rete

ENCODER = [
    ("_casuale", "casuale", BLU),
    ("_spinto", "spinto (lr 3e-4)", ARANCIO),
    ("", "completa (lr 3e-5)", ACQUA),
]


def carica(tag):
    p = os.path.join(OUT_DIR, f"results_vit_small_L2-7-11{tag}.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def salva(fig, nome):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, f"{nome}.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def etichetta(ax, x, y, testo, dy=0.004, **kw):
    """Valore scritto accanto alla marca. Il testo usa l'inchiostro, mai il
    colore della serie: il colore identifica la marca, non la scritta."""
    ax.text(x, y + dy, testo, ha="center", va="bottom", fontsize=8.5,
            color=INK_MID, **kw)


# --------------------------------------------------------------------------
# 1. Il soffitto del problema
# --------------------------------------------------------------------------
def fig_soffitto():
    """
    La figura piu' importante: quanto aggiunge la rete rispetto a misurare
    la lesione con un righello.

    Barre orizzontali perche' le etichette sono frasi, non categorie brevi:
    scritte in orizzontale si leggono senza ruotare la testa. Serie unica,
    quindi nessuna legenda - il titolo dice gia' cosa si sta guardando.
    """
    dati = [("classificatore costante\n(pavimento)", PAVIMENTO, INK_SOFT),
            ("due soglie sulla bbox\nnessuna rete", SOGLIE_BBOX, INK_MID)]

    c = carica("_casuale")
    if c:
        migliore = max(r["macro_f1_mean"] for r in c)
        dati.append(("miglior modello\nencoder congelato", migliore, BLU))

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(dati))
    ax.barh(y, [d[1] for d in dati], height=0.5,
            color=[d[2] for d in dati], zorder=3)
    for i, (_, v, _) in enumerate(dati):
        ax.text(v + 0.012, i, f"{v:.4f}", va="center", fontsize=9.5,
                color=INK, fontweight="bold")

    ax.set_yticks(y, [d[0] for d in dati], fontsize=9, color=INK_MID)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("macro-F1 sul test")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_title("Tutta la rete aggiunge "
                 f"{dati[-1][1] - SOGLIE_BBOX:+.4f} su un righello"
                 if len(dati) == 3 else "Il soffitto del problema")
    return salva(fig, "fin1_soffitto")


# --------------------------------------------------------------------------
# 2. I tre encoder
# --------------------------------------------------------------------------
def fig_encoder():
    """
    Ogni configurazione come un punto, la media come barra piatta.

    Non barre con deviazione standard: con dieci configurazioni per encoder
    la forma della distribuzione e' l'informazione - mostra che le tre nuvole
    si sovrappongono, che e' esattamente la conclusione. Una barra con i
    baffi la nasconderebbe dietro due numeri.
    """
    presenti = [(t, n, c) for t, n, c in ENCODER if carica(t)]
    if len(presenti) < 2:
        print("  salto fin2_encoder: servono almeno due encoder misurati")
        return None

    fig, assi = plt.subplots(1, 2, figsize=(9.8, 4.3),
                             constrained_layout=True)
    for ax, chiave, titolo in (
            (assi[0], "pr_auc_pai5_mean", "PR-AUC su PAI 5\n(metrica primaria del brief)"),
            (assi[1], "macro_f1_mean", "macro-F1")):
        for i, (tag, nome, colore) in enumerate(presenti):
            v = np.array([r[chiave] for r in carica(tag)])
            x = np.random.default_rng(0).normal(i, 0.055, len(v))
            ax.scatter(x, v, s=26, color=colore, alpha=0.55,
                       edgecolors="white", linewidths=0.8, zorder=3)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color=colore,
                      linewidth=2.4, zorder=4)
            # il valore va ACCANTO alla media, non sopra la nuvola: sopra
            # finisce addosso al titolo e a chi ha la nuvola piu' alta
            ax.text(i + 0.31, v.mean(), f"{v.mean():.4f}", va="center",
                    ha="left", fontsize=8.5, color=INK_MID)

        ax.set_xticks(range(len(presenti)),
                      [n for _, n, _ in presenti], fontsize=9, color=INK_MID)
        ax.set_xlim(-0.5, len(presenti) - 0.12)
        ax.set_title(titolo, pad=10)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    assi[0].set_ylabel("valore sul test, 5 seed")
    fig.suptitle("Dieci configurazioni per encoder: le nuvole si sovrappongono",
                 fontsize=11.5, fontweight="bold", color=INK)
    return salva(fig, "fin2_encoder")


# --------------------------------------------------------------------------
# 3. La novita' contro le baseline
# --------------------------------------------------------------------------
def fig_novita():
    """
    F1 sulla classe rara sopra, precisione sotto, come PUNTI non barre.

    La forma non e' un dettaglio estetico. Le differenze qui stanno sulla
    terza cifra: con delle barre o l'asse parte da zero e le schiaccia tutte
    a sembrare identiche, oppure lo si taglia - e una barra tagliata mente,
    perche' la sua lunghezza dice qualcosa che non e' vero. Un punto codifica
    una POSIZIONE, non una lunghezza, quindi un asse che non parte da zero e'
    legittimo e le differenze restano leggibili.

    Sotto la precisione e non la recall: tutti i metodi alzano la recall -
    basta abbassare la soglia - ma la pagano in precisione. La novita' no, ed
    e' il pannello inferiore a dimostrarlo.
    """
    dati = {t: carica(t) for t, _, _ in ENCODER if carica(t)}
    if not dati:
        print("  salto fin3_novita: nessun risultato")
        return None

    metodi = ["none", "class_weighted", "focal", "oversample", "balanced_tokens"]
    nomi = ["CE semplice", "pesi di classe", "focal", "oversampling",
            "balanced token sampling"]
    tag_nome = {t: n for t, n, _ in ENCODER}
    colori = {t: c for t, _, c in ENCODER}

    fig, assi = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True,
                             constrained_layout=True)

    for ax, chiave, titolo in (
            (assi[0], "f1_pai5_mean", "F1 sulla classe PAI 5"),
            (assi[1], "precision_pai5_mean", "precisione su PAI 5")):
        for j_, (tag, righe) in enumerate(dati.items()):
            per_metodo = {r["method"]: r for r in righe if r["head"] == "flat"}
            y = np.arange(len(metodi)) + (j_ - (len(dati) - 1) / 2) * 0.22
            v = [per_metodo[m][chiave] if m in per_metodo else np.nan
                 for m in metodi]
            ax.scatter(v, y, s=54, color=colori[tag], zorder=3,
                       edgecolors="white", linewidths=1.1,
                       label=tag_nome[tag] if ax is assi[0] else None)

        # la riga della novita' si stacca dallo sfondo, cosi' l'occhio la
        # trova senza doverla cercare fra cinque etichette
        ax.axhspan(len(metodi) - 1.42, len(metodi) - 0.58,
                   color=INK_SOFT, alpha=0.07, zorder=0)
        ax.set_title(titolo, pad=10)
        ax.xaxis.grid(True, zorder=1)
        ax.set_axisbelow(True)

    assi[0].set_yticks(range(len(metodi)), nomi, fontsize=9.5, color=INK_MID)
    assi[0].set_ylim(-0.6, len(metodi) - 0.4)
    assi[0].invert_yaxis()
    # legenda fuori dai pannelli: dentro copriva proprio i punti della riga
    # evidenziata, che e' quella che la figura deve far vedere
    fig.legend(loc="outside lower center", ncol=len(dati), fontsize=9.5,
               labelcolor=INK_MID, handletextpad=0.4, columnspacing=2.2)
    fig.suptitle("La novita' alza la classe rara senza pagarla in precisione",
                 fontsize=11.5, fontweight="bold", color=INK)
    return salva(fig, "fin3_novita")


# --------------------------------------------------------------------------
# 4. La traiettoria del pre-training
# --------------------------------------------------------------------------
def traiettoria(*logs):
    """
    (epoche, downstream) da uno o piu' log di pre-training.

    Piu' log perche' un run ripreso dopo un'interruzione ne scrive uno nuovo:
    la configurazione `completa` sta in SSL_completa.log fino all'epoca 150 e
    in SSL_completa_150-300.log fino alla 232. Leggendone uno solo la curva
    si fermerebbe a meta' senza dirlo.
    """
    ep_tot, v_tot = [], []
    for log in logs:
        if not os.path.isfile(log):
            continue
        t = open(log, encoding="utf-8", errors="replace").read()
        d = re.findall(r"\[downstream\] macroF1=([\d.]+)", t)
        ep = [int(x) + 1 for x in re.findall(r"\[monitor\] ep(\d+)", t)]
        dieci = [e for e in ep if e % 10 == 0]
        v = [float(x) for x in d[1:]]      # la prima misura e' il riferimento
        n = min(len(v), len(dieci))
        ep_tot += dieci[:n]
        v_tot += v[:n]
    if not ep_tot:
        return None, None
    ordine = np.argsort(ep_tot)
    return [ep_tot[i] for i in ordine], [v_tot[i] for i in ordine]


def fig_traiettoria():
    """
    Il downstream lungo le epoche, contro la banda del riferimento casuale.

    La banda invece di una linea: il riferimento ha una sua incertezza
    (+-0.0111 su 5 seed), e disegnarlo come linea netta suggerirebbe una
    precisione che non ha.
    """
    serie = []
    for logs, nome, colore in (
            (("logs/SSL_completa.log", "logs/SSL_completa_150-300.log"),
             "paper, 53.8% mascherato", ACQUA),
            (("logs/SSL_mask80.log",), "80% mascherato", ARANCIO),
            (("logs/SSL_spinto.log",), "lr 3e-4", BLU)):
        e, v = traiettoria(*logs)
        if e:
            serie.append((e, v, nome, colore))
    if not serie:
        print("  salto fin4_traiettoria: nessun log di pre-training")
        return None

    fig, ax = plt.subplots(figsize=(9, 3.9))
    rif, err = 0.7512, 0.0111
    ax.axhspan(rif - err, rif + err, color=INK_SOFT, alpha=0.13, zorder=1)
    ax.axhline(rif, color=INK_SOFT, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(4, rif + err + 0.001, "encoder casuale, 0.7512 +- 0.0111",
            fontsize=8.5, color=INK_MID, va="bottom")

    for e, v, nome, colore in serie:
        ax.plot(e, v, linewidth=2, color=colore, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8,
                label=nome, zorder=3)

    ax.set_xlabel("epoca di pre-training")
    ax.set_ylabel("downstream, macro-F1 su validation")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, labelcolor=INK_MID, loc="lower right")
    ax.set_title("Nessuna tendenza: l'oscillazione supera il vantaggio cercato")
    return salva(fig, "fin4_traiettoria")


# --------------------------------------------------------------------------
# 5. Lo sweep di alpha
# --------------------------------------------------------------------------
def fig_alpha():
    """
    L'ablation su alpha, quando il file c'e'.

    Si accettano solo i file nel formato di exp_alpha.py (un dizionario con
    "screening"): in runs/ possono restarne di vecchi, prodotti da una
    versione precedente che salvava una lista. Caricarli e' un errore
    silenzioso - il grafico uscirebbe con dati di un altro esperimento.
    """
    buoni = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "sweep_alpha*.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("screening"):
            buoni.append((p, d))
    if not buoni:
        print("  salto fin5_alpha: sweep non ancora eseguito")
        return None

    percorso, d = buoni[-1]
    scr = d["screening"]
    chiavi = sorted(scr, key=float)
    a = [float(k) for k in chiavi]
    m = np.array([scr[k]["pr_auc"][0] for k in chiavi])
    s_dev = np.array([scr[k]["pr_auc"][1] for k in chiavi])

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.axhline(0.8813, color=INK_SOFT, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(a[0], 0.8813 + 0.0006, "alpha 0.5 misurato a 5 seed: 0.8813",
            fontsize=8.5, color=INK_MID, va="bottom")
    ax.errorbar(a, m, yerr=s_dev, color=BLU, linewidth=2, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=0.9,
                capsize=4, zorder=3)
    for x, y in zip(a, m):
        etichetta(ax, x, y, f"{y:.4f}", dy=0.0012)

    ax.set_xlabel("alpha (0 = nessun ribilanciamento, 1 = pareggio effettivo)")
    ax.set_ylabel("PR-AUC su PAI 5")
    ax.set_xticks(a)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Ablation su alpha, screening")
    print("  fin5_alpha da", os.path.basename(percorso))
    return salva(fig, "fin5_alpha")


if __name__ == "__main__":
    fatte = []
    for f in (fig_soffitto, fig_encoder, fig_novita, fig_traiettoria, fig_alpha):
        p = f()
        if p:
            fatte.append(p)
    print("\nFigure generate:")
    for p in fatte:
        print("  ", p)
