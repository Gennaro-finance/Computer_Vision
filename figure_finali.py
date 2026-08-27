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

from utils import leggi_righe_risultati
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
BIANCO_CARTA = "#ffffff"

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
    return leggi_righe_risultati(p)


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
def fig_sonde():
    """
    Le due figure che il ribaltamento richiede, affiancate.

    A SINISTRA la curva di apprendimento: la qualita' della rappresentazione
    contro le epoche di pre-training, nel protocollo cieco alla dimensione.
    E' la dimostrazione diretta dell'obiettivo 1 - non un confronto a due
    punti ma una tendenza, e per giunta non satura a 179 epoche.

    A DESTRA il decadimento del segnale geometrico con la profondita'. Nel
    casuale e' piatto perche' qualunque proiezione casuale conserva l'area;
    nell'addestrato scende, perche' il masked prediction costruisce
    invarianza e l'invarianza costa la geometria. Le due curve del cieco
    sono piatte e distanti: l'informazione d'aspetto c'e' gia' tutta al
    blocco 2 e non cresce.

    Tutto senza un solo parametro addestrato: e' una sonda k-NN sulla media
    dei token dentro la bbox. Serve a rispondere in anticipo a "non sara' la
    testa a fare il lavoro?".
    """
    percorso = os.path.join(OUT_DIR, "sonde_vit_small.json")
    if not os.path.isfile(percorso):
        print("  salto fin7_sonde: lanciare prima exp_sonde.py")
        return None
    with open(percorso, encoding="utf-8") as f:
        d = json.load(f)["sonde"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))

    # ---- sinistra: curva di apprendimento, solo la config confrontabile ----
    curva = [(r["epoche"], r["tutti"]["macro_f1"])
             for n, r in d["cieco"].items() if n in ("casuale", "notte", "completa")]
    curva.sort()
    x, y = zip(*curva)
    ax1.plot(x, y, color=BLU, linewidth=2.4, marker="o", markersize=7,
             markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    # Il primo punto E' l'encoder casuale: una riga di riferimento alla sua
    # altezza sarebbe ridondante col punto stesso, e ci finiva sopra
    # l'etichetta. Si etichetta il punto e basta.
    for i, (xi, yi) in enumerate(curva):
        ax1.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                     xytext=(14 if i == 0 else 0, 4 if i == 0 else 11),
                     ha="left" if i == 0 else "center",
                     fontsize=9, color=INK_MID)
    ax1.annotate("0 epoche = encoder casuale", (x[0], y[0]),
                 textcoords="offset points", xytext=(14, -13),
                 ha="left", fontsize=8.5, color=INK_SOFT)
    ax1.set_xlabel("epoche di pre-training I-JEPA")
    ax1.set_ylabel("macro-F1, sonda k-NN")
    ax1.set_title("La rappresentazione migliora, e non &egrave; satura"
                  .replace("&egrave;", "e'"))
    ax1.set_ylim(min(y) - 0.05, max(y) + 0.06)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    # ---- destra: per blocco, nei due protocolli ----
    bl = ["b2", "b7", "b11"]
    xs = [2, 7, 11]
    stile = {("geometrico", "casuale"):  dict(color=ARANCIO, ls="-",  marker="o"),
             ("geometrico", "completa"): dict(color=ARANCIO, ls="--", marker="s"),
             ("cieco", "casuale"):       dict(color=BLU, ls="-",  marker="o"),
             ("cieco", "completa"):      dict(color=BLU, ls="--", marker="s")}
    for (prot, enc), kw in stile.items():
        if enc not in d.get(prot, {}):
            continue
        v = [d[prot][enc][b]["macro_f1"] for b in bl]
        ax2.plot(xs, v, linewidth=2, markersize=6, markeredgecolor="white",
                 markeredgewidth=0.9, zorder=3,
                 label=f"{prot}, {enc}", **kw)
    ax2.set_xticks(xs)
    ax2.set_xlabel("blocco del ViT")
    ax2.set_ylabel("macro-F1, sonda k-NN")
    ax2.set_title("Il segnale geometrico decade, l'aspetto no")
    # La legenda va sotto: con quattro serie qualunque posizione interna
    # finisce addosso a una curva, e qui lo spazio verticale libero non c'e'.
    ax2.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=2, columnspacing=1.4, handlelength=2.2)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    print("  fin7_sonde da", os.path.basename(percorso))
    return salva(fig, "fin7_sonde")


def fig_mascheramento():
    """
    Il canale nascosto: cosa succede quando la maschera smette di dire la
    dimensione della lesione.

    A SINISTRA il confronto fra protocolli. La barra grigia in cima e' la
    MASCHERA DA SOLA, one-hot, senza un solo pixel: nel protocollo del
    brief batte il vettore completo dell'encoder casuale. E' il numero che
    spiega tutto il resto.

    A DESTRA lo sweep su K con K uguale per tutte le classi. Il casuale
    resta schiacciato a ogni K; il divario si apre invece di chiudersi.

    Le barre non partono da zero perche' il pavimento utile non e' zero ma
    0,2589, la macro-F1 di un classificatore costante: sotto quella soglia
    non c'e' niente da leggere. La riga tratteggiata la marca.
    """
    percorso = os.path.join(OUT_DIR, "mascheramento_vit_small.json")
    if not os.path.isfile(percorso):
        print("  salto fin8_mascheramento: lanciare prima exp_mascheramento.py")
        return None
    with open(percorso, encoding="utf-8") as f:
        d = json.load(f)
    P, tags = d["protocolli"], ["_casuale", "_geo_completa"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- sinistra: protocolli a confronto ----
    ordine = ["P1_bbox", "P2d_conteggio_scorrelato", "P2c_centrale_fisso",
              "P2b_griglia_fissa", "P2a_casuale_K36"]
    nomi = {"P1_bbox": "maschera bbox\n(16/36/64 token)",
            "P2d_conteggio_scorrelato": "conteggio scorrelato\ndalla classe",
            "P2c_centrale_fisso": "blocco centrale\nfisso, 36",
            "P2b_griglia_fissa": "griglia fissa\n6x6, 36",
            "P2a_casuale_K36": "36 token a caso,\nbbox ignorata"}
    y = np.arange(len(ordine))
    h = 0.36
    ax1.barh(y + h/2, [P[k][tags[0]]["macro_f1"] for k in ordine], h,
             color=ARANCIO, label="encoder casuale", zorder=3)
    ax1.barh(y - h/2, [P[k][tags[1]]["macro_f1"] for k in ordine], h,
             color=BLU, label="I-JEPA", zorder=3)
    for i, k in enumerate(ordine):
        for dy, t in ((h/2, tags[0]), (-h/2, tags[1])):
            v = P[k][t]["macro_f1"]
            ax1.text(v + 0.008, i + dy, f"{v:.3f}", va="center",
                     fontsize=8.5, color=INK_MID)
    # La riga della "sola maschera" e' il numero chiave: si etichetta in
    # orizzontale sopra l'asse, non ruotata dentro il grafico, dove finiva
    # addosso al titolo e alle barre.
    sm = d["solo_maschera_macro_f1"]
    ax1.axvline(sm, color=INK, linestyle="-", linewidth=1.6, zorder=4)
    ax1.annotate(f"la sola maschera, zero pixel: {sm:.4f}",
                 xy=(sm, 2.0), xytext=(-9, 0),
                 textcoords="offset points", ha="right", va="center",
                 fontsize=8.5, color=INK,
                 bbox=dict(boxstyle="round,pad=0.28", fc=BIANCO_CARTA,
                           ec=GRIGLIA, lw=0.8))
    ax1.axvline(PAVIMENTO, color=INK_SOFT, linestyle=":", linewidth=1.1, zorder=2)
    ax1.text(PAVIMENTO + 0.006, -0.68, "classificatore costante",
             fontsize=8, color=INK_SOFT)
    ax1.set_yticks(y, [nomi[k] for k in ordine], fontsize=8.5)
    ax1.set_xlim(0.22, 0.90)
    ax1.set_ylim(-0.85, len(ordine) - 0.15)
    ax1.set_xlabel("macro-F1, sonda k-NN senza parametri")
    ax1.set_title("Tolta la dimensione dalla maschera, il segno si ribalta")
    # La legenda in basso a destra finiva sulle barre lunghe di P1.
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.95,
               bbox_to_anchor=(1.0, 1.02))
    ax1.xaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    # ---- destra: sweep su K ----
    Ks = [9, 16, 36, 64, 100]
    for t, col, eti in ((tags[0], ARANCIO, "encoder casuale"),
                        (tags[1], BLU, "I-JEPA")):
        ax2.plot(Ks, [P[f"P3_K{k}"][t]["macro_f1"] for k in Ks], color=col,
                 linewidth=2.2, marker="o", markersize=6,
                 markeredgecolor="white", markeredgewidth=0.9,
                 label=eti, zorder=3)
    ax2.axhline(P["P1_bbox"][tags[0]]["macro_f1"], color=ARANCIO,
                linestyle="--", linewidth=1.3, zorder=2)
    ax2.text(100, P["P1_bbox"][tags[0]]["macro_f1"] - 0.016,
             "casuale col protocollo originale", ha="right", va="top",
             fontsize=8.5, color=ARANCIO)
    ax2.set_xscale("log")
    ax2.set_xticks(Ks, [str(k) for k in Ks])
    ax2.set_xlabel("K, uguale per tutte le classi")
    ax2.set_ylabel("macro-F1, sonda k-NN")
    ax2.set_title("A ogni K, con K scorrelato dalla classe")
    ax2.legend(fontsize=8.5, loc="center right")
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    print("  fin8_mascheramento da", os.path.basename(percorso))
    return salva(fig, "fin8_mascheramento")


def fig_curve_pr():
    """
    Le curve precision-recall su PAI 5 - la metrica primaria, non compressa.

    DUE PANNELLI. Sopra la curva intera, sotto uno zoom sulla fascia di
    recall che interessa davvero (0.6-0.95): li' le curve si separano di
    pochi punti percentuali, e sul pannello intero quella differenza e' un
    pelo di spessore. Uno zoom non e' un abbellimento, e' l'unico modo di
    far vedere la differenza che il numero riporta.

    Le curve sono etichettate DIRETTAMENTE oltre che in legenda: con cinque
    metodi la sola legenda costringe a fare la spola fra il riquadro e le
    curve, e su una slide nessuno la fa.
    """
    percorsi = sorted(glob.glob(os.path.join(OUT_DIR, "curve_pr_*.json")))
    if not percorsi:
        print("  salto fin6_curve_pr: lanciare prima exp_curve_pr.py")
        return None
    with open(percorsi[-1], encoding="utf-8") as f:
        d = json.load(f)

    ordine = [m for m in ("balanced_tokens", "none", "class_weighted",
                          "oversample", "focal") if m in d["curve"]]
    nomi = {"balanced_tokens": "balanced tokens (novita')", "none": "CE semplice",
            "class_weighted": "CE pesata", "oversample": "oversample",
            "focal": "focal loss"}
    # La novita' e' l'unica in blu pieno: le baseline sono il contesto, non
    # cinque protagonisti. Dare a ognuna un colore proprio fa perdere la sola
    # cosa che il grafico deve dire.
    stile = {"balanced_tokens": dict(color=BLU, linewidth=2.4, zorder=5),
             "none": dict(color=ARANCIO, linewidth=1.8, zorder=4)}
    grigi = [dict(color=INK_SOFT, linewidth=1.2, linestyle=ls, zorder=3)
             for ls in ("--", ":", "-.")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.4),
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    i_grigio = 0
    for m in ordine:
        c = d["curve"][m]
        r, p_ = np.array(c["recall"]), np.array(c["precision"])
        if m in stile:
            kw = stile[m]
        else:
            kw = grigi[i_grigio % len(grigi)]
            i_grigio += 1
        for ax in (ax1, ax2):
            ax.plot(r, p_, label=nomi.get(m, m), **kw)
        if m == "balanced_tokens":
            # La banda solo sulla novita': cinque bande sovrapposte sono
            # illeggibili, e la dispersione che serve vedere e' la sua.
            dev = np.array(c["dev"])
            for ax in (ax1, ax2):
                ax.fill_between(r, p_ - dev, p_ + dev, color=BLU, alpha=0.13,
                                linewidth=0, zorder=2)

    prev = d["prevalenza_pai5"]
    ax1.axhline(prev, color=INK_SOFT, linestyle="--", linewidth=1)
    ax1.text(0.985, prev + 0.02, f"classificatore casuale = prevalenza {prev:.3f}",
             ha="right", va="bottom", fontsize=8.5, color=INK_MID)

    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("precisione")
    ax1.set_title("Curve precision-recall su PAI 5, media di "
                  f"{len(d['seeds'])} seed, test")
    ax1.legend(loc="lower left", fontsize=8.5)

    # Zoom sulla fascia di lavoro. I limiti in y si ricavano dai dati dentro
    # la fascia: fissarli a mano vorrebbe dire ritagliarli attorno al
    # risultato, che e' un modo di mentire con un grafico onesto.
    lo, hi = 0.60, 0.95
    dentro = []
    for m in ordine:
        c = d["curve"][m]
        r, p_ = np.array(c["recall"]), np.array(c["precision"])
        dentro.append(p_[(r >= lo) & (r <= hi)])
    dentro = np.concatenate(dentro)
    m0, m1 = float(dentro.min()), float(dentro.max())
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(m0 - 0.02, m1 + 0.03)
    ax2.set_xlabel("recall su PAI 5")
    ax2.set_ylabel("precisione")
    ax2.set_title(f"Zoom sulla fascia di lavoro clinica, recall {lo:.2f}-{hi:.2f}")

    # Precisione alle recall di lavoro, scritta: e' quello che si cita a voce.
    pr = d["precisione_a_recall"]
    for r in d["recall_lavoro"]:
        if not (lo <= r <= hi):
            continue
        ax2.axvline(r, color=GRIGLIA, linewidth=1, zorder=1)
        b = pr.get("balanced_tokens", {}).get(f"{r:.2f}")
        n_ = pr.get("none", {}).get(f"{r:.2f}")
        if b is not None and n_ is not None:
            ax2.annotate(f"r={r:.2f}\n{b:.3f} vs {n_:.3f}",
                         (r, m1 + 0.012), ha="center", va="top",
                         fontsize=8, color=INK_MID)

    for ax in (ax1, ax2):
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    print("  fin6_curve_pr da", os.path.basename(percorsi[-1]))
    return salva(fig, "fin6_curve_pr")


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
    ax.errorbar(a, m, yerr=s_dev, color=BLU, linewidth=2, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=0.9,
                capsize=4, zorder=3)
    # i valori sotto il punto, sotto la barra d'errore: sopra finiscono
    # addosso alla riga di riferimento, che passa proprio all'altezza del
    # massimo
    for x, y, e in zip(a, m, s_dev):
        ax.text(x, y - e - 0.0007, f"{y:.4f}", ha="center", va="top",
                fontsize=8.5, color=INK_MID)
    # l'annotazione della riga va in basso a sinistra, dove non c'e' nulla
    ax.text(0.02, 0.04, "riga tratteggiata: alpha 0.5 misurato\na 5 seed nella griglia, 0.8813",
            transform=ax.transAxes, fontsize=8.5, color=INK_MID,
            va="bottom", ha="left")

    ax.set_xlabel("alpha (0 = nessun ribilanciamento, 1 = pareggio effettivo)")
    ax.set_ylabel("PR-AUC su PAI 5")
    ax.set_xticks(a)
    # margine sotto: le etichette stanno sotto la barra d'errore e senza
    # questo l'ultima, la piu' bassa, finisce tagliata dall'asse
    ax.set_ylim((m - s_dev).min() - 0.0028, (m + s_dev).max() + 0.0008)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Ablation su alpha, screening")
    print("  fin5_alpha da", os.path.basename(percorso))
    return salva(fig, "fin5_alpha")


if __name__ == "__main__":
    fatte = []
    for f in (fig_soffitto, fig_encoder, fig_novita, fig_traiettoria,
              fig_alpha, fig_curve_pr, fig_sonde,
              fig_mascheramento):
        p = f()
        if p:
            fatte.append(p)
    print("\nFigure generate:")
    for p in fatte:
        print("  ", p)
