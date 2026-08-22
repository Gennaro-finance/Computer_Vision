"""
Evaluation - la figura che spiega la correzione dei crop.

Sezione "Evaluation" della struttura richiesta dal corso.

COSA MOSTRA. Le STESSE tre lesioni, una per grado, ritagliate nei due modi:

  relative  finestra proporzionale alla bbox, poi ridimensionata a 224.
            Il fattore di scala varia da lesione a lesione ESATTAMENTE in
            modo da annullare la differenza di dimensione: la bbox finisce
            sempre al centro, sempre larga 74.7 px, per tutte le lesioni.
            La dimensione - che e' il segnale piu' predittivo del dataset -
            viene cancellata prima che la rete veda l'immagine.

  fixed     finestra costante di 224 px nativi, nessun ricampionamento.
            Una lesione grande APPARE grande.

Misurato sul train, lato mediano della bbox per grado: 57 / 81 / 126 px.
Due sole soglie su quel numero danno macro-F1 0.7567 sul test, piu' di
qualunque encoder provato prima della correzione.

Uso:
    python make_fig_crop.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import LesionCropDataset, load_splits, parse_annotations
from globals import FIG_DIR, PAI_GRADES
from network import bbox_to_token_mask

BLU, ROSSO = "#1f4e79", "#c1272d"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.size": 10})


def main():
    records = parse_annotations(verbose=False)
    splits = load_splits()

    ds_rel = LesionCropDataset(records, splits["val"], mode="relative")
    ds_fix = LesionCropDataset(records, splits["val"], mode="fixed")

    # Una lesione per grado, scegliendo quella con la bbox piu' vicina alla
    # mediana della sua classe: un esempio tipico, non il piu' vistoso.
    lati = {c: [] for c in range(3)}
    for i, it in enumerate(ds_fix.items):
        lati[__import__("globals").GRADE_TO_IDX[it["grade"]]].append(
            (max(it["xmax"] - it["xmin"], it["ymax"] - it["ymin"]), i))
    scelti = []
    for c in range(3):
        v = sorted(lati[c])
        scelti.append(v[len(v) // 2][1])

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.6))
    for col, idx in enumerate(scelti):
        for row, (ds, nome) in enumerate([(ds_rel, "relative"), (ds_fix, "fixed")]):
            d = ds[idx]
            ax = axes[row][col]
            ax.imshow(d["image"][0].numpy() * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
            b = d["bbox"].numpy()
            lato = max(b[2] - b[0], b[3] - b[1])
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, edgecolor=ROSSO, lw=2.2))
            n_tok = int(bbox_to_token_mask(d["bbox"][None], 14).sum())
            ax.set_title(f"PAI {PAI_GRADES[d['label']]} · lato {lato:.0f} px · "
                         f"{n_tok} token", fontsize=9.5)
            ax.axis("off")
            if col == 0:
                ax.text(-0.08, 0.5, nome, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12, fontweight="bold",
                        color=BLU if row else "#777")

    fig.suptitle("Stesse lesioni, due ritagli: sopra la dimensione e' annullata, "
                 "sotto e' preservata", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig5_crop_confronto.png")
    fig.savefig(p1, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p1}")

    # Seconda figura: la distribuzione del lato della bbox nel crop.
    # E' il grafico che rende evidente il difetto: una riga verticale contro
    # una distribuzione vera.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    misure = {}
    for ds, nome in [(ds_rel, "relative"), (ds_fix, "fixed")]:
        v = []
        for i in range(0, len(ds), 3):
            b = ds[i]["bbox"]
            v.append(max((b[2] - b[0]).item(), (b[3] - b[1]).item()))
        misure[nome] = np.array(v)

    ax.hist(misure["fixed"], bins=40, alpha=.85, color=BLU, edgecolor="white",
            linewidth=.5,
            label=f"fixed (dopo) · dev.std {misure['fixed'].std():.1f} px")

    # 'relative' e' degenere: TUTTI i valori coincidono. Disegnarla come
    # istogramma la fa sparire dietro l'altra, e sparirebbe proprio il punto
    # della figura. Si traccia come riga verticale, che e' cio' che e'.
    x0 = misure["relative"].mean()
    ax.axvline(x0, color=ROSSO, lw=2.6, ls="--",
               label=f"relative (prima) · TUTTE a {x0:.1f} px · dev.std "
                     f"{misure['relative'].std():.1f}")
    ax.annotate("ogni lesione,\nqualunque grado,\narrivava identica",
                xy=(x0, ax.get_ylim()[1] * .72),
                xytext=(x0 + 38, ax.get_ylim()[1] * .80),
                fontsize=9.5, color=ROSSO,
                arrowprops=dict(arrowstyle="->", color=ROSSO, lw=1.4))
    ax.set_xlabel("lato della bbox nell'immagine data alla rete (px)")
    ax.set_ylabel("numero di lesioni")
    ax.set_title("Prima, ogni lesione arrivava alla rete della stessa dimensione",
                 fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig6_distribuzione_scala.png")
    fig.savefig(p2, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {p2}")


if __name__ == "__main__":
    print("Figure generate:")
    main()
