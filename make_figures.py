"""
Evaluation - figure finali per la presentazione.

Sezione "Evaluation" della struttura richiesta dal corso.

Le guidelines chiedono sfondo bianco e alto contrasto negli elementi
grafici, e vietano l'accuracy globale come metrica riportata. Qui si
producono le quattro figure che servono a raccontare il progetto:

  1. confronto dei tre bracci      -> la domanda di ricerca del brief
  2. curva dose-risposta su alpha  -> l'ablation della novita' (obiettivo 4)
  3. confusion matrix              -> richiesta esplicitamente dal brief
  4. curve di pre-training         -> rango e k-NN, il collasso monitorato

Uso:
    python make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from globals import FIG_DIR, NUM_CLASSES, OUT_DIR, PAI_GRADES

# Palette ad alto contrasto, leggibile anche stampata in scala di grigi.
BLU, ARANCIO, GRIGIO, VERDE = "#1f4e79", "#c55a11", "#7f7f7f", "#2e7d32"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _carica():
    import glob
    cand = sorted(glob.glob(os.path.join(OUT_DIR, "results_vit_small*.json")),
                  key=os.path.getmtime)
    p = cand[-1] if cand else ""
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _salva(fig, nome):
    path = os.path.join(FIG_DIR, f"{nome}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")
    return path


# ==========================================================================
# 2. Ablation della novita': la curva dose-risposta
# ==========================================================================
def fig_alpha():
    """
    L'evidenza piu' forte dell'obiettivo 3: alpha regola quante viste riceve
    ogni classe, e la recall sulla minoritaria lo segue in modo monotono.
    Si riporta accanto la F1 su PAI 3 per mostrare il prezzo pagato: un
    guadagno sulla minoritaria ottenuto erodendo la maggioritaria non e' un
    guadagno, ed e' giusto che si veda.
    """
    p = os.path.join(OUT_DIR, "sweep_alpha_ijepa_vit_small.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if r["head"] == "flat"]
    rows.sort(key=lambda r: r["alpha"])

    a = [r["alpha"] for r in rows]
    rec = [r["recall_pai5_mean"] for r in rows]
    rec_e = [r["recall_pai5_std"] for r in rows]
    f1_3 = [r["f1_pai3_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(a, rec, yerr=rec_e, marker="o", ms=7, lw=2.2, capsize=5,
                color=ARANCIO, label="recall PAI 5 (minoritaria)")
    ax.plot(a, f1_3, marker="s", ms=6, lw=2.2, ls="--", color=BLU,
            label="F1 PAI 3 (maggioritaria)")

    ax.annotate(f"+{(rec[-2] - rec[0]) / rec[0]:.0%}",
                xy=(a[-2], rec[-2]), xytext=(a[-2] - .18, rec[-2] + .07),
                fontweight="bold", color=ARANCIO,
                arrowprops=dict(arrowstyle="->", color=ARANCIO))

    ax.set_xlabel(r"$\alpha$  (viste per classe: $\lceil (n_{max}/n_c)^{\alpha} \rceil$)")
    ax.set_ylabel("metrica sul test (5 seed)")
    ax.set_title("Balanced token sampling: piu' viste sulla rara, piu' recall",
                 fontsize=11.5)
    ax.set_xticks(a)
    # In basso a destra e' l'unica zona libera: la curva arancione sale da
    # sinistra e la blu occupa la fascia alta.
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0.25, 0.86)
    ax.grid(alpha=.25)
    return _salva(fig, "fig2_alpha")


# ==========================================================================
# 3. Confusion matrix
# ==========================================================================
def fig_confusioni():
    """
    Richiesta esplicitamente dal brief. Si mostrano affiancate la
    configurazione migliore in assoluto e il braccio del progetto, cosi'
    la differenza si legge sulla diagonale invece che in una tabella.
    """
    import torch

    from evaluation import evaluate_split
    from train_downstream import load_latents, train_head

    # Le due estremita' dell'ablation dell'obiettivo 4: la cross-entropy
    # semplice contro la novita' proposta, a encoder identico e congelato.
    casi = [("none", "flat", "CE semplice"),
            ("balanced_tokens", "ordinal", "balanced token sampling")]

    fig, axes = plt.subplots(1, len(casi), figsize=(10.5, 4.6))
    for ax, (method, head, titolo) in zip(axes, casi):
        try:
            cached = load_latents("vit_small")
        except FileNotFoundError:
            ax.axis("off")
            continue
        clf, _ = train_head(cached, method, head, seed=0)
        res = evaluate_split(clf, cached["data"]["test"], head)
        cm = np.array(res["confusion_matrix"], dtype=float)
        del clf
        torch.cuda.empty_cache()

        # Normalizzata per riga: senza, la classe maggioritaria domina il
        # colore e non si vede piu' nulla delle altre due.
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.0%}", ha="center",
                        va="center", fontsize=10,
                        color="white" if cmn[i, j] > .55 else "black")
        et = [f"PAI {g}" for g in PAI_GRADES]
        ax.set_xticks(range(NUM_CLASSES), et)
        ax.set_yticks(range(NUM_CLASSES), et)
        ax.set_xlabel("predetto")
        ax.set_ylabel("vero")
        # Va dichiarato che e' UN seed: la tabella riporta medie su 5 e i
        # numeri non coincidono (0.720 qui contro 0.7121 di media). Senza
        # questa nota sembra una discrepanza invece di una singola
        # realizzazione.
        ax.set_title(f"{titolo}\nmacro-F1 {res['macro_f1']:.3f} · "
                     f"kappa {res['quadratic_kappa']:.3f}  (seed 0)",
                     fontsize=10.5)
    fig.colorbar(im, ax=axes, fraction=.025, label="quota della riga")
    return _salva(fig, "fig3_confusioni")


# ==========================================================================
# 4. Il pre-training: rango e k-NN
# ==========================================================================
def fig_pretraining():
    """
    Il monitoraggio del collasso, che e' il modo in cui questo progetto
    poteva fallire silenziosamente. Si mostrano insieme rango effettivo e
    k-NN probe perche' il punto e' proprio che non coincidono.
    """
    p = os.path.join(FIG_DIR, "ijepa_vit_small_monitor.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        h = json.load(f)

    ep = [e["epoch"] for e in h]
    rango = [e["eff_rank"] for e in h]
    kep = [e["epoch"] for e in h if "knn_f1" in e]
    knn = [e["knn_f1"] for e in h if "knn_f1" in e]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(ep, rango, lw=2.2, color=BLU, label="rango effettivo (su 280)")
    ax.set_xlabel("epoca")
    ax.set_ylabel("rango effettivo", color=BLU)
    ax.tick_params(axis="y", labelcolor=BLU)
    ax.grid(alpha=.25)

    ax2 = ax.twinx()
    ax2.plot(kep, knn, marker="o", ms=6, lw=2.2, color=ARANCIO,
             label="k-NN probe (macro-F1)")
    ax2.axhline(0.2530, ls="--", c="black", lw=1.1)
    # A sinistra: a destra finirebbe sotto la legenda.
    ax2.text(ep[0] + 1, 0.2565, "pavimento 0.253", ha="left", fontsize=9)
    ax2.set_ylabel("k-NN probe, macro-F1", color=ARANCIO)
    ax2.tick_params(axis="y", labelcolor=ARANCIO)
    ax2.spines["top"].set_visible(False)

    linee = ax.get_lines() + ax2.get_lines()[:1]
    ax.legend(linee, [l.get_label() for l in linee], frameon=False,
              loc="upper left", bbox_to_anchor=(.02, .98))
    # Il titolo dice cio' che il grafico mostra davvero: il rango cresce in
    # modo continuo, il k-NN sale ma a gradini, con un plateau fra le epoche
    # 19 e 59 e un salto dopo. Le due misure non si muovono insieme, ed e'
    # il motivo per cui il rango da solo non basta a giudicare.
    ax.set_title("Rango e utilita' delle feature non crescono insieme",
                 fontsize=11.5)
    return _salva(fig, "fig4_pretraining")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Figure generate:")
    fig_alpha()
    fig_pretraining()
    fig_confusioni()
