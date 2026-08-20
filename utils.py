"""
Utils - seed, checkpoint, monitoraggio del collasso, k-NN probe, plot.

Sezione "Utils" della struttura richiesta dal corso.
"""

import json
import os
import random

import numpy as np
import torch

from globals import CKPT_DIR, FIG_DIR, KNN_K, SEED


# --------------------------------------------------------- riproducibilita'
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# -------------------------------------------------------------- checkpoint
# Il pre-training SSL supera facilmente le 12 h di una sessione Kaggle
# (300 epoche ~ 6 h, 600 ~ 12.5 h). Il resume non e' opzionale.
def save_checkpoint(state: dict, name: str) -> str:
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    torch.save(state, path)
    return path


def load_checkpoint(name: str, map_location=None):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ==========================================================================
# MONITORAGGIO DEL COLLASSO
# ==========================================================================
# Il collasso della rappresentazione e' il modo in cui questo progetto
# fallisce, ed e' silenzioso: la loss I-JEPA scende regolarmente mentre tutti
# gli embedding convergono a una costante, perche' predire un target costante
# e' banale. Se ve ne accorgete il 5 settembre, il progetto e' finito.
#
# Queste tre funzioni vanno chiamate a ogni epoca. Costano nulla.
# ==========================================================================
@torch.no_grad()
def embedding_std(embeddings: torch.Tensor) -> float:
    """
    Deviazione standard media per dimensione, su embedding L2-normalizzati.

    Collasso completo -> 0. Riferimento sano: per embedding normalizzati di
    dimensione d, una distribuzione isotropa da' circa 1/sqrt(d).
    """
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    return z.std(dim=0).mean().item()


@torch.no_grad()
def effective_rank(embeddings: torch.Tensor) -> float:
    """
    Rango effettivo via participation ratio: (sum l)^2 / sum l^2 sugli
    autovalori della matrice dei momenti secondi.

    Dice quante direzioni dello spazio sono realmente usate.
    Collasso -> 1. Isotropia perfetta su d dimensioni -> d.

    DUE DETTAGLI DI IMPLEMENTAZIONE CHE CONTANO, e su cui e' facile
    sbagliare (ci sono cascato scrivendo la prima versione):

    1. Si L2-normalizza prima. Senza, la metrica confonde "embedding tutti
       nella stessa direzione ma di norma diversa" con una distribuzione
       ricca: la varianza radiale gonfia il rango.

    2. NON si centra sulla media. Usando la covarianza centrata, embedding
       tutti identici piu' un epsilon di rumore danno rango ~d invece di 1,
       perche' dopo il centraggio resta solo il rumore, che e' isotropo.
       Il collasso costante - cioe' la modalita' di collasso PIU' CLASSICA -
       passerebbe inosservato. Con la matrice dei momenti secondi, vettori
       identici danno una matrice di rango 1 e il participation ratio va a 1
       come deve.

    Verificato su quattro casi noti nel blocco __main__ di questo file.
    """
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    second_moment = (z.T @ z) / max(z.shape[0], 1)
    eigvals = torch.linalg.eigvalsh(second_moment).clamp(min=0)
    s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
    return (s1 ** 2 / s2).item() if s2 > 0 else 1.0


@torch.no_grad()
def rank_reference(n_samples: int, dim: int, seed: int = 0) -> float:
    """
    Rango effettivo di embedding perfettamente sani, con QUELLO stesso numero
    di campioni e dimensioni.

    Serve perche' il rango effettivo e' limitato dal numero di campioni, non
    solo dal collasso: con 128 campioni in 384 dimensioni, embedding isotropi
    danno ~97, non 384. Senza questo riferimento un valore di 97 sembra un
    collasso in corso quando invece e' il massimo ottenibile.

    Il rapporto misurato/riferimento e' la quantita' leggibile: ~1.0 sano,
    vicino a 0 collassato.
    """
    g = torch.Generator().manual_seed(seed)
    return effective_rank(torch.randn(n_samples, dim, generator=g))


@torch.no_grad()
def knn_probe(train_feats, train_labels, test_feats, test_labels, k=KNN_K):
    """
    Probe k-NN sulle feature congelate - il segnale d'allarme piu' onesto.

    Ogni ~20 epoche: se dopo 100 epoche resta al livello della classe
    maggioritaria (0.612 per PAI 3), il pre-training non sta imparando
    niente di utile e va cambiato qualcosa PRIMA di aver bruciato giorni.

    Ritorna: (accuracy, macro-F1). Guardate la macro-F1: con 61% di PAI 3
    l'accuracy da' sola un'impressione ottimistica.
    """
    tr = torch.nn.functional.normalize(train_feats.float(), dim=-1)
    te = torch.nn.functional.normalize(test_feats.float(), dim=-1)
    sims = te @ tr.T
    idx = sims.topk(min(k, tr.shape[0]), dim=-1).indices
    votes = train_labels[idx]

    n_cls = int(max(train_labels.max().item(), test_labels.max().item())) + 1
    onehot = torch.zeros(votes.shape[0], n_cls)
    for c in range(n_cls):
        onehot[:, c] = (votes == c).sum(dim=-1).float()
    pred = onehot.argmax(dim=-1)

    acc = (pred == test_labels).float().mean().item()

    f1s = []
    for c in range(n_cls):
        tp = ((pred == c) & (test_labels == c)).sum().item()
        fp = ((pred == c) & (test_labels != c)).sum().item()
        fn = ((pred != c) & (test_labels == c)).sum().item()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    return acc, float(np.mean(f1s))


class CollapseMonitor:
    """
    Tiene la storia dei segnali di collasso e avvisa quando degradano.

    Uso nel loop di training:
        mon = CollapseMonitor()
        ...
        mon.update(epoch, loss, embeddings)
        if mon.is_collapsing():
            print("Fermarsi e cambiare qualcosa.")
    """

    def __init__(self, std_floor=1e-3, rank_ratio_floor=0.15, min_epoch=None):
        self.history = []
        self.std_floor = std_floor
        # Soglia sul RAPPORTO rango misurato / rango di riferimento, non sul
        # valore assoluto: vedi rank_reference().
        self.rank_ratio_floor = rank_ratio_floor
        # Guardia di warmup: all'inizio del training gli embedding sono
        # legittimamente quasi identici - la rete non ha ancora imparato a
        # distinguere niente, quindi il rango effettivo parte basso. Senza
        # questa guardia is_collapsing() scatta all'epoca 2 e interrompe
        # ogni run prima che possa convergere.
        #
        # Il valore era SSL_WARMUP_EPOCHS (15), cioe' esattamente la fine del
        # warmup: la guardia si armava nell'istante in cui il LR arrivava al
        # massimo e uccideva ogni run all'epoca 15 senza un solo step a LR
        # pieno. Ora usa COLLAPSE_MIN_EPOCH, allineato al criterio dichiarato
        # dal progetto (~100 epoche). Vedi il commento in globals.py.
        from globals import COLLAPSE_MIN_EPOCH
        self.min_epoch = COLLAPSE_MIN_EPOCH if min_epoch is None else min_epoch

    def update(self, epoch, loss, embeddings, knn=None):
        n, d = embeddings.shape[0], embeddings.shape[-1]
        std = embedding_std(embeddings)
        rank = effective_rank(embeddings)
        rif = rank_reference(n, d)
        ratio = rank / rif if rif > 0 else 0.0

        entry = {"epoch": epoch, "loss": float(loss), "std": std,
                 "eff_rank": rank, "rank_ref": rif, "rank_ratio": ratio,
                 "n": n, "dim": d}
        if knn is not None:
            entry["knn_acc"], entry["knn_f1"] = knn
        self.history.append(entry)

        flag = ""
        if std < self.std_floor or ratio < self.rank_ratio_floor:
            flag = "   <-- COLLASSO"
        print(f"  [monitor] ep{epoch:03d} loss={loss:.4f} std={std:.5f} "
              f"rango={rank:.0f}/{rif:.0f} ({ratio:.0%} del sano){flag}")
        return entry

    def is_collapsing(self, patience=None):
        """
        True solo se i segnali restano degradati per `patience` epoche
        consecutive E siamo oltre il warmup. Le due condizioni servono
        entrambe: la prima evita i falsi allarmi da rumore di una singola
        epoca, la seconda evita di scambiare l'inizializzazione per collasso.

        Terza condizione, aggiunta dopo i run del 19 ago: se i segnali stanno
        MIGLIORANDO non e' collasso, e' apprendimento lento. Un run che sale
        da rango 1 a rango 4 sta uscendo dal collasso, non entrandoci, e
        interromperlo butta via l'unica cosa che stava funzionando.
        """
        if patience is None:
            from globals import COLLAPSE_PATIENCE
            patience = COLLAPSE_PATIENCE
        if len(self.history) < patience:
            return False
        if self.history[-1]["epoch"] < self.min_epoch:
            return False
        recent = self.history[-patience:]

        degradati = all(e["std"] < self.std_floor
                        or e["rank_ratio"] < self.rank_ratio_floor
                        for e in recent)
        if not degradati:
            return False

        # I segnali sono bassi, ma stanno salendo? Allora il modello sta
        # uscendo dal collasso e va lasciato lavorare. Si confronta la prima
        # meta' della finestra con la seconda: serve un miglioramento netto
        # (>5%) di almeno uno dei due segnali, non il rumore di un'epoca.
        meta = max(len(recent) // 2, 1)
        for chiave in ("rank_ratio", "std"):
            prima = sum(e[chiave] for e in recent[:meta]) / meta
            dopo = sum(e[chiave] for e in recent[meta:]) / max(len(recent) - meta, 1)
            if dopo > prima * 1.05:
                return False
        return True

    def save(self, path):
        save_json(self.history, path)

    def plot(self, name="collapse_monitor"):
        import matplotlib.pyplot as plt

        if not self.history:
            return None
        ep = [e["epoch"] for e in self.history]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        axes[0].plot(ep, [e["loss"] for e in self.history], color="#1f77b4")
        axes[0].set_title("Loss I-JEPA")
        axes[1].plot(ep, [e["std"] for e in self.history], color="#d62728")
        axes[1].axhline(self.std_floor, ls="--", c="gray", lw=1)
        axes[1].set_title("Std degli embedding")
        axes[2].plot(ep, [100 * e["rank_ratio"] for e in self.history], color="#2ca02c")
        axes[2].axhline(100 * self.rank_ratio_floor, ls="--", c="gray", lw=1)
        axes[2].set_ylim(0, 110)
        axes[2].set_title("Rango effettivo (% del sano)")
        for a in axes:
            a.set_xlabel("epoca")
            a.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return path


if __name__ == "__main__":
    # Verifica del monitoraggio del collasso su casi con esito noto.
    # Lanciare con: python utils.py
    print("=== effective_rank / embedding_std su casi noti ===")
    D, N = 192, 512
    torch.manual_seed(0)
    casi = {
        "isotropo (sano)":        torch.randn(N, D),
        "costante (collasso)":    torch.randn(1, D).repeat(N, 1) + 1e-6 * torch.randn(N, D),
        "rango 1":                torch.randn(N, 1) @ torch.randn(1, D),
        "rango 8":                torch.randn(N, 8) @ torch.randn(8, D),
    }
    attesi = {"isotropo (sano)": "~D", "costante (collasso)": "~1",
              "rango 1": "~1", "rango 8": "~8"}
    for nome, e in casi.items():
        print(f"  {nome:22s} std={embedding_std(e):.6f}  "
              f"eff_rank={effective_rank(e):7.2f} / {D}   atteso {attesi[nome]}")

    print("\n=== CollapseMonitor: loss che scende, embedding collassati ===")
    # min_epoch=0 per testare la logica di rilevamento: con il default la
    # guardia di warmup sopprime l'allarme nelle prime epoche, ed e' giusto
    # cosi' in training ma qui maschererebbe il test.
    # Servono almeno COLLAPSE_PATIENCE epoche perche' la guardia possa
    # esprimersi: con meno campioni is_collapsing() si astiene per design.
    from globals import COLLAPSE_PATIENCE

    n_ep = COLLAPSE_PATIENCE + 1
    mon = CollapseMonitor(min_epoch=0)
    for ep in range(n_ep):
        mon.update(ep, 0.5 / (ep + 1), casi["costante (collasso)"])
    print(f"  is_collapsing() = {mon.is_collapsing()}   (atteso True)")
    print("  ^ la loss scendeva regolarmente: e' il fallimento silenzioso.")

    mon2 = CollapseMonitor(min_epoch=0)
    for ep in range(n_ep):
        mon2.update(ep, 0.5 / (ep + 1), casi["isotropo (sano)"])
    print(f"  su embedding sani: is_collapsing() = {mon2.is_collapsing()}   (atteso False)")

    # Terzo caso, quello che ha causato le interruzioni premature del 19 ago:
    # segnali bassi ma in MIGLIORAMENTO. Non e' collasso, e' apprendimento
    # lento, e la guardia non deve interrompere.
    mon3 = CollapseMonitor(min_epoch=0)
    for ep in range(n_ep):
        k = 1 + ep                     # rango che cresce 1, 2, 3, ...
        e = torch.randn(N, k) @ torch.randn(k, D)
        mon3.update(ep, 0.5 / (ep + 1), e)
    print(f"  su segnali BASSI ma in salita: is_collapsing() = {mon3.is_collapsing()}   (atteso False)")
