"""
verify_setup - controllo completo dell'ambiente prima di iniziare.

Gira sia in locale su Windows sia su Kaggle. Non richiede il dataset: se non
c'e', lo segnala come passo successivo invece di fallire.

Uso:
    python verify_setup.py
"""

import importlib
import os
import platform
import sys
import traceback

OK, KO, WARN = "[ OK ]", "[FAIL]", "[ !! ]"
_esiti = []


def check(nome, fn, critico=True):
    try:
        msg = fn()
        print(f"{OK} {nome:42s} {msg if msg else ''}")
        _esiti.append(True)
        return True
    except Exception as exc:
        tag = KO if critico else WARN
        print(f"{tag} {nome:42s} {type(exc).__name__}: {exc}")
        _esiti.append(not critico)
        return False


# ==========================================================================
def c_python():
    v = sys.version_info
    if v < (3, 9):
        raise RuntimeError(f"Python {v.major}.{v.minor} troppo vecchio, serve >= 3.9")
    return f"{v.major}.{v.minor}.{v.micro} su {platform.system()}"


def c_venv():
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv and not os.path.isdir("/kaggle"):
        raise RuntimeError("non sei in un virtual environment "
                           "(attiva .venv\\Scripts\\Activate.ps1)")
    return os.path.basename(sys.prefix)


def c_pacchetti():
    mancanti = []
    versioni = []
    for mod, nome in [("torch", "torch"), ("torchvision", "torchvision"),
                      ("numpy", "numpy"), ("PIL", "pillow"),
                      ("matplotlib", "matplotlib")]:
        try:
            m = importlib.import_module(mod)
            versioni.append(f"{nome} {getattr(m, '__version__', '?')}")
        except ImportError:
            mancanti.append(nome)
    if mancanti:
        raise RuntimeError(f"mancano: {', '.join(mancanti)}")
    return " | ".join(versioni)


def c_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: girera' su CPU (lentissimo per il SSL)")
    p = torch.cuda.get_device_properties(0)
    if p.major >= 8:
        amp = "bf16 supportato (niente GradScaler)"
    elif p.major == 7:
        amp = "fp16 con GradScaler"
    else:
        amp = "Pascal: AMP rende poco"
    return f"{p.name} {p.total_memory/1024**3:.1f} GB, cc {p.major}.{p.minor}, {amp}"


def c_moduli():
    for m in ("globals", "utils", "data", "network", "imbalance",
              "train_ssl", "train_downstream", "evaluation"):
        importlib.import_module(m)
    return "8 moduli importati"


def c_collasso():
    """Il monitoraggio del collasso deve distinguere 4 casi noti."""
    import torch
    from utils import effective_rank, embedding_std

    D, N = 192, 512
    torch.manual_seed(0)
    costante = torch.randn(1, D).repeat(N, 1) + 1e-6 * torch.randn(N, D)
    isotropo = torch.randn(N, D)
    rango8 = torch.randn(N, 8) @ torch.randn(8, D)

    r_cost, r_iso, r_8 = map(effective_rank, (costante, isotropo, rango8))
    assert r_cost < 2, f"collasso costante non rilevato (rango {r_cost:.1f})"
    assert r_iso > 50, f"embedding sani segnalati come collassati (rango {r_iso:.1f})"
    assert 5 < r_8 < 12, f"rango 8 stimato male ({r_8:.1f})"
    assert embedding_std(costante) < 1e-3
    return f"costante={r_cost:.1f}  rango8={r_8:.1f}  isotropo={r_iso:.0f}/{D}"


def c_ijepa():
    """Forward completo di I-JEPA: se questo passa, la pipeline regge."""
    import torch
    import globals as G
    from network import bbox_to_token_mask, build_ijepa, count_params

    m = build_ijepa(G.DEFAULT_VARIANT)
    loss, emb = m(torch.randn(2, 3, G.TILE_SIZE, G.TILE_SIZE))
    assert torch.isfinite(loss), "loss non finita"
    assert emb.shape == (2, m.embed_dim)

    n_tok = bbox_to_token_mask(torch.tensor([[40., 40., 120., 120.]]), m.grid).sum().item()
    assert n_tok > 0
    return (f"{G.DEFAULT_VARIANT} {count_params(m)/1e6:.2f}M, griglia {m.grid}x{m.grid}, "
            f"loss={loss.item():.3f}, {n_tok} token in bbox")


def c_metriche():
    """Le metriche devono dare il valore atteso su casi noti."""
    import numpy as np
    from evaluation import confusion_matrix, macro_f1, pr_auc, quadratic_weighted_kappa

    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    assert abs(macro_f1(confusion_matrix(y, y)) - 1.0) < 1e-9
    assert abs(quadratic_weighted_kappa(y, y) - 1.0) < 1e-9
    assert abs(quadratic_weighted_kappa(y, np.zeros_like(y))) < 1e-9

    vicino, lontano = y.copy(), y.copy()
    vicino[y == 2] = 1
    lontano[y == 2] = 0
    k_v, k_l = quadratic_weighted_kappa(y, vicino), quadratic_weighted_kappa(y, lontano)
    assert k_v > k_l, "il kappa quadratico non penalizza l'errore a due gradi"
    return f"kappa 1-grado={k_v:.3f} > 2-gradi={k_l:.3f}, PR-AUC ok"


def c_sbilanciamento():
    import torch
    from imbalance import class_counts, compute_loss, inverse_frequency_weights

    lab = torch.cat([torch.zeros(3691), torch.ones(1817), torch.full((521,), 2)]).long()
    c = class_counts(lab)
    ratio = (c.max() / c.min()).item()
    assert abs(ratio - 7.08) < 0.05, f"rapporto {ratio:.2f}, atteso 7.08"
    w = inverse_frequency_weights(lab)
    assert w[2] > w[0], "i pesi non favoriscono la minoritaria"
    for metodo in ("none", "class_weighted", "focal"):
        loss = compute_loss(torch.randn(8, 3), torch.randint(0, 3, (8,)), metodo,
                            train_labels=lab)
        assert torch.isfinite(loss)
    return f"sbilanciamento {ratio:.2f}:1, pesi e loss coerenti"


def c_dataset():
    import globals as G
    if not os.path.isdir(G.DATA_ROOT):
        raise FileNotFoundError(
            f"{G.DATA_ROOT} non esiste - scaricatelo da "
            "data.mendeley.com/datasets/kx52tk2ddj/3")
    import glob
    tutto = glob.glob(os.path.join(G.DATA_ROOT, "**", "*"), recursive=True)
    n_xml = len(glob.glob(os.path.join(G.DATA_ROOT, "**", "*.xml"), recursive=True))
    n_img = len(glob.glob(os.path.join(G.DATA_ROOT, "**", "*.jpg"), recursive=True))

    if not tutto:
        raise FileNotFoundError(
            "la cartella e' vuota - il dataset non e' ancora stato scaricato")
    if n_xml == 0:
        raise FileNotFoundError(
            f"{len(tutto)} file presenti ma nessun XML: probabilmente l'archivio "
            "e' stato scompattato in una sottocartella sbagliata")
    return f"{n_xml} XML, {n_img} JPG"


# ==========================================================================
def main():
    print("=" * 78)
    print(" Verifica ambiente - Progetto 8 (JEPA / grading PAI)")
    print("=" * 78)

    print("\n--- Ambiente ---")
    check("Versione Python", c_python)
    check("Virtual environment", c_venv, critico=False)
    check("Pacchetti richiesti", c_pacchetti)
    check("GPU CUDA", c_gpu, critico=False)

    print("\n--- Codice del progetto ---")
    if not check("Import dei moduli", c_moduli):
        print("\nGli import falliscono: il resto non ha senso. "
              "Controllate di essere nella cartella del progetto.")
        return 1
    check("Rilevamento del collasso", c_collasso)
    check("Forward I-JEPA", c_ijepa)
    check("Metriche ordinali", c_metriche)
    check("Gestione dello sbilanciamento", c_sbilanciamento)

    print("\n--- Dati ---")
    ha_dati = check("Dataset periapicale", c_dataset, critico=False)

    print("\n" + "=" * 78)
    if all(_esiti):
        print(" AMBIENTE PRONTO")
        if ha_dati:
            print("\n Prossimi passi:")
            print("   python data.py --inspect")
            print("   python data.py --bbox-stats     <- decide l'architettura")
            print("   python data.py --splits")
            print("   python train_ssl.py --smoke --epochs 1 --batch-size 8")
        else:
            print("\n Manca solo il dataset:")
            print("   1. scaricatelo da data.mendeley.com/datasets/kx52tk2ddj/3")
            print("   2. scompattatelo in data\\periapical\\")
            print("   3. rilanciate questo script")
    else:
        print(" CI SONO PROBLEMI - vedi le righe [FAIL] sopra")
    print("=" * 78)
    return 0 if all(_esiti) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
