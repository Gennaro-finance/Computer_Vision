"""
Train (stadio 1) - pre-training I-JEPA sui tile, con monitoraggio del collasso.

Sezione "Train" della struttura richiesta dal corso.

Uso:
    python train_ssl.py --variant vit_tiny --epochs 300
    python train_ssl.py --resume
    python train_ssl.py --smoke          # verifica che tutto girii, 20 step
"""

import argparse
import math
import os
import time

import torch

from data import LesionCropDataset, TileDataset, load_splits, make_loader, parse_annotations
from globals import (
    AMP, CKPT_DIR, GRAD_CLIP, MONITOR_SAMPLES, NUM_CLASSES, amp_dtype, DEFAULT_VARIANT, DEVICE, FIG_DIR, KNN_PROBE_EVERY, KNN_SUBSET,
    LESION_CROP_MODE, SSL_BATCH_SIZE, SSL_EMA_END, SSL_EMA_START, SSL_EPOCHS,
    SSL_LR, SSL_WARMUP_EPOCHS, SSL_WEIGHT_DECAY, TILE_SIZE,
)
from network import build_ijepa, count_params
from utils import (
    AverageMeter, CollapseMonitor, knn_probe, load_checkpoint, save_checkpoint,
    set_seed,
)


def ema_momentum(step, total_steps):
    """Momentum EMA con schedule da SSL_EMA_START a SSL_EMA_END (coseno)."""
    p = min(step / max(total_steps, 1), 1.0)
    return SSL_EMA_END - (SSL_EMA_END - SSL_EMA_START) * (math.cos(math.pi * p) + 1) / 2


@torch.no_grad()
def extract_for_probe(model, loader, max_items=KNN_SUBSET):
    """Feature medie sui token + etichette, per il k-NN probe."""
    model.eval()
    feats, labels = [], []
    n = 0
    for batch in loader:
        tokens = model.encode(batch["image"].to(DEVICE))
        feats.append(tokens.mean(dim=1).float().cpu())
        labels.append(batch["label"])
        n += tokens.shape[0]
        if n >= max_items:
            break
    model.train()
    return torch.cat(feats), torch.cat(labels)


def run_probe(model, records, splits):
    """
    k-NN probe: il segnale d'allarme piu' onesto.

    Se dopo ~100 epoche resta al livello della classe maggioritaria (0.612),
    il pre-training non sta imparando niente di utile. Meglio scoprirlo ora
    che il 5 settembre.
    """
    # num_workers=0: loader usa e getta. Con i worker persistenti, e' il
    # break dentro extract_for_probe a lasciarli vivi, e a ogni probe se ne
    # accumulano altri finche' la memoria condivisa non finisce.
    tr = make_loader(LesionCropDataset(records, splits["train"]),
                     batch_size=64, num_workers=0)
    va = make_loader(LesionCropDataset(records, splits["val"]),
                     batch_size=64, num_workers=0)
    ftr, ltr = extract_for_probe(model, tr)
    fva, lva = extract_for_probe(model, va)
    acc, f1 = knn_probe(ftr, ltr, fva, lva)

    # I riferimenti si MISURANO sulle etichette di validazione, non si
    # assumono: sono cio' che otterrebbe un modello che predice SEMPRE la
    # maggioritaria. Il valore dichiarato nel brief non coincide con quello
    # reale del dataset (vedi globals), e da questo numero dipende il gate.
    quota = torch.bincount(lva.long()).max().item() / len(lva)

    # Il verdetto si basa sulla MACRO-F1, non sull'accuracy. Il brief vieta
    # l'accuracy globale proprio perche' con il 61% di PAI 3 e' ingannevole:
    # un modello costante fa 0.61 di accuracy e sembra decente. La stessa
    # previsione costante fa invece macro-F1 = 2q/(1+q)/K, cioe' ~0.25 su tre
    # classi - ed e' quello il pavimento onesto da superare.
    # Sul vecchio criterio (acc > quota + 0.02) un encoder che imparava
    # davvero risultava "al livello del caso" mentre la macro-F1 saliva.
    f1_maggioritaria = (2 * quota / (1 + quota)) / NUM_CLASSES

    verdict = "OK" if f1 > f1_maggioritaria * 1.10 else "<-- AL LIVELLO DEL CASO"
    print(f"  [k-NN probe] acc={acc:.4f} macroF1={f1:.4f}  "
          f"(costante: acc={quota:.4f} macroF1={f1_maggioritaria:.4f}) {verdict}")
    return acc, f1


def probe_only(variant=DEFAULT_VARIANT, tag="", arch="ijepa"):
    """
    Rimisura la sonda k-NN su un checkpoint gia' addestrato, senza allenare.

    PERCHE' ESISTE. run_probe costruisce le sue feature con
    LesionCropDataset, quindi ogni numero di sonda prodotto prima di
    LESION_CROP_MODE='fixed' e' stato misurato attraverso il crop che
    annullava la scala della lesione - cioe' la caratteristica piu'
    predittiva del grado PAI (lato mediano 57/81/126 px per PAI 3/4/5).
    Quei verdetti non dicono quanto vale l'encoder: dicono quanto vale
    l'encoder visto da una lente che aveva gia' cancellato il segnale.

    La prova che erano sbagliati e' gia' agli atti: l'encoder che la sonda
    dava per fermo alla maggioritaria rende macro-F1 0.7415 a valle appena
    il crop e' corretto, contro 0.5302 con quello vecchio.

    Costa minuti e non tocca i pesi: va rilanciata sui checkpoint esistenti
    PRIMA di decidere se un altro pre-training serve davvero, e verso dove.
    """
    set_seed()
    run_name = f"{arch}_{variant}{('_' + tag) if tag else ''}"

    ckpt = load_checkpoint(run_name, map_location=DEVICE)
    if ckpt is None:
        raise FileNotFoundError(
            f"Nessun checkpoint '{run_name}' in {CKPT_DIR}. I .pt sono "
            f"gitignored: o lo generate qui, o ve lo fate passare da chi ha "
            f"lanciato il run."
        )

    model = build_ijepa(ckpt.get("variant", variant), arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model"])

    print(f"Checkpoint: {run_name}  (epoca {ckpt.get('epoch', '?')})")
    print(f"Crop del downstream: LESION_CROP_MODE={LESION_CROP_MODE!r}"
          f"{'  <-- scala preservata' if LESION_CROP_MODE == 'fixed' else '  <-- scala ANNULLATA, il numero non vale'}")

    records = parse_annotations(verbose=False)
    splits = load_splits()
    return run_probe(model, records, splits)


def train(variant=DEFAULT_VARIANT, epochs=SSL_EPOCHS, batch_size=SSL_BATCH_SIZE,
          resume=False, smoke=False, tag="", arch="ijepa"):
    set_seed()
    # Il tag tiene separati i checkpoint di run paralleli: senza, due varianti
    # lanciate insieme si sovrascrivono a vicenda lo stesso file.
    run_name = f"{arch}_{variant}{('_' + tag) if tag else ''}"

    records = parse_annotations(verbose=False)
    splits = load_splits()

    # Pre-training SOLO sui tile delle immagini di train, e SOLO dagli
    # originali: la cartella di augmentation del dataset e' ignorata (il SSL
    # fa le proprie augmentation e quelle pre-generate creano solo occasioni
    # di leakage).
    train_ds = TileDataset(records, splits["train"], augment=True)
    # Il dataset rende k crop per item da una sola decodifica, quindi il
    # batch va chiesto in ITEM e non in tile, altrimenti il batch effettivo
    # sarebbe k volte quello voluto (e andrebbe in OOM).
    k = train_ds.k
    loader = make_loader(train_ds, shuffle=True, batch_size=max(batch_size // k, 1))
    n_tile = len(train_ds) * k
    print(f"Tile di pre-training: {n_tile} da {len(splits['train'])} immagini "
          f"({len(train_ds)} item x {k} crop per decodifica)")
    print(f"Step per epoca: {len(loader)}  (batch = {max(batch_size // k, 1)} img x {k} = {max(batch_size // k, 1) * k} tile)")

    model = build_ijepa(variant, arch=arch).to(DEVICE)
    print(f"{variant}: {count_params(model)/1e6:.2f}M parametri addestrabili")

    # In I-JEPA si ottimizzano context encoder e predictor: il target
    # encoder segue per EMA e non deve ricevere gradienti. In LeJEPA c'e' un
    # solo encoder e va ottimizzato tutto, quindi si prendono direttamente i
    # parametri che richiedono gradiente.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=SSL_LR, weight_decay=SSL_WEIGHT_DECAY)

    total_steps = max(epochs * len(loader), 1)
    warmup = SSL_WARMUP_EPOCHS * len(loader)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    # Il GradScaler serve SOLO con float16: il bfloat16 ha il range dinamico
    # del float32 e non ha bisogno di scalare la loss. Abilitarlo comunque
    # non romperebbe nulla, ma maschera i problemi numerici veri.
    dtype = amp_dtype()
    use_amp = AMP and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    print(f"Precisione: {dtype} (GradScaler {'attivo' if scaler.is_enabled() else 'non necessario'})")

    monitor = CollapseMonitor()

    start_epoch, gstep = 0, 0
    if resume:
        ckpt = load_checkpoint(run_name, map_location=DEVICE)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch, gstep = ckpt["epoch"] + 1, ckpt["gstep"]
            monitor.history = ckpt.get("monitor", [])
            print(f"Ripreso dall'epoca {start_epoch}")

    for epoch in range(start_epoch, epochs):
        t_epoca = time.time()
        model.train()
        meter = AverageMeter()
        # Il monitor vuole abbastanza campioni: con un solo batch il rango
        # effettivo e' limitato dal batch, non dalla salute del modello.
        emb_epoca = []

        for step, batch in enumerate(loader):
            if smoke and step >= 20:
                break
            images = batch["image"].to(DEVICE, non_blocking=True)
            if images.dim() == 5:      # (B, k, C, H, W) -> (B*k, C, H, W)
                images = images.flatten(0, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                loss, embeddings = model(images)

            scaler.scale(loss).backward()

            # Clipping del gradiente: I-JEPA lo usa, e qui serve davvero.
            # Un picco di gradiente all'inizio del training manda la loss a
            # NaN e brucia una nottata di GPU senza avvisare.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.update_target(ema_momentum(gstep, total_steps))
            gstep += 1
            meter.update(loss.item(), images.size(0))

            # Segnale di vita: senza, un'epoca da decine di secondi sembra
            # un blocco. Poche righe per epoca, non intasa il log.
            if step % 25 == 0:
                el = time.time() - t_epoca
                print(f"    ep{epoch:03d} step {step:4d}/{len(loader)}  "
                      f"loss={loss.item():.4f}  {el:.0f}s", flush=True)

            if sum(e.shape[0] for e in emb_epoca) < MONITOR_SAMPLES:
                emb_epoca.append(embeddings.detach().float().cpu())

        knn = None
        if (epoch + 1) % KNN_PROBE_EVERY == 0 or epoch == epochs - 1:
            knn = run_probe(model, records, splits)

        if not emb_epoca:
            print("  [monitor] nessun batch elaborato: dataset vuoto?")
            break
        entry = monitor.update(epoch, meter.avg, torch.cat(emb_epoca), knn)
        dt = time.time() - t_epoca
        rimanenti = epochs - epoch - 1
        print(f"            epoca in {dt:.0f}s   restano {rimanenti} epoche "
              f"(~{rimanenti * dt / 3600:.1f} h)", flush=True)

        if monitor.is_collapsing():
            print("\n  COLLASSO RILEVATO. Non insistete: cambiate qualcosa.")
            print("  Ordine di intervento suggerito:")
            print("   1. abbassare il learning rate (fattore 3)")
            print("   2. alzare SSL_EMA_START verso 0.999")
            print("   3. ridurre la capacita' del predictor (e' troppo forte)")
            print("   4. passare al braccio SIGReg (network.sigreg_loss)")
            break

        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "gstep": gstep, "variant": variant,
            "monitor": monitor.history,
        }, run_name)

    monitor.save(os.path.join(FIG_DIR, f"{run_name}_monitor.json"))
    print(f"\nFigura monitoraggio: {monitor.plot(f'{run_name}_monitor')}")
    print(f"Checkpoint: {CKPT_DIR}/{run_name}.pt")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--epochs", type=int, default=SSL_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=SSL_BATCH_SIZE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="carica il checkpoint e rimisura SOLO la sonda k-NN, "
                         "senza allenare: serve a rileggere i run fatti col crop rotto")
    ap.add_argument("--smoke", action="store_true")
    # Override per gli esperimenti in parallelo. Restano fuori da globals.py
    # apposta: globals descrive la configurazione di riferimento, questi
    # servono a esplorarne le varianti senza toccarla.
    ap.add_argument("--tag", default="", help="suffisso del run (checkpoint separati)")
    ap.add_argument("--context-scale", type=float, nargs=2, default=None,
                    help="es. 0.4 0.7 - contesto piu' piccolo = compito piu' difficile")
    ap.add_argument("--target-scale", type=float, nargs=2, default=None)
    ap.add_argument("--sigreg-lambda", type=float, default=None)
    ap.add_argument("--predictor-dim", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--arch", default="ijepa", choices=["ijepa", "lejepa"],
                    help="ijepa = obiettivo 1; lejepa = braccio di confronto (ref [2])")
    a = ap.parse_args()

    # Si scrivono nei moduli che li leggono a ogni chiamata, cosi' l'override
    # vale per l'intero run senza duplicare la configurazione.
    import network
    import data as data_mod
    if a.context_scale:
        network.CONTEXT_SCALE = tuple(a.context_scale)
    if a.target_scale:
        network.TARGET_SCALE = tuple(a.target_scale)
    if a.sigreg_lambda is not None:
        network.SIGREG_LAMBDA = a.sigreg_lambda
    if a.predictor_dim is not None:
        network.PREDICTOR_DIM = a.predictor_dim
    if a.workers is not None:
        data_mod.NUM_WORKERS = a.workers

    print(f"[config] arch={a.arch} tag={a.tag or '-'} context={network.CONTEXT_SCALE} "
          f"target={network.TARGET_SCALE} sigreg_lambda={network.SIGREG_LAMBDA} "
          f"predictor_dim={network.PREDICTOR_DIM} workers={data_mod.NUM_WORKERS}")
    if a.probe_only:
        probe_only(a.variant, a.tag, a.arch)
    else:
        train(a.variant, a.epochs, a.batch_size, a.resume, a.smoke, a.tag, a.arch)
