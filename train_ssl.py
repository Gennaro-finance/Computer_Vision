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
from sklearn.linear_model import LogisticRegression

from data import LesionCropDataset, TileDataset, load_splits, make_loader, parse_annotations
from globals import (
    AMP, CKPT_DIR, GRAD_CLIP, MONITOR_SAMPLES, NUM_CLASSES, amp_dtype, DEFAULT_VARIANT, DEVICE, FIG_DIR, KNN_PROBE_EVERY, KNN_SUBSET,
    GATE_CROLLO, GATE_EPOCH, GATE_MARGINE, GATE_SONDE_SOTTO, OUT_DIR, SSL_BATCH_SIZE, SSL_EMA_END, SSL_EMA_START,
    SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS, SSL_WEIGHT_DECAY, TILE_SIZE,
)
from evaluation import confusion_matrix, macro_f1
import network
from network import bbox_to_token_mask, build_ijepa, count_params

# Sovrascrivibili da riga di comando: sono i parametri che si esplorano per
# uscire dal collasso, e vanno cambiati senza toccare globals.py, che
# descrive la configurazione di riferimento.
EMA_START = SSL_EMA_START
EMA_END = SSL_EMA_END
LR = SSL_LR
GATE_AT = GATE_EPOCH
from utils import (
    AverageMeter, CollapseMonitor, knn_probe, load_checkpoint, save_checkpoint,
    set_seed,
)


def ema_momentum(step, total_steps):
    """Momentum EMA con schedule da EMA_START a EMA_END (coseno).

    EMA_START e' la leva principale contro il collasso: piu' e' alto, piu'
    il target encoder e' lento, e piu' e' difficile per il context encoder
    inseguirlo fino alla soluzione costante.
    """
    p = min(step / max(total_steps, 1), 1.0)
    return EMA_END - (EMA_END - EMA_START) * (math.cos(math.pi * p) + 1) / 2


@torch.no_grad()
def extract_for_probe(model, loader, max_items=KNN_SUBSET):
    """
    Feature aggregate sui token DENTRO LA BBOX, piu' le etichette.

    PERCHE' LA MASCHERA. La versione precedente mediava tutti i 196 token
    dell'immagine. Funzionava per caso: col vecchio crop 'relative' la
    lesione occupava sempre un terzo esatto del riquadro, quindi la media
    portava comunque segnale. Col crop a finestra fissa la lesione e' 8-20
    token su 196 e la media e' dominata dallo sfondo: rimisurando i
    checkpoint esistenti il 22 ago, tutti crollavano a ridosso del pavimento
    (0.27-0.28 contro 0.2530) mentre gli stessi encoder rendono 0.7415 a
    valle, dove l'attention pooling usa la bbox.

    Non era l'encoder a essere peggiorato: era la sonda a misurare
    prevalentemente osso sano. Qui si aggrega come fa il downstream, cosi'
    la sonda torna a essere un anticipo di quel numero e non di altro.
    """
    model.eval()
    feats, labels = [], []
    n = 0
    for batch in loader:
        tokens = model.encode(batch["image"].to(DEVICE))
        msk = bbox_to_token_mask(batch["bbox"].to(DEVICE), model.grid)
        w = msk.float().unsqueeze(-1)
        pooled = (tokens * w).sum(1) / w.sum(1).clamp(min=1)
        feats.append(pooled.float().cpu())
        labels.append(batch["label"])
        n += tokens.shape[0]
        if n >= max_items:
            break
    model.train()
    return torch.cat(feats), torch.cat(labels)


def probe_lineare(ftr, ltr, fva, lva):
    """
    Sonda LINEARE: una direzione APPRESA nello spazio delle feature.

    E' la sonda che conta, e sostituisce il k-NN come criterio di giudizio.

    PERCHE' IL k-NN ERA LA MISURA SBAGLIATA. Il k-NN classifica per
    DISTANZA euclidea, quindi premia una geometria precisa: che i vicini
    piu' prossimi abbiano lo stesso grado. Un addestramento puo' conservare
    tutta l'informazione e cambiare la geometria - riallocando la varianza
    su direzioni che servono al compito di pre-training - e il k-NN crolla
    lo stesso.

    E' esattamente cio' che succede qui. Misurato il 22 ago sul checkpoint
    dell'epoca 40, contro l'encoder casuale:

        informazione presente (R^2 di una regressione lineare)
          intensita' media nella lesione   0.991 -> 0.992
          log area della bbox              0.886 -> 0.884
        lettura della stessa rappresentazione
          k-NN     (distanza)              0.7299 -> 0.5422
          lineare  (appresa)               0.7212 -> 0.7329

    L'informazione e' intatta. Solo la distanza euclidea smette di
    rifletterla. E il downstream di questo progetto NON usa distanze: usa
    attention pooling piu' una testa addestrata, cioe' una lettura appresa.
    Il k-NN misurava quindi una proprieta' che al progetto non serve, e ha
    fatto interrompere run che stavano migliorando.
    """
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-8
    ztr, zva = ((ftr - mu) / sd).numpy(), ((fva - mu) / sd).numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(ztr, ltr.numpy())
    return macro_f1(confusion_matrix(lva.numpy(), clf.predict(zva)))


def run_probe(model, records, splits):
    """
    Sonda di monitoraggio. Riporta ENTRAMBE le letture e giudica sulla
    lineare: vedi probe_lineare() per il perche'.
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
    f1_lin = probe_lineare(ftr, ltr, fva, lva)

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

    verdict = "OK" if f1_lin > f1_maggioritaria * 1.10 else "<-- AL LIVELLO DEL CASO"
    print(f"  [sonda] lineare={f1_lin:.4f}  k-NN={f1:.4f}  "
          f"(costante {f1_maggioritaria:.4f}) {verdict}")
    return f1_lin, f1   # (criterio, diagnostica)


def train(variant=DEFAULT_VARIANT, epochs=SSL_EPOCHS, batch_size=SSL_BATCH_SIZE,
          resume=False, smoke=False, tag=""):
    set_seed()
    # Il tag tiene separati i checkpoint di run paralleli: senza, due varianti
    # lanciate insieme si sovrascrivono a vicenda lo stesso file.
    run_name = f"ijepa_{variant}{('_' + tag) if tag else ''}"

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

    model = build_ijepa(variant).to(DEVICE)
    print(f"{variant}: {count_params(model)/1e6:.2f}M parametri addestrabili")
    print(f"[iperparametri] lr={LR:.2e} ema={EMA_START}->{EMA_END} "
          f"predictor_dim={network.PREDICTOR_DIM}")

    # In I-JEPA si ottimizzano context encoder e predictor: il target
    # encoder segue per EMA (obiettivo 1 del brief) e non riceve gradienti,
    # quindi si prendono solo i parametri che li richiedono.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=SSL_WEIGHT_DECAY)

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

    # TensorBoard: curve dal vivo. Mentre un run gira, in un altro terminale:
    #     .venv\Scripts\tensorboard --logdir runs\tb
    # poi apri http://localhost:6006. Scrive per epoca loss, rango effettivo,
    # rapporto di rango, std, k-NN, durata e temperatura GPU. Costa nulla, e se
    # tensorboard non e' installato il run prosegue senza.
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(OUT_DIR, "tb", run_name))
    except Exception as e:
        print(f"  [tensorboard non attivo: {e}]")
        tb = None

    start_epoch, gstep = 0, 0
    knn_ref = None
    if resume:
        ckpt = load_checkpoint(run_name, map_location=DEVICE)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch, gstep = ckpt["epoch"] + 1, ckpt["gstep"]
            monitor.history = ckpt.get("monitor", [])
            knn_ref = ckpt.get("knn_ref")
            print(f"Ripreso dall'epoca {start_epoch}")

            # Se il checkpoint e' stato scritto con iperparametri diversi da
            # quelli attivi ora, si RIFIUTA di proseguire: continuare
            # silenziosamente produrrebbe un run ibrido, impossibile da
            # descrivere in presentazione.
            atteso = {"lr": LR, "ema_start": EMA_START,
                      "predictor_dim": network.PREDICTOR_DIM}
            diverso = {k: (ckpt[k], v) for k, v in atteso.items()
                       if k in ckpt and ckpt[k] != v}
            if diverso:
                print("ERRORE: il checkpoint usa iperparametri diversi da questi.")
                for k, (era, ora) in diverso.items():
                    print(f"  {k}: checkpoint={era}  riga di comando={ora}")
                print("Rilanciate con gli stessi valori, oppure senza --resume")
                print("e con un --tag nuovo per iniziare un run separato.")
                raise SystemExit(1)

    # RIFERIMENTO: la sonda k-NN sull'encoder non ancora addestrato. E' il
    # "modello casuale" - la cosa da battere. Misurarlo QUI, con lo stesso
    # protocollo usato dopo, e' l'unico modo per sapere se il pre-training
    # aggiunge o toglie. Costa una manciata di secondi.
    if knn_ref is None:
        print(""
              "Riferimento (encoder casuale, pesi non addestrati):")
        knn_ref = run_probe(model, records, splits)[0]
    print(f"Da battere: macro-F1 della sonda LINEARE = {knn_ref:.4f}"
          f"   (cancello: se a {GATE_AT} epoche non e' superato, ci si ferma)")
    knn_best = 0.0
    sotto = 0

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

        if tb is not None:
            tb.add_scalar("loss", meter.avg, epoch)
            # Si logga qualunque scalare numerico ci sia nella voce del
            # monitor (std, eff_rank, rank_ratio...), senza assumerne i nomi.
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    tb.add_scalar(f"monitor/{k}", v, epoch)
            if knn is not None:
                tb.add_scalar("sonda/lineare", knn[0], epoch)
                tb.add_scalar("sonda/knn", knn[1], epoch)
            tb.add_scalar("sistema/epoca_secondi", dt, epoch)
            tb.add_scalar("sistema/learning_rate", scheduler.get_last_lr()[0], epoch)
            tb.flush()

        if knn is not None:
            # CHECKPOINT MIGLIORE, tenuto a parte.
            # Il checkpoint normale viene sovrascritto a ogni epoca: se la
            # sonda peggiora - e in questo progetto peggiora sempre, da un
            # certo punto in poi - alla fine resta salvato l'encoder PEGGIORE
            # e quello buono e' perduto. Successo il 22 ago: la sonda
            # migliore era all'epoca 10 (0.7067) ma sul disco e' rimasta
            # l'epoca 39 (0.4280).
            #
            # Il modello che si consegna e' questo, non l'ultimo: nulla nel
            # brief chiede di addestrare fino all'ultima epoca, e scegliere
            # sulla base di una sonda misurata sul VALIDATION e' un criterio
            # di selezione onesto, da dichiarare in presentazione.
            if knn[0] > knn_best:
                save_checkpoint({
                    "model": model.state_dict(), "epoch": epoch,
                    "gstep": gstep, "variant": variant,
                    "probe_lineare": knn[0], "probe_knn": knn[1],
                    "probe_ref": knn_ref,
                    "lr": LR, "ema_start": EMA_START,
                    "predictor_dim": network.PREDICTOR_DIM,
                }, run_name + "_best")
                print(f"            [migliore] nuovo record {knn[0]:.4f} "
                      f"all'epoca {epoch}: salvato {run_name}_best")
            knn_best = max(knn_best, knn[0])
            delta = knn[0] - knn_ref
            soglia = knn_ref - GATE_MARGINE

            # Si giudica la sonda CORRENTE, non la migliore mai vista.
            # La versione precedente confrontava knn_best con la soglia: una
            # sola sonda fortunata all'inizio disarmava il cancello per
            # sempre. Successo il 22 ago - k-NN 0.7067 all'epoca 10, poi
            # 0.6520, 0.5548, 0.4280 - e il run e' proseguito verso altre
            # 4.7 ore di degrado con il cancello che taceva, perche' il
            # massimo storico restava sopra la soglia.
            sotto = sotto + 1 if knn[0] < soglia else 0
            print(f"            [cancello] lineare {knn[0]:.4f} vs casuale {knn_ref:.4f}"
                  f"  -> {delta:+.4f}   miglior finora {knn_best:.4f}"
                  f"   sonde sotto di fila: {sotto}")

            crollo = knn[0] < knn_ref - GATE_CROLLO * GATE_MARGINE
            if epoch + 1 >= GATE_AT and (sotto >= GATE_SONDE_SOTTO or crollo):
                motivo = (f"crollo netto ({delta:+.4f}, oltre {GATE_CROLLO} margini)"
                          if crollo else
                          f"{sotto} sonde consecutive sotto il riferimento")
                print(f"  CANCELLO all'epoca {epoch+1}: {motivo}.")
                print(f"  Sonda lineare {knn[0]:.4f} contro encoder casuale {knn_ref:.4f}.")
                print("  Il pre-training sta DEGRADANDO le rappresentazioni: ci si")
                print("  ferma qui invece di consumare le epoche restanti.")
                break

        if monitor.is_collapsing():
            print("\n  COLLASSO RILEVATO. Non insistete: cambiate qualcosa.")
            print("  Ordine di intervento suggerito:")
            print("   1. abbassare il learning rate (fattore 3)")
            print("   2. alzare SSL_EMA_START verso 0.999")
            print("   3. ridurre la capacita' del predictor (e' troppo forte)")
            break

        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "gstep": gstep, "variant": variant,
            "monitor": monitor.history, "knn_ref": knn_ref,
            # Gli iperparametri passati da riga di comando vanno nel
            # checkpoint: senza, un --resume che dimentica --ema-start
            # ripartirebbe con un valore DIVERSO e il run cambierebbe
            # regime a meta' senza dirlo. E' lo stesso tipo di confusione
            # dello scheduler troncato di agosto, che aveva nascosto per
            # giorni il vero effetto di una configurazione.
            "lr": LR, "ema_start": EMA_START,
            "predictor_dim": network.PREDICTOR_DIM,
        }, run_name)

    if tb is not None:
        tb.close()
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
    ap.add_argument("--smoke", action="store_true")
    # Override per gli esperimenti in parallelo. Restano fuori da globals.py
    # apposta: globals descrive la configurazione di riferimento, questi
    # servono a esplorarne le varianti senza toccarla.
    ap.add_argument("--tag", default="", help="suffisso del run (checkpoint separati)")
    ap.add_argument("--context-scale", type=float, nargs=2, default=None,
                    help="es. 0.4 0.7 - contesto piu' piccolo = compito piu' difficile")
    ap.add_argument("--target-scale", type=float, nargs=2, default=None)
    ap.add_argument("--predictor-dim", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema-start", type=float, default=None,
                    help="momentum EMA iniziale: piu' alto = target piu' lento = meno collasso")
    ap.add_argument("--gate-epoch", type=int, default=None,
                    help="epoche dopo cui fermarsi se la sonda non batte l'encoder casuale")
    a = ap.parse_args()

    # Si scrivono nei moduli che li leggono a ogni chiamata, cosi' l'override
    # vale per l'intero run senza duplicare la configurazione.
    import network
    import data as data_mod
    if a.lr is not None:
        LR = a.lr
    if a.ema_start is not None:
        EMA_START = a.ema_start
    if a.gate_epoch is not None:
        GATE_AT = a.gate_epoch
    if a.context_scale:
        network.CONTEXT_SCALE = tuple(a.context_scale)
    if a.target_scale:
        network.TARGET_SCALE = tuple(a.target_scale)
    if a.predictor_dim is not None:
        network.PREDICTOR_DIM = a.predictor_dim
    if a.workers is not None:
        data_mod.NUM_WORKERS = a.workers

    print(f"[config] tag={a.tag or '-'} context={network.CONTEXT_SCALE} "
          f"target={network.TARGET_SCALE} predictor_dim={network.PREDICTOR_DIM} "
          f"workers={data_mod.NUM_WORKERS}")
    train(a.variant, a.epochs, a.batch_size, a.resume, a.smoke, a.tag)
