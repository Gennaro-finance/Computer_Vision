"""
INFRASTRUTTURA — orchestra pre-training ed esperimenti senza interazione.

Catena notturna: pre-training completo e tutte le misure che ne dipendono,
senza interazione.

PERCHE' ESISTE. Il pre-training dura ore e le misure a valle altrettante.
Lanciarle a mano significa stare svegli a guardare una barra di avanzamento,
e - come e' gia' successo il 24 agosto - perdere lo stato se qualcosa si
ferma nel mezzo. Qui la sequenza e' dichiarata una volta, ogni stadio sa da
cosa dipende, e chi si sveglia legge un resoconto invece di ricostruire.

TRE REGOLE, e sono quelle che rendono la catena sicura da lanciare al buio:

  1. NON SOVRASCRIVE NIENTE. Ogni stadio scrive su un tag nuovo (`finale`).
     I checkpoint, i latenti e i risultati dei bracci gia' misurati restano
     dove sono. Dove uno script scriverebbe su un file condiviso si passa
     un --esito dedicato.

  2. SI RIPRENDE. Ogni stadio dichiara il file che produce: se c'e' gia',
     lo stadio si salta. Rilanciare la catena dopo un'interruzione riparte
     da dove si era fermata invece di rifare ore di lavoro.

  3. UN FALLIMENTO NON PROPAGA IL DANNO. Chi dipende da uno stadio fallito
     viene saltato e dichiarato tale; chi non dipende prosegue. Alla fine
     `runs/catena/esito.md` dice cosa e' andato e cosa no.

Tutto quello che tocca la GPU passa da sorveglia.py con il tetto di potenza:
il blocco dei clock ha retto 15 ore, ma la rete di sicurezza resta.

Uso:
    python catena.py                # esegue
    python catena.py --secco        # stampa cosa farebbe e esce
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

CARTELLA = os.path.join("runs", "catena")
REGISTRO = os.path.join(CARTELLA, "registro.txt")
ESITO = os.path.join(CARTELLA, "esito.md")

TAG = "finale"                       # tag del run SSL: ijepa_vit_small_finale
GEO = "_geo_finale"                  # latenti, protocollo del brief
CIECO = "_cieco_finale"              # latenti, ciechi alla dimensione
LAYERS = ["2", "7", "11"]
SEEDS = ["0", "1", "2", "3", "4"]
TETTO, TETTO_TEMP = "95", "86"

CKPT = f"runs/checkpoints/ijepa_vit_small_{TAG}_best.pt"


def interprete():
    """
    Il python del .venv, non quello di sistema, e NON sys.executable.

    PERCHE' ESPLICITO. Il 28 agosto la catena e' morta in due secondi con
    ModuleNotFoundError: sklearn. Su questa macchina convivono due stack:

        sistema  torch 2.12.0, senza sklearn
        .venv    torch 2.13.0, sklearn 1.9.0   <- ha prodotto TUTTI i risultati

    Usare sys.executable significa ereditare l'interprete di chi lancia, che
    dipende dal PATH della shell. Non e' solo un rischio di crash: il
    confronto sull'lr era gia' finito sul torch 2.12 senza che nessuno se ne
    accorgesse, e un numero misurato su uno stack diverso non si puo'
    affiancare agli altri. Meglio fallire subito e rumorosamente.
    """
    p = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.isfile(p):
        p = os.path.join(".venv", "bin", "python")
    if not os.path.isfile(p):
        raise SystemExit(f"Nessun interprete in .venv: cercato {p}")
    r = subprocess.run([p, "-c", "import torch, sklearn, numpy"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"L'interprete {p} non importa le dipendenze:\n"
                         f"{r.stderr.strip()}")
    return p


PY = interprete()
LAT_GEO = f"runs/cache/latents_vit_small_L2-7-11{GEO}.pt"
LAT_CIECO = f"runs/cache/latents_vit_small_L2-7-11{CIECO}.pt"

# L'ULTIMA epoca, non la migliore. train_ssl salva due checkpoint distinti:
# `_best` (scelto sulla sonda) e quello senza suffisso, riscritto a ogni
# epoca e quindi fermo all'ultima. Il secondo di solito non serve a nessuno;
# qui serve, perche' e' l'encoder che ha disimparato di piu' la bbox.
ULTIMA = "_geo_ultima"
LAT_ULTIMA = f"runs/cache/latents_vit_small_L2-7-11{ULTIMA}.pt"


# --------------------------------------------------------- attesa GPU
def memoria_gpu():
    """MiB occupati sulla GPU, o None se nvidia-smi non risponde."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def attendi_gpu_libera(minuti_max=120, soglia_mib=2500):
    """
    Non parte finche' un'altra misura sta usando la scheda.

    SERVE DAVVERO. La catena viene lanciata mentre il confronto sull'lr sta
    ancora girando: partire subito significherebbe due processi sulla stessa
    GPU, che non e' solo lento - e' il modo in cui si superava il tetto di
    potenza e la macchina si spegneva. E aspettare risolve anche una seconda
    cosa: scegli_lr() legge un file che quel processo sta ancora scrivendo,
    e leggerlo dopo significa leggerlo completo.

    Se dopo minuti_max e' ancora occupata si parte lo stesso: meglio una
    catena lenta che una catena che non parte mai perche' qualcuno ha
    lasciato aperto un notebook.

    LA SOGLIA NON E' ZERO, e il primo tentativo l'aveva messa a 1000 MiB
    sbagliando. Su questa macchina il desktop da solo tiene ~1195 MiB
    sparsi su una venticinquina di processi (browser, compositor, Explorer):
    con la soglia a 1000 la GPU risulta occupata SEMPRE, e la catena
    avrebbe aspettato i 120 minuti pieni per poi partire lo stesso.
    2500 sta comodamente in mezzo fra il fondo del desktop (~1200) e una
    misura vera (5348 MiB durante il confronto sull'lr).
    """
    t0 = time.time()
    while time.time() - t0 < minuti_max * 60:
        m = memoria_gpu()
        if m is None:
            scrivi("nvidia-smi non risponde: parto senza aspettare")
            return
        if m < soglia_mib:
            if time.time() - t0 > 30:
                scrivi(f"GPU libera ({m} MiB) dopo "
                       f"{(time.time()-t0)/60:.0f} min di attesa")
            return
        if int(time.time() - t0) % 600 < 31:
            scrivi(f"attendo la GPU: {m} MiB occupati "
                   f"({(time.time()-t0)/60:.0f} min)")
        time.sleep(30)
    scrivi(f"la GPU e' ancora occupata dopo {minuti_max} min: parto comunque")


# ----------------------------------------------------------------- lr
def scegli_lr():
    """
    La sola scelta della catena, e la regola e' scritta prima di vedere il
    numero.

    Confronta `spinto` (lr 3e-4) e `completa` (lr 3e-5) su P3_K16, misurati
    su VALIDATION - il test non si usa per scegliere. Si cambia lr SOLO se
    spinto vince in modo separabile:

        divario >= 0.01 di macro-F1  E  |z| >= 2.31

    2.31 e' il quantile di Student a 8 gradi di liberta' (due campioni da 5
    seed), non 1.96: con cinque ripetizioni la normale sottostima la coda.

    Se non separano, si resta a 3e-5. Non e' timidezza: e' la configurazione
    che ha prodotto TUTTI i risultati gia' misurati, e cambiarla senza prova
    romperebbe la confrontabilita' con l'intero progetto. L'onere sta su chi
    cambia.
    """
    percorso = "runs/fixedk_lr_val.json"
    if not os.path.isfile(percorso):
        return 3e-5, "misura assente: resto sulla configurazione di sempre"
    with open(percorso, encoding="utf-8") as f:
        d = json.load(f)["risultati"]
    a = d.get("P3_K16|_geo_completa")
    b = d.get("P3_K16|_spinto")
    if not (a and b):
        return 3e-5, "celle incomplete: resto sulla configurazione di sempre"
    ma, sa = a["macro_f1"]
    mb, sb = b["macro_f1"]
    div = mb - ma
    se = math.sqrt(sa ** 2 + sb ** 2) / math.sqrt(len(SEEDS))
    z = div / se if se > 0 else 0.0
    testo = (f"completa(3e-5)={ma:.4f}+-{sa:.4f}  spinto(3e-4)={mb:.4f}+-{sb:.4f}"
             f"  divario={div:+.4f}  z={z:+.2f}"
             f"   [ATTENZIONE: misurato su torch 2.12 del python di sistema,"
             f" non sul .venv del progetto. Il divario e' interno e regge;"
             f" i valori assoluti li rimisura lo stadio verifica_lr]")
    if div >= 0.01 and z >= 2.31:
        return 3e-4, f"spinto vince in modo separabile -> lr 3e-4.  {testo}"
    if div <= -0.01 and z <= -2.31:
        # La regola e' a UNA CODA - cambia solo se spinto vince - ma il
        # motivo scritto nel resoconto dev'essere quello vero: qui i due
        # bracci separano benissimo, e separano CONTRO spinto.
        return 3e-5, (f"spinto e' PEGGIORE in modo separabile -> lr 3e-5 "
                      f"confermato dalla misura, non per prudenza.  {testo}")
    return 3e-5, f"non separano (serve div>=0.01 e z>=2.31) -> lr 3e-5.  {testo}"


# ------------------------------------------------------------- stadi
def costruisci(lr):
    def gpu(*cmd):
        return [PY, "sorveglia.py", "--tetto", TETTO,
                "--tetto-temp", TETTO_TEMP, "--", PY, *cmd]

    return [
        dict(nome="pretrain", dipende=None, produce=CKPT, ore=9.0,
             cosa="pre-training I-JEPA, 300 epoche, sonda di selezione su P3_K16",
             cmd=gpu("train_ssl.py", "--variant", "vit_small",
                     "--epochs", "300", "--tag", TAG,
                     "--lr", repr(lr), "--ema-start", "0.9996")),

        dict(nome="estrai_geo", dipende="pretrain", produce=LAT_GEO, ore=0.1,
             cosa="latenti nel protocollo del brief (finestra fissa 224 px)",
             cmd=gpu("train_downstream.py", "--cache", "--layers", *LAYERS,
                     "--ckpt-tag", f"{TAG}_best", "--tag", GEO)),

        dict(nome="estrai_cieco", dipende="pretrain", produce=LAT_CIECO, ore=0.1,
             cosa="latenti ciechi alla dimensione (finestra 3x la bbox)",
             cmd=gpu("train_downstream.py", "--cache", "--layers", *LAYERS,
                     "--ckpt-tag", f"{TAG}_best", "--tag", CIECO,
                     "--context-factor", "3.0")),

        dict(nome="fixedk_flat", dipende="estrai_geo", ore=2.7,
             produce=f"{CARTELLA}/fixedk_flat_test.json",
             cosa="i cinque protocolli di mascheramento, testa flat, sul test",
             cmd=gpu("exp_fixedk.py", "--tag", "_casuale", GEO,
                     "--protocolli", "P1_bbox", "P2b_griglia_fissa",
                     "P3_K16", "P3_K36", "P3_K64",
                     "--head", "flat", "--split", "test",
                     "--esito", f"{CARTELLA}/fixedk_flat_test.json")),

        dict(nome="fixedk_mil", dipende="estrai_geo", ore=1.1,
             produce=f"{CARTELLA}/fixedk_mil_test.json",
             cosa="MIL per token: il margine piu' grande del progetto",
             cmd=gpu("exp_fixedk.py", "--tag", "_casuale", GEO,
                     "--protocolli", "P1_bbox", "P3_K16",
                     "--head", "mil", "--split", "test",
                     "--esito", f"{CARTELLA}/fixedk_mil_test.json")),

        dict(nome="fewshot", dipende="estrai_geo", ore=1.0, produce=None,
             cosa="few-shot 1/5/10/25/100% sui due protocolli",
             cmd=gpu("exp_fewshot.py", "--tag", "_casuale", GEO,
                     "--protocolli", "P1_bbox", "P3_K16", "--head", "flat")),

        dict(nome="griglia_geo", dipende="estrai_geo", ore=2.0,
             produce=f"runs/results_vit_small_L2-7-11{GEO}.json",
             cosa="griglia principale: 5 metodi x 2 teste, protocollo del brief",
             cmd=gpu("train_downstream.py", "--grid", "--layers", *LAYERS,
                     "--tag", GEO)),

        dict(nome="griglia_cieca", dipende="estrai_cieco", ore=0.3,
             produce=f"runs/results_vit_small_L2-7-11{CIECO}.json",
             cosa="stessa griglia nel protocollo cieco, solo metodo none",
             cmd=gpu("train_downstream.py", "--grid", "--layers", *LAYERS,
                     "--tag", CIECO, "--metodi", "none", "--teste", "flat")),

        dict(nome="novita_K", dipende="estrai_geo", ore=1.0, produce=None,
             cosa="obiettivo 3, la novita', misurata DOVE si vede (P3_K16)",
             cmd=gpu("exp_novita_K.py", "--tag", GEO, "--K", "16")),

        # ---- misure che erano state fatte su `completa` e vanno rifatte ----
        # Sostituire il checkpoint non basta: ogni numero misurato sul vecchio
        # resta vecchio finche' non lo si rifa'. Questi cinque erano fuori
        # dalla prima stesura della catena, e senza di loro la presentazione
        # sarebbe meta' su un encoder e meta' sull'altro.
        dict(nome="mascheramento", dipende="estrai_geo", ore=0.8, produce=None,
             cosa="protocolli di mascheramento, sonda k-NN senza parametri",
             cmd=gpu("exp_mascheramento.py", "--tag", "_casuale", GEO)),

        dict(nome="stratificata", dipende="estrai_geo", ore=0.6, produce=None,
             cosa="prestazione stratificata per dimensione della lesione",
             cmd=gpu("exp_stratificata.py", "--tag", "_casuale", GEO)),

        dict(nome="testa_pooling", dipende="estrai_geo", ore=1.2, produce=None,
             cosa="confronto fra i pooling, protocollo del brief",
             cmd=gpu("exp_testa.py", "--tag", GEO, "--controllo", "_casuale")),

        dict(nome="testa_pooling_K16", dipende="estrai_geo", ore=1.2,
             produce=None,
             cosa="stesso confronto a conteggio fisso, dove il metro vede",
             cmd=gpu("exp_testa.py", "--tag", GEO, "--controllo", "_casuale",
                     "--protocollo", "P3_K16")),

        dict(nome="traiettoria", dipende="estrai_cieco", ore=0.6, produce=None,
             cosa="traiettoria della testa nel protocollo cieco",
             cmd=gpu("exp_traiettoria_testa.py", "--tag", "_cieco_casuale",
                     CIECO, "--nome", "cieco")),

        # Dipende da ENTRAMBE le estrazioni: legge i tag geo e cieco insieme.
        # Il modello di dipendenze ne ammette una sola, quindi si dichiara la
        # cieca (che e' l'ultima delle due) e se la geo manca lo stadio
        # fallisce da solo, senza sporcare niente.
        dict(nome="sonde", dipende="estrai_cieco", ore=0.5, produce=None,
             cosa="sonde k-NN su tutti gli encoder, curva di apprendimento",
             cmd=gpu("exp_sonde.py")),

        # ---- la previsione della traiettoria, verificata sulla pipeline ----
        # Durante il pre-training la sonda ha misurato due cose opposte fra
        # le prime 10 e le ultime 10 rilevazioni: K16 fermo (+0.0023, z=+0.31)
        # e P1_bbox in calo (-0.0190, z=-6.73). Cioe': la qualita' satura
        # verso l'epoca 70, e le duecento epoche successive servono a
        # DISIMPARARE la bounding box.
        #
        # Se e' vero, i due checkpoint che il run lascia sul disco devono
        # comportarsi in modo diverso e prevedibile:
        #     finale_best  epoca ~69   massima qualita'
        #     finale       epoca 288   massimo disimparamento
        # stesso K16, P1 piu' basso. Qui lo si verifica dove il progetto
        # lavora davvero - attention pooling piu' testa addestrata - invece
        # che sulla sonda interna da 2500 campioni.
        #
        # Se la previsione fallisse sarebbe un risultato lo stesso: vorrebbe
        # dire che il calo di P1 vive solo nella sonda e non nella pipeline.
        dict(nome="estrai_ultima", dipende="pretrain", produce=LAT_ULTIMA,
             ore=0.1,
             cosa="latenti dall'ULTIMA epoca (non dal best): il piu' disimparato",
             cmd=gpu("train_downstream.py", "--cache", "--layers", *LAYERS,
                     "--ckpt-tag", TAG, "--tag", ULTIMA)),

        # Scrive nello STESSO file di fixedk_flat: le celle di _casuale e
        # _geo_finale sono gia' li' e vengono riusate: si calcolano solo le
        # due nuove. Stessa testa, stesso split, stessi seed - altrimenti la
        # guardia di ripresa azzererebbe il file invece di completarlo.
        dict(nome="fixedk_ultima", dipende="estrai_ultima", ore=0.6,
             produce=None,
             cosa="stesso K16 ma P1 piu' basso? la previsione sulla pipeline",
             cmd=gpu("exp_fixedk.py", "--tag", "_casuale", GEO, ULTIMA,
                     "--protocolli", "P1_bbox", "P3_K16",
                     "--head", "flat", "--split", "test",
                     "--esito", f"{CARTELLA}/fixedk_flat_test.json")),

        # Indagine, non misura: non dipende da niente e non produce numeri
        # da riportare. Cerca l'origine dello scarto di 0,0140 sul braccio
        # casuale fra i file di agosto (0,7565) e fixedk di oggi (0,7705).
        # Conta perche' il 27 agosto quello scarto e' stato diagnosticato
        # come "celle stantie" e la griglia e' stata rifatta: se la diagnosi
        # era sbagliata, la correzione ha peggiorato i dati.
        dict(nome="scarto", dipende=None, ore=0.9,
             produce=f"{CARTELLA}/scarto.json",
             cosa="da dove viene lo scarto di 0,0140 sul casuale",
             cmd=gpu("exp_scarto.py", "--tag", "_casuale",
                     "--esito", f"{CARTELLA}/scarto.json")),

        # NON dipende da niente: e' una riparazione di provenienza, e deve
        # girare anche se il pre-training e' fallito. Il confronto lr e'
        # stato misurato per sbaglio sul python di sistema (torch 2.12); qui
        # si rifa' sullo stack del progetto (2.13) perche' il numero sia
        # affiancabile agli altri. La DECISIONE non cambiera' - z = -4.88 non
        # lo ribalta una versione minore - ma il valore citabile si'.
        dict(nome="verifica_lr", dipende=None, ore=0.6,
             produce=f"{CARTELLA}/fixedk_lr_val_venv.json",
             cosa="rimisura del confronto lr sullo stack giusto (provenienza)",
             cmd=gpu("exp_fixedk.py", "--tag", "_geo_completa", "_spinto",
                     "--protocolli", "P3_K16", "--head", "flat",
                     "--split", "val",
                     "--esito", f"{CARTELLA}/fixedk_lr_val_venv.json")),
    ]


# ------------------------------------------------------------ motore
def marcatore(nome):
    """
    Traccia di uno stadio riuscito che non produce un file suo.

    SERVE PERCHE' LA CATENA SI RILANCIA. Gli stadi che scrivono dentro un
    JSON condiviso (few-shot, novita', sonde) non hanno un file "loro" da
    controllare: senza marcatore verrebbero rieseguiti a ogni passata, e con
    un file gia' pieno il salvataggio interno li fa uscire subito - il che e'
    peggio, perche' sembrano fatti senza esserlo. Con il marcatore, la
    seconda passata esegue solo cio' che manca davvero.
    """
    return os.path.join(CARTELLA, f".fatto_{nome}")


# Le prove a secco NON sporcano il registro: quello e' il file che si legge
# la mattina dopo per capire com'e' andata, e riempirlo di piani mai
# eseguiti lo rende illeggibile proprio quando serve.
SOLO_STAMPA = False


def scrivi(riga):
    stampa = f"[{time.strftime('%H:%M:%S')}] {riga}"
    print(stampa, flush=True)
    if SOLO_STAMPA:
        return
    with open(REGISTRO, "a", encoding="utf-8") as f:
        f.write(stampa + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secco", action="store_true",
                    help="stampa il piano e esce, senza eseguire niente")
    a = ap.parse_args()

    os.makedirs(CARTELLA, exist_ok=True)
    globals()["SOLO_STAMPA"] = a.secco

    # L'ORDINE CONTA: prima si aspetta che la GPU si liberi, POI si sceglie
    # l'lr. Il file su cui si sceglie e' scritto dal processo che stiamo
    # aspettando; leggerlo prima significherebbe leggerlo a meta'.
    if not a.secco:
        scrivi(f"CATENA - lanciata {time.strftime('%d/%m %H:%M')}, "
               f"controllo che la GPU sia libera")
        attendi_gpu_libera()

    lr, perche = scegli_lr()
    stadi = costruisci(lr)

    scrivi("=" * 74)
    scrivi(f"CATENA - avvio {time.strftime('%d/%m %H:%M')}")
    scrivi(f"lr scelto: {lr:g}")
    scrivi(f"  motivo: {perche}")
    scrivi(f"stima totale: {sum(s['ore'] for s in stadi):.1f} h")
    scrivi("=" * 74)

    if a.secco:
        for s in stadi:
            fatto = os.path.isfile(marcatore(s["nome"]))
            print(f"  {s['nome']:14s} {'[GIA FATTO]' if fatto else '':12s} "
                  f"~{s['ore']:.1f}h  {s['cosa']}")
            print(f"       {' '.join(s['cmd'][2:])}")
        return

    stato = {}
    for s in stadi:
        n = s["nome"]
        if s["dipende"] and stato.get(s["dipende"]) != "ok":
            stato[n] = "saltato"
            scrivi(f"{n}: SALTATO (dipende da {s['dipende']}, "
                   f"che e' {stato.get(s['dipende'], 'mai eseguito')})")
            continue
        # SI SALTA SOLO SUL MARCATORE, mai sull'esistenza del file prodotto.
        #
        # La prima versione guardava `produce`, e il dry-run del 28 agosto ha
        # mostrato "pretrain [GIA FATTO]" mentre il pre-training era
        # all'epoca 39 di 300: il checkpoint `_best` viene scritto al primo
        # record utile, non alla fine. Un rilancio dopo un crash avrebbe
        # saltato l'addestramento e sarebbe andato avanti con un encoder
        # mezzo cotto, senza dire niente. Vale per ogni stadio che scrive
        # presto o in streaming, estrazioni comprese: un file a meta' esiste
        # eccome.
        #
        # `produce` resta, ma come POST-CONDIZIONE: sotto si verifica che
        # dopo un'uscita pulita il file ci sia davvero.
        if os.path.isfile(marcatore(n)):
            stato[n] = "ok"
            scrivi(f"{n}: gia' fatto in una passata precedente, salto")
            continue

        scrivi(f"{n}: AVVIO - {s['cosa']}  (stima {s['ore']:.1f} h)")
        t0 = time.time()
        log = os.path.join(CARTELLA, f"{n}.log")
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(s["cmd"], stdout=f, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 3600

        if r.returncode != 0:
            stato[n] = "fallito"
            scrivi(f"{n}: FALLITO (codice {r.returncode}) dopo {dt:.2f} h "
                   f"- vedi {log}")
        elif s["produce"] and not os.path.isfile(s["produce"]):
            stato[n] = "fallito"
            scrivi(f"{n}: uscito bene ma {s['produce']} non c'e' - "
                   f"trattato come fallito, vedi {log}")
        else:
            stato[n] = "ok"
            open(marcatore(n), "w").close()
            scrivi(f"{n}: fatto in {dt:.2f} h")

    # ------------------------------------------------------- resoconto
    righe = ["# Esito della catena", "",
             f"Avviata e conclusa il {time.strftime('%d/%m/%Y, fine alle %H:%M')}.",
             "", f"**lr scelto: {lr:g}** — {perche}", "",
             "| stadio | esito | cosa produceva |", "|---|---|---|"]
    for s in stadi:
        e = stato.get(s["nome"], "mai eseguito")
        simbolo = {"ok": "fatto", "fallito": "**FALLITO**",
                   "saltato": "saltato"}.get(e, e)
        righe.append(f"| `{s['nome']}` | {simbolo} | {s['cosa']} |")
    falliti = [k for k, v in stato.items() if v != "ok"]
    righe += ["", ("Tutto a posto." if not falliti else
                   "Da guardare: " + ", ".join(f"`{k}`" for k in falliti) +
                   f". I log stanno in `{CARTELLA}/`.")]
    righe += ["", "Il checkpoint nuovo e' `" + CKPT + "`. I bracci gia'",
              "misurati non sono stati toccati: la catena scrive solo su tag",
              "`finale`."]
    with open(ESITO, "w", encoding="utf-8") as f:
        f.write("\n".join(righe) + "\n")
    scrivi(f"resoconto in {ESITO}")


if __name__ == "__main__":
    main()
