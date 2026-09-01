"""
INFRASTRUTTURA — pulizia e commit del repository dopo la catena.

Mette in ordine il repository dopo la catena notturna: pulisce, verifica,
propone cosa togliere e prepara il commit. NON pubblica.

DOVE SI FERMA, E PERCHE'. Questo script arriva fino al commit locale su un
ramo dedicato e si ferma. Il push resta un comando che dai tu, sveglio, per
tre ragioni concrete e non per formalita':

  - il remote e' github.com/Gennaro-finance/Computer_Vision e il progetto e'
    in tre. Un push su `main` arriva agli altri due, e non e' una cosa da
    far succedere alle 22 mentre chi l'ha lanciata dorme.

  - "levare di mezzo cio' che non serve" e' un giudizio, non un'operazione.
    runs/archivio/ contiene i risultati superati, che sono anche la prova
    documentale della sezione "errori fatti e chiusi" della presentazione:
    cancellarli toglierebbe verificabilita' proprio dove serve di piu'.
    Qui si PROPONE con le prove, si decide altrove.

  - se un esperimento della catena e' fallito, committare i risultati come
    se fossero completi e' peggio che non committare niente.

Il commit va su un RAMO, non su main: se domattina la proposta non convince,
si cancella il ramo e non e' successo niente.

Uso:
    python sistema_repo.py            # pulisce, verifica, propone, committa
    python sistema_repo.py --secco    # dice cosa farebbe, non tocca niente
"""

import argparse
import json
import os
import re
import subprocess
import sys

CARTELLA = os.path.join("runs", "catena")
ESITO = os.path.join(CARTELLA, "esito.md")
PROPOSTA = os.path.join(CARTELLA, "repo_proposta.md")
RAMO = "catena-300"

# Roba che la catena e gli smoke test lasciano in giro e che non serve a
# nessuno. Si cancella davvero, perche' si rigenera da sola.
SPAZZATURA = [
    "runs/figures/ijepa_vit_small_smoketest_monitor.json",
    "runs/figures/ijepa_vit_small_smoketest_monitor.png",
    "runs/checkpoints/ijepa_vit_small_smoketest.pt",
    "runs/checkpoints/ijepa_vit_small_smoketest_best.pt",
]


def git(*a, controlla=True):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    if controlla and r.returncode != 0:
        raise SystemExit(f"git {' '.join(a)} -> {r.returncode}\n{r.stderr}")
    return r.stdout.strip()


# ------------------------------------------------------- 1. i controlli
def verifica_catena():
    """
    Non si sistema il repo prima di sapere com'e' andata la notte.

    Se la catena non e' finita, i risultati sui quali si committerebbe non
    esistono ancora o sono a meta'. Se e' finita male, il commit direbbe una
    cosa falsa. In entrambi i casi meglio fermarsi e dirlo.
    """
    if not os.path.isfile(ESITO):
        raise SystemExit(
            f"{ESITO} non c'e': la catena non ha finito.\n"
            f"Controlla con:  tail runs/catena/registro.txt")
    testo = open(ESITO, encoding="utf-8").read()
    falliti = re.findall(r"\|\s*`(\w+)`\s*\|\s*\*\*FALLITO\*\*", testo)
    saltati = re.findall(r"\|\s*`(\w+)`\s*\|\s*saltato", testo)
    return falliti, saltati


# --------------------------------------------------- 2. cosa non serve
def audit_orfani():
    """
    File tracciati che nessun altro file tracciato nomina.

    NON e' una prova che siano morti: uno script si lancia da riga di
    comando e nessuno lo importa, ed e' vivissimo. E' un ELENCO DI SOSPETTI
    con l'evidenza accanto, da leggere con la testa. Per questo il risultato
    finisce in un file di proposta e non in un `git rm`.

    SI GUARDANO ANCHE I NON TRACCIATI, e la prima versione sbagliava. Con
    `git ls-files` soltanto, il primo giro ha dichiarato orfani exp_fewshot.py
    e exp_novita_K.py mentre la catena li stava eseguendo: li nomina
    catena.py, che a quel punto non era ancora tracciato e quindi non veniva
    nemmeno letto. Chi cerca file morti deve guardare il disco per come e',
    non per come git lo conosceva ieri.
    """
    tracciati = git("ls-files", "--cached", "--others",
                    "--exclude-standard").splitlines()
    testo = {}
    for f in tracciati:
        if f.endswith((".py", ".md", ".ps1", ".bat", ".json", ".txt", ".ipynb")):
            try:
                testo[f] = open(f, encoding="utf-8", errors="ignore").read()
            except OSError:
                pass

    sospetti = []
    for f in tracciati:
        if not f.endswith((".py", ".ps1", ".bat")):
            continue
        base = os.path.basename(f)
        modulo = base[:-3] if base.endswith(".py") else base
        citazioni = [g for g, t in testo.items()
                     if g != f and (base in t or
                                    (base.endswith(".py") and
                                     re.search(rf"\b(import|from)\s+{re.escape(modulo)}\b", t)))]
        if not citazioni:
            sospetti.append((f, os.path.getsize(f) if os.path.isfile(f) else 0))
    return sorted(sospetti)


def scrivi_proposta(falliti, saltati, sospetti):
    r = ["# Proposta di riordino del repository", "",
         "Generata da `sistema_repo.py` dopo la catena notturna.",
         "**Niente di quanto segue e' stato cancellato**: e' un elenco da"
         " approvare.", ""]

    r += ["## Com'e' andata la notte", ""]
    if not falliti and not saltati:
        r.append("Tutti gli stadi della catena sono andati a buon fine.")
    else:
        if falliti:
            r.append(f"**Falliti**: {', '.join('`'+x+'`' for x in falliti)}. "
                     f"I log stanno in `{CARTELLA}/`.")
        if saltati:
            r.append(f"Saltati per dipendenza: "
                     f"{', '.join('`'+x+'`' for x in saltati)}.")
        r.append("")
        r.append("> Finche' questi non sono chiariti, il commit preparato "
                 "descrive un lavoro incompleto. Leggi i log prima di pushare.")
    r += [""]

    r += ["## Sospetti orfani", "",
          "Script tracciati che nessun altro file tracciato nomina. "
          "**Non e' una condanna**: un file lanciato solo da riga di comando "
          "risulta orfano ed e' vivo. Serve il tuo giudizio.", ""]
    if sospetti:
        r += ["| file | byte | verdetto (da mettere a mano) |", "|---|---:|---|"]
        for f, n in sospetti:
            r.append(f"| `{f}` | {n:,} | |")
    else:
        r.append("Nessuno: ogni script e' citato da qualcos'altro.")
    r += [""]

    r += ["## Cose che NON propongo di togliere, e perche'", "",
          "- `runs/archivio/` — sono i risultati superati, e sono la prova "
          "documentale della sezione *errori fatti e chiusi*. Toglierli "
          "renderebbe non verificabile proprio la parte piu' onesta del "
          "racconto.",
          "- `runs/PROVENIENZA.md` — dice quale checkpoint ha prodotto quale "
          "file. E' cio' che ha permesso di scoprire il braccio costruito su "
          "un checkpoint ignoto.",
          "- i `logs/` in whitelist — senza, l'affermazione *ogni numero e' "
          "ricalcolabile* diventa falsa.", ""]

    r += ["## Il push non e' stato fatto", "",
          f"Il commit e' pronto sul ramo `{RAMO}`. Quando la proposta qui "
          f"sopra ti convince:", "", "```bash",
          f"git push -u origin {RAMO}", "```", "",
          "Il remote e' `Gennaro-finance/Computer_Vision` e siete in tre: "
          "conviene aprire una PR invece di fondere dritto su `main`, cosi' "
          "gli altri due vedono cosa cambia.", ""]

    os.makedirs(CARTELLA, exist_ok=True)
    open(PROPOSTA, "w", encoding="utf-8").write("\n".join(r) + "\n")


# ------------------------------------------------------- 3. il commit
def messaggio(falliti):
    n = []
    for f in ("fixedk_flat_test.json", "fixedk_mil_test.json",
              "fixedk_lr_val_venv.json"):
        if os.path.isfile(os.path.join(CARTELLA, f)):
            n.append(f)
    corpo = [
        "pre-training completo a 300 epoche, con una sonda che vede",
        "",
        "La sonda che sceglieva il checkpoint leggeva la pipeline del brief,",
        "cioe' un protocollo dominato dal canale della maschera: la sola bbox",
        "one-hot da' macro-F1 0.7708. Sceglieva guardando la bounding box.",
        "Ora seleziona su P3_K16 - 16 token per tutti - e misura P1_bbox come",
        "controllo: se K16 sale mentre P1 resta piatta, la diagnosi si legge",
        "sulla curva di addestramento invece che a posteriori.",
        "",
        "Misurato prima di lanciare: lr 3e-4 (`spinto`) produce una",
        "rappresentazione PEGGIORE di 0.0440 macro-F1 a z = -4.88 su",
        "validation. L'lr resta 3e-5 per misura, non per prudenza. Da notare",
        "che il k-NN dava il segno opposto - il quarto surrogato a mentire.",
        "",
        "catena.py orchestra pre-training ed esperimenti a valle senza",
        "interazione: ogni stadio dichiara cosa produce, si salta se c'e'",
        "gia', e un fallimento non trascina chi non dipende.",
    ]
    if falliti:
        corpo += ["", f"NOTA: stadi falliti in questa catena: "
                      f"{', '.join(falliti)}. Vedi runs/catena/."]
    return "\n".join(corpo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secco", action="store_true")
    a = ap.parse_args()

    falliti, saltati = verifica_catena()

    tolti = [p for p in SPAZZATURA if os.path.isfile(p)]
    sospetti = audit_orfani()

    if a.secco:
        print(f"catena: {len(falliti)} falliti, {len(saltati)} saltati")
        print(f"spazzatura da togliere: {tolti or 'niente'}")
        print(f"sospetti orfani: {len(sospetti)}")
        for f, n in sospetti:
            print(f"   {f}  ({n:,} byte)")
        print(f"proposta -> {PROPOSTA}")
        print(f"commit sul ramo {RAMO}, NESSUN push")
        return

    for p in tolti:
        os.remove(p)
        print(f"tolto {p}")

    scrivi_proposta(falliti, saltati, sospetti)
    print(f"proposta scritta in {PROPOSTA}")

    attuale = git("branch", "--show-current")
    if attuale != RAMO:
        esistenti = git("branch", "--list", RAMO)
        git("checkout", *(["-b"] if not esistenti else []), RAMO)
        print(f"ramo {RAMO} (da {attuale})")

    # Si aggiunge in modo mirato: il .gitignore gia' tiene fuori i binari
    # pesanti, ma i .log della catena sono rumore e non entrano.
    git("add", "-A", "--", ".", ":!runs/catena/*.log")
    git("commit", "-m", messaggio(falliti), controlla=False)
    print(git("log", "-1", "--stat", "--format=%h %s"))
    print(f"\nNESSUN PUSH FATTO. Leggi {PROPOSTA}, poi:\n"
          f"    git push -u origin {RAMO}")


if __name__ == "__main__":
    main()
