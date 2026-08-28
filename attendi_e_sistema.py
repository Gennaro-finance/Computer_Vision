"""
Aspetta la catena, poi esegue cio' che alla prima passata mancava, poi mette
in ordine il repository.

PERCHE' UN PROCESSO A PARTE. catena.py e' gia' in esecuzione: aggiungere
stadi adesso non ha effetto, perche' il modulo e' stato caricato al lancio.
E questo va avviato STACCATO dalla sessione che lo crea - la catena dura
quasi un giorno, molto piu' di qualunque sessione interattiva, e un processo
figlio morirebbe insieme al genitore.

TRE PASSI:

  1. attende `runs/catena/esito.md`, che la prima passata scrive alla fine.

  2. RICOSTRUISCE I MARCATORI dagli esiti, poi rilancia catena.py. Il
     meccanismo dei marcatori e' stato aggiunto DOPO l'avvio della prima
     passata, quindi quella non ne scrive nessuno: senza questo passo la
     seconda rifarebbe da zero ore di misure gia' fatte. Si marcano solo gli
     stadi che `esito.md` dichiara riusciti - un fallito viene giustamente
     ritentato.

  3. chiama sistema_repo.py, che si ferma al commit locale su un ramo.
     Nessun push: quello lo decide un umano sveglio.
"""

import os
import re
import subprocess
import sys
import time

CARTELLA = os.path.join("runs", "catena")
ESITO = os.path.join(CARTELLA, "esito.md")
DIARIO = os.path.join(CARTELLA, "attesa_repo.log")
ORE_MAX = 30
PASSO = 300          # 5 minuti: la catena dura ore, non serve di piu'


def nota(riga):
    testo = f"[{time.strftime('%d/%m %H:%M:%S')}] {riga}"
    print(testo, flush=True)
    with open(DIARIO, "a", encoding="utf-8") as f:
        f.write(testo + "\n")


def ricostruisci_marcatori():
    """Un marcatore per ogni stadio che esito.md dichiara riuscito."""
    testo = open(ESITO, encoding="utf-8").read()
    fatti = re.findall(r"\|\s*`(\w+)`\s*\|\s*fatto\s*\|", testo)
    for nome in fatti:
        p = os.path.join(CARTELLA, f".fatto_{nome}")
        if not os.path.isfile(p):
            open(p, "w").close()
    nota(f"marcatori ricostruiti per {len(fatti)} stadi riusciti: "
         f"{', '.join(fatti) if fatti else '(nessuno)'}")
    return fatti


def esegui(cosa, *cmd):
    nota(f"{cosa}: avvio")
    t0 = time.time()
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    coda = "\n".join(r.stdout.strip().splitlines()[-25:])
    nota(f"{cosa}: codice {r.returncode} dopo {(time.time()-t0)/3600:.2f} h\n"
         f"{coda}")
    if r.stderr.strip():
        nota(f"{cosa} STDERR: " + r.stderr.strip()[-2000:])
    return r.returncode


def main():
    nota(f"in attesa di {ESITO} (limite {ORE_MAX} h, controllo ogni "
         f"{PASSO//60} min)")
    t0 = time.time()
    while time.time() - t0 < ORE_MAX * 3600:
        if os.path.isfile(ESITO):
            nota(f"prima passata finita dopo {(time.time()-t0)/3600:.1f} h "
                 f"di attesa")
            ricostruisci_marcatori()
            esegui("seconda passata (stadi aggiunti dopo l'avvio)",
                   "catena.py")
            esegui("riordino del repository", "sistema_repo.py")
            nota("FINITO. Nessun push fatto: leggi "
                 "runs/catena/repo_proposta.md, poi decidi tu.")
            return
        time.sleep(PASSO)
    nota(f"{ORE_MAX} h senza che la catena finisse: mi fermo senza toccare "
         f"il repository. Lancia a mano  python catena.py  e poi  "
         f"python sistema_repo.py")


if __name__ == "__main__":
    main()
