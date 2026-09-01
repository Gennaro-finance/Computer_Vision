"""
INFRASTRUTTURA — pagina di stato della coda degli esperimenti.

Stato della coda in una pagina: quali stadi sono chiusi, quale gira, quanto
resta. Rigenerabile a piacere - e' una fotografia, non si aggiorna da sola.

Il tempo che resta NON e' la somma delle stime: le stime venivano da una
sola misura estrapolata a tutti gli script, e si sono rivelate larghe. Qui
si moltiplica per il rapporto reale/stimato osservato sugli stadi gia'
conclusi, che e' l'unica calibrazione che abbiamo.

Uso:  python stato.py     poi apri runs/catena/coda.html
"""

import html
import json
import os
import re
import time

import catena

REG = "runs/catena/registro.txt"
PAGINA = "runs/catena/coda.html"


def raccogli():
    """Stato di ogni stadio, letto dal registro invece che indovinato."""
    stadi = catena.costruisci(3e-5)
    reg = open(REG, encoding="utf-8").read() if os.path.isfile(REG) else ""
    durate = {n: float(d) for n, d in re.findall(r"(\w+): fatto in ([\d.]+) h", reg)}
    avvii = {n: t for t, n in re.findall(r"\[(\d\d:\d\d:\d\d)\] (\w+): AVVIO", reg)}
    falliti = set(re.findall(r"(\w+): FALLITO", reg))
    # "in corso" = avviato e non ancora concluso. Il marcatore non basta:
    # la passata in esecuzione potrebbe non scriverne (vedi catena.py).
    corrente = next((x["nome"] for x in stadi
                     if x["nome"] in avvii and x["nome"] not in durate
                     and x["nome"] not in falliti), None)
    out = []
    for x in stadi:
        n = x["nome"]
        # L'ORDINE DI QUESTI CONTROLLI E' IL PUNTO. "fatto" viene PRIMA di
        # "fallito", perche' il registro e' cumulativo e conserva anche i
        # fallimenti superati: pretrain risulta FALLITO (sforamento di
        # potenza all'epoca 289) ma ha un marcatore scritto dopo, a mano,
        # una volta accertato che i checkpoint erano integri. Con la
        # precedenza invertita risultava non fatto, e le sue 9 ore
        # rientravano nel tempo residuo spostando la fine attesa di
        # quattro ore e mezza.
        st = ("fatto" if n in durate or os.path.isfile(catena.marcatore(n)) else
              "fallito" if n in falliti else
              "corso" if n == corrente else "coda")
        out.append({"nome": n, "cosa": x["cosa"], "ore": x["ore"],
                    "stato": st, "reale": durate.get(n)})
    return out


def scrivi(d):
    sp = [(x["ore"], x["reale"]) for x in d if x["reale"]]
    fatt = (sum(r for _, r in sp) / sum(s for s, _ in sp)) if sp else 1.0
    resta = sum(x["ore"] for x in d if x["stato"] not in ("fatto",)) * fatt
    fine = time.strftime("%H:%M", time.localtime(time.time() + resta * 3600))
    nomi = {"fatto": "fatto", "corso": "in corso", "coda": "in coda",
            "fallito": "fallito"}
    righe = "".join(
        f'<tr class="{x["stato"]}"><td class="n">{html.escape(x["nome"])}</td>'
        f'<td class="c">{html.escape(x["cosa"])}</td>'
        f'<td><span class="b {x["stato"]}">{nomi[x["stato"]]}</span></td>'
        f'<td class="t">~{x["ore"]:.1f} h</td>'
        f'<td class="t r">{f"{x['reale']:.2f} h" if x["reale"] else "—"}</td></tr>'
        for x in d)
    fatti = sum(1 for x in d if x["stato"] == "fatto")
    corso = next((x["nome"] for x in d if x["stato"] == "corso"), "—")
    stile = open(os.path.join(os.path.dirname(PAGINA), "stile.css"),
                 encoding="utf-8").read()
    open(PAGINA, "w", encoding="utf-8").write(
        f"<title>Coda della catena</title>\n<style>{stile}</style>\n"
        f'<div class="wrap"><h1>Coda della catena</h1>'
        f'<p class="sub">Istantanea delle {time.strftime("%H:%M")} '
        f"&middot; non si aggiorna da sola</p>"
        f'<div class="tiles">'
        f'<div class="tile"><div class="k">fatti</div><div class="v">{fatti}/{len(d)}</div></div>'
        f'<div class="tile"><div class="k">in corso</div>'
        f'<div class="v" style="font-size:15px">{html.escape(corso)}</div></div>'
        f'<div class="tile"><div class="k">resta (stima)</div><div class="v">{resta:.1f} h</div></div>'
        f'<div class="tile"><div class="k">fine attesa</div><div class="v">{fine}</div></div>'
        f"</div>"
        f'<div class="scroll"><table><thead><tr><th>stadio</th><th>cosa misura</th>'
        f"<th>stato</th><th>stima</th><th>reale</th></tr></thead>"
        f"<tbody>{righe}</tbody></table></div>"
        f'<p class="nota">Gli stadi conclusi hanno impiegato <b>{fatt:.0%}</b> del tempo '
        f"stimato, e la fine attesa applica quel fattore a ci&ograve; che resta.<br>"
        f"Al termine parte da solo il riordino del repository: commit sul ramo "
        f"<b>catena-300</b>, <b>nessun push</b>.</p></div>")
    return fatti, len(d), fatt, fine


if __name__ == "__main__":
    d = raccogli()
    f, n, fatt, fine = scrivi(d)
    for x in d:
        print(f"  {x['stato']:8s} {x['nome']:18s} ~{x['ore']:.1f}h"
              + (f"  reale {x['reale']:.2f}h" if x["reale"] else ""))
    print(f"\n{f}/{n} fatti   fattore reale/stima {fatt:.2f}   fine ~{fine}")
    print(f"pagina in {PAGINA}")
