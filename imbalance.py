"""
Imbalance - metodi per lo sbilanciamento di classe, baseline e la novita'.

Copre l'obiettivo 3 del brief: "formulate and integrate an original
algorithmic novelty specifically designed to combat class imbalance in the
latent space".

--------------------------------------------------------------------------
LA NOVITA' E' VOSTRA. Qui trovate:
  - le baseline complete e funzionanti (CE, pesata, focal, oversampling)
  - la baseline scomoda, SMOTE latente, come TODO strutturato
  - la novita' proposta, `balanced_tokens`, come TODO con il ragionamento

Lo sbilanciamento e' 7:1, non 100:1. E' serio ma non estremo, quindi le
baseline standard funzioneranno decentemente e la vostra novita' deve
batterle su un margine ristretto. Progettate la valutazione per rilevare
differenze piccole: N_SEEDS seed e intervalli di confidenza, non un run.
--------------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F

from globals import FOCAL_GAMMA, NUM_CLASSES


# ==========================================================================
# Pesi di classe
# ==========================================================================
def class_counts(labels, num_classes=NUM_CLASSES):
    return torch.bincount(labels.long(), minlength=num_classes).float()


def inverse_frequency_weights(labels, num_classes=NUM_CLASSES):
    c = class_counts(labels, num_classes).clamp(min=1)
    w = c.sum() / (num_classes * c)
    return w / w.mean()


def ldam_margins(labels, num_classes=NUM_CLASSES, max_margin=0.5):
    """
    Margini in stile LDAM: delta_c proporzionale a n_c^(-1/4).

    Non e' la novita' (e' del 2019), ma e' il termine di confronto corretto
    se la vostra novita' agisce sui margini nello spazio latente.
    """
    c = class_counts(labels, num_classes).clamp(min=1)
    m = c.pow(-0.25)
    return m / m.max() * max_margin


# ==========================================================================
# Loss
# ==========================================================================
def focal_loss(logits, targets, gamma=FOCAL_GAMMA, weight=None):
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, targets[:, None]).squeeze(1)
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def ordinal_loss(cum_logits, targets, num_classes=NUM_CLASSES, weight=None):
    """
    BCE sui logit cumulativi, per la testa ordinale (stile CORAL).

    Penalizza correttamente gli errori a due gradi di distanza: e' il motivo
    per cui la testa ordinale ha senso su una scala come il PAI.
    """
    lv = torch.arange(num_classes - 1, device=targets.device)[None, :]
    t = (targets[:, None] > lv).float()
    loss = F.binary_cross_entropy_with_logits(cum_logits, t, reduction="none")
    if weight is not None:
        loss = loss * weight[targets][:, None]
    return loss.mean()


# ==========================================================================
# Sampler
# ==========================================================================
def balanced_sampler_weights(labels, num_classes=NUM_CLASSES):
    """Pesi per WeightedRandomSampler: oversampling della minoritaria."""
    c = class_counts(labels, num_classes).clamp(min=1)
    return (1.0 / c)[labels.long()]


# ==========================================================================
# BASELINE SCOMODA - SMOTE latente
# ==========================================================================
def _smote_pairs(feats_c, k, n_new, gen):
    """
    Sceglie (ancora, vicino, lambda) per n_new campioni sintetici.

    Isolato dal resto perche' lo usano sia la versione su vettori sia quella
    sui token: la logica dei vicini e' la stessa, cambia solo cosa si
    interpola.
    """
    n = feats_c.shape[0]
    # k vicini ESCLUSO se stesso: con topk(k+1) il primo e' sempre il punto
    # stesso (distanza 0), quindi si scarta la colonna 0.
    kk = min(k + 1, n)
    d = torch.cdist(feats_c, feats_c)
    nn_idx = d.topk(kk, largest=False).indices[:, 1:]   # (n, kk-1)
    if nn_idx.shape[1] == 0:                            # classe con 1 solo campione
        nn_idx = torch.zeros(n, 1, dtype=torch.long)

    anchors = torch.randint(0, n, (n_new,), generator=gen)
    picks = torch.randint(0, nn_idx.shape[1], (n_new,), generator=gen)
    neighbors = nn_idx[anchors, picks]
    lam = torch.rand(n_new, generator=gen)
    return anchors, neighbors, lam


def latent_smote(features, labels, num_classes=NUM_CLASSES, k=5, seed=0):
    """
    SMOTE nello spazio latente - la baseline che vi mette alla prova.

    Con l'encoder congelato i vostri campioni SONO vettori di feature, quindi
    SMOTE si applica direttamente nello spazio latente, senza generare pixel:

      1. per ogni classe minoritaria, trovare i k vicini piu' prossimi
         all'interno della stessa classe
      2. generare campioni sintetici interpolando:
             x_new = x_i + lambda * (x_vicino - x_i),  lambda ~ U(0,1)
      3. generarne quanti bastano a pareggiare la classe maggioritaria

    Perche' conta: e' notoriamente forte e costa venti righe. Se la vostra
    novita' non la batte, dovete saperlo PRIMA della presentazione, non
    durante le domande. Riportare onestamente "la nostra novita' pareggia
    SMOTE latente ma non lo supera" e' molto meglio che ometterlo - se lo
    omettete, sara' la prima cosa che vi chiedono.

    ATTENZIONE al leakage: generate i sintetici SOLO dal train split. Se
    interpolate campioni di validation nel train, i risultati sono falsi.

    Ritorna: (features_aumentate, labels_aumentate)
    """
    gen = torch.Generator().manual_seed(seed)
    feats = features.float()
    counts = class_counts(labels, num_classes)
    target = int(counts.max().item())

    extra_f, extra_y = [], []
    for c in range(num_classes):
        sel = (labels == c).nonzero(as_tuple=True)[0]
        n_new = target - sel.numel()
        if sel.numel() < 2 or n_new <= 0:
            continue
        fc = feats[sel]
        a, nb, lam = _smote_pairs(fc, k, n_new, gen)
        extra_f.append(fc[a] + lam[:, None] * (fc[nb] - fc[a]))
        extra_y.append(torch.full((n_new,), c, dtype=labels.dtype))

    if not extra_f:
        return features, labels
    return (torch.cat([feats] + extra_f), torch.cat([labels] + extra_y))


def latent_smote_tokens(tokens, token_mask, labels, num_classes=NUM_CLASSES,
                        k=5, seed=0, max_per_class=None):
    """
    SMOTE applicato alle MAPPE DI TOKEN, non ai vettori gia' aggregati.

    Perche' questa variante esiste: la testa del progetto lavora su token
    (B, T, D) piu' maschera, non su un vettore per lesione. Riducendo tutto a
    un vettore prima di SMOTE si cambierebbe l'architettura solo per una
    baseline, e il confronto con la novita' non sarebbe piu' alla pari.
    Qui i vicini si cercano sui token aggregati (media mascherata, che e'
    la stessa cosa che vede il pooling) ma si interpolano le mappe intere;
    la maschera dell'ancora viene mantenuta, cosi' la geometria resta
    plausibile invece di mescolare due bbox diverse.

    Ritorna: (tokens_aumentati, mask_aumentata, labels_aumentate)
    """
    gen = torch.Generator().manual_seed(seed)
    m = token_mask.float().unsqueeze(-1)
    pooled = (tokens.float() * m).sum(1) / m.sum(1).clamp(min=1)

    counts = class_counts(labels, num_classes)
    target = int(counts.max().item())
    if max_per_class is not None:
        target = min(target, max_per_class)

    tk, mk, yk = [tokens], [token_mask], [labels]
    for c in range(num_classes):
        sel = (labels == c).nonzero(as_tuple=True)[0]
        n_new = target - sel.numel()
        if sel.numel() < 2 or n_new <= 0:
            continue
        a, nb, lam = _smote_pairs(pooled[sel], k, n_new, gen)
        ta, tb = tokens[sel][a].float(), tokens[sel][nb].float()
        new = ta + lam[:, None, None] * (tb - ta)
        tk.append(new.to(tokens.dtype))
        mk.append(token_mask[sel][a])
        yk.append(torch.full((n_new,), c, dtype=labels.dtype))

    return torch.cat(tk), torch.cat(mk), torch.cat(yk)


# ==========================================================================
# NOVITA' PROPOSTA - Balanced token sampling
# ==========================================================================
def n_views_per_class(counts, alpha=0.5, num_classes=NUM_CLASSES):
    """
    Quante viste per classe: n_c = ceil((max_count / count_c) ** alpha).

    alpha e' il grado di aggressivita' del ribilanciamento ed e' il parametro
    da mettere a sweep nell'ablation: 0 = nessun ribilanciamento (una vista
    per tutti), 1 = pareggio completo delle frequenze.
    """
    c = counts.float().clamp(min=1)
    v = (c.max() / c) ** alpha
    return v.ceil().long()


def balanced_token_sampling(tokens, token_mask, labels, counts,
                            num_classes=NUM_CLASSES, generator=None,
                            alpha=0.5, p_min=0.6, p_max=1.0, min_tokens=4,
                            attn_weights=None):
    """
    Balanced token sampling - la novita' metodologica del progetto.

    Il brief nomina esplicitamente le "balanced token-sampling strategies"
    tra gli esempi, ed e' la piu' elegante delle tre opzioni discusse
    nell'analisi.

    L'IDEA
    L'attention pooling aggrega i token che cadono dentro la bbox della
    lesione. Una bbox ne contiene diversi (verificatelo: dovrebbero essere
    ~10-36 alla risoluzione corretta). Invece di aggregarli sempre tutti,
    per le classi minoritarie si campionano SOTTOINSIEMI DIVERSI di token
    dalla stessa lesione, e ogni sottoinsieme diventa un'istanza di training
    distinta.

    E' oversampling nello spazio dei token:
      - non duplica immagini (a differenza dell'oversampling classico, che
        ripresenta lo stesso vettore identico e invita all'overfitting)
      - non genera pixel sintetici (a differenza di SMOTE, che interpola
        in un latente dove l'interpolazione puo' non avere senso anatomico)
      - ogni istanza e' una vista GENUINA della stessa lesione reale

    IMPLEMENTAZIONE
      1. numero di viste per classe inversamente proporzionale alla
         frequenza: n_views_c = ceil((max_count / count_c) ** alpha)
         Con 7:1 e alpha=0.5, PAI 5 riceve ~3 viste e PAI 3 una sola.
      2. per ogni vista, tenere una frazione casuale p ~ U(p_min, p_max) dei
         token dentro la maschera (es. 0.6-1.0)
      3. garantire un minimo di token per vista, altrimenti su bbox piccole
         si aggrega rumore
      4. restituire le viste espanse con le etichette replicate

    SCELTE DA GIUSTIFICARE IN PRESENTAZIONE (ve le chiederanno)
      - alpha: quanto aggressivo e' il ribilanciamento? Sweep su
        {0.25, 0.5, 0.75, 1.0}.
      - p_min/p_max: se togliete troppi token perdete il segnale, se ne
        togliete troppo pochi le viste sono quasi identiche e non aggiungono
        niente. C'e' un optimum e va mostrato.
      - campionamento casuale uniforme o pesato per il peso di attenzione?
        Quest'ultimo e' piu' interessante: tenere i token a cui il pooling
        da' meno peso forza il modello a non dipendere da un solo token.
      - applicarlo solo in training, MAI in valutazione. In test si usano
        tutti i token: altrimenti confrontate cose diverse.

    ABLATION RICHIESTO DALL'OBIETTIVO 4
      - senza la novita' (CE semplice)
      - la novita' con alpha variabile
      - la novita' contro ogni baseline di questo file
      - metriche sulla minoritaria: Macro-F1, PR-AUC, confusion matrix
        (non l'accuracy globale: PAI 3 e' il 61%, un modello costante fa 61%)

    NOTA SULL'USO: va chiamata PER BATCH dentro il ciclo di training, non una
    volta sola sul dataset. Due motivi: le viste cambiano a ogni epoca (che e'
    il punto - viste identiche non aggiungono niente) e non si duplica in
    memoria l'intero tensore dei token.

    Ritorna: (tokens_espansi, mask_espansa, labels_espanse)
    """
    views = n_views_per_class(counts, alpha, num_classes).to(labels.device)
    per_sample = views[labels.long()]                      # viste per campione
    idx = torch.repeat_interleave(
        torch.arange(labels.shape[0], device=labels.device), per_sample
    )

    tok = tokens[idx]
    base = token_mask[idx]
    y = labels[idx]

    # La PRIMA vista di ogni campione resta integra: e' l'istanza originale.
    # Senza questa garanzia anche la maggioritaria verrebbe sottocampionata e
    # il confronto con le baseline non sarebbe piu' alla pari.
    first = torch.zeros(idx.shape[0], dtype=torch.bool, device=labels.device)
    first[torch.cumsum(per_sample, 0) - per_sample] = True

    n, t = base.shape
    p = torch.empty(n, device=base.device).uniform_(p_min, p_max, generator=generator)

    if attn_weights is None:
        score = torch.rand(n, t, device=base.device, generator=generator)
    else:
        # Campionamento pesato al CONTRARIO dell'attenzione: si tengono di
        # preferenza i token a cui il pooling da' meno peso, cosi' il modello
        # non puo' appoggiarsi a un singolo token dominante. E' la variante
        # piu' interessante da difendere in presentazione.
        w = attn_weights[idx].clamp(min=1e-6)
        score = torch.rand(n, t, device=base.device, generator=generator) / w

    # Si ordina solo dentro la maschera: i token fuori bbox non sono candidati.
    score = score.masked_fill(~base, float("inf"))
    n_in = base.sum(1)
    keep_n = torch.maximum((n_in.float() * p).long(),
                           torch.minimum(n_in, torch.full_like(n_in, min_tokens)))

    order = score.argsort(dim=1)
    ranks = order.argsort(dim=1)
    new_mask = ranks < keep_n[:, None]
    new_mask &= base
    new_mask[first] = base[first]

    # Salvagente: una vista senza token non e' aggregabile.
    vuote = ~new_mask.any(dim=1)
    if vuote.any():
        new_mask[vuote] = base[vuote]

    return tok, new_mask, y


# ==========================================================================
# Dispatcher
# ==========================================================================
def compute_loss(logits, targets, method="none", head_type="flat",
                 train_labels=None):
    """
    Contratto unico verso train_downstream.py.

    Definite questa firma il giorno 1 e non cambiatela: e' cio' che permette
    alla Persona B di lavorare sulle baseline mentre la Persona A sta ancora
    addestrando il SSL.
    """
    weight = None
    if method in ("class_weighted", "focal") and train_labels is not None:
        weight = inverse_frequency_weights(train_labels).to(logits.device)

    if head_type == "ordinal":
        return ordinal_loss(logits, targets, weight=weight)

    if method == "focal":
        return focal_loss(logits, targets, weight=weight)
    return F.cross_entropy(logits, targets, weight=weight)


if __name__ == "__main__":
    # distribuzione reale attesa del dataset
    labels = torch.cat([
        torch.zeros(3691), torch.ones(1817), torch.full((521,), 2)
    ]).long()

    print("Distribuzione:", class_counts(labels).tolist())
    print(f"Sbilanciamento max:min = {3691/521:.2f} : 1")
    print("Pesi inverse-frequency:", [round(v, 3) for v in inverse_frequency_weights(labels).tolist()])
    print("Margini LDAM           :", [round(v, 3) for v in ldam_margins(labels).tolist()])

    logits = torch.randn(8, 3)
    t = torch.randint(0, 3, (8,))
    print(f"\nCE          : {compute_loss(logits, t, 'none').item():.4f}")
    print(f"CE pesata   : {compute_loss(logits, t, 'class_weighted', train_labels=labels).item():.4f}")
    print(f"Focal       : {compute_loss(logits, t, 'focal', train_labels=labels).item():.4f}")
    print(f"Ordinale    : {compute_loss(torch.randn(8,2), t, 'none', 'ordinal').item():.4f}")

    w = balanced_sampler_weights(labels)
    print(f"\nPesi sampler: PAI3={w[0]:.2e}  PAI4={w[4000]:.2e}  PAI5={w[-1]:.2e}")
    print(f"  rapporto PAI5/PAI3 = {(w[-1]/w[0]).item():.2f} (atteso ~7.08)")
