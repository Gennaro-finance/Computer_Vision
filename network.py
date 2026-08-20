"""
Network - ViT compatto, I-JEPA (context + target EMA + predictor), teste.

Sezione "Network" della struttura richiesta dal corso.

Obiettivo 1 del brief, alla lettera: "a Context Encoder, a Target Encoder
updated via Exponential Moving Average (EMA), and a shallow Predictor
network". E' quello che c'e' qui.

Nota strategica: LeJEPA (ref [2] del vostro brief) rimuove esattamente questi
componenti - niente EMA, niente stop-gradient, niente teacher-student - e li
sostituisce con SIGReg. Implementate I-JEPA come primario per soddisfare
l'obiettivo 1, e tenete SIGReg come braccio di confronto E come assicurazione
sul collasso. Vedi il segnaposto in fondo al file.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from globals import (
    ATTN_POOL_HEADS, CONTEXT_SCALE, NUM_CLASSES, NUM_TARGET_BLOCKS,
    PATCH_SIZE, PREDICTOR_DEPTH, PREDICTOR_DIM, PREDICTOR_HEADS, SIGREG_LAMBDA,
    SIGREG_PROJECTIONS, TARGET_ASPECT, TARGET_SCALE, TILE_SIZE, VIT_VARIANTS,
)


# ==========================================================================
# 1. Blocchi ViT
# ==========================================================================
def sincos_pos_embed(dim, grid_h, grid_w):
    """Positional embedding sinusoidale 2D (fisso, non appreso)."""
    def _1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2.0))
        out = pos.flatten()[:, None] * omega[None, :]
        return torch.cat([out.sin(), out.cos()], dim=1)

    gh = torch.arange(grid_h, dtype=torch.float32)
    gw = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(gw, gh, indexing="xy")
    emb = torch.cat([_1d(dim // 2, grid[0]), _1d(dim // 2, grid[1])], dim=1)
    return emb  # (grid_h*grid_w, dim)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """
    ViT che accetta un SOTTOINSIEME di token.

    Il supporto ai sottoinsiemi e' il requisito centrale di I-JEPA: il
    context encoder deve vedere solo le patch di contesto, non l'immagine
    intera con dei token mascherati. Da qui il parametro `keep_indices`.
    """

    def __init__(self, img_size=TILE_SIZE, patch_size=PATCH_SIZE, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.register_buffer(
            "pos_embed", sincos_pos_embed(embed_dim, self.grid, self.grid)[None],
            persistent=False,
        )
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, keep_indices=None):
        """
        x: (B, C, H, W)
        keep_indices: (B, K) indici dei token da tenere, oppure None per tutti.
        Ritorna: (B, K o N, D)
        """
        t = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, D)
        t = t + self.pos_embed

        if keep_indices is not None:
            idx = keep_indices.unsqueeze(-1).expand(-1, -1, t.shape[-1])
            t = torch.gather(t, 1, idx)

        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)


# ==========================================================================
# 2. Block masking (strategia I-JEPA)
# ==========================================================================
def sample_block(grid, scale_range, aspect_range, generator=None):
    """Campiona un blocco rettangolare di token; ritorna gli indici piatti."""
    n = grid * grid
    scale = torch.empty(1).uniform_(*scale_range, generator=generator).item()
    aspect = torch.empty(1).uniform_(*aspect_range, generator=generator).item()

    target_area = scale * n
    h = max(1, min(grid, int(round(math.sqrt(target_area / aspect)))))
    w = max(1, min(grid, int(round(math.sqrt(target_area * aspect)))))

    top = torch.randint(0, grid - h + 1, (1,), generator=generator).item()
    left = torch.randint(0, grid - w + 1, (1,), generator=generator).item()

    rows = torch.arange(top, top + h)
    cols = torch.arange(left, left + w)
    return (rows[:, None] * grid + cols[None, :]).flatten()


def sample_masks(grid, num_targets=NUM_TARGET_BLOCKS, generator=None):
    """
    Un blocco di contesto ampio + `num_targets` blocchi target piccoli,
    con i target RIMOSSI dal contesto (altrimenti il compito e' banale).

    Ritorna: (context_indices, [target_indices, ...])
    """
    n = grid * grid
    targets = [sample_block(grid, TARGET_SCALE, TARGET_ASPECT, generator)
               for _ in range(num_targets)]

    context = sample_block(grid, CONTEXT_SCALE, (1.0, 1.0), generator)
    forbidden = torch.zeros(n, dtype=torch.bool)
    for t in targets:
        forbidden[t] = True
    context = context[~forbidden[context]]

    if context.numel() == 0:   # degenere: teniamo tutto cio' che non e' target
        context = torch.arange(n)[~forbidden]
    if context.numel() == 0:
        context = torch.arange(n)[:1]
    return context, targets


# ==========================================================================
# 3. Predictor
# ==========================================================================
class Predictor(nn.Module):
    """
    Predictor shallow e stretto (obiettivo 1: "a shallow Predictor network").

    Prende i token di contesto codificati piu' dei mask token posizionati
    dove stanno i target, e predice le rappresentazioni target nello spazio
    latente. Deve restare *piccolo*: se ha troppa capacita' risolve il
    compito senza costringere l'encoder a imparare nulla.
    """

    def __init__(self, embed_dim, pred_dim=None, depth=None,
                 heads=PREDICTOR_HEADS, num_patches=196):
        super().__init__()
        # Risolti a runtime e non come default dell'argomento: i default si
        # fissano alla definizione della classe, e gli override degli
        # esperimenti (train_ssl --predictor-dim) non avrebbero effetto.
        pred_dim = PREDICTOR_DIM if pred_dim is None else pred_dim
        depth = PREDICTOR_DEPTH if depth is None else depth
        self.proj_in = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        grid = int(math.sqrt(num_patches))
        self.register_buffer(
            "pos_embed", sincos_pos_embed(pred_dim, grid, grid)[None], persistent=False
        )
        self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj_out = nn.Linear(pred_dim, embed_dim)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        """
        ctx_tokens: (B, Kc, D) uscita del context encoder
        ctx_idx:    (B, Kc) posizioni dei token di contesto
        tgt_idx:    (B, Kt) posizioni da predire
        Ritorna:    (B, Kt, D)
        """
        b, d = ctx_tokens.shape[0], self.mask_token.shape[-1]
        x = self.proj_in(ctx_tokens)
        x = x + torch.gather(
            self.pos_embed.expand(b, -1, -1), 1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, d)
        )

        m = self.mask_token.expand(b, tgt_idx.shape[1], -1)
        m = m + torch.gather(
            self.pos_embed.expand(b, -1, -1), 1,
            tgt_idx.unsqueeze(-1).expand(-1, -1, d)
        )

        z = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            z = blk(z)
        return self.proj_out(self.norm(z[:, x.shape[1]:]))


# ==========================================================================
# 4. I-JEPA
# ==========================================================================
class IJEPA(nn.Module):
    """
    Pipeline I-JEPA completa: context encoder + target encoder EMA + predictor.

    Il target encoder e' una copia dei pesi del context encoder aggiornata per
    media mobile esponenziale e MAI dai gradienti - e' il meccanismo che
    impedisce la soluzione banale, ed e' anche il pezzo piu' fragile fuori dal
    regime di iperparametri di ImageNet. Monitorate il collasso (utils.py).
    """

    def __init__(self, variant="vit_tiny", img_size=TILE_SIZE, patch_size=PATCH_SIZE):
        super().__init__()
        cfg = VIT_VARIANTS[variant]
        self.context_encoder = VisionTransformer(img_size, patch_size, **cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            cfg["embed_dim"], num_patches=self.context_encoder.num_patches
        )
        self.grid = self.context_encoder.grid
        self.embed_dim = cfg["embed_dim"]

    @torch.no_grad()
    def update_target(self, momentum: float):
        """EMA: theta_target <- m * theta_target + (1-m) * theta_context."""
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)
        for bt, bc in zip(self.target_encoder.buffers(),
                          self.context_encoder.buffers()):
            bt.copy_(bc)

    def forward(self, images, generator=None):
        """
        Ritorna (loss, embeddings_per_monitoraggio).

        Gli embedding restituiti servono a utils.CollapseMonitor: sono la
        media dei token del target encoder, cioe' la rappresentazione che
        userete a valle. Se collassano, collassa il progetto.
        """
        b = images.shape[0]
        device = images.device

        ctx_idx, tgt_blocks = sample_masks(self.grid, generator=generator)
        ctx_idx = ctx_idx.to(device)[None].expand(b, -1)

        ctx_tokens = self.context_encoder(images, ctx_idx)

        with torch.no_grad():
            full = self.target_encoder(images)          # (B, N, D)
            full = F.layer_norm(full, (full.shape[-1],))

        loss = images.new_zeros(())
        for tgt in tgt_blocks:
            tgt_idx = tgt.to(device)[None].expand(b, -1)
            target = torch.gather(
                full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
            )
            pred = self.predictor(ctx_tokens, ctx_idx, tgt_idx)
            loss = loss + F.smooth_l1_loss(pred, target)

        loss = loss / len(tgt_blocks)

        # Assicurazione sul collasso (vedi ANALISI_PROGETTO_8.md sez.3 e 5):
        # vincola gli embedding del context encoder - quello che riceve i
        # gradienti - a restare isotropi. A differenza di EMA/predictor, che
        # prevengono il collasso solo indirettamente bilanciando le due reti,
        # SIGReg lo penalizza direttamente: una rappresentazione collassata ha
        # varianza ~0 in ogni proiezione, che e' lontanissima da una N(0,1).
        if SIGREG_LAMBDA > 0:
            ctx_pooled = ctx_tokens.mean(dim=1)
            loss = loss + SIGREG_LAMBDA * sigreg_loss(ctx_pooled, SIGREG_PROJECTIONS)

        return loss, full.mean(dim=1).detach()

    @torch.no_grad()
    def encode(self, images):
        """Encoder congelato per il downstream: token del target encoder."""
        return self.target_encoder(images)


# ==========================================================================
# 4b. Braccio di confronto: encoder ImageNet congelato
# ==========================================================================
class FrozenImageNetEncoder(nn.Module):
    """
    ViT-B/16 pre-addestrato su ImageNet, congelato ed esposto con la STESSA
    interfaccia di IJEPA (encode / grid / embed_dim).

    E' il braccio 2 della sez.9 dell'analisi, quello definito "critico e non
    negoziabile": se il JEPA in-domain su ~4k immagini non batte il transfer
    da ImageNet, quello E' il risultato del progetto e va detto. Senza questo
    confronto, un numero come "Macro-F1 0.62" non dimostra niente, ed e' la
    prima domanda che arriva in sede d'esame.

    Il patch da 16 su tile da 224 da' la stessa griglia 14x14 del nostro ViT,
    quindi bbox_to_token_mask e l'attention pooling funzionano identici e il
    confronto e' davvero alla pari: cambia l'encoder, nient'altro.
    """

    def __init__(self, img_size=TILE_SIZE, patch_size=PATCH_SIZE):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.net = vit_b_16(weights=weights)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.grid = img_size // patch_size
        self.embed_dim = self.net.hidden_dim

        # I tile arrivano da data.py in [-1, 1] (grayscale replicato su 3
        # canali). Il ViT di torchvision vuole invece le statistiche di
        # ImageNet: si torna in [0, 1] e si ri-normalizza. Saltare questo
        # passaggio non da' errore, da' solo feature peggiori - cioe'
        # sabotarebbe silenziosamente proprio il braccio di confronto.
        t = weights.transforms()
        self.register_buffer("mean", torch.tensor(t.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(t.std).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, images):
        x = (images * 0.5 + 0.5 - self.mean) / self.std
        x = self.net._process_input(x)
        cls = self.net.class_token.expand(x.shape[0], -1, -1)
        x = self.net.encoder(torch.cat([cls, x], dim=1))
        return x[:, 1:]        # via il class token: restano i token di patch


# ==========================================================================
# 5. Teste downstream
# ==========================================================================
class AttentionPooling(nn.Module):
    """
    Attention pooling sui token, con maschera opzionale.

    La maschera limita l'aggregazione ai token che cadono dentro la bbox
    della lesione - che e' quello che chiede il brief ("extract the latent
    vectors corresponding to the lesion areas").
    """

    def __init__(self, dim, heads=ATTN_POOL_HEADS):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, token_mask=None):
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        kpm = ~token_mask if token_mask is not None else None
        out, w = self.attn(q, tokens, tokens, key_padding_mask=kpm)
        return self.norm(out.squeeze(1)), w


def bbox_to_token_mask(bbox, grid, patch_size=PATCH_SIZE):
    """
    Converte bbox in coordinate pixel in una maschera booleana sui token.

    Se la maschera risulta vuota per qualche campione, la bbox e' piu'
    piccola di un token: e' il sintomo del problema di scala descritto in
    ANALISI_PROGETTO_8.md sez.2. Qui teniamo almeno il token del centro, ma se
    succede spesso la risoluzione e' sbagliata.
    """
    b = bbox.shape[0]
    device = bbox.device
    x0 = (bbox[:, 0] / patch_size).floor().long().clamp(0, grid - 1)
    y0 = (bbox[:, 1] / patch_size).floor().long().clamp(0, grid - 1)
    x1 = (bbox[:, 2] / patch_size).ceil().long().clamp(1, grid)
    y1 = (bbox[:, 3] / patch_size).ceil().long().clamp(1, grid)

    cols = torch.arange(grid, device=device)[None, :]
    mask_x = (cols >= x0[:, None]) & (cols < x1[:, None])
    mask_y = (cols >= y0[:, None]) & (cols < y1[:, None])
    mask = (mask_y[:, :, None] & mask_x[:, None, :]).reshape(b, -1)

    empty = ~mask.any(dim=1)
    if empty.any():
        cx = ((bbox[:, 0] + bbox[:, 2]) / 2 / patch_size).long().clamp(0, grid - 1)
        cy = ((bbox[:, 1] + bbox[:, 3]) / 2 / patch_size).long().clamp(0, grid - 1)
        mask[empty, (cy * grid + cx)[empty]] = True
    return mask


class FlatHead(nn.Module):
    """Softmax a 3 vie - quello che chiede il brief."""

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class OrdinalHead(nn.Module):
    """
    Testa ordinale in stile CORAL - piu' appropriata al PAI.

    Il PAI e' una scala ordinale: 3 < 4 < 5. Confondere PAI 3 con PAI 5 e'
    clinicamente peggio che confondere 4 con 5, ma una softmax piatta li
    tratta allo stesso modo. Qui si predicono K-1 soglie cumulative
    P(y > 3), P(y > 4) con un peso condiviso e bias separati, il che impone
    la monotonicita'.

    Tenete comunque FlatHead come braccio di confronto: la scelta va
    argomentata con i numeri, non per principio.
    """

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.shared = nn.Linear(dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        self.num_classes = num_classes

    def forward(self, x):
        return self.shared(x) + self.biases     # (B, K-1) logit cumulativi

    @staticmethod
    def logits_to_class(cum_logits):
        return (cum_logits > 0).sum(dim=1)

    @staticmethod
    def targets(labels, num_classes=NUM_CLASSES):
        """label k -> [1]*k + [0]*(K-1-k)"""
        lv = torch.arange(num_classes - 1, device=labels.device)[None, :]
        return (labels[:, None] > lv).float()


class LesionClassifier(nn.Module):
    """Encoder CONGELATO + attention pooling + testa. Solo pooling e testa si addestrano."""

    def __init__(self, embed_dim, grid, head_type="flat"):
        super().__init__()
        self.pool = AttentionPooling(embed_dim)
        self.head = FlatHead(embed_dim) if head_type == "flat" else OrdinalHead(embed_dim)
        self.head_type = head_type
        self.grid = grid

    def forward(self, tokens, bbox=None, token_mask=None):
        if token_mask is None and bbox is not None:
            token_mask = bbox_to_token_mask(bbox, self.grid)
        pooled, attn = self.pool(tokens, token_mask)
        return self.head(pooled), pooled, attn


# ==========================================================================
# 6. Braccio di confronto LeJEPA / SIGReg
# ==========================================================================
def sigreg_loss(embeddings, num_projections=64):
    """
    SIGReg (Sketched Isotropic Gaussian Regularization), da LeJEPA
    (Balestriero & LeCun, 2025 - ref [2] del brief).

    Si campionano `num_projections` direzioni casuali unitarie in R^D, si
    proiettano gli embedding su ciascuna (-> distribuzioni 1-D), e si misura
    la discrepanza di ognuna da una N(0,1) con il criterio di
    Cramer-von Mises: CDF empirica dei valori proiettati vs CDF della
    gaussiana standard. E' l'approssimazione descritta nello schema
    originale (un'alternativa a Epps-Pulley/funzione caratteristica),
    scelta perche' e' differenziabile via erf ed economica.

    Una rappresentazione collassata ha varianza ~0 in ogni proiezione: la
    sua CDF empirica e' quasi un gradino attorno alla media, lontanissima
    dalla CDF di una N(0,1). Il criterio la penalizza forte, il che e'
    esattamente perche' funziona da assicurazione sul collasso (vedi
    ANALISI_PROGETTO_8.md sez.3 e 5) oltre che da braccio di confronto
    LeJEPA nell'ablation.

    embeddings: (B, D). Richiede B >= 2 (la CDF empirica non e' definita
    per un solo campione).

    Scorciatoia legittima per validare l'implementazione: `pip install
    lejepa` espone la loss ufficiale come riferimento. Consegnare solo
    quella al posto dell'obiettivo 1 no.
    """
    b, d = embeddings.shape
    device = embeddings.device

    directions = F.normalize(torch.randn(d, num_projections, device=device), dim=0)
    proj = embeddings.float() @ directions   # (B, P)

    sorted_proj, _ = torch.sort(proj, dim=0)
    normal = torch.distributions.Normal(0.0, 1.0)
    cdf = normal.cdf(sorted_proj)            # (B, P)

    ranks = torch.arange(1, b + 1, device=device, dtype=cdf.dtype).unsqueeze(1)
    empirical_cdf = (2 * ranks - 1) / (2 * b)

    cvm = ((cdf - empirical_cdf) ** 2).sum(dim=0) + 1.0 / (12 * b)
    return cvm.mean()


def build_ijepa(variant="vit_tiny"):
    return IJEPA(variant)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    for variant in VIT_VARIANTS:
        model = build_ijepa(variant)
        x = torch.randn(2, 3, TILE_SIZE, TILE_SIZE)
        loss, emb = model(x)
        print(f"{variant:10s} params={count_params(model)/1e6:5.2f}M  "
              f"grid={model.grid}x{model.grid}  loss={loss.item():.4f}  "
              f"emb={tuple(emb.shape)}")

    m = build_ijepa("vit_tiny")
    clf = LesionClassifier(m.embed_dim, m.grid, "ordinal")
    tokens = m.encode(torch.randn(4, 3, TILE_SIZE, TILE_SIZE))
    bbox = torch.tensor([[40., 40., 120., 120.]] * 4)
    logits, pooled, _ = clf(tokens, bbox)
    print(f"\ntoken={tuple(tokens.shape)} -> pooled={tuple(pooled.shape)} "
          f"-> logit ordinali={tuple(logits.shape)}")
    print(f"maschera token dentro bbox: {bbox_to_token_mask(bbox, m.grid).sum(1).tolist()}")
