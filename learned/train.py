"""behavior cloning. two losses:

- "bce": binary cross-entropy on every legal edge, target = did milp fire along it. each edge is
  judged alone, so what a score means depends on how many other edges the effector has.
- "listwise": per effector, a softmax over its legal edges plus a fixed "hold" logit of 0, target
  = the edge milp fired along, or hold. milp fires at most once per effector per tick, so that's
  an exact categorical. a logit is then log-odds vs holding fire, which shouldn't drift w graph
  size, and sigmoid(logit) >= 0.5 still means "better than holding" at decode time.

train/val split is by scenario, not by tick. ticks from the same episode are near-duplicates, so a
tick-level split would leak and make val look better than it is.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from learned.dataset import Batch
from learned.model import EdgeScorer


@dataclass
class TrainConfig:
    hidden: int = 64
    rounds: int = 3
    degree: bool = False  # see EdgeScorer
    loss: str = "bce"  # or "listwise", see module docstring
    lr: float = 1e-3
    epochs: int = 30
    graphs_per_step: int = 64
    val_frac: float = 0.1
    seed: int = 0


def tensors(b: Batch, ids):
    """stack graphs `ids` into one big disconnected graph, offsetting edge indices."""
    eff, thr, edges, ex, y = [], [], [], [], []
    e_off = t_off = 0
    for i in ids:
        es, ee = b.eff_ptr[i], b.eff_ptr[i + 1]
        ts, te = b.thr_ptr[i], b.thr_ptr[i + 1]
        ds, de = b.edge_ptr[i], b.edge_ptr[i + 1]
        eff.append(b.eff_x[es:ee])
        thr.append(b.thr_x[ts:te])
        edges.append(b.edges[:, ds:de] + np.array([[e_off], [t_off]]))
        ex.append(b.edge_x[ds:de])
        y.append(b.y[ds:de])
        e_off += ee - es
        t_off += te - ts
    t = torch.from_numpy
    return (
        t(np.concatenate(eff)),
        t(np.concatenate(thr)),
        t(np.concatenate(edges, axis=1)),
        t(np.concatenate(ex)),
        t(np.concatenate(y)),
    )


def split(b: Batch, val_frac: float, seed: int):
    scns = np.unique(b.scn)
    rng = np.random.default_rng(seed)
    val_scns = rng.choice(scns, max(1, int(len(scns) * val_frac)), replace=False)
    is_val = np.isin(b.scn, val_scns)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def listwise_loss(logits, y, src, n_eff):
    """mean over effectors w a legal edge of -log softmax([edge logits..., 0 = hold])[target]."""
    h = torch.zeros(n_eff, dtype=logits.dtype)
    # per-effector max (incl. the hold logit 0) for a stable logsumexp
    m = h.scatter_reduce(0, src, logits.detach(), reduce="amax", include_self=True)
    den = torch.exp(-m).index_add(0, src, torch.exp(logits - m[src]))
    lse = torch.log(den) + m
    target = h.index_add(0, src, y * logits)  # chosen edge's logit, or 0 if it held
    has = h.index_add(0, src, torch.ones_like(logits)) > 0
    return (lse - target)[has].mean()


def evaluate(model, b: Batch, ids, loss_fn) -> dict:
    model.eval()
    with torch.no_grad():
        eff, thr, edges, ex, y = tensors(b, ids)
        logits = model(eff, thr, edges, ex)
        pred = (logits > 0).float()
    return {
        "loss": float(loss_fn(logits, y, edges[0], len(eff))),
        "edge_acc": float((pred == y).float().mean()),
        # of the shots milp took, how many did the model also take
        "recall": float((pred * y).sum() / y.sum().clamp(min=1)),
        "precision": float((pred * y).sum() / pred.sum().clamp(min=1)),
    }


def train(b: Batch, cfg: TrainConfig, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tr, va = split(b, cfg.val_frac, cfg.seed)
    model = EdgeScorer(cfg.hidden, cfg.rounds, cfg.degree)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # most legal edges aren't taken, so upweight the positives to balance the classes
    pos = float(
        b.y[np.concatenate([np.arange(b.edge_ptr[i], b.edge_ptr[i + 1]) for i in tr])].mean()
    )
    if cfg.loss == "listwise":
        loss_fn = listwise_loss
    elif cfg.loss == "bce":
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
        loss_fn = lambda logits, y, src, n_eff: bce(logits, y)  # noqa: E731
    else:
        raise ValueError(f"unknown loss {cfg.loss!r}")
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), cfg.graphs_per_step):
            eff, thr, edges, ex, y = tensors(b, order[k : k + cfg.graphs_per_step])
            loss = loss_fn(model(eff, thr, edges, ex), y, edges[0], len(eff))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = {"epoch": epoch, **{f"val_{k}": v for k, v in evaluate(model, b, va, loss_fn).items()}}
        history.append(m)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    return model, history
