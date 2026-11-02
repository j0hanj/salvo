"""edge-scoring GNN over the effector/threat bipartite graph. plain torch, no graph library.

each round of message passing:
  1. every edge updates from (itself, its effector, its threat)
  2. every threat updates from the mean + max of its edges (who could shoot me, how well)
  3. every effector updates from the mean + max of its edges (what could i shoot, how good)
then each edge gets a logit = "the expert would fire along this edge".

mean + max instead of sum so the numbers don't scale w how many neighbors a node has. that's what
should let it run on graphs way bigger than it trained on. no node ids anywhere, so it's
permutation equivariant: shuffle the threats and the scores shuffle the same way.
"""

import torch
from torch import nn

from learned.features import EDGE_FEATURES, EFF_FEATURES, THR_FEATURES


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def scatter_mean_max(src, index, n):
    """mean and max of src rows grouped by index into n buckets. empty buckets get 0."""
    h = src.shape[1]
    total = torch.zeros(n, h, dtype=src.dtype).index_add_(0, index, src)
    count = torch.zeros(n, dtype=src.dtype).index_add_(
        0, index, torch.ones_like(index, dtype=src.dtype)
    )
    mean = total / count.clamp(min=1).unsqueeze(1)
    mx = torch.zeros(n, h, dtype=src.dtype).scatter_reduce(
        0, index.unsqueeze(1).expand(-1, h), src, reduce="amax", include_self=False
    )
    return mean, mx


class Round(nn.Module):
    def __init__(self, h, degree: bool = False):
        super().__init__()
        d = int(degree)
        self.edge = mlp(3 * h, h, h)
        self.thr = mlp(3 * h + d, h, h)
        self.eff = mlp(3 * h + d, h, h)

    def forward(self, he, ht, hd, src, dst, t_deg=None, e_deg=None):
        """t_deg / e_deg: (n, 1) log(1 + legal edges) per node, or None to leave degree out."""
        # residual updates everywhere so stacking rounds doesn't wash out the input features
        hd = hd + self.edge(torch.cat([hd, he[src], ht[dst]], 1))
        t_mean, t_max = scatter_mean_max(hd, dst, len(ht))
        ht = ht + self.thr(
            torch.cat([ht, t_mean, t_max, *([t_deg] if t_deg is not None else [])], 1)
        )
        e_mean, e_max = scatter_mean_max(hd, src, len(he))
        he = he + self.eff(
            torch.cat([he, e_mean, e_max, *([e_deg] if e_deg is not None else [])], 1)
        )
        return he, ht, hd


class EdgeScorer(nn.Module):
    def __init__(self, hidden: int = 64, rounds: int = 3, degree: bool = False):
        """degree=True adds log(1 + #legal edges) to every node update. mean + max alone can't
        tell 3 competing shooters from 50, which is the suspected reason it over-stacks at scale."""
        super().__init__()
        self.degree = degree
        self.enc_eff = mlp(len(EFF_FEATURES), hidden, hidden)
        self.enc_thr = mlp(len(THR_FEATURES), hidden, hidden)
        self.enc_edge = mlp(len(EDGE_FEATURES), hidden, hidden)
        self.rounds = nn.ModuleList(Round(hidden, degree) for _ in range(rounds))
        self.head = mlp(3 * hidden, hidden, 1)

    def forward(self, eff_x, thr_x, edges, edge_x):
        """edges: (2, E) long, row 0 indexes eff_x, row 1 indexes thr_x. returns (E,) logits."""
        src, dst = edges
        he, ht, hd = self.enc_eff(eff_x), self.enc_thr(thr_x), self.enc_edge(edge_x)
        t_deg = e_deg = None
        if self.degree:
            ones = torch.ones_like(src, dtype=he.dtype)
            t_deg = torch.log1p(torch.zeros(len(ht)).index_add_(0, dst, ones)).unsqueeze(1)
            e_deg = torch.log1p(torch.zeros(len(he)).index_add_(0, src, ones)).unsqueeze(1)
        for r in self.rounds:
            he, ht, hd = r(he, ht, hd, src, dst, t_deg, e_deg)
        return self.head(torch.cat([hd, he[src], ht[dst]], 1)).squeeze(1)
