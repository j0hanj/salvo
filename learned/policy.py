"""wrap a trained EdgeScorer as a sim policy: featurize, score every legal shot, then each effector
fires along its best edge if the model's confident enough. effectors can stack on one threat, same
as milp."""

import numpy as np
import torch

from learned.features import featurize
from sim.engine import Obs


class GNNPolicy:
    def __init__(self, model, threshold: float = 0.5):
        self.model = model.eval()
        self.threshold = threshold
        self._threads_set = False

    def __call__(self, obs: Obs) -> list[tuple[int, int]]:
        if not self._threads_set:
            # one thread per process: fair latency vs the other baselines + no oversubscription
            # when the harness runs scenarios in parallel
            torch.set_num_threads(1)
            self._threads_set = True
        g = featurize(obs)
        if g.edges.shape[1] == 0:
            return []
        with torch.no_grad():
            p = torch.sigmoid(
                self.model(
                    torch.from_numpy(g.eff_x),
                    torch.from_numpy(g.thr_x),
                    torch.from_numpy(g.edges),
                    torch.from_numpy(g.edge_x),
                )
            ).numpy()
        pairs = []
        for e in np.unique(g.edges[0]):
            mine = np.flatnonzero(g.edges[0] == e)
            best = mine[np.argmax(p[mine])]
            if p[best] >= self.threshold:
                pairs.append((int(e), int(g.edges[1, best])))
        return pairs
