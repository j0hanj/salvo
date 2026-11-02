"""wrap a trained EdgeScorer as a sim policy: featurize, score every legal shot, decode to shots.

decode="argmax": each effector fires along its best edge if the model's confident enough. every
  effector decides on its own, so nothing stops 20 of them picking the same threat.
decode="capped": walk edges from most to least confident and take one if the effector is still
  free and the threat isn't already covered (expected survival from the shots assigned so far
  below `survival_floor`). an effector whose favorite is covered falls through to its next pick.
"""

import numpy as np
import torch

from learned.features import featurize
from sim.engine import Obs


class GNNPolicy:
    def __init__(
        self, model, threshold: float = 0.5, decode: str = "argmax", survival_floor: float = 0.1
    ):
        if decode not in ("argmax", "capped"):
            raise ValueError(f"unknown decode {decode!r}")
        self.model = model.eval()
        self.threshold = threshold
        self.decode = decode
        self.survival_floor = survival_floor
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
        if self.decode == "capped":
            return self._capped(obs, g, p)
        pairs = []
        for e in np.unique(g.edges[0]):
            mine = np.flatnonzero(g.edges[0] == e)
            best = mine[np.argmax(p[mine])]
            if p[best] >= self.threshold:
                pairs.append((int(e), int(g.edges[1, best])))
        return pairs

    def _capped(self, obs, g, p):
        pk = np.array([obs.types[k].p_kill for k in obs.eff_type])
        survive = np.ones(len(obs.threat_value))
        used = set()
        pairs = []
        # stable sort so ties break the same way every run
        for i in np.argsort(-p, kind="stable"):
            if p[i] < self.threshold:
                break
            e, j = int(g.edges[0, i]), int(g.edges[1, i])
            if e in used or survive[j] <= self.survival_floor:
                continue
            used.add(e)
            survive[j] *= 1 - pk[e]
            pairs.append((e, j))
        return pairs
