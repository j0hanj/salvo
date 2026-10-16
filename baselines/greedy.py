"""greedy baselines. dumb on purpose: they're the floor everything else has to beat."""

import numpy as np

from baselines.common import feasible
from sim.engine import Obs


def nearest(obs: Obs) -> list[tuple[int, int]]:
    """every ready effector shoots the threat closest to the asset that's in its range and nobody's
    already shooting at. one interceptor per threat at a time."""
    ok = feasible(obs)
    to_asset = np.linalg.norm(obs.threat_pos, axis=1)
    pairs = []
    for e in np.flatnonzero(ok.any(1)):
        cand = np.flatnonzero(ok[e])
        if len(cand):
            j = int(cand[np.argmin(to_asset[cand])])
            pairs.append((int(e), j))
            ok[:, j] = False
    return pairs
