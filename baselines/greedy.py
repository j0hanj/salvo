"""greedy baselines. dumb on purpose: they're the floor everything else has to beat."""

import numpy as np

from baselines.common import feasible, shot_value
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


def value(obs: Obs, cost_weight: float = 0.0) -> list[tuple[int, int]]:
    """take the single best (effector, threat) shot by expected value, lock both out, repeat.
    the classic greedy WTA heuristic, myopic: it never trades a good shot now for a better
    combined plan. shots worth <= 0 after the cost penalty don't get taken."""
    v = shot_value(obs, cost_weight)
    used_e, used_j = set(), set()
    pairs = []
    # stable sort so ties break the same way every run
    for flat in np.argsort(-v, axis=None, kind="stable"):
        e, j = (int(i) for i in np.unravel_index(flat, v.shape))
        if not v[e, j] > 0:
            break
        if e in used_e or j in used_j:
            continue
        used_e.add(e)
        used_j.add(j)
        pairs.append((e, j))
    return pairs
