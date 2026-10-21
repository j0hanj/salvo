"""hungarian: the best one-to-one matching of ready effectors to threats for this snapshot.

same per-shot score as greedy-value (value x kill chance - cost penalty), but optimized jointly, so
it'll give up the single best shot if that frees up a better overall plan. still myopic in time:
it only sees this tick, not what's coming next.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from baselines.common import shot_value
from sim.engine import Obs


def hungarian(obs: Obs, cost_weight: float = 0.0) -> list[tuple[int, int]]:
    v = shot_value(obs, cost_weight)
    if v.size == 0 or not np.isfinite(v).any():
        return []
    # scipy always returns a full matching, but holding fire is a real option worth 0. so clamp
    # illegal / negative shots to 0: the solver then never trades a good pair for a forced bad one
    rows, cols = linear_sum_assignment(np.clip(v, 0, None), maximize=True)
    return [(int(e), int(j)) for e, j in zip(rows, cols, strict=True) if v[e, j] > 0]
