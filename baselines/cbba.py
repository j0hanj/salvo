"""CBBA (choi, brunet & how 2009), the standard decentralized auction for task assignment.

every effector is its own agent. no central solver: each one bids on threats from its own scores,
then agents swap what they think the winning bids are w their neighbors and drop any threat
they've been outbid on. repeat until nobody changes their mind.

bundle length is 1 here. an effector fires at most one interceptor per tick and everything gets
re-planned every tick, so each agent only ever holds one task. that's the single-assignment
version from the same paper (CBAA): same auction + max-consensus, no bundle bookkeeping.

`auction` takes the comm graph as an adjacency matrix so phase 4 can thin it out / drop messages.
w everyone connected it converges to the same assignment greedy-value picks (the paper proves
this for the fully connected case), which is what the tests check.
"""

import numpy as np

from baselines.common import shot_value
from sim.engine import Obs


def auction(v: np.ndarray, adj: np.ndarray | None = None, max_rounds: int = 1000):
    """v: (m, n) bid of agent e on task j, -inf / <= 0 = won't bid. adj: (m, m) bool, who hears
    whom (self loop implied), default everyone. returns (task per agent or -1, rounds used).

    ties on bids break toward the lower agent id, so the outcome doesn't depend on message order.
    """
    m, n = v.shape
    if adj is None:
        adj = np.ones((m, m), dtype=bool)
    adj = adj | np.eye(m, dtype=bool)
    x = np.full(m, -1)  # each agent's own task
    y = np.zeros((m, n))  # agent i's view of the winning bid on every task
    z = np.full((m, n), -1)  # agent i's view of who holds every task

    for rnd in range(1, max_rounds + 1):
        changed = False
        # 1. bid: a free agent takes its best task where it'd beat the current winner it knows of
        for i in np.flatnonzero(x < 0):
            beats = (v[i] > y[i]) | ((v[i] == y[i]) & ((z[i] < 0) | (i < z[i])))
            ok = beats & (v[i] > 0)
            if ok.any():
                j = int(np.argmax(np.where(ok, v[i], -np.inf)))
                x[i], y[i, j], z[i, j] = j, v[i, j], i
                changed = True
        # 2. consensus: everyone takes the best (bid, lowest id) per task over what neighbors said
        y_new, z_new = y.copy(), z.copy()
        for i in range(m):
            for k in np.flatnonzero(adj[i]):
                better = (y[k] > y_new[i]) | (
                    (y[k] == y_new[i]) & (z[k] >= 0) & ((z_new[i] < 0) | (z[k] < z_new[i]))
                )
                y_new[i] = np.where(better, y[k], y_new[i])
                z_new[i] = np.where(better, z[k], z_new[i])
        changed |= not (np.array_equal(y_new, y) and np.array_equal(z_new, z))
        y, z = y_new, z_new
        # 3. outbid agents drop their task and bid again next round
        for i in np.flatnonzero(x >= 0):
            if z[i, x[i]] != i:
                x[i] = -1
                changed = True
        if not changed:
            return x, rnd
    return x, max_rounds


def cbba(obs: Obs, cost_weight: float = 0.0) -> list[tuple[int, int]]:
    """fully connected comms, i.e. the best case for CBBA. phase 4 adds loss."""
    v = shot_value(obs, cost_weight)
    if v.size == 0:
        return []
    x, _ = auction(v)
    return [(int(e), int(j)) for e, j in enumerate(x) if j >= 0]
