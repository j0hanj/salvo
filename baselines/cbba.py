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
from sim.comms import GilbertElliott
from sim.engine import Obs


def auction(
    v: np.ndarray,
    adj: np.ndarray | None = None,
    max_rounds: int = 1000,
    p_drop: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """v: (m, n) bid of agent e on task j, -inf / <= 0 = won't bid. adj: (m, m) bool, who hears
    whom (self loop implied), default everyone. returns (task per agent or -1, rounds used).

    p_drop: chance each message is lost, per link per round. w loss an agent can't tell "nothing
    changed" from "i heard nothing", so it runs the full max_rounds budget and everyone acts on
    whatever they hold at the end. conflicts (two agents on one threat) are possible then, which
    is the point: that's the degradation phase 4 measures.

    ties on bids break toward the lower agent id, so the outcome doesn't depend on message order.
    """
    if p_drop > 0 and rng is None:
        raise ValueError("p_drop > 0 needs an rng, so runs stay reproducible")
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
        # 2. consensus: everyone takes the best (bid, lowest id) per task over what it heard
        heard = adj
        if p_drop > 0:
            # each directed message independently lost; you always hear yourself
            heard = adj & ((rng.random((m, m)) >= p_drop) | np.eye(m, dtype=bool))
        y_new, z_new = np.empty_like(y), np.empty_like(z)
        for i in range(m):
            yk, zk = y[heard[i]], z[heard[i]]
            best = yk.max(0)
            tied = (yk == best) & (zk >= 0)
            zb = np.where(tied, zk, m).min(0)
            y_new[i] = best
            z_new[i] = np.where(zb < m, zb, -1)
        changed |= not (np.array_equal(y_new, y) and np.array_equal(z_new, z))
        y, z = y_new, z_new
        # 3. outbid agents drop their task and bid again next round
        for i in np.flatnonzero(x >= 0):
            if z[i, x[i]] != i:
                x[i] = -1
                changed = True
        if not changed and p_drop == 0:
            return x, rnd
    return x, max_rounds


def cbba(obs: Obs, cost_weight: float = 0.0) -> list[tuple[int, int]]:
    """fully connected comms, i.e. the best case for CBBA. phase 4 adds loss."""
    v = shot_value(obs, cost_weight)
    if v.size == 0:
        return []
    x, _ = auction(v)
    return [(int(e), int(j)) for e, j in enumerate(x) if j >= 0]


class LossyCBBA:
    """cbba where every message is dropped w prob p_drop, and agents get `rounds` rounds of
    talking per tick before they have to act. the rng is reseeded per scenario (`reset`) so a
    scenario's losses don't depend on which other scenarios ran before it in the same process."""

    def __init__(self, p_drop: float, rounds: int = 10, cost_weight: float = 0.0):
        self.p_drop, self.rounds, self.cost_weight = p_drop, rounds, cost_weight
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: Obs) -> list[tuple[int, int]]:
        v = shot_value(obs, self.cost_weight)
        if v.size == 0 or not np.isfinite(v).any():
            return []
        x, _ = auction(v, max_rounds=self.rounds, p_drop=self.p_drop, rng=self.rng)
        return [(int(e), int(j)) for e, j in enumerate(x) if j >= 0]


class BurstyCBBA:
    """cbba over gilbert-elliott links (`sim.comms`): a link that's down is down for every round
    of this tick, so talking longer can only route around it through other effectors. the link
    chain steps once per replan, whether or not there's anything to decide."""

    def __init__(self, loss: float, burst: float, rounds: int = 10, cost_weight: float = 0.0):
        self.loss, self.burst, self.rounds, self.cost_weight = loss, burst, rounds, cost_weight
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.links = None  # built on first call, once the effector count is known

    def __call__(self, obs: Obs) -> list[tuple[int, int]]:
        if self.links is None:
            self.links = GilbertElliott(len(obs.eff_pos), self.loss, self.burst, self.rng)
        up = self.links.step()
        v = shot_value(obs, self.cost_weight)
        if v.size == 0 or not np.isfinite(v).any():
            return []
        x, _ = auction(v, adj=up, max_rounds=self.rounds)
        return [(int(e), int(j)) for e, j in enumerate(x) if j >= 0]
