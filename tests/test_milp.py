import itertools

import numpy as np

from baselines.common import feasible
from baselines.hungarian import hungarian
from baselines.milp import milp, objective
from tests.test_baselines import check_shooters, kind, obs


def brute_force(o, cost_weight):
    """every effector picks one reachable threat or holds fire. score w the real objective."""
    ok = feasible(o)
    options = [[None, *np.flatnonzero(ok[e])] for e in range(len(o.eff_pos))]
    best = 0.0
    for choice in itertools.product(*options):
        pairs = [(e, j) for e, j in enumerate(choice) if j is not None]
        best = max(best, objective(o, pairs, cost_weight))
    return best


def test_doubles_up_on_a_big_threat():
    # one threat worth 10, two 50% shooters. both fire: 7.5 expected. hungarian can only send one: 5
    o = obs([[0, 1000]], [10], [[0, 0], [10, 0]], [0, 0], [kind("a", 0.5)])
    pairs = milp(o)
    assert sorted(pairs) == [(0, 0), (1, 0)]
    assert objective(o, pairs) > objective(o, hungarian(o))


def test_holds_fire_when_not_worth_it():
    o = obs([[0, 1000]], [1], [[0, 0]], [0], [kind("pricey", 0.9, cost=20)])
    assert milp(o, cost_weight=0.05) == []


def test_matches_brute_force():
    rng = np.random.default_rng(2)
    types = [kind("a", 0.7, cost=1, rng=2000), kind("b", 0.9, cost=20, rng=4000)]
    for _ in range(60):
        n, m = rng.integers(1, 5), rng.integers(1, 5)
        o = obs(
            rng.uniform(-4000, 4000, (n, 2)),
            rng.lognormal(0, 1, n),
            rng.uniform(-500, 500, (m, 2)),
            rng.integers(0, 2, m),
            types,
        )
        cw = rng.choice([0.0, 0.05])
        pairs = milp(o, cw)
        check_shooters(pairs, feasible(o))
        assert np.isclose(objective(o, pairs, cw), brute_force(o, cw))
