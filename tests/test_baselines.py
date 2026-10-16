import itertools

import numpy as np

from baselines.common import shot_value
from baselines.greedy import nearest, value
from baselines.hungarian import hungarian
from sim.engine import FLYING, Obs
from sim.types import EffectorType


def kind(name, p_kill, cost=1.0, rng=5000.0):
    return EffectorType(
        name, range=rng, speed=200, p_kill=p_kill, cost=cost, magazine=4, reload=1.0, max_flight=30
    )


def obs(threat_pos, threat_value, eff_pos, eff_type, types, ready=None):
    n, m = len(threat_pos), len(eff_pos)
    return Obs(
        t=0.0,
        threat_pos=np.array(threat_pos, float),
        threat_vel=np.zeros((n, 2)),
        threat_status=np.full(n, FLYING),
        threat_value=np.array(threat_value, float),
        engaged=np.zeros(n, int),
        eff_pos=np.array(eff_pos, float),
        eff_type=np.array(eff_type),
        eff_ammo=np.full(m, 4),
        eff_ready=np.ones(m, bool) if ready is None else np.array(ready),
        types=types,
    )


def check_legal(pairs, m, n):
    es = [e for e, _ in pairs]
    js = [j for _, j in pairs]
    assert len(set(es)) == len(es) and len(set(js)) == len(js)
    assert all(0 <= e < m and 0 <= j < n for e, j in pairs)


def test_nearest_picks_closest_to_asset():
    o = obs([[0, 3000], [0, 1000]], [1, 1], [[0, 0]], [0], [kind("a", 0.5)])
    assert nearest(o) == [(0, 1)]


def test_value_goes_for_the_big_one():
    o = obs([[0, 1000], [0, 3000]], [1, 10], [[0, 0]], [0], [kind("a", 0.5)])
    assert value(o) == [(0, 1)]


def test_value_respects_range_and_readiness():
    types = [kind("short", 0.9, rng=1500)]
    o = obs([[0, 3000]], [10], [[0, 0]], [0], types)
    assert value(o) == []
    o = obs([[0, 1000]], [10], [[0, 0]], [0], types, ready=[False])
    assert value(o) == []


def test_value_cost_weight_skips_pricey_shot():
    # cheap so-so effector vs expensive good one, one threat. w cost ignored the good one fires,
    # w cost weighted the cheap one does
    types = [kind("cheap", 0.6, cost=1), kind("pricey", 0.9, cost=20)]
    o = obs([[0, 1000]], [1], [[0, 0], [10, 0]], [0, 1], types)
    assert value(o) == [(1, 0)]
    assert value(o, cost_weight=0.05) == [(0, 0)]
    # only the pricey one in range and it's not worth its cost -> hold fire
    o = obs([[0, 1000]], [1], [[10, 0]], [1], types)
    assert value(o, cost_weight=0.05) == []


def test_random_snapshots_stay_legal():
    rng = np.random.default_rng(0)
    types = [kind("a", 0.7, rng=2000), kind("b", 0.9, rng=4000)]
    for _ in range(50):
        n, m = rng.integers(1, 12), rng.integers(1, 6)
        o = obs(
            rng.uniform(-4000, 4000, (n, 2)),
            rng.lognormal(0, 1, n),
            rng.uniform(-500, 500, (m, 2)),
            rng.integers(0, 2, m),
            types,
        )
        for policy in (nearest, value, hungarian):
            check_legal(policy(o), m, n)


def best_total(v):
    """brute force the best one-to-one matching, tiny cases only."""
    m, n = v.shape
    best = 0.0
    for perm in itertools.permutations(range(max(m, n)), m):
        tot = sum(v[e, j] for e, j in enumerate(perm) if j < n and v[e, j] > 0)
        best = max(best, tot)
    return best


def test_hungarian_beats_greedy_when_range_matters():
    # long-range site can reach both, short-range only the close one. greedy grabs the close
    # high-value one w the long site and strands the short site. hungarian splits them up.
    types = [kind("long", 0.9, rng=5000), kind("short", 0.5, rng=1500)]
    o = obs([[0, 1000], [0, 4000]], [10, 5], [[0, 0], [10, 0]], [0, 1], types)
    v = shot_value(o)
    total = lambda pairs: sum(v[e, j] for e, j in pairs)  # noqa: E731
    assert total(value(o)) < total(hungarian(o))
    assert np.isclose(total(hungarian(o)), best_total(v))


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(1)
    types = [kind("a", 0.7, cost=1, rng=2000), kind("b", 0.9, cost=20, rng=4000)]
    for _ in range(100):
        n, m = rng.integers(1, 6), rng.integers(1, 5)
        o = obs(
            rng.uniform(-4000, 4000, (n, 2)),
            rng.lognormal(0, 1, n),
            rng.uniform(-500, 500, (m, 2)),
            rng.integers(0, 2, m),
            types,
        )
        cw = rng.choice([0.0, 0.05])
        pairs = hungarian(o, cw)
        check_legal(pairs, m, n)
        v = shot_value(o, cw)
        assert np.isclose(sum(v[e, j] for e, j in pairs), best_total(v))
