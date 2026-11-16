import numpy as np

from baselines.cbba import auction, cbba
from baselines.greedy import value
from eval.suites import load_suite
from sim.engine import run


def random_bids(rng, m, n, p_legal=0.6):
    v = rng.uniform(0.1, 10, (m, n))
    return np.where(rng.random((m, n)) < p_legal, v, -np.inf)


def greedy_on(v):
    """greedy-value's selection rule on a raw bid matrix."""
    used_e, used_j, out = set(), set(), {}
    for flat in np.argsort(-v, axis=None, kind="stable"):
        e, j = (int(i) for i in np.unravel_index(flat, v.shape))
        if not v[e, j] > 0:
            break
        if e not in used_e and j not in used_j:
            used_e.add(e)
            used_j.add(j)
            out[e] = j
    return out


def test_fully_connected_matches_greedy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m, n = rng.integers(1, 12), rng.integers(1, 12)
        v = random_bids(rng, m, n)
        x, _ = auction(v)
        assert {e: int(j) for e, j in enumerate(x) if j >= 0} == greedy_on(v)


def test_line_graph_converges_conflict_free():
    rng = np.random.default_rng(1)
    for _ in range(100):
        m, n = rng.integers(2, 10), rng.integers(1, 10)
        v = random_bids(rng, m, n)
        adj = np.eye(m, k=1, dtype=bool) | np.eye(m, k=-1, dtype=bool)
        x, rounds = auction(v, adj)
        taken = x[x >= 0]
        assert len(taken) == len(set(taken.tolist()))  # nobody shares a threat
        assert all(np.isfinite(v[e, j]) and v[e, j] > 0 for e, j in enumerate(x) if j >= 0)
        assert rounds < 1000


def test_policy_matches_greedy_value_in_sim():
    suite = load_suite("suite_small_v2")
    sc = suite.scenarios[0]
    seen = []

    def both(obs):
        a, b = sorted(cbba(obs)), sorted(value(obs))
        seen.append(a == b)
        return a

    run(sc, suite.cfg.sim, suite.cfg.types, both)
    assert seen and all(seen)
