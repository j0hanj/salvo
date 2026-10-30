import numpy as np

from baselines.common import feasible
from learned.features import EDGE_FEATURES, EFF_FEATURES, THR_FEATURES, featurize
from tests.test_baselines import kind, obs


def snapshot():
    types = [kind("a", 0.7, rng=2000), kind("b", 0.9, cost=20, rng=4000)]
    return obs([[0, 1000], [0, 3000], [0, 9000]], [1, 5, 2], [[0, 0], [10, 0]], [0, 1], types)


def test_shapes():
    g = featurize(snapshot())
    assert g.eff_x.shape == (2, len(EFF_FEATURES))
    assert g.thr_x.shape == (3, len(THR_FEATURES))
    assert g.edges.shape[0] == 2 and g.edge_x.shape == (g.edges.shape[1], len(EDGE_FEATURES))
    assert all(np.isfinite(a).all() for a in (g.eff_x, g.thr_x, g.edge_x))


def test_edges_are_exactly_the_legal_shots():
    o = snapshot()
    g = featurize(o)
    ok = feasible(o)
    assert set(zip(*g.edges.tolist(), strict=True)) == set(zip(*np.nonzero(ok), strict=True))
    # 9km threat is out of everyone's range
    assert 2 not in g.edges[1]


def test_no_legal_shots_is_an_empty_graph():
    o = snapshot()
    o.eff_ready[:] = False
    g = featurize(o)
    assert g.edges.shape == (2, 0) and g.edge_x.shape == (0, len(EDGE_FEATURES))
