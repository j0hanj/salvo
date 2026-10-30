import numpy as np

from baselines.greedy import nearest
from baselines.milp import milp
from eval.suites import load_suite
from learned.dataset import Batch, collect, merge


def test_collect_and_roundtrip(tmp_path):
    suite = load_suite("suite_small_v2")
    b = collect(suite, driver=nearest, labeler=milp, limit=3)
    assert len(b) > 0
    assert b.edge_ptr[-1] == len(b.y) == b.edges.shape[1]
    assert set(np.unique(b.y)) <= {0.0, 1.0} and b.y.sum() > 0
    # local edge indices stay inside their own graph's node ranges
    for i in range(len(b)):
        es = b.edges[:, b.edge_ptr[i] : b.edge_ptr[i + 1]]
        assert es[0].max(initial=-1) < b.eff_ptr[i + 1] - b.eff_ptr[i]
        assert es[1].max(initial=-1) < b.thr_ptr[i + 1] - b.thr_ptr[i]
    b.save(tmp_path / "d.npz")
    back = Batch.load(tmp_path / "d.npz")
    assert np.array_equal(back.y, b.y) and np.array_equal(back.edges, b.edges)


def test_parallel_matches_serial():
    suite = load_suite("suite_small_v2")
    a = collect(suite, driver=nearest, labeler=milp, limit=4, jobs=1)
    b = collect(suite, driver=nearest, labeler=milp, limit=4, jobs=2)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.edge_x, b.edge_x)


def test_merge_keeps_every_graph_intact():
    suite = load_suite("suite_small_v2")
    a = collect(suite, driver=nearest, labeler=milp, limit=2)
    b = collect(suite, driver=milp, labeler=milp, limit=3)
    m = merge(a, b)
    assert len(m) == len(a) + len(b)
    # last graph of the merged set is exactly b's last graph
    i, j = len(m) - 1, len(b) - 1
    assert np.array_equal(
        m.edges[:, m.edge_ptr[i] : m.edge_ptr[i + 1]], b.edges[:, b.edge_ptr[j] : b.edge_ptr[j + 1]]
    )
    assert np.array_equal(m.thr_x[m.thr_ptr[i] :], b.thr_x[b.thr_ptr[j] :])
