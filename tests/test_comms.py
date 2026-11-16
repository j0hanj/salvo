import numpy as np

from sim.comms import GilbertElliott


def run_chain(loss, burst, m=20, ticks=4000, seed=0):
    ge = GilbertElliott(m, loss, burst, np.random.default_rng(seed))
    return np.stack([~ge.step() for _ in range(ticks)])  # (T, m, m) True = down


def test_long_run_loss_and_burst_length():
    for loss, burst in [(0.3, 1.0), (0.5, 5.0), (0.9, 20.0)]:
        down = run_chain(loss, burst)
        off = ~np.eye(down.shape[1], dtype=bool)
        assert abs(down[:, off].mean() - loss) < 0.03
        # mean length of a bad spell on one link, over all links
        lens = []
        for i, k in zip(*np.nonzero(off), strict=True):
            d = down[:, i, k].astype(int)
            starts = np.flatnonzero(np.diff(np.r_[0, d]) == 1)
            ends = np.flatnonzero(np.diff(np.r_[d, 0]) == -1)
            lens += (ends - starts + 1).tolist()
        assert abs(np.mean(lens) - burst) / burst < 0.1


def test_no_loss_and_self_links():
    down = run_chain(0.0, 5.0, ticks=50)
    assert not down.any()
    down = run_chain(0.9, 5.0, ticks=50)
    assert not down[:, np.arange(20), np.arange(20)].any()


def test_reproducible():
    a = run_chain(0.5, 3.0, ticks=30, seed=4)
    assert np.array_equal(a, run_chain(0.5, 3.0, ticks=30, seed=4))
