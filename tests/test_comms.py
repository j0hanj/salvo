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
    down = run_chain(0.9, 10.0, ticks=50)
    assert not down[:, np.arange(20), np.arange(20)].any()


def test_rejects_loss_burst_combos_it_cant_hit():
    # 90% loss w 5-tick bursts would need p(good -> bad) = 1.8. used to get clipped to 1, which
    # quietly ran ~83% loss instead
    for loss, burst in [(0.9, 5.0), (0.9, 1.0), (0.99, 20.0)]:
        try:
            GilbertElliott(4, loss, burst, np.random.default_rng(0))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for loss={loss} burst={burst}")
    # right at the edge is fine: good spells exactly 1 tick
    down = run_chain(0.9, 9.0)
    assert abs(down[:, ~np.eye(20, dtype=bool)].mean() - 0.9) < 0.03


def test_reproducible():
    a = run_chain(0.5, 3.0, ticks=30, seed=4)
    assert np.array_equal(a, run_chain(0.5, 3.0, ticks=30, seed=4))


def test_node_outage_cuts_whole_nodes():
    from sim.comms import NodeOutage

    m = 12
    no = NodeOutage(m, 0.3, 5.0, np.random.default_rng(0))
    dark_frac = []
    for _ in range(3000):
        up = no.step()
        off = up & ~np.eye(m, dtype=bool)
        # lit nodes all hear each other, dark ones hear (and are heard by) nobody but themselves
        assert up[np.ix_(~no.dark, ~no.dark)].all()
        assert not off[no.dark].any() and not off[:, no.dark].any()
        assert np.diag(up).all()
        dark_frac.append(no.dark.mean())
    assert abs(np.mean(dark_frac) - 0.3) < 0.03


def test_flood_relays_over_rounds():
    from sim.comms import flood

    # line 0 - 1 - 2 - 3, both directions. k rounds reach k hops
    m = 4
    line = np.eye(m, dtype=bool) | np.eye(m, k=1, dtype=bool) | np.eye(m, k=-1, dtype=bool)
    assert flood([line]).tolist() == line.tolist()
    two = flood([line, line])
    assert two[0, 2] and not two[0, 3]
    assert flood([line] * 3).all()
    # nobody hears anybody: you only know yourself
    assert flood([np.eye(m, dtype=bool)] * 5).tolist() == np.eye(m, dtype=bool).tolist()
    # order matters: a link that's up before your neighbor heard something doesn't carry it
    a = np.eye(3, dtype=bool)
    a[0, 1] = True  # round 1: 0 hears 1
    b = np.eye(3, dtype=bool)
    b[1, 2] = True  # round 2: 1 hears 2
    assert not flood([a, b])[0, 2] and flood([b, a])[0, 2]


def test_channel_matches_the_models_it_wraps():
    from sim.comms import Channel, NodeOutage

    ch = Channel("node", 0.4, 5.0, rounds=3)
    ch.reset(11)
    ref = NodeOutage(6, 0.4, 5.0, np.random.default_rng(11))
    for _ in range(20):
        masks = ch.tick(6)
        assert len(masks) == 3 and all(np.array_equal(x, masks[0]) for x in masks)
        assert np.array_equal(masks[0], ref.step())
    full = Channel("full", rounds=2).tick(4)
    assert len(full) == 2 and all(x.all() for x in full)
    iid = Channel("iid", 1.0, rounds=2)
    assert all(np.array_equal(x, np.eye(5, dtype=bool)) for x in iid.tick(5))
