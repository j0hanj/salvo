import numpy as np
import pytest

from eval.stats import bootstrap_ci, paired_diff


def test_ci_covers_true_mean_most_of_the_time():
    rng = np.random.default_rng(0)
    hits = 0
    for i in range(200):
        x = rng.normal(3.0, 1.0, 50)
        _, lo, hi = bootstrap_ci(x, n_boot=2000, seed=i)
        hits += lo <= 3.0 <= hi
    # 95% interval, allow some slack for bootstrap undercoverage at n=50
    assert 0.88 <= hits / 200 <= 0.99


def test_paired_is_tighter_when_scenarios_vary():
    rng = np.random.default_rng(1)
    difficulty = rng.uniform(0, 1, 200)  # shared per-scenario effect
    a = difficulty + 0.05 + rng.normal(0, 0.02, 200)
    b = difficulty + rng.normal(0, 0.02, 200)
    _, lo_a, hi_a = bootstrap_ci(a)
    _, lo_d, hi_d = paired_diff(a, b)
    assert (hi_d - lo_d) < (hi_a - lo_a) / 5
    assert lo_d > 0  # the 0.05 edge is clearly detected


def test_nan_dropped_and_seeded():
    x = [1.0, np.nan, 3.0, 2.0]
    assert bootstrap_ci(x)[0] == 2.0
    assert bootstrap_ci(x, seed=5) == bootstrap_ci(x, seed=5)


def test_misaligned_pair_fails():
    with pytest.raises(ValueError):
        paired_diff([1, 2], [1, 2, 3])
