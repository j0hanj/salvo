"""confidence intervals. percentile bootstrap over scenarios, seeded so tables are reproducible.

paired_diff is the one that matters for comparing methods: every method ran on the same scenarios,
so diff them scenario by scenario first. scenario difficulty cancels out and the interval gets a lot
tighter than comparing two separate intervals.
"""

import numpy as np


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """(mean, lo, hi) of the mean of x. nans dropped."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff(a, b, **kw):
    """(mean, lo, hi) of mean(a - b), scenario by scenario. a and b must be aligned."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired diff needs the same scenarios in the same order")
    return bootstrap_ci(a - b, **kw)
