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


def ratio_ci(num, den, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """(ratio, lo, hi) of sum(num) / sum(den), resampling scenarios. for stuff like cost per kill,
    where averaging per-scenario ratios would blow up on scenarios w ~0 kills."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(num.sum() / den.sum()), float(lo), float(hi)
