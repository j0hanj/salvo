"""run methods on a suite, one row of metrics per (method, scenario).

every method sees the exact same scenarios w the exact same seeds, so per-scenario rows line up and
methods can be compared pairwise.
"""

import csv
import functools
import os
import time
from multiprocessing import Pool

import numpy as np

from baselines.common import feasible
from baselines.greedy import nearest, value
from baselines.hungarian import hungarian
from baselines.milp import milp
from eval.suites import load_suite
from sim.engine import FLYING, KILLED, LEAKED, PENDING, run

METHODS = {
    "greedy-nearest": nearest,
    "greedy-value": value,
    "hungarian": hungarian,
    "milp": milp,
}


@functools.cache
def resolve(method: str):
    """baseline name, or "gnn:<path to model.pt>" for a trained policy."""
    if method.startswith("gnn:"):
        # imported here so the baselines don't need torch
        import torch

        from learned.model import EdgeScorer
        from learned.policy import GNNPolicy

        ckpt = torch.load(method[4:], weights_only=True)
        cfg = ckpt["config"]
        model = EdgeScorer(cfg["hidden"], cfg["rounds"])
        model.load_state_dict(ckpt["state_dict"])
        return GNNPolicy(model)
    return METHODS[method]


class Timed:
    """wraps a policy and clocks it. wall time, so it includes python overhead.

    only ticks w at least one legal shot count. most ticks have nothing to decide and return in
    microseconds, which would drown out the calls that actually matter.
    """

    def __init__(self, policy):
        self.policy = policy
        self.ms = []

    def __call__(self, obs):
        live = feasible(obs).any()  # checked outside the timer
        t0 = time.perf_counter()
        out = self.policy(obs)
        if live:
            self.ms.append((time.perf_counter() - t0) * 1e3)
        return out


def run_one(suite, idx: int, method: str) -> dict:
    sc = suite.scenarios[idx]
    policy = Timed(resolve(method))
    res = run(sc, suite.cfg.sim, suite.cfg.types, policy)
    stopped = res.status == KILLED
    n_stop = int(stopped.sum())
    ms = np.array(policy.ms)
    return {
        "suite": suite.id,
        "scenario": idx,
        "method": method,
        "n_threats": sc.n_threats,
        "n_effectors": sc.n_effectors,
        "stopped": n_stop,
        "leaked": int((res.status == LEAKED).sum()),
        # still in play when t_max hit. should be 0, nonzero means the suite's t_max is too short
        "unresolved": int(np.isin(res.status, (PENDING, FLYING)).sum()),
        "stopped_frac": n_stop / sc.n_threats,
        "value_frac": float(sc.threat_value[stopped].sum() / sc.threat_value.sum()),
        "shots": int(res.shots.sum()),
        "cost": res.cost,
        "cost_per_stop": res.cost / n_stop if n_stop else float("nan"),
        "decisions": len(ms),
        "lat_mean_ms": float(ms.mean()) if len(ms) else float("nan"),
        "lat_p95_ms": float(np.percentile(ms, 95)) if len(ms) else float("nan"),
        "lat_max_ms": float(ms.max()) if len(ms) else float("nan"),
    }


_suite = None


def _init(suite_id):
    global _suite
    _suite = load_suite(suite_id)


def _work(job):
    return run_one(_suite, *job)


def evaluate(
    suite_id: str, methods: list[str], jobs: int | None = None, limit: int | None = None
) -> list[dict]:
    """jobs=1 runs in-process. more jobs is faster but latency gets noisier from cpu contention.
    limit = only the first k scenarios, for quick checks."""
    suite = load_suite(suite_id)
    n = len(suite.scenarios) if limit is None else min(limit, len(suite.scenarios))
    work = [(i, m) for m in methods for i in range(n)]
    jobs = jobs or max(1, (os.cpu_count() or 2) // 2)
    if jobs == 1:
        return [run_one(suite, *w) for w in work]
    with Pool(jobs, initializer=_init, initargs=(suite_id,)) as pool:
        return pool.map(_work, work, chunksize=4)


def write_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("suite", "method"):
                r[k] = float(v)
    return rows
