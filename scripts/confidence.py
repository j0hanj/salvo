"""w capped decode the gnn stops stacking but still stops far fewer threats than hungarian at scale.
are effectors holding fire because the model's scores shrink as the graph grows, so nothing clears
the fixed 0.5 threshold?

runs capped decode on the scaling suites and logs, per decision tick, every free effector that has
at least one legal shot: its best edge score, whether it fired, and whether it sat idle while a
threat it could reach was still uncovered (= the threshold was the only thing stopping it).

uv run python -m scripts.confidence --model results/<run>-train/model.pt
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.harness import resolve
from eval.runlog import start_run
from eval.stats import bootstrap_ci
from eval.suites import load_suite
from scripts.compare import splice_readme
from sim.engine import run


@dataclass
class ConfidenceConfig:
    model: str = ""
    sizes: list[int] = field(default_factory=lambda: [10, 50, 100, 500])
    limit: int | None = None
    n_boot: int = 10_000
    seed: int = 0


class Probe:
    """runs the capped gnn policy and records per-effector stats on every tick w a legal shot."""

    def __init__(self, policy):
        self.policy = policy
        self.best, self.fired, self.idle_open = [], [], []

    def __call__(self, obs):
        g, p = self.policy.scores(obs)
        if g is None:
            return []
        pairs = self.policy.decode_scores(obs, g, p)
        shooters = {e for e, _ in pairs}
        pk = np.array([obs.types[k].p_kill for k in obs.eff_type])
        survive = np.ones(len(obs.threat_value))
        for e, j in pairs:
            survive[j] *= 1 - pk[e]
        uncovered = survive > self.policy.survival_floor
        for e in np.unique(g.edges[0]):
            mine = g.edges[0] == e
            self.best.append(float(p[mine].max()))
            fired = int(e) in shooters
            self.fired.append(fired)
            self.idle_open.append(not fired and bool(uncovered[g.edges[1, mine]].any()))
        return pairs


def fmt(c, d=1):
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = ConfidenceConfig(model=args.model, limit=args.limit)
    out = start_run("confidence", cfg, seed=cfg.seed)
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}

    lines = [
        "| threats | best edge score, median [p10, p90] | effectors w best score < 0.5 % "
        "| idle w an uncovered legal target % | fired % |",
        "|---|---|---|---|---|",
    ]
    n_scn = 0
    for n in cfg.sizes:
        print(f"n={n}", flush=True)
        suite = load_suite(f"suite_scale_n{n}_v1")
        scns = suite.scenarios[: cfg.limit]
        best, low, idle, fired = [], [], [], []
        for sc in scns:
            # fresh policy per scenario (resolve is cached, the probe isn't)
            probe = Probe(resolve(f"gnn-capped:{cfg.model}"))
            run(sc, suite.cfg.sim, suite.cfg.types, probe)
            b = np.array(probe.best)
            best.append(b)
            # per-scenario shares so the ci resamples scenarios, not correlated ticks
            low.append(100 * (b < 0.5).mean())
            idle.append(100 * np.mean(probe.idle_open))
            fired.append(100 * np.mean(probe.fired))
        b = np.concatenate(best)
        q = np.percentile(b, [50, 10, 90])
        lines.append(
            f"| {n} | {q[0]:.2f} [{q[1]:.2f}, {q[2]:.2f}] "
            f"| {fmt(bootstrap_ci(np.array(low), **kw))} "
            f"| {fmt(bootstrap_ci(np.array(idle), **kw))} "
            f"| {fmt(bootstrap_ci(np.array(fired), **kw))} |"
        )
        np.save(out / f"best_scores_n{n}.npy", b)
        n_scn = len(scns)

    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    lines += [
        "",
        f"`suite_scale_n*_v1`, {n_scn} scenarios per size, gnn w capped decode at threshold "
        "0.5. one row = every (tick, free effector w at least one legal shot). idle w an uncovered "
        "legal target = didn't fire even though some threat it could reach was still below the "
        "coverage floor after decode, so only the threshold held it back. 95% bootstrap cis over "
        f"scenarios. model `{Path(cfg.model).parent.name}`. made by `scripts/confidence.py` at "
        f"`{commit[:7]}`.",
    ]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    if cfg.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/confidence.md").write_text(md)
        splice_readme("confidence", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
