"""phase 4, first half: how does cbba hold up as messages get dropped? every message is lost w
prob p_drop, agents get a fixed number of talking rounds per tick, then act on whatever they
hold. swept over both, since w enough rounds independent loss barely matters. hungarian
(central, perfect info) is the reference line. the gnn joins this sweep once it can run
decentralized.

uv run python -m scripts.loss
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.harness import evaluate, write_csv
from eval.runlog import start_run
from eval.stats import bootstrap_ci, paired_diff
from scripts.compare import splice_readme


@dataclass
class LossConfig:
    suites: list[str] = field(default_factory=lambda: ["suite_small_v2", "suite_scale_n100_v1"])
    p_drops: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.9, 0.99, 1.0])
    # talking rounds per tick before everyone has to commit. max-consensus spreads a winning bid
    # through anyone who heard it, so w enough rounds even heavy independent loss gets through
    rounds: list[int] = field(default_factory=lambda: [1, 3, 10])
    reference: str = "hungarian"
    n_boot: int = 10_000
    seed: int = 0


def fmt(c, d=1):
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = LossConfig()
    out = start_run("loss", cfg, seed=cfg.seed)
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}
    names = {
        f"cbba@{p}/{r}": f"cbba, {r} round{'s' * (r > 1)}, {100 * p:g}% lost"
        for r in cfg.rounds
        for p in cfg.p_drops
    }

    rows = []
    for s in cfg.suites:
        print(s, flush=True)
        rows += evaluate(s, [cfg.reference, *names], jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")

    lines = [
        f"| suite | method | threats stopped % | value defended % | Δ value vs {cfg.reference} "
        "| shooters per targeted threat | shots per threat |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in cfg.suites:
        at = sorted((r for r in rows if r["suite"] == s), key=lambda r: r["scenario"])
        ref_v = np.array([100 * r["value_frac"] for r in at if r["method"] == cfg.reference])
        for m in [cfg.reference, *names]:
            rs = [r for r in at if r["method"] == m]
            col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
            delta = (
                "—"
                if m == cfg.reference
                else fmt(paired_diff(100 * col("value_frac"), ref_v, **kw))
            )
            lines.append(
                f"| {s} | {names.get(m, m)} "
                f"| {fmt(bootstrap_ci(100 * col('stopped_frac'), **kw))} "
                f"| {fmt(bootstrap_ci(100 * col('value_frac'), **kw))} | {delta} "
                f"| {fmt(bootstrap_ci(col('stack_mean'), **kw), 2)} "
                f"| {fmt(bootstrap_ci(col('shots') / col('n_threats'), **kw), 2)} |"
            )
    n_scn = {
        s: sum(r["suite"] == s and r["method"] == cfg.reference for r in rows) for s in cfg.suites
    }
    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    lines += [
        "",
        f"scenarios per suite: {', '.join(f'{s} {k}' for s, k in n_scn.items())}, same scenarios "
        "for every method. cbba gets the given rounds of message passing per tick, every message "
        "lost independently w the given prob, then each effector fires on whatever it holds. "
        "100% lost = no comms at all, every effector takes its own favorite. hungarian is "
        "central w perfect info, the no-jamming ceiling. shooters per targeted threat > 1 = "
        "effectors that didn't hear they'd been outbid. 95% bootstrap cis, "
        f"Δ paired per scenario. made by `scripts/loss.py` at `{commit[:7]}`.",
    ]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/loss.md").write_text(md)
        splice_readme("loss", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
