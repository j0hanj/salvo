"""same average loss, different shape. `scripts/loss.py` drops every message independently, and w
10 rounds per tick cbba routes around that up to 90% loss. here links go down in bursts
(gilbert-elliott, `sim/comms.py`): a down link stays down for every round of the tick and usually
for the next few ticks too. compared at matching long-run loss, 10 rounds per tick.

uv run python -m scripts.burst
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
class BurstConfig:
    suite: str = "suite_scale_n100_v1"
    losses: list[float] = field(default_factory=lambda: [0.5, 0.9, 0.95])
    # mean ticks down. a combo only runs if the chain can actually hit that loss, see sim/comms.py
    bursts: list[float] = field(default_factory=lambda: [10.0, 20.0, 50.0])
    rounds: int = 10
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
    cfg = BurstConfig()
    out = start_run("burst", cfg, seed=cfg.seed)
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}

    names = {}
    for p in cfg.losses:
        names[f"cbba@{p}/{cfg.rounds}"] = (p, f"independent per message, {100 * p:g}% lost")
        for b in cfg.bursts:
            if b < p / (1 - p) - 1e-9:
                continue  # mean good spell would be under a tick
            names[f"cbba-burst@{p}/{b}"] = (p, f"bursty links, {100 * p:g}% down, ~{b:g} ticks")
    rows = evaluate(cfg.suite, [cfg.reference, *names], jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")

    rows.sort(key=lambda r: r["scenario"])
    ref_v = np.array([100 * r["value_frac"] for r in rows if r["method"] == cfg.reference])
    lines = [
        f"| loss model | threats stopped % | value defended % | Δ value vs {cfg.reference} "
        "| shooters per targeted threat |",
        "|---|---|---|---|---|",
    ]
    for m in [cfg.reference, *names]:
        rs = [r for r in rows if r["method"] == m]
        col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        label = f"{cfg.reference} (central, no loss)" if m == cfg.reference else names[m][1]
        delta = (
            "—" if m == cfg.reference else fmt(paired_diff(100 * col("value_frac"), ref_v, **kw))
        )
        lines.append(
            f"| {label} | {fmt(bootstrap_ci(100 * col('stopped_frac'), **kw))} "
            f"| {fmt(bootstrap_ci(100 * col('value_frac'), **kw))} | {delta} "
            f"| {fmt(bootstrap_ci(col('stack_mean'), **kw), 2)} |"
        )
    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    lines += [
        "",
        f"`{cfg.suite}`, {len(ref_v)} scenarios, same scenarios for every method. cbba w "
        f"{cfg.rounds} rounds of message passing per tick. independent = every message lost w "
        "the given prob, fresh each round. bursty = each directed link is a gilbert-elliott "
        "chain stepped once per replan tick (1 s), down the given share of ticks in spells of "
        "the given mean length, and down for every round of a tick it's down in. 95% bootstrap "
        f"cis, Δ paired per scenario. made by `scripts/burst.py` at `{commit[:7]}`.",
    ]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/burst.md").write_text(md)
        splice_readme("burst", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
