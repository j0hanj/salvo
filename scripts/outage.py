"""whole effectors going dark vs independent bursty links, at the same pairwise loss. a node
that's dark `down` of the time leaves any given link down 1 - (1 - down)^2 of the time, so each
node row gets a link row w exactly that loss. same mean spell length for both. the no-comms row
(every message lost) is the floor.

uv run python -m scripts.outage
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
class OutageConfig:
    suite: str = "suite_scale_n100_v1"
    downs: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    burst: float = 20.0  # mean ticks per dark spell, for nodes and links alike
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
    cfg = OutageConfig()
    out = start_run("outage", cfg, seed=cfg.seed)
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}

    names = {}
    for d in cfg.downs:
        link = round(1 - (1 - d) ** 2, 4)
        names[f"cbba-node@{d}/{cfg.burst}"] = f"nodes dark {100 * d:g}% of ticks"
        names[f"cbba-burst@{link}/{cfg.burst}"] = f"links down {100 * link:g}% (same pairwise loss)"
    names[f"cbba@1.0/{cfg.rounds}"] = "no comms at all"
    rows = evaluate(cfg.suite, [cfg.reference, *names], jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")

    rows.sort(key=lambda r: r["scenario"])
    ref_v = np.array([100 * r["value_frac"] for r in rows if r["method"] == cfg.reference])
    lines = [
        f"| comms | threats stopped % | value defended % | Δ value vs {cfg.reference} "
        "| shooters per targeted threat |",
        "|---|---|---|---|---|",
    ]
    for m in [cfg.reference, *names]:
        rs = [r for r in rows if r["method"] == m]
        col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        label = f"{cfg.reference} (central, no loss)" if m == cfg.reference else names[m]
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
        f"{cfg.rounds} rounds of message passing per tick. node rows: each effector is dark the "
        "given share of ticks (hears nobody, heard by nobody, still fires on its own pick). link "
        "rows: every directed link independently down 1 - (1 - down)^2 of ticks, the same "
        f"pairwise loss. both bursty w a mean spell of {cfg.burst:g} ticks (1 s each). 95% "
        f"bootstrap cis, Δ paired per scenario. made by `scripts/outage.py` at `{commit[:7]}`.",
    ]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/outage.md").write_text(md)
        splice_readme("outage", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
