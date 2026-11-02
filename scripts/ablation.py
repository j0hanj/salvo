"""does the gnn's over-stacking at scale come from mean+max aggregation (can't count competing
shooters) or from independent per-effector decoding? 2x2: {plain, +degree channel} x {argmax,
capped decode}, against hungarian (and milp up to 100 threats) on the scaling suites.

uv run python -m scripts.ablation --base <plain model.pt> --degree <degree model.pt>
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
class AblationConfig:
    base: str = ""  # model trained w/o the degree channel
    degree: str = ""  # same recipe + --degree
    sizes: list[int] = field(default_factory=lambda: [10, 50, 100, 500])
    milp_max: int = 100  # milp gets slow past this, and it's not what's being tested
    reference: str = "hungarian"
    n_boot: int = 10_000
    seed: int = 0


def variants(cfg: AblationConfig) -> dict[str, str]:
    return {
        "gnn": f"gnn:{cfg.base}",
        "gnn capped": f"gnn-capped:{cfg.base}",
        "gnn+deg": f"gnn:{cfg.degree}",
        "gnn+deg capped": f"gnn-capped:{cfg.degree}",
    }


def fmt(c, d=1):
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def table(rows, cfg: AblationConfig, n_scn: int, commit: str, model_commits: dict) -> str:
    names = {v: k for k, v in variants(cfg).items()}
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}
    lines = [
        f"| threats | method | value defended % | Δ value vs {cfg.reference} | threats stopped % "
        "| targeted threats w 3+ shooters % | decision latency ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in cfg.sizes:
        at = sorted((r for r in rows if r["n_threats"] == n), key=lambda r: r["scenario"])
        methods = list(dict.fromkeys(r["method"] for r in at))
        ref = np.array([r["value_frac"] for r in at if r["method"] == cfg.reference])
        for m in methods:
            rs = [r for r in at if r["method"] == m]
            col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
            delta = (
                "—"
                if m == cfg.reference
                else fmt(paired_diff(100 * col("value_frac"), 100 * ref, **kw))
            )
            lines.append(
                f"| {n} | {names.get(m, m)} | {fmt(bootstrap_ci(100 * col('value_frac'), **kw))} "
                f"| {delta} | {fmt(bootstrap_ci(100 * col('stopped_frac'), **kw))} "
                f"| {fmt(bootstrap_ci(100 * col('stack3_frac'), **kw))} "
                f"| {fmt(bootstrap_ci(col('lat_mean_ms'), **kw), 2)} |"
            )
    lines += [
        "",
        f"`suite_scale_n*_v1`, {n_scn} scenarios per size, same scenarios for every method. 95% "
        "bootstrap cis, Δ paired per scenario. 3+ shooters = share of (tick, targeted threat) "
        "pairs w 3 or more interceptors launched at it, averaged per scenario. models from "
        f"`scripts/train.py` at `{model_commits['base'][:7]}` (plain) and "
        f"`{model_commits['degree'][:7]}` (+deg). made by `scripts/ablation.py` at `{commit[:7]}`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--degree", required=True)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = AblationConfig(base=args.base, degree=args.degree)
    commits = {
        k: json.loads((Path(p).parent / "meta.json").read_text())["git_commit"]
        for k, p in (("base", cfg.base), ("degree", cfg.degree))
    }

    out = start_run("ablation", cfg, seed=cfg.seed)
    rows = []
    for n in cfg.sizes:
        print(f"n={n}", flush=True)
        methods = [cfg.reference, *(["milp"] if n <= cfg.milp_max else []), *variants(cfg).values()]
        rows += evaluate(f"suite_scale_n{n}_v1", methods, jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")
    n_scn = len([r for r in rows if r["method"] == cfg.reference]) // len(cfg.sizes)
    md = table(rows, cfg, n_scn, json.loads((out / "meta.json").read_text())["git_commit"], commits)
    (out / "summary.md").write_text(md)
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/ablation.md").write_text(md)
        splice_readme("ablation", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
