"""does the capped gnn's leftover gap at scale come from the fixed 0.5 fire threshold?
`scripts/confidence.py` shows best edge scores drop as graphs grow and most free effectors sit idle
w a reachable uncovered threat. sweep the threshold and see if threats stopped comes back.

caveat: this sweeps on the eval suites, so the best row is picked on test data. it's a diagnostic
(is the threshold the bottleneck?), not a tuned headline number.

uv run python -m scripts.threshold --model results/<run>-train/model.pt
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from eval.harness import evaluate, write_csv
from eval.runlog import start_run
from eval.stats import bootstrap_ci, paired_diff
from scripts.compare import splice_readme


@dataclass
class ThresholdConfig:
    model: str = ""
    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.1, 0.0])
    # capped decode's survival floor: keep adding shooters to a threat until its chance of
    # surviving them all is <= this. 0.1 is the policy default
    floors: list[float] = field(default_factory=lambda: [0.1])
    sizes: list[int] = field(default_factory=lambda: [10, 50, 100, 500])
    reference: str = "hungarian"
    # at threshold 0 capped decode is just "rank edges by score, fill greedily", so the fair
    # question is whether the gnn's ranking beats a hand-written one
    others: list[str] = field(default_factory=lambda: ["greedy-value"])
    n_boot: int = 10_000
    seed: int = 0


def fmt(c, d=1):
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    ap.add_argument("--floors", type=float, nargs="+", default=None)
    ap.add_argument(
        "--tag", default="", help="writes threshold_<tag>.md + readme block, for a second model"
    )
    args = ap.parse_args()
    cfg = ThresholdConfig(model=args.model)
    if args.thresholds:
        cfg.thresholds = args.thresholds
    if args.floors:
        cfg.floors = args.floors
    model_commit = json.loads((Path(cfg.model).parent / "meta.json").read_text())["git_commit"]
    tcfg = yaml.safe_load((Path(cfg.model).parent / "config.yaml").read_text())
    # older runs had a single `suite`
    train_suites = tcfg.get("suites") or [tcfg["suite"]]
    out = start_run("threshold", cfg, seed=cfg.seed)
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}
    if cfg.floors == [0.1]:  # default floor, keep the old names so old tables still match
        names = {f"gnn-capped@{t}:{cfg.model}": f"gnn capped @{t}" for t in cfg.thresholds}
    else:
        names = {
            f"gnn-capped@{t}/{f}:{cfg.model}": f"gnn capped @{t} floor {f}"
            for t in cfg.thresholds
            for f in cfg.floors
        }

    rows = []
    for n in cfg.sizes:
        print(f"n={n}", flush=True)
        methods = [cfg.reference, *cfg.others, *names]
        rows += evaluate(f"suite_scale_n{n}_v1", methods, jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")

    lines = [
        f"| threats | method | value defended % | Δ value vs {cfg.reference} | threats stopped % "
        f"| shots vs {cfg.reference} | targeted threats w 3+ shooters % |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in cfg.sizes:
        at = sorted((r for r in rows if r["n_threats"] == n), key=lambda r: r["scenario"])
        ref = [r for r in at if r["method"] == cfg.reference]
        ref_v = np.array([r["value_frac"] for r in ref])
        ref_s = np.array([r["shots"] for r in ref], dtype=float)
        for m in [cfg.reference, *cfg.others, *names]:
            rs = [r for r in at if r["method"] == m]
            col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
            delta = (
                "—"
                if m == cfg.reference
                else fmt(paired_diff(100 * col("value_frac"), 100 * ref_v, **kw))
            )
            lines.append(
                f"| {n} | {names.get(m, m)} | {fmt(bootstrap_ci(100 * col('value_frac'), **kw))} "
                f"| {delta} | {fmt(bootstrap_ci(100 * col('stopped_frac'), **kw))} "
                f"| {fmt(bootstrap_ci(col('shots') / ref_s, **kw), 2)}x "
                f"| {fmt(bootstrap_ci(100 * col('stack3_frac'), **kw))} |"
            )
    n_scn = len(rows) // (len(cfg.sizes) * (1 + len(cfg.others) + len(names)))
    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    lines += [
        "",
        f"`suite_scale_n*_v1`, {n_scn} scenarios per size, same scenarios for every method. "
        "gnn w capped decode at each fire threshold (and survival floor, default 0.1). shots vs "
        "hungarian = interceptors launched per scenario / hungarian's on the same scenario. 95% "
        "bootstrap cis, Δ paired per scenario. "
        "swept on the eval suites, so read it as a diagnostic, not a tuned result. "
        f"model `{Path(cfg.model).parent.name}`, trained on {', '.join(train_suites)} by "
        f"`scripts/train.py` at `{model_commit[:7]}`. made by `scripts/threshold.py` at "
        f"`{commit[:7]}`.",
    ]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        key = f"threshold_{args.tag}" if args.tag else "threshold"
        Path(f"docs/tables/{key}.md").write_text(md)
        splice_readme(key, md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
