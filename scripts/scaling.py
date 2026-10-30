"""zero-shot scaling: every method on every suite_scale_n*. gnn only ever trained on 5-10 threats.
writes a logged run, the readme table + the scaling figure.

uv run python -m scripts.scaling --model results/<run>-train/model.pt
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
from viz.scaling import draw
from viz.style import save_both


@dataclass
class ScalingConfig:
    sizes: list[int] = field(default_factory=lambda: [10, 25, 50, 100, 250, 500])
    baselines: list[str] = field(
        default_factory=lambda: ["greedy-nearest", "greedy-value", "hungarian", "milp"]
    )
    model: str = ""
    reference: str = "hungarian"  # the only exact-ish method that's cheap at every size
    n_boot: int = 10_000
    seed: int = 0


def label(method: str) -> str:
    return "gnn" if method.startswith("gnn:") else method


def latest_model() -> str:
    runs = sorted(Path("results").glob("*-train/model.pt"))
    if not runs:
        raise SystemExit("no trained model in results/, run `make train` first")
    return str(runs[-1])


def summarize(rows, cfg: ScalingConfig) -> list[dict]:
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}
    out = []
    for n in cfg.sizes:
        at = [r for r in rows if r["n_threats"] == n]
        by = {}
        for r in sorted(at, key=lambda r: r["scenario"]):
            by.setdefault(label(r["method"]), []).append(r)
        ref = np.array([r["value_frac"] for r in by[cfg.reference]])
        for m, rs in by.items():
            col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
            out.append(
                {
                    "n": n,
                    "method": m,
                    "value": bootstrap_ci(100 * col("value_frac"), **kw),
                    "stopped": bootstrap_ci(100 * col("stopped_frac"), **kw),
                    "vs_ref": paired_diff(100 * col("value_frac"), 100 * ref, **kw),
                    "latency": bootstrap_ci(col("lat_mean_ms"), **kw),
                    "lat_max": float(np.nanmax(col("lat_max_ms"))),
                }
            )
    return out


def table(summary, cfg: ScalingConfig, n_scn: int, commit: str, model_commit: str) -> str:
    fmt = lambda c, d=1: f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"  # noqa: E731
    lines = [
        f"| threats | method | value defended % | Δ value vs {cfg.reference} "
        "| threats stopped % | decision latency ms |",
        "|---|---|---|---|---|---|",
    ]
    for s in summary:
        delta = "—" if s["method"] == cfg.reference else fmt(s["vs_ref"])
        lines.append(
            f"| {s['n']} | {s['method']} | {fmt(s['value'])} | {delta} | {fmt(s['stopped'])} "
            f"| {fmt(s['latency'], 2)} |"
        )
    lines += [
        "",
        f"`suite_scale_n*_v1`, {n_scn} scenarios per size, same scenarios for every method. "
        "95% bootstrap cis, Δ paired per scenario. milp solves are capped at 5s each. "
        f"gnn trained on 5-10 threats only (`scripts/train.py` at `{model_commit[:7]}`). "
        f"made by `scripts/scaling.py` at `{commit[:7]}`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first k scenarios per size only")
    args = ap.parse_args()
    cfg = ScalingConfig(model=args.model or latest_model())
    if args.sizes:
        cfg.sizes = args.sizes
    model_meta = json.loads((Path(cfg.model).parent / "meta.json").read_text())

    out = start_run("scaling", cfg, seed=cfg.seed)
    methods = [*cfg.baselines, f"gnn:{cfg.model}"]
    rows = []
    for n in cfg.sizes:
        print(f"n={n}", flush=True)
        rows += evaluate(f"suite_scale_n{n}_v1", methods, jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")
    summary = summarize(rows, cfg)
    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    n_scn = len(rows) // (len(cfg.sizes) * len(methods))
    md = table(summary, cfg, n_scn, commit, model_meta["git_commit"])
    (out / "summary.md").write_text(md)

    if args.limit is None and args.sizes is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/scaling.md").write_text(md)
        save_both(lambda th: draw(summary, cfg, th), "docs/figures/scaling")
        splice_readme("scaling", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
