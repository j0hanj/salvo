"""run every baseline on a suite, write a logged run + the readme table + the comparison figure.

uv run python -m scripts.compare --suite suite_small_v2
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.harness import METHODS, evaluate, write_csv
from eval.runlog import start_run
from eval.stats import bootstrap_ci, paired_diff, ratio_ci
from viz.compare import draw
from viz.style import save_both


@dataclass
class CompareConfig:
    suite: str = "suite_small_v2"
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    reference: str = "milp"
    n_boot: int = 10_000
    seed: int = 0  # bootstrap seed. the sim seeds live in the suite


def summarize(rows: list[dict], cfg: CompareConfig) -> list[dict]:
    by = {
        m: sorted((r for r in rows if r["method"] == m), key=lambda r: r["scenario"])
        for m in cfg.methods
    }
    ref = by[cfg.reference]
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}
    out = []
    for m, rs in by.items():
        col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        ref_col = lambda k: np.array([r[k] for r in ref], dtype=float)  # noqa: E731
        out.append(
            {
                "method": m,
                "stopped": bootstrap_ci(100 * col("stopped_frac"), **kw),
                "value": bootstrap_ci(100 * col("value_frac"), **kw),
                "vs_ref": paired_diff(
                    100 * col("stopped_frac"), 100 * ref_col("stopped_frac"), **kw
                ),
                "vs_ref_value": paired_diff(
                    100 * col("value_frac"), 100 * ref_col("value_frac"), **kw
                ),
                "cost_per_stop": ratio_ci(col("cost"), col("stopped"), **kw),
                "latency": bootstrap_ci(col("lat_mean_ms"), **kw),
            }
        )
    return out


def fmt(ci, digits=1):
    m, lo, hi = ci
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(summary: list[dict], cfg: CompareConfig, n: int, commit: str) -> str:
    lines = [
        f"| method | threats stopped % | value defended % | Δ stopped vs {cfg.reference} "
        f"| Δ value vs {cfg.reference} | cost per threat stopped | decision latency ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        is_ref = s["method"] == cfg.reference
        delta = "—" if is_ref else fmt(s["vs_ref"])
        delta_v = "—" if is_ref else fmt(s["vs_ref_value"])
        lines.append(
            f"| {s['method']} | {fmt(s['stopped'])} | {fmt(s['value'])} | {delta} | {delta_v} "
            f"| {fmt(s['cost_per_stop'])} | {fmt(s['latency'], 3)} |"
        )
    lines += [
        "",
        f"`{cfg.suite}`, {n} scenarios, every method on the same ones. means w 95% bootstrap cis "
        "over scenarios. Δs are paired per scenario, in pct points. latency = mean wall time "
        f"of ticks that had a real decision. made by `scripts/compare.py` at `{commit[:7]}`.",
    ]
    return "\n".join(lines) + "\n"


def splice_readme(suite: str, md: str, path="README.md") -> None:
    """swap the generated table into the readme between its markers, so readme numbers can only
    come from this script."""
    start, end = f"<!-- compare:{suite} -->", f"<!-- /compare:{suite} -->"
    text = Path(path).read_text()
    if start not in text:
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    Path(path).write_text(f"{head}{start}\n{md}{end}{tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite_small_v2")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument(
        "--limit", type=int, default=None, help="first k scenarios only, for quick looks"
    )
    args = ap.parse_args()
    cfg = CompareConfig(suite=args.suite)

    out = start_run(f"compare-{cfg.suite}", cfg, seed=cfg.seed)
    rows = evaluate(cfg.suite, cfg.methods, jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")
    n = len(rows) // len(cfg.methods)
    summary = summarize(rows, cfg)
    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    md = table(summary, cfg, n, commit)
    (out / "summary.md").write_text(md)

    # the committed copies the readme points at. partial runs never overwrite them
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path(f"docs/tables/compare_{cfg.suite}.md").write_text(md)
        save_both(lambda th: draw(summary, cfg, n, th), f"docs/figures/compare_{cfg.suite}")
        splice_readme(cfg.suite, md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
