"""Δ value defended vs hungarian as the capped gnn's fire threshold drops, one line per raid size.
reads a `scripts/threshold.py` run, so the figure can only show numbers that run produced.

uv run python -m viz.threshold --run results/<run>-threshold
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from eval.harness import read_csv
from eval.stats import paired_diff
from viz.scaling import SERIES, _style
from viz.style import FONT, save_both


def summarize(rows, reference="hungarian"):
    """{n: {"gnn": [(thr, (mean, lo, hi)), ...], "greedy-value": (mean, lo, hi)}} as Δ value %."""
    out = {}
    for n in sorted({int(r["n_threats"]) for r in rows}):
        at = sorted((r for r in rows if r["n_threats"] == n), key=lambda r: r["scenario"])
        ref = np.array([100 * r["value_frac"] for r in at if r["method"] == reference])
        d = {"gnn": []}
        for m in dict.fromkeys(r["method"] for r in at):
            if m == reference:
                continue
            v = np.array([100 * r["value_frac"] for r in at if r["method"] == m])
            ci = paired_diff(v, ref, n_boot=10_000, seed=0)
            if m.startswith("gnn-capped@"):
                d["gnn"].append((float(m.split("@", 1)[1].split(":", 1)[0]), ci))
            else:
                d[m] = ci
        d["gnn"].sort(reverse=True)  # 0.5 first, the default
        out[n] = d
    return out


def draw(summary, theme, commit: str):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    dark = theme["surface"] == "#1a1a19"
    colors = SERIES["dark" if dark else "light"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=theme["surface"])
    _style(ax, theme)
    ax.axhline(0, color=theme["muted"], lw=1, ls="--")
    ends = []
    for c, (n, d) in zip(colors, summary.items(), strict=False):
        t = np.array([x[0] for x in d["gnn"]])
        ci = np.array([x[1] for x in d["gnn"]])
        ax.fill_between(t, ci[:, 1], ci[:, 2], color=c, alpha=0.12, lw=0)
        ax.plot(t, ci[:, 0], "o-", color=c, lw=2, ms=5, mec=theme["surface"], mew=1.2)
        ends.append([ci[-1, 0], t[-1], c, n])
    # direct labels at the right end, pushed apart so lines that converge stay readable
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.6)
    for y, x, c, n in ends:
        ax.annotate(
            f"{n} threats",
            (x, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=c,
        )
    ax.invert_xaxis()  # left to right = more willing to fire
    ax.set_xticks(sorted({x[0] for d in summary.values() for x in d["gnn"]}))
    ax.set_xlabel("fire threshold (lower = fires on less confident edges)")
    ax.set_ylabel("Δ threat value defended vs hungarian (pts)")
    ax.text(
        ax.get_xlim()[0],
        0,
        "hungarian",
        va="top",
        ha="left",
        fontsize=8,
        color=theme["muted"],
    )

    big = max(summary)
    g = summary[big]["gnn"]
    gv = summary[big].get("greedy-value")
    fig.text(
        0.0,
        1.08,
        f"at {big} threats, dropping the threshold {g[0][0]:g} -> {g[-1][0]:g} "
        f"takes the gap from {g[0][1][0]:.1f} to {g[-1][1][0]:.1f} pts",
        fontsize=13,
        fontweight="bold",
        color=theme["ink"],
        ha="left",
    )
    sub = "capped decode · 50 scenarios per size · bands = 95% paired bootstrap ci"
    if gv is not None:
        sub += f" · greedy-value at {big}: {gv[0]:+.1f}"
    fig.text(0.0, 1.01, sub, fontsize=9.5, color=theme["ink2"], ha="left")
    fig.text(
        1.0,
        -0.02,
        f"scripts/threshold.py at {commit[:7]} · swept on the eval suites, diagnostic only",
        fontsize=8,
        color=theme["muted"],
        ha="right",
    )
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a results/<stamp>-threshold dir")
    args = ap.parse_args()
    run = Path(args.run)
    rows = read_csv(run / "per_scenario.csv")
    ref = yaml.safe_load((run / "config.yaml").read_text())["reference"]
    commit = json.loads((run / "meta.json").read_text())["git_commit"]
    summary = summarize(rows, ref)
    for p in save_both(lambda th: draw(summary, th, commit), "docs/figures/threshold"):
        print(p)


if __name__ == "__main__":
    main()
