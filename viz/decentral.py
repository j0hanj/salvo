"""value defended vs share of nodes jammed dark: cbba vs the decentralized gnn, hungarian (central,
no jamming) as the ceiling. one panel per raid size. reads a `scripts/decentral.py` run, so the
figure can only show numbers that run produced.

uv run python -m viz.decentral --run results/<stamp>-decentral
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval.harness import read_csv
from eval.stats import bootstrap_ci
from viz.scaling import SERIES, _style
from viz.style import FONT, save_both

FAMILIES = {"cbba-node@": "cbba", "dgnn-node@": "gnn (decentral)"}


def summarize(rows, reference="hungarian"):
    """{n: {"ref": ci, family: [(dark %, ci of value %), ...]}}"""
    out = {}
    for n in sorted({int(r["n_threats"]) for r in rows}):
        at = [r for r in rows if int(r["n_threats"]) == n]
        d = {
            "ref": bootstrap_ci(
                np.array([100 * r["value_frac"] for r in at if r["method"] == reference])
            )
        }
        for m in dict.fromkeys(r["method"] for r in at):
            for prefix, fam in FAMILIES.items():
                if m.startswith(prefix):
                    dark = 100 * float(m[len(prefix) :].split("/", 1)[0])
                    v = np.array([100 * r["value_frac"] for r in at if r["method"] == m])
                    d.setdefault(fam, []).append((dark, bootstrap_ci(v)))
        for fam in FAMILIES.values():
            d[fam].sort()
        out[n] = d
    return out


def draw(summary, theme, commit: str):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    dark_theme = theme["surface"] == "#1a1a19"
    colors = SERIES["dark" if dark_theme else "light"]
    # same colors as the scaling figure: gnn blue, hungarian green. cbba gets the next free one
    col = {"gnn (decentral)": colors[0], "cbba": colors[3], "ref": colors[2]}
    sizes = list(summary)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(6 * len(sizes), 4.4), facecolor=theme["surface"]
    )
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, sizes, strict=True):
        _style(ax, theme)
        d = summary[n]
        ax.axhline(d["ref"][0], color=col["ref"], lw=1.4, ls="--")
        ax.text(
            0,
            d["ref"][0],
            " hungarian, central, no jamming",
            va="bottom",
            fontsize=8,
            color=col["ref"],
        )
        for fam in FAMILIES.values():
            x = np.array([p[0] for p in d[fam]])
            ci = np.array([p[1] for p in d[fam]])
            lw = 2.6 if fam.startswith("gnn") else 1.8
            ax.fill_between(x, ci[:, 1], ci[:, 2], color=col[fam], alpha=0.12, lw=0)
            ax.plot(
                x,
                ci[:, 0],
                "o-",
                color=col[fam],
                lw=lw,
                ms=5,
                mec=theme["surface"],
                mew=1.2,
                label=fam,
            )
        ax.set_xlabel("share of ticks each effector is jammed dark (%)")
        ax.set_ylabel("threat value defended (%)")
        ax.set_title(f"{n} threats", loc="left", fontsize=10, color=theme["ink2"])
    axes[0].legend(loc="lower left", fontsize=9, frameon=False, labelcolor=theme["ink2"])

    n = max(sizes)
    worst = max(summary[n]["cbba"])[0]
    c = dict(summary[n]["cbba"])[worst][0]
    g = dict(summary[n]["gnn (decentral)"])[worst][0]
    fig.text(
        0.0,
        1.07,
        f"at {n} threats w effectors dark {worst:g}% of the time: decentral gnn {g:.1f}%, "
        f"cbba {c:.1f}% of threat value defended",
        fontsize=13,
        fontweight="bold",
        color=theme["ink"],
        ha="left",
    )
    fig.text(
        0.0,
        1.01,
        "same scenarios + same dark spells for both · lines = mean, bands = 95% bootstrap ci",
        fontsize=9.5,
        color=theme["ink2"],
        ha="left",
    )
    fig.text(
        1.0,
        -0.03,
        f"scripts/decentral.py at {commit[:7]}",
        fontsize=8,
        color=theme["muted"],
        ha="right",
    )
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a results/<stamp>-decentral dir")
    args = ap.parse_args()
    run = Path(args.run)
    rows = read_csv(run / "per_scenario.csv")
    commit = json.loads((run / "meta.json").read_text())["git_commit"]
    summary = summarize(rows)
    for p in save_both(lambda th: draw(summary, th, commit), "docs/figures/decentral"):
        print(p)


if __name__ == "__main__":
    main()
