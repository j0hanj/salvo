"""quality + latency vs swarm size. two panels, two separate y axes, never one shared dual axis."""

import matplotlib.pyplot as plt
import numpy as np

from viz.style import FONT

# fixed categorical order so a method keeps its color across every figure. light / dark steps
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
ORDER = ["gnn", "milp", "hungarian", "greedy-value", "greedy-nearest"]


def _style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=3)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])
    ax.grid(color=theme["hairline"], lw=0.8)
    ax.set_axisbelow(True)


def draw(summary, cfg, theme):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    dark = theme["surface"] == "#1a1a19"
    colors = dict(zip(ORDER, SERIES["dark" if dark else "light"], strict=True))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=theme["surface"])
    methods = [m for m in ORDER if any(s["method"] == m for s in summary)]

    for m in methods:
        rows = sorted((s for s in summary if s["method"] == m), key=lambda s: s["n"])
        n = np.array([s["n"] for s in rows])
        v = np.array([s["value"] for s in rows])
        lat = np.array([s["latency"] for s in rows])
        c = colors[m]
        lw = 2.6 if m == "gnn" else 1.8
        a.fill_between(n, v[:, 1], v[:, 2], color=c, alpha=0.12, lw=0)
        a.plot(n, v[:, 0], "o-", color=c, lw=lw, ms=5, mec=theme["surface"], mew=1.2, label=m)
        b.fill_between(n, lat[:, 1], lat[:, 2], color=c, alpha=0.12, lw=0)
        b.plot(n, lat[:, 0], "o-", color=c, lw=lw, ms=5, mec=theme["surface"], mew=1.2)

    for ax in (a, b):
        _style(ax, theme)
        ax.set_xscale("log")
        ax.set_xticks(cfg.sizes, [str(s) for s in cfg.sizes])
        ax.minorticks_off()
        ax.set_xlabel("threats in the raid (log scale)")
        # shade the sizes the gnn trained on, so zero-shot is obvious at a glance
        ax.axvspan(min(cfg.sizes) * 0.85, 10, color=theme["hairline"], alpha=0.6, lw=0)
    a.set_ylabel("threat value defended (%)")
    b.set_ylabel("decision latency, ms (log scale)")
    b.set_yscale("log")
    a.annotate(
        "gnn trained\nin here",
        (10, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-4, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=theme["muted"],
    )

    a.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=theme["ink2"])
    biggest = max(cfg.sizes)
    g = next(s for s in summary if s["method"] == "gnn" and s["n"] == biggest)
    ref = next(s for s in summary if s["method"] == cfg.reference and s["n"] == biggest)
    fig.text(
        0.0,
        1.07,
        f"at {biggest} threats the gnn defends {g['value'][0]:.0f}% of threat "
        f"value vs {ref['value'][0]:.0f}% for {cfg.reference}",
        fontsize=14,
        fontweight="bold",
        color=theme["ink"],
        ha="left",
    )
    fig.text(
        0.0,
        1.0,
        "trained on 5-10 threat fights only · 50 scenarios per size · lines = mean, "
        "bands = 95% bootstrap ci",
        fontsize=10,
        color=theme["ink2"],
        ha="left",
    )
    fig.tight_layout()
    return fig
