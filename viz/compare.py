"""method comparison: dot + 95% ci per method, one panel per metric, never two scales on one axis"""

import matplotlib.pyplot as plt

from viz.style import FONT


def _panel(ax, theme, names, cis, ref_idx, xlabel, digits):
    for i, (m, lo, hi) in enumerate(cis):
        c = theme["ink"] if i == ref_idx else theme["series1"]
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(m, i, "o", ms=8, color=c, mec=theme["surface"], mew=1.5, zorder=3)
        ax.annotate(
            f"{m:.{digits}f}",
            (hi, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=theme["ink2"],
        )
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(axis="x", colors=theme["muted"], labelsize=9, length=3)
    ax.tick_params(axis="y", colors=theme["ink"], labelsize=10, length=0)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.grid(axis="x", color=theme["hairline"], lw=0.8)
    ax.set_axisbelow(True)
    # leave room for the value labels past the whiskers
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.12)


def draw(summary, cfg, n, theme):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, facecolor=theme["surface"])
    names = [s["method"] for s in summary]
    ref = names.index(cfg.reference)
    panels = [
        ("value", "threat value defended (%)"),
        ("stopped", "threats stopped (%)"),
        ("cost_per_stop", "cost per threat stopped (lower is better)"),
    ]
    for ax, (key, label) in zip(axes, panels, strict=True):
        _panel(ax, theme, names, [s[key] for s in summary], ref, label, 1)

    # headline on value defended: that's where the methods actually separate
    best = max(summary, key=lambda s: s["value"][0])
    worst = min(summary, key=lambda s: s["value"][0])
    fig.text(
        0.0,
        1.06,
        f"{best['method']} defends {best['value'][0]:.0f}% of threat value, "
        f"{worst['method']} {worst['value'][0]:.0f}%",
        fontsize=14,
        fontweight="bold",
        color=theme["ink"],
        ha="left",
    )
    fig.text(
        0.0,
        0.99,
        f"{cfg.suite} · {n} scenarios · same scenarios for every method · "
        "dots = mean, bars = 95% bootstrap ci",
        fontsize=10,
        color=theme["ink2"],
        ha="left",
    )
    fig.tight_layout()
    return fig
