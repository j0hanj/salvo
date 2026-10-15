"""one look for every figure. light + dark variants so the readme matches github's theme.

colors are role-based (outcome, interceptor, ink), not per-chart picks. the blue/red pair passes
colorblind separation checks on both surfaces, and every colored thing also has a label or shape so
color is never the only cue.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "hairline": "#e1e0d9",
        "axis": "#c3c2b7",
        "series1": "#2a78d6",  # blue: our stuff (interceptors, methods)
        "critical": "#d03b3b",  # red: bad outcome (leaked)
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "hairline": "#2c2c2a",
        "axis": "#383835",
        "series1": "#3987e5",
        "critical": "#d03b3b",
    },
}

FONT = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def figure(theme: dict, size=(8, 8)):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    fig, ax = plt.subplots(figsize=size, facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=3)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])
    return fig, ax


def titles(fig, ax, theme, title, subtitle):
    # left-aligned headline + one line of context, reads like a sentence
    ax.set_title(subtitle, loc="left", fontsize=10, color=theme["ink2"], pad=10)
    fig.text(
        ax.get_position().x0,
        ax.get_position().y1 + 0.045,
        title,
        fontsize=15,
        fontweight="bold",
        color=theme["ink"],
        ha="left",
    )


def save_both(draw, out_stem: str | Path) -> list[Path]:
    """draw(theme) -> fig. writes <stem>_light.png and <stem>_dark.png."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, theme in THEMES.items():
        fig = draw(theme)
        p = out_stem.with_name(f"{out_stem.name}_{name}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=theme["surface"])
        plt.close(fig)
        paths.append(p)
    return paths
