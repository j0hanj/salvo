"""draw one recorded engagement as a top-down map."""

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from sim.engine import KILLED, LEAKED, Result
from sim.types import EffectorType, Scenario, SimConfig
from viz.style import figure, titles

KM = 1e-3
SITE_MARKERS = ["o", "s", "D", "^"]  # site type = shape + label, color is saved for outcomes


def _heading_arrow(ax, track, color, frac=0.35):
    """small arrowhead partway along a track so you can tell which way it's going."""
    pts = track[np.isfinite(track[:, 0])]
    if len(pts) < 3:
        return
    i = max(1, int(len(pts) * frac))
    ax.annotate(
        "",
        xy=pts[i] * KM,
        xytext=pts[i - 1] * KM,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0, "mutation_scale": 14},
    )


def draw(
    sc: Scenario,
    res: Result,
    cfg: SimConfig,
    types: list[EffectorType],
    theme: dict,
    title: str,
    subtitle: str,
):
    fig, ax = figure(theme)
    ring = [pe.withStroke(linewidth=3.5, foreground=theme["surface"])]
    h = res.history

    # site ranges first so everything else draws on top
    for k in sorted(set(sc.eff_type)):
        for pos in sc.eff_pos[sc.eff_type == k]:
            ax.add_patch(
                Circle(
                    pos * KM,
                    types[k].range * KM,
                    fill=False,
                    lw=0.8,
                    ls=(0, (4, 3)),
                    color=theme["axis"],
                    zorder=1,
                )
            )

    th = h["threat"]
    for j in range(sc.n_threats):
        leaked = res.status[j] == LEAKED
        c = theme["critical"] if leaked else theme["muted"]
        ax.plot(
            th[:, j, 0] * KM,
            th[:, j, 1] * KM,
            color=c,
            lw=2.0 if leaked else 1.4,
            solid_capstyle="round",
            zorder=3 if leaked else 2,
        )
        track = th[:, j]
        start = track[np.isfinite(track[:, 0])][0]
        ax.plot(*(start * KM), "o", ms=5, mfc=theme["surface"], mec=c, mew=1.4, zorder=4)
        _heading_arrow(ax, th[:, j], c)

    ih = h["intercept"]
    for s in range(ih.shape[1]):
        if np.isfinite(ih[:, s, 0]).any():
            ax.plot(
                ih[:, s, 0] * KM,
                ih[:, s, 1] * KM,
                color=theme["series1"],
                lw=1.1,
                alpha=0.9,
                zorder=2.5,
            )

    if len(h["kills"]):
        ax.plot(
            h["kills"][:, 0] * KM,
            h["kills"][:, 1] * KM,
            "X",
            ms=10,
            color=theme["ink"],
            mec=theme["surface"],
            mew=1.2,
            zorder=6,
        )

    # asset + sites. sites are packed in close to the asset, so they're named by shape in the
    # legend instead of text labels that'd pile on top of each other
    ax.plot(0, 0, "*", ms=18, color=theme["ink"], mec=theme["surface"], mew=1.5, zorder=8)
    ax.annotate(
        "defended asset",
        (0, 0),
        xytext=(0, -24),
        textcoords="offset points",
        fontsize=9,
        color=theme["ink"],
        ha="center",
        va="top",
        path_effects=ring,
        zorder=9,
    )
    for pos, k in zip(sc.eff_pos, sc.eff_type, strict=True):
        ax.plot(
            *(pos * KM),
            SITE_MARKERS[k % len(SITE_MARKERS)],
            ms=9,
            color=theme["ink2"],
            mec=theme["surface"],
            mew=1.5,
            zorder=7,
        )

    legend = [
        Line2D([], [], color=theme["muted"], lw=1.4, label="threat, stopped"),
        Line2D([], [], color=theme["critical"], lw=2.0, label="threat, leaked"),
        Line2D([], [], color=theme["series1"], lw=1.1, label="interceptor"),
        Line2D([], [], ls="", marker="X", ms=9, color=theme["ink"], label="kill"),
    ]
    for k in sorted(set(sc.eff_type)):
        legend.append(
            Line2D(
                [],
                [],
                ls="",
                marker=SITE_MARKERS[k % len(SITE_MARKERS)],
                ms=8,
                color=theme["ink2"],
                label=f"{types[k].name}-range site",
            )
        )
    legend.append(Line2D([], [], color=theme["axis"], lw=0.8, ls=(0, (4, 3)), label="site range"))
    # legend under the map so it never covers a track
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        fontsize=8.5,
        frameon=False,
        labelcolor=theme["ink2"],
        handlelength=1.8,
        columnspacing=1.6,
    )

    # frame the action (threat spawns + asset), not the whole long-range circle
    pts = np.concatenate([sc.threat_start, sc.eff_pos, [[0, 0]]]) * KM
    pad = 0.6
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    titles(fig, ax, theme, title, subtitle)
    return fig


def headline(res: Result, sc: Scenario, policy: str) -> tuple[str, str]:
    n_kill = int((res.status == KILLED).sum())
    n_leak = int((res.status == LEAKED).sum())
    title = f"{policy}: {n_kill} of {sc.n_threats} threats stopped"
    sub = f"{n_leak} leaked · {int(res.shots.sum())} shots · {res.cost:g} cost units spent"
    return title, sub
