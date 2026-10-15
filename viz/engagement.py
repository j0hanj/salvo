"""draw one engagement from a recorded run."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim.engine import KILLED, LEAKED, Result  # noqa: E402
from sim.types import EffectorType, Scenario, SimConfig  # noqa: E402


def plot(sc: Scenario, res: Result, cfg: SimConfig, types: list[EffectorType], title="", ax=None):
    ax = ax or plt.subplots(figsize=(7, 7))[1]
    h = res.history
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.add_patch(plt.Circle((0, 0), cfg.asset_radius, color="k", alpha=0.8))
    for e, (pos, k) in enumerate(zip(sc.eff_pos, sc.eff_type, strict=True)):
        c = colors[k % len(colors)]
        ax.add_patch(plt.Circle(pos, types[k].range, fill=False, ls="--", lw=0.8, color=c))
        ax.plot(
            *pos,
            "s",
            color=c,
            ms=7,
            label=f"{types[k].name} site" if e == list(sc.eff_type).index(k) else None,
        )

    th = h["threat"]  # (steps, n, 2), nan before launch
    for j in range(sc.n_threats):
        leaked = res.status[j] == LEAKED
        ax.plot(th[:, j, 0], th[:, j, 1], color="tab:red" if leaked else "0.55", lw=1.2)
    ih = h["intercept"]
    for s in range(ih.shape[1]):
        if np.isfinite(ih[:, s, 0]).any():
            ax.plot(ih[:, s, 0], ih[:, s, 1], color="tab:blue", lw=0.7, alpha=0.8)
    if len(h["kills"]):
        ax.plot(h["kills"][:, 0], h["kills"][:, 1], "x", color="k", ms=8, label="kill")

    ax.plot([], [], color="0.55", label="threat (stopped)")
    ax.plot([], [], color="tab:red", label="threat (leaked)")
    ax.plot([], [], color="tab:blue", label="interceptor")
    n_kill = int((res.status == KILLED).sum())
    ax.set_title(f"{title}  stopped {n_kill}/{sc.n_threats}, cost {res.cost:g}".strip())
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", fontsize=8)
    return ax
