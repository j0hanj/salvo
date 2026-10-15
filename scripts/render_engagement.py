"""render one engagement from a suite as a png.

uv run python -m scripts.render_engagement --suite suite_small_v1 --index 3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from baselines.greedy import nearest
from eval.suites import load_suite
from sim.engine import run
from sim.render import plot

POLICIES = {"nearest": nearest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite_small_v1")
    ap.add_argument("--index", type=int, default=3)
    ap.add_argument("--policy", default="nearest", choices=POLICIES)
    ap.add_argument("--out", default="docs/figures/engagement_example.png")
    args = ap.parse_args()

    suite = load_suite(args.suite)
    sc = suite.scenarios[args.index]
    res = run(sc, suite.cfg.sim, suite.cfg.types, POLICIES[args.policy], record=True)
    ax = plot(
        sc, res, suite.cfg.sim, suite.cfg.types, title=f"{suite.id} #{args.index} {args.policy}"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
