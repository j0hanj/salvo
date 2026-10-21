"""render one engagement from a suite as a png (light + dark).

uv run python -m scripts.render_engagement --suite suite_small_v2 --index 11
"""

import argparse

from baselines.greedy import nearest, value
from baselines.hungarian import hungarian
from eval.suites import load_suite
from sim.engine import run
from viz.engagement import draw, headline
from viz.style import save_both

POLICIES = {"greedy-nearest": nearest, "greedy-value": value, "hungarian": hungarian}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite_small_v2")
    ap.add_argument("--index", type=int, default=11)
    ap.add_argument("--policy", default="greedy-nearest", choices=POLICIES)
    ap.add_argument("--out", default="docs/figures/engagement_example")
    args = ap.parse_args()

    suite = load_suite(args.suite)
    sc = suite.scenarios[args.index]
    res = run(sc, suite.cfg.sim, suite.cfg.types, POLICIES[args.policy], record=True)
    title, sub = headline(res, sc, args.policy)
    sub = f"{suite.id} #{args.index} · {sub}"
    paths = save_both(
        lambda theme: draw(sc, res, suite.cfg.sim, suite.cfg.types, theme, title, sub), args.out
    )
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
