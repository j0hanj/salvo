"""how often does the assigner face a real choice? counts decision ticks w any legal shot, w more
than one, and where the baselines disagree. a suite where they never disagree can't rank methods.

uv run python -m scripts.decision_stats --suite suite_small_v1 --limit 40
"""

import argparse

from baselines.common import feasible
from baselines.greedy import nearest, value
from baselines.hungarian import hungarian
from eval.suites import load_suite
from sim.engine import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite_small_v2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    suite = load_suite(args.suite)
    scenarios = suite.scenarios[: args.limit]
    counts = {"live": 0, "multi": 0, "disagree": 0}

    def policy(obs):
        ok = feasible(obs)
        if ok.any():
            counts["live"] += 1
            counts["multi"] += int(ok.sum() > 1)
            picks = {tuple(sorted(p(obs))) for p in (nearest, value, hungarian)}
            counts["disagree"] += int(len(picks) > 1)
        return nearest(obs)  # drive the run w nearest so every suite gets the same treatment

    for sc in scenarios:
        run(sc, suite.cfg.sim, suite.cfg.types, policy)
    print(
        f"{suite.id}, {len(scenarios)} scenarios, driven by greedy-nearest: "
        f"{counts['live']} ticks w a legal shot, {counts['multi']} w more than one, "
        f"{counts['disagree']} where nearest / value / hungarian disagree"
    )


if __name__ == "__main__":
    main()
