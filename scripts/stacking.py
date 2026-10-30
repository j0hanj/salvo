"""how many effectors pick the same threat on the same tick. the gnn's failure mode at scale is
piling onto threats, so compare it to milp at a few sizes.

uv run python -m scripts.stacking --model results/<run>-train/model.pt --sizes 10 100 --limit 5
"""

import argparse
from collections import Counter

import numpy as np

from eval.harness import resolve
from eval.suites import load_suite
from sim.engine import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    for n in args.sizes:
        suite = load_suite(f"suite_scale_n{n}_v1")
        for name in ["milp", f"gnn:{args.model}"]:
            policy = resolve(name)
            stack = []

            def counted(obs, policy=policy, stack=stack):
                pairs = policy(obs)
                stack.extend(Counter(j for _, j in pairs).values())
                return pairs

            for sc in suite.scenarios[: args.limit]:
                run(sc, suite.cfg.sim, suite.cfg.types, counted)
            s = np.array(stack)
            label = "gnn" if name.startswith("gnn:") else name
            print(
                f"n={n} {label}: {s.mean():.2f} shooters per targeted threat per tick, "
                f"max {s.max()}, {100 * (s >= 3).mean():.1f}% w 3+ "
                f"(first {args.limit} scenarios of suite_scale_n{n}_v1)"
            )


if __name__ == "__main__":
    main()
