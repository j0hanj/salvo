"""expand every suite yaml and write/verify its .lock hash.

uv run python -m scripts.build_suites          # verify all, write locks for new suites
uv run python -m scripts.build_suites --force  # overwrite a bad lock (or just bump the version)
"""

import argparse
import json

from configs.load import load
from eval.suites import SUITE_DIR, SuiteConfig, build, suite_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    bad = False
    for path in sorted(SUITE_DIR.glob("*.yaml")):
        suite = build(load(SuiteConfig, path))
        h = suite_hash(suite)
        lock = path.with_suffix(".lock")
        n_threats = sum(s.n_threats for s in suite.scenarios)
        if lock.exists() and json.loads(lock.read_text())["hash"] != h and not args.force:
            print(f"MISMATCH {suite.id}: lock says one thing, code says another")
            bad = True
            continue
        lock.write_text(
            json.dumps(
                {"hash": h, "n_scenarios": len(suite.scenarios), "n_threats_total": n_threats},
                indent=2,
            )
            + "\n"
        )
        print(f"ok {suite.id} {h[:12]} ({len(suite.scenarios)} scenarios, {n_threats} threats)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
