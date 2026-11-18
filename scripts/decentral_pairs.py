"""head to head from a `scripts/decentral.py` run: gnn minus cbba, paired per scenario (same
scenario, same dark spells), at every raid size and dark level. reads the run's per-scenario csv,
nothing is re-simulated.

uv run python -m scripts.decentral_pairs --run results/<stamp>-decentral
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.harness import read_csv
from eval.stats import paired_diff
from scripts.compare import splice_readme


def fmt(c, d=1):
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def dark_of(method: str) -> float:
    return float(method.split("@", 1)[1].split("/", 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    rows = read_csv(run / "per_scenario.csv")
    commit = json.loads((run / "meta.json").read_text())["git_commit"]
    kw = {"n_boot": 10_000, "seed": 0}

    lines = [
        "| threats | nodes dark % | Δ value, gnn - cbba | Δ threats stopped, gnn - cbba |",
        "|---|---|---|---|",
    ]
    for n in sorted({int(r["n_threats"]) for r in rows}):
        at = sorted((r for r in rows if int(r["n_threats"]) == n), key=lambda r: r["scenario"])
        cb = {dark_of(r["method"]) for r in at if r["method"].startswith("cbba-node@")}
        for d in sorted(cb):

            def col(prefix, key, d=d):
                return np.array(
                    [
                        100 * r[key]
                        for r in at
                        if r["method"].startswith(prefix) and dark_of(r["method"]) == d
                    ]
                )

            dv = paired_diff(col("dgnn-node@", "value_frac"), col("cbba-node@", "value_frac"), **kw)
            ds = paired_diff(
                col("dgnn-node@", "stopped_frac"), col("cbba-node@", "stopped_frac"), **kw
            )
            lines.append(f"| {n} | {100 * d:g} | {fmt(dv)} | {fmt(ds)} |")
    lines += [
        "",
        "positive = the decentral gnn did better. paired per scenario (same scenario, same dark "
        "spells), 95% bootstrap cis. from the per-scenario rows of "
        f"`{run.name}` (`scripts/decentral.py` at `{commit[:7]}`), made by "
        "`scripts/decentral_pairs.py`.",
    ]
    md = "\n".join(lines) + "\n"
    Path("docs/tables/decentral_pairs.md").write_text(md)
    splice_readme("decentral_pairs", md)
    print(md)


if __name__ == "__main__":
    main()
