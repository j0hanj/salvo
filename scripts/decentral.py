"""phase 4 headline: cbba vs the decentralized gnn as whole effectors get jammed dark.

both see the same node outages (same seed -> same dark spells, see sim.comms.Channel). cbba: 10
rounds of auction + consensus among whoever's lit. dgnn: every node builds its graph from the
effectors it heard and decodes alone (learned/decentral.py). hungarian = central, no jamming,
the ceiling.

uv run python -m scripts.decentral --model results/<run>-train/model.pt
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from eval.harness import evaluate, write_csv
from eval.runlog import start_run
from eval.stats import bootstrap_ci, paired_diff
from scripts.compare import splice_readme


@dataclass
class DecentralConfig:
    model: str = ""
    suites: list[str] = field(
        default_factory=lambda: ["suite_scale_n100_v1", "suite_scale_n500_v1"]
    )
    downs: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7])
    burst: float = 20.0  # mean ticks per dark spell
    reference: str = "hungarian"
    n_boot: int = 10_000
    seed: int = 0


def fmt(c, d=1):
    return f"{c[0]:.{d}f} [{c[1]:.{d}f}, {c[2]:.{d}f}]"


def methods(cfg: DecentralConfig) -> dict[str, tuple[str, float]]:
    """method string -> (family, share of ticks a node is dark)"""
    out = {}
    for d in cfg.downs:
        out[f"cbba-node@{d}/{cfg.burst}"] = ("cbba", d)
        out[f"dgnn-node@{d}/{cfg.burst}:{cfg.model}"] = ("gnn (decentral)", d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = DecentralConfig(model=args.model)
    run_dir = Path(cfg.model).parent
    model_commit = json.loads((run_dir / "meta.json").read_text())["git_commit"]
    tcfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    train_suites = tcfg.get("suites") or [tcfg["suite"]]
    out = start_run("decentral", cfg, seed=cfg.seed)
    kw = {"n_boot": cfg.n_boot, "seed": cfg.seed}
    ms = methods(cfg)

    rows = []
    for s in cfg.suites:
        print(s, flush=True)
        rows += evaluate(s, [cfg.reference, *ms], jobs=args.jobs, limit=args.limit)
    write_csv(rows, out / "per_scenario.csv")

    lines = [
        f"| threats | nodes dark % | method | threats stopped % | value defended % "
        f"| Δ value vs {cfg.reference} | shooters per targeted threat | decision latency ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in cfg.suites:
        at = sorted((r for r in rows if r["suite"] == s), key=lambda r: r["scenario"])
        n = int(at[0]["n_threats"])
        ref_v = np.array([100 * r["value_frac"] for r in at if r["method"] == cfg.reference])
        order = [(cfg.reference, (cfg.reference, 0.0)), *ms.items()]
        for m, (fam, d) in order:
            rs = [r for r in at if r["method"] == m]
            col = lambda k, rs=rs: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
            delta = (
                "—"
                if m == cfg.reference
                else fmt(paired_diff(100 * col("value_frac"), ref_v, **kw))
            )
            dark = "— (central)" if m == cfg.reference else f"{100 * d:g}"
            lines.append(
                f"| {n} | {dark} | {fam} | {fmt(bootstrap_ci(100 * col('stopped_frac'), **kw))} "
                f"| {fmt(bootstrap_ci(100 * col('value_frac'), **kw))} | {delta} "
                f"| {fmt(bootstrap_ci(col('stack_mean'), **kw), 2)} "
                f"| {fmt(bootstrap_ci(col('lat_mean_ms'), **kw), 2)} |"
            )
    n_scn = len(rows) // (len(cfg.suites) * (1 + len(ms)))
    commit = json.loads((out / "meta.json").read_text())["git_commit"]
    lines += [
        "",
        f"`suite_scale_n*_v1`, {n_scn} scenarios per size, same scenarios and the same dark "
        f"spells (mean {cfg.burst:g} ticks) for cbba and the gnn. a dark node hears nobody and "
        "nobody hears it, it still fires on its own pick. cbba: 10 rounds of auction + "
        "consensus per tick. gnn: every node decodes its own local graph (threshold 0, floor "
        f"0.35) and fires only its own pick. gnn `{run_dir.name}`, trained on "
        f"{', '.join(train_suites)} by `scripts/train.py` at `{model_commit[:7]}`. latency = "
        "wall time per decision tick for all nodes together on one cpu thread. 95% bootstrap "
        f"cis, Δ paired per scenario. made by `scripts/decentral.py` at `{commit[:7]}`.",
    ]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    if args.limit is None:
        Path("docs/tables").mkdir(parents=True, exist_ok=True)
        Path("docs/tables/decentral.md").write_text(md)
        splice_readme("decentral", md)
    print(md)
    print(f"run logged to {out}")


if __name__ == "__main__":
    main()
