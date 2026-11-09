"""imitation learning end to end. milp plays every training scenario and labels every tick, the gnn
trains on it (behavior cloning). then `dagger_rounds` times: the gnn plays the scenarios itself,
milp labels the states the gnn actually reached, add those to the pile, retrain from scratch.
writes model.pt (final) + model_round{k}.pt + history.json into a logged run dir.

uv run python -m scripts.train
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

import torch

from baselines.milp import milp
from eval.runlog import start_run
from eval.suites import load_suite
from learned.dataset import collect, merge
from learned.policy import GNNPolicy
from learned.train import TrainConfig, train


@dataclass
class RunConfig:
    # more than one = mixed sizes. scenario ids get offset per suite so the train/val split
    # (by scenario) can't mix up scenario 3 of one suite w scenario 3 of another
    suites: list[str] = field(default_factory=lambda: ["suite_train_v1"])
    limit: int | None = None
    dagger_rounds: int = 3
    train: TrainConfig = field(default_factory=TrainConfig)


SCN_STRIDE = 1_000_000


def collect_all(suites, driver, limit, jobs):
    """milp labels every suite, scenario ids offset by SCN_STRIDE per suite so they stay unique."""
    out = None
    for k, suite in enumerate(suites):
        b = collect(suite, driver=driver, labeler=milp, limit=limit, jobs=jobs)
        b.scn = b.scn + k * SCN_STRIDE
        out = b if out is None else merge(out, b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dagger", type=int, default=None, help="dagger rounds, 0 = plain bc")
    ap.add_argument("--degree", action="store_true", help="add the log-degree channel")
    ap.add_argument("--loss", choices=["bce", "listwise"], default="bce")
    ap.add_argument("--suites", nargs="+", default=None, help="training suites, default 5-10 only")
    args = ap.parse_args()
    cfg = RunConfig(limit=args.limit)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.dagger is not None:
        cfg.dagger_rounds = args.dagger
    cfg.train.degree = args.degree
    cfg.train.loss = args.loss
    if args.suites:
        cfg.suites = args.suites

    out = start_run("train", cfg, seed=cfg.train.seed)
    suites = [load_suite(s) for s in cfg.suites]
    jobs = os.cpu_count() or 1
    history = []

    with open(out / "train.log", "w") as f:

        def log(line):
            print(line)
            f.write(line + "\n")

        def fit(k, data):
            log(
                f"round {k}: {len(data)} labeled ticks, {len(data.y)} edges, "
                f"{data.y.mean():.3f} positive"
            )
            model, hist = train(data, cfg.train, log=log)
            history.append({"round": k, "ticks": len(data), "epochs": hist})
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(cfg.train)},
                out / f"model_round{k}.pt",
            )
            return model

        # round 0 = plain behavior cloning on milp's own episodes
        data = collect_all(suites, milp, cfg.limit, jobs)
        model = fit(0, data)
        for k in range(1, cfg.dagger_rounds + 1):
            # the model drives, milp only labels. these are the states plain bc never sees
            data = merge(data, collect_all(suites, GNNPolicy(model), cfg.limit, jobs))
            model = fit(k, data)

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg.train)}, out / "model.pt")
    data.save(out / "data.npz")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
