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
    suite: str = "suite_train_v1"
    limit: int | None = None
    dagger_rounds: int = 3
    train: TrainConfig = field(default_factory=TrainConfig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dagger", type=int, default=None, help="dagger rounds, 0 = plain bc")
    args = ap.parse_args()
    cfg = RunConfig(limit=args.limit)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.dagger is not None:
        cfg.dagger_rounds = args.dagger

    out = start_run("train", cfg, seed=cfg.train.seed)
    suite = load_suite(cfg.suite)
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
        data = collect(suite, driver=milp, labeler=milp, limit=cfg.limit, jobs=jobs)
        model = fit(0, data)
        for k in range(1, cfg.dagger_rounds + 1):
            # the model drives, milp only labels. these are the states plain bc never sees
            new = collect(suite, driver=GNNPolicy(model), labeler=milp, limit=cfg.limit, jobs=jobs)
            data = merge(data, new)
            model = fit(k, data)

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg.train)}, out / "model.pt")
    data.save(out / "data.npz")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
