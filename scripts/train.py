"""behavior cloning end to end: milp plays every training scenario and labels every tick, then the
gnn trains on it. writes model.pt + history.json into a logged run dir.

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
from learned.dataset import collect
from learned.train import TrainConfig, train


@dataclass
class RunConfig:
    suite: str = "suite_train_v1"
    limit: int | None = None
    train: TrainConfig = field(default_factory=TrainConfig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = RunConfig(limit=args.limit)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    out = start_run("train-bc", cfg, seed=cfg.train.seed)
    suite = load_suite(cfg.suite)
    data = collect(suite, driver=milp, labeler=milp, limit=cfg.limit, jobs=os.cpu_count() or 1)
    data.save(out / "data.npz")
    print(
        f"{len(data)} labeled ticks from {len(set(data.scn))} scenarios, "
        f"{len(data.y)} edges, {data.y.mean():.3f} positive"
    )

    with open(out / "train.log", "w") as f:

        def log(line):
            print(line)
            f.write(line + "\n")

        model, history = train(data, cfg.train, log=log)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg.train)}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
