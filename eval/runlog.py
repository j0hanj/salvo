"""every run gets its own folder w the exact config, seed and git hash. any number traces back."""

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from configs.load import dump

RESULTS = Path(__file__).resolve().parent.parent / "results"


def git_state() -> tuple[str, bool]:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return run("rev-parse", "HEAD").strip(), bool(run("status", "--porcelain").strip())


def start_run(name: str, cfg, seed: int, root: Path = RESULTS) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = root / f"{stamp}-{name}"
    out.mkdir(parents=True)
    commit, dirty = git_state()
    dump(cfg, out / "config.yaml")
    meta = {
        "name": name,
        "seed": seed,
        "git_commit": commit,
        # dirty = uncommitted changes, so the hash alone doesn't fully reproduce this run
        "git_dirty": dirty,
        "time": stamp,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
