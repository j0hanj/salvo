import json
from dataclasses import dataclass

from configs.load import load
from eval.runlog import start_run


@dataclass
class Cfg:
    n: int = 4


def test_run_dir_has_config_and_meta(tmp_path):
    out = start_run("t", Cfg(7), seed=123, root=tmp_path)
    assert load(Cfg, out / "config.yaml") == Cfg(7)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["seed"] == 123
    assert len(meta["git_commit"]) == 40
