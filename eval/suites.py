"""fixed eval suites. a suite = a yaml config that always expands into the same list of scenarios.

each suite has a committed .lock file holding a hash of everything in it. if the code or config ever
changes what a suite expands to, the hash check fails, so old results never silently stop matching.
want different scenarios? bump the version, don't edit v1.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from configs.load import load
from sim.scenario import GenConfig, generate
from sim.types import EffectorType, Scenario, SimConfig

SUITE_DIR = Path(__file__).resolve().parent.parent / "configs" / "suites"


@dataclass
class SuiteConfig:
    name: str
    version: int
    seed: int
    n_scenarios: int
    types: list[EffectorType]
    sim: SimConfig = field(default_factory=SimConfig)
    gen: GenConfig = field(default_factory=GenConfig)


@dataclass
class Suite:
    cfg: SuiteConfig
    scenarios: list[Scenario]

    @property
    def id(self) -> str:
        return f"{self.cfg.name}_v{self.cfg.version}"


def build(cfg: SuiteConfig) -> Suite:
    names = [t.name for t in cfg.types]
    children = np.random.SeedSequence(cfg.seed).spawn(cfg.n_scenarios)
    return Suite(cfg, [generate(cfg.gen, names, c) for c in children])


def suite_hash(suite: Suite) -> str:
    h = hashlib.sha256()
    h.update(yaml.safe_dump(dataclasses.asdict(suite.cfg), sort_keys=True).encode())
    for sc in suite.scenarios:
        for f in dataclasses.fields(sc):
            # fixed dtypes so the hash doesn't depend on platform default int size. floats get
            # rounded first: sin/cos/exp can differ in the last bit between mac and linux math
            # libs, which is noise, not a different suite
            v = np.asarray(getattr(sc, f.name))
            if v.dtype.kind == "f":
                v = np.round(v.astype(np.float64), 6) + 0.0  # + 0.0 turns -0.0 into 0.0
            h.update(v.astype(np.float64 if v.dtype.kind == "f" else np.int64).tobytes())
    return h.hexdigest()


def load_suite(suite_id: str, check=True) -> Suite:
    suite = build(load(SuiteConfig, SUITE_DIR / f"{suite_id}.yaml"))
    if check:
        lock = json.loads((SUITE_DIR / f"{suite_id}.lock").read_text())
        got = suite_hash(suite)
        if got != lock["hash"]:
            raise RuntimeError(
                f"{suite_id} doesn't match its lock ({got[:12]} vs {lock['hash'][:12]})"
            )
    return suite
