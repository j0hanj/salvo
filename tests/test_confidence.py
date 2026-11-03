import torch

from eval.suites import load_suite
from learned.model import EdgeScorer
from learned.policy import GNNPolicy
from scripts.confidence import Probe
from sim.engine import run


def test_probe_idle_only_when_threshold_blocks():
    torch.manual_seed(0)
    model = EdgeScorer(hidden=8, rounds=1)
    suite = load_suite("suite_small_v2")
    sc = suite.scenarios[0]

    # threshold 0: capped decode takes every uncovered legal edge, so nobody can be idle w one
    p0 = Probe(GNNPolicy(model, threshold=0.0, decode="capped"))
    run(sc, suite.cfg.sim, suite.cfg.types, p0)
    assert p0.best and not any(p0.idle_open)

    # threshold above any sigmoid: nobody fires, and everyone w a legal shot is idle w a target
    p1 = Probe(GNNPolicy(model, threshold=1.1, decode="capped"))
    run(sc, suite.cfg.sim, suite.cfg.types, p1)
    assert not any(p1.fired) and all(p1.idle_open)
