import numpy as np
import pytest

from baselines.common import feasible
from baselines.milp import milp
from eval.suites import load_suite
from learned.dataset import collect
from learned.policy import GNNPolicy
from learned.train import TrainConfig, split, train
from sim.engine import run


@pytest.fixture(scope="module")
def data():
    suite = load_suite("suite_train_v1")
    return suite, collect(suite, driver=milp, labeler=milp, limit=30)


def test_split_never_shares_a_scenario(data):
    _, b = data
    tr, va = split(b, 0.2, 0)
    assert len(va) > 0
    assert not set(b.scn[tr]) & set(b.scn[va])


def test_loss_goes_down(data):
    _, b = data
    _, hist = train(b, TrainConfig(hidden=16, rounds=1, epochs=8, val_frac=0.2), log=lambda _: 0)
    assert hist[-1]["val_loss"] < hist[0]["val_loss"]


def test_policy_only_takes_legal_shots(data):
    suite, b = data
    model, _ = train(b, TrainConfig(hidden=16, rounds=1, epochs=2), log=lambda _: 0)
    policy = GNNPolicy(model)

    def checked(obs):
        ok = feasible(obs)
        pairs = policy(obs)
        es = [e for e, _ in pairs]
        assert len(set(es)) == len(es) and all(ok[e, j] for e, j in pairs)
        return pairs

    for sc in suite.scenarios[:3]:
        run(sc, suite.cfg.sim, suite.cfg.types, checked)
    assert np.isfinite(model.head[0].weight.detach().numpy()).all()
