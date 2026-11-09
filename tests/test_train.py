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


def test_capped_decode_spreads_shooters(data):
    suite, b = data
    model, _ = train(b, TrainConfig(hidden=16, rounds=1, epochs=2), log=lambda _: 0)
    # threshold 0 so every effector wants to fire, floor 0.5 so one 0.7+ shooter covers a threat
    policy = GNNPolicy(model, threshold=0.0, decode="capped", survival_floor=0.5)

    def checked(obs):
        ok = feasible(obs)
        pairs = policy(obs)
        es = [e for e, _ in pairs]
        assert len(set(es)) == len(es) and all(ok[e, j] for e, j in pairs)
        # every kill chance in this suite is >= 0.7, so nobody should ever double up
        js = [j for _, j in pairs]
        assert len(set(js)) == len(js)
        return pairs

    for sc in suite.scenarios[:3]:
        run(sc, suite.cfg.sim, suite.cfg.types, checked)


def test_collect_all_keeps_scenarios_apart():
    from baselines.milp import milp
    from eval.suites import load_suite
    from scripts.train import SCN_STRIDE, collect_all

    suites = [load_suite("suite_train_v1"), load_suite("suite_small_v2")]
    b = collect_all(suites, milp, limit=1, jobs=1)
    # scenario 0 of each suite, kept distinct so a by-scenario split can't straddle them
    assert set(b.scn.tolist()) == {0, SCN_STRIDE}
