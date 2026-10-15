import numpy as np

from baselines.greedy import nearest
from sim.engine import KILLED, LEAKED, Engagement, run
from sim.types import EffectorType, Scenario, SimConfig

CFG = SimConfig()


def kind(p_kill=1.0, magazine=4):
    return EffectorType(
        "k",
        range=3000,
        speed=300,
        p_kill=p_kill,
        cost=2.0,
        magazine=magazine,
        reload=1.0,
        max_flight=30,
    )


def one_on_one(seed=0):
    return Scenario(
        threat_start=np.array([[0.0, 4000]]),
        threat_launch=np.zeros(1),
        threat_speed=np.array([50.0]),
        threat_value=np.ones(1),
        eff_pos=np.array([[300.0, 0]]),
        eff_type=np.zeros(1, dtype=int),
        seed=seed,
    )


def hold_fire(obs):
    return []


def test_no_shots_means_leak():
    r = run(one_on_one(), CFG, [kind()], hold_fire)
    assert r.status[0] == LEAKED
    assert r.cost == 0


def test_sure_kill():
    r = run(one_on_one(), CFG, [kind(p_kill=1.0)], nearest)
    assert r.status[0] == KILLED
    assert r.shots[0] == 1
    assert r.cost == 2.0


def test_zero_pk_uses_whole_magazine_then_leaks():
    r = run(one_on_one(), CFG, [kind(p_kill=0.0, magazine=3)], nearest)
    assert r.status[0] == LEAKED
    assert r.shots[0] == 3


def test_threat_moves_at_its_speed():
    eng = Engagement(one_on_one(), CFG, [kind()])
    eng.step()  # goes pending -> flying and moves on the same tick
    for _ in range(99):
        eng.step()
    assert np.isclose(eng.t_pos[0, 1], 4000 - 50 * 10.0)


def test_out_of_range_shot_is_ignored():
    eng = Engagement(one_on_one(), CFG, [kind()])
    eng.step()
    assert eng.launch([(0, 0)]) == 0  # threat is ~4km out, range is 3km
    assert eng.ammo[0] == 4


def test_same_seed_same_result():
    types = [kind(p_kill=0.5, magazine=8)]
    a = run(one_on_one(seed=7), CFG, types, nearest, record=True)
    b = run(one_on_one(seed=7), CFG, types, nearest, record=True)
    assert np.array_equal(a.status, b.status)
    assert np.array_equal(a.shots, b.shots)
    assert np.array_equal(a.history["intercept"], b.history["intercept"], equal_nan=True)
