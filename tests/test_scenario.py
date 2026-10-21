import numpy as np
import pytest

from sim.scenario import GenConfig, generate

NAMES = ["short", "long"]


def same(a, b):
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in vars(a))


def test_same_seed_same_scenario():
    assert same(generate(GenConfig(), NAMES, 42), generate(GenConfig(), NAMES, 42))


def test_different_seed_different_scenario():
    assert not same(generate(GenConfig(), NAMES, 1), generate(GenConfig(), NAMES, 2))


def test_knobs_respected():
    cfg = GenConfig(n_threats=[8, 8], n_effectors=[4, 4], sector_deg=60, spawn_radius=[5000, 5000])
    sc = generate(cfg, NAMES, 0)
    assert sc.n_threats == 8 and sc.n_effectors == 4
    assert np.allclose(np.linalg.norm(sc.threat_start, axis=1), 5000)
    bearing = np.degrees(np.arctan2(sc.threat_start[:, 1], sc.threat_start[:, 0]))
    assert np.all(np.abs(bearing - 90) <= 30 + 1e-9)


def test_waves_are_bunched():
    def busy_bins(launch):
        # 5s bins w at least 5 launches in them
        counts, _ = np.histogram(launch, bins=np.arange(0, 75, 5))
        return (counts >= 5).sum()

    spread = generate(GenConfig(n_threats=[200, 200]), NAMES, 0)
    waves = generate(GenConfig(n_threats=[200, 200], arrival="waves", n_waves=2), NAMES, 0)
    assert busy_bins(spread.threat_launch) >= 11
    assert busy_bins(waves.threat_launch) <= 8


def test_type_mix():
    sc = generate(GenConfig(n_effectors=[50, 50], type_mix={"long": 1.0}), NAMES, 0)
    assert np.all(sc.eff_type == NAMES.index("long"))


def test_bad_knob_fails_loud():
    with pytest.raises(ValueError):
        generate(GenConfig(arrival="nope"), NAMES, 0)


def test_impact_waves_arrive_together():
    cfg = GenConfig(n_threats=[100, 100], arrival="impact_waves", n_waves=1, wave_jitter=2.0)
    sc = generate(cfg, NAMES, 0)
    arrive = sc.threat_launch + np.linalg.norm(sc.threat_start, axis=1) / sc.threat_speed
    assert arrive.std() < 3.0
    assert sc.threat_launch.min() >= 0
    # same knobs w launch-time waves: arrivals smear out a lot more
    lw = generate(GenConfig(n_threats=[100, 100], arrival="waves", n_waves=1), NAMES, 0)
    arrive_lw = lw.threat_launch + np.linalg.norm(lw.threat_start, axis=1) / lw.threat_speed
    assert arrive_lw.std() > 5 * arrive.std()
