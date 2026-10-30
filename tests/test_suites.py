import dataclasses

import pytest

from eval.suites import SUITE_DIR, build, load_suite, suite_hash


@pytest.mark.parametrize("suite_id", sorted(p.stem for p in SUITE_DIR.glob("*.yaml")))
def test_suites_match_lock(suite_id):
    suite = load_suite(suite_id)  # raises if the hash drifted
    assert len(suite.scenarios) == suite.cfg.n_scenarios


def test_any_config_change_changes_hash():
    suite = load_suite("suite_small_v1")
    cfg = dataclasses.replace(suite.cfg, seed=suite.cfg.seed + 1)
    assert suite_hash(build(cfg)) != suite_hash(suite)
    sim = dataclasses.replace(suite.cfg.sim, kill_radius=11.0)
    assert suite_hash(build(dataclasses.replace(suite.cfg, sim=sim))) != suite_hash(suite)


def test_missing_suite():
    with pytest.raises(FileNotFoundError):
        load_suite("nope_v9")
