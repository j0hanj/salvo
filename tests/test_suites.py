import dataclasses

import pytest

from eval.suites import build, load_suite, suite_hash


def test_suite_small_matches_lock():
    suite = load_suite("suite_small_v1")  # raises if the hash drifted
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
