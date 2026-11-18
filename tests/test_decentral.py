import numpy as np

from baselines.common import feasible, shot_value
from eval.suites import load_suite
from learned.decentral import restrict
from sim.engine import run


def some_obs(n=6):
    """a few real mid-episode observations w at least one legal shot."""
    suite = load_suite("suite_scale_n50_v1")
    out = []

    def grab(obs):
        if feasible(obs).any() and len(out) < n:
            out.append(obs)
        return []

    run(suite.scenarios[0], suite.cfg.sim, suite.cfg.types, grab)
    return out


def test_restrict_keeps_threats_and_picks_rows():
    for obs in some_obs():
        m = len(obs.eff_pos)
        effs = np.array([0, 2, m - 1])
        sub = restrict(obs, effs)
        assert len(sub.eff_pos) == 3 and np.array_equal(sub.threat_pos, obs.threat_pos)
        # legality and shot values are per effector, so they're just the matching rows
        assert np.array_equal(feasible(sub), feasible(obs)[effs])
        assert np.array_equal(shot_value(sub), shot_value(obs)[effs])
