"""shared bits for snapshot policies."""

import numpy as np

from sim.engine import FLYING, Obs


def feasible(obs: Obs) -> np.ndarray:
    """(m, n) bool: effector e can legally shoot threat j right now.

    ready (ammo + not reloading), threat in the air, nobody already on it, and in range.
    """
    open_ = (obs.threat_status == FLYING) & (obs.engaged == 0)
    rng = np.array([obs.types[k].range for k in obs.eff_type])
    d = np.linalg.norm(obs.eff_pos[:, None] - obs.threat_pos[None], axis=-1)
    return obs.eff_ready[:, None] & open_[None] & (d <= rng[:, None])


def p_kill(obs: Obs) -> np.ndarray:
    """(m,) kill chance per effector."""
    return np.array([obs.types[k].p_kill for k in obs.eff_type])


def cost(obs: Obs) -> np.ndarray:
    """(m,) cost per shot per effector."""
    return np.array([obs.types[k].cost for k in obs.eff_type], dtype=float)
