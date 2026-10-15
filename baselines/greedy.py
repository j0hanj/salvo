"""greedy baselines. dumb on purpose: they're the floor everything else has to beat."""

import numpy as np

from sim.engine import FLYING, Obs


def nearest(obs: Obs) -> list[tuple[int, int]]:
    """every ready effector shoots the threat closest to the asset that's in its range and nobody's
    already shooting at. one interceptor per threat at a time."""
    open_ = (obs.threat_status == FLYING) & (obs.engaged == 0)
    rng = np.array([obs.types[k].range for k in obs.eff_type])
    to_asset = np.linalg.norm(obs.threat_pos, axis=1)
    pairs = []
    for e in np.flatnonzero(obs.eff_ready):
        d = np.linalg.norm(obs.threat_pos - obs.eff_pos[e], axis=1)
        ok = open_ & (d <= rng[e])
        if ok.any():
            j = int(np.flatnonzero(ok)[np.argmin(to_asset[ok])])
            pairs.append((int(e), j))
            open_[j] = False
    return pairs
