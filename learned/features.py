"""turn one decision tick into a bipartite graph: effector nodes, threat nodes, an edge for every
legal shot. the model scores edges, so it can only ever pick legal shots.

everything is described by its parameters (kill chance, range, speed, value...), never by a type id
or an index. that's what lets a model trained on 10-threat fights run on 500-threat ones, or on
effector types it never saw.
"""

from dataclasses import dataclass

import numpy as np

from baselines.common import feasible
from sim.engine import Obs

EFF_FEATURES = ["p_kill", "log_cost", "range_km", "speed_kms", "ammo", "reload_10s", "ready"]
THR_FEATURES = ["log_value", "dist_km", "time_to_impact_min", "speed_100ms", "engaged"]
EDGE_FEATURES = ["dist_over_range", "expected_value", "time_to_reach_10s"]


@dataclass
class Graph:
    eff_x: np.ndarray  # (m, len(EFF_FEATURES)) float32
    thr_x: np.ndarray  # (n, len(THR_FEATURES)) float32
    edges: np.ndarray  # (2, E) int64, row 0 = effector idx, row 1 = threat idx
    edge_x: np.ndarray  # (E, len(EDGE_FEATURES)) float32


def featurize(obs: Obs) -> Graph:
    t = [obs.types[k] for k in obs.eff_type]
    pk = np.array([x.p_kill for x in t])
    rng = np.array([x.range for x in t])
    spd = np.array([x.speed for x in t])
    eff_x = np.stack(
        [
            pk,
            np.log1p([x.cost for x in t]),
            rng / 1e3,
            spd / 1e3,
            obs.eff_ammo,
            np.array([x.reload for x in t]) / 10,
            obs.eff_ready.astype(float),
        ],
        1,
    )

    dist = np.linalg.norm(obs.threat_pos, axis=1)
    tspeed = np.maximum(np.linalg.norm(obs.threat_vel, axis=1), 1e-6)
    thr_x = np.stack(
        [
            np.log(obs.threat_value),
            dist / 1e3,
            # threats not launched yet have zero velocity, clip so they read as "far off" not inf
            np.minimum(dist / tspeed / 60, 10.0),
            tspeed / 100,
            obs.engaged,
        ],
        1,
    )

    e, j = np.nonzero(feasible(obs))
    d = np.linalg.norm(obs.eff_pos[e] - obs.threat_pos[j], axis=1)
    edge_x = np.stack([d / rng[e], obs.threat_value[j] * pk[e], d / spd[e] / 10], 1)
    return Graph(
        eff_x=eff_x.astype(np.float32),
        thr_x=thr_x.astype(np.float32),
        edges=np.stack([e, j]).astype(np.int64),
        edge_x=edge_x.astype(np.float32).reshape(-1, len(EDGE_FEATURES)),
    )
