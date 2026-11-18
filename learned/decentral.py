"""the gnn, decentralized. no central node: every effector builds its own graph and decides for
itself, firing only its own pick.

what a node sees on a tick: every threat (shared sensor picture, same as the centralized case),
itself, and every effector whose state reached it over this tick's talking rounds
(`sim.comms.flood`). it runs the model on that graph, decodes w capped decode exactly like the
centralized policy, and keeps the one shot it assigned to itself.

if every node heard everyone, every node builds the same graph and makes the same decode, so the
union of their own picks is exactly the centralized policy's output. a node that heard nobody
decodes a graph w just itself in it, i.e. takes its own best shot.
"""

import dataclasses

import numpy as np

from baselines.common import feasible
from sim.comms import flood
from sim.engine import Obs

EFF_FIELDS = ("eff_pos", "eff_type", "eff_ammo", "eff_ready")


def restrict(obs: Obs, effs: np.ndarray) -> Obs:
    """obs w only the effectors in `effs` (sorted indices), every threat kept. index k in the
    result is effector effs[k] in the original."""
    return dataclasses.replace(obs, **{f: getattr(obs, f)[effs] for f in EFF_FIELDS})


class DecentralGNN:
    """`policy` is a centralized GNNPolicy (its decode settings are what every node uses),
    `channel` a sim.comms.Channel. call it like any policy: it returns the union of every node's
    own pick, which is what the sim launches."""

    def __init__(self, policy, channel):
        self.policy, self.channel = policy, channel

    def reset(self, seed: int) -> None:
        self.channel.reset(seed)

    def __call__(self, obs: Obs) -> list[tuple[int, int]]:
        m = len(obs.eff_pos)
        # comms step every tick, decision or not, so link state lines up w cbba on the same seed
        knows = flood(self.channel.tick(m))
        active = np.flatnonzero(feasible(obs).any(1))
        out = []
        # nodes that heard exactly the same set build the same graph: decode it once
        groups: dict[bytes, list[int]] = {}
        for i in active:
            groups.setdefault(knows[i].tobytes(), []).append(int(i))
        for key, members in groups.items():
            effs = np.flatnonzero(np.frombuffer(key, dtype=bool))
            for k, j in self.policy(restrict(obs, effs)):
                if int(effs[k]) in members:
                    out.append((int(effs[k]), int(j)))
        return out
