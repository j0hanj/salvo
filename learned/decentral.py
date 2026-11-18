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

from sim.engine import Obs

EFF_FIELDS = ("eff_pos", "eff_type", "eff_ammo", "eff_ready")


def restrict(obs: Obs, effs: np.ndarray) -> Obs:
    """obs w only the effectors in `effs` (sorted indices), every threat kept. index k in the
    result is effector effs[k] in the original."""
    return dataclasses.replace(obs, **{f: getattr(obs, f)[effs] for f in EFF_FIELDS})
