"""config + scenario containers. the numbers live in yaml under configs/, not here."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EffectorType:
    """a kind of defender. all numbers are made up / generic on purpose."""

    name: str
    range: float  # max distance from the site to the threat at launch (m)
    speed: float  # interceptor speed (m/s), constant
    p_kill: float  # chance an interceptor that gets close actually kills it
    cost: float  # cost per shot, arbitrary units
    magazine: int  # shots per site
    reload: float  # seconds between shots from one site
    max_flight: float  # interceptor gives up after this long (s)


@dataclass
class SimConfig:
    dt: float = 0.1
    t_max: float = 400.0
    asset_radius: float = 50.0  # threat this close to the asset (at origin) = leaked
    kill_radius: float = 10.0  # interceptor this close to its target = roll p_kill
    nav_gain: float = 3.0
    max_accel: float = 300.0
    replan_every: float = 1.0  # how often (s) the assignment policy gets called


@dataclass
class Scenario:
    """one engagement. arrays only, so it's cheap to hash and save."""

    threat_start: np.ndarray  # (n, 2) spawn point
    threat_launch: np.ndarray  # (n,) time it starts flying
    threat_speed: np.ndarray  # (n,)
    threat_value: np.ndarray  # (n,) how bad it is if this one gets through
    eff_pos: np.ndarray  # (m, 2)
    eff_type: np.ndarray  # (m,) index into the suite's type list
    seed: int  # drives the p_kill rolls

    @property
    def n_threats(self) -> int:
        return len(self.threat_speed)

    @property
    def n_effectors(self) -> int:
        return len(self.eff_type)
