"""random scenario generator. same seed in, same scenario out, always."""

from dataclasses import dataclass, field

import numpy as np

from sim.types import Scenario


@dataclass
class GenConfig:
    n_threats: list[int] = field(default_factory=lambda: [5, 10])  # [lo, hi] inclusive
    n_effectors: list[int] = field(default_factory=lambda: [3, 5])
    type_mix: dict[str, float] = field(default_factory=lambda: {"short": 0.7, "long": 0.3})
    # "spread": launches uniform over the window.
    # "waves": launch times bunched into n_waves groups. threats start at different ranges and
    #   speeds though, so they still arrive spread out. kept so suite_small_v1 expands the same.
    # "impact_waves": *arrival* times at the asset bunched into n_waves groups, launch is
    #   backed out from range / speed. this is what actually saturates the defense.
    arrival: str = "spread"
    window: float = 60.0  # s
    n_waves: int = 3
    wave_jitter: float = 3.0  # s, std dev inside a wave
    # "uniform": every threat worth 1. "lognormal": a few are worth way more than the rest
    value: str = "uniform"
    value_sigma: float = 1.0
    spawn_radius: list[float] = field(default_factory=lambda: [4000.0, 6000.0])
    sector_deg: float = 120.0  # threats come from inside this arc, centered on +y
    threat_speed: list[float] = field(default_factory=lambda: [30.0, 60.0])
    defense_radius: list[float] = field(default_factory=lambda: [200.0, 600.0])


def generate(cfg: GenConfig, type_names: list[str], seed) -> Scenario:
    """seed can be an int or a SeedSequence (suites pass spawned children)."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    geo_ss, sim_ss = ss.spawn(2)  # separate streams so scenario layout and kill rolls don't couple
    rng = np.random.default_rng(geo_ss)

    n = int(rng.integers(cfg.n_threats[0], cfg.n_threats[1] + 1))
    m = int(rng.integers(cfg.n_effectors[0], cfg.n_effectors[1] + 1))

    half = np.deg2rad(cfg.sector_deg) / 2
    bearing = np.pi / 2 + rng.uniform(-half, half, n)
    r = rng.uniform(*cfg.spawn_radius, n)
    start = r[:, None] * np.stack([np.cos(bearing), np.sin(bearing)], 1)

    if cfg.arrival == "spread":
        launch = rng.uniform(0, cfg.window, n)
    elif cfg.arrival == "impact_waves":
        centers = rng.uniform(0, cfg.window, cfg.n_waves)
        impact = centers[rng.integers(0, cfg.n_waves, n)] + rng.normal(0, cfg.wave_jitter, n)
        launch = None  # needs speed, which gets drawn below
    elif cfg.arrival == "waves":
        centers = rng.uniform(0, cfg.window, cfg.n_waves)
        launch = centers[rng.integers(0, cfg.n_waves, n)] + rng.normal(0, cfg.wave_jitter, n)
        launch = np.clip(launch, 0, None)
    else:
        raise ValueError(f"unknown arrival pattern {cfg.arrival!r}")

    if cfg.value == "uniform":
        value = np.ones(n)
    elif cfg.value == "lognormal":
        value = rng.lognormal(0, cfg.value_sigma, n)
    else:
        raise ValueError(f"unknown value dist {cfg.value!r}")

    names = list(cfg.type_mix)
    p = np.array([cfg.type_mix[k] for k in names], dtype=float)
    eff_type = np.array(
        [type_names.index(names[i]) for i in rng.choice(len(names), m, p=p / p.sum())]
    )

    # sites sit in a ring around the asset, evenly spaced w some jitter
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False) + rng.uniform(0, 2 * np.pi)
    ang += rng.normal(0, 0.2, m)
    rad = rng.uniform(*cfg.defense_radius, m)
    eff_pos = rad[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)

    speed = rng.uniform(*cfg.threat_speed, n)
    if launch is None:
        launch = impact - r / speed  # time to fly r meters at that speed
        launch -= min(launch.min(), 0.0)  # shift the whole raid so nobody launches before t=0

    return Scenario(
        threat_start=start,
        threat_launch=launch,
        threat_speed=speed,
        threat_value=value,
        eff_pos=eff_pos,
        eff_type=eff_type,
        seed=int(sim_ss.generate_state(1)[0]),
    )
