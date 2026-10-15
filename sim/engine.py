"""the engagement loop. threats fly at the asset, policy picks shots, interceptors chase w pro-nav.

order inside one tick: move threats, move interceptors, check intercepts, then check leaks. so if a
threat gets hit and reaches the asset in the same tick, the hit wins (it happened on the way in).
"""

from dataclasses import dataclass, field

import numpy as np

from sim.kinematics import closest_approach, pro_nav_turn_rate
from sim.types import EffectorType, Scenario, SimConfig

# threat status
PENDING, FLYING, KILLED, LEAKED = 0, 1, 2, 3


@dataclass
class Obs:
    """what a centralized policy gets to see each replan. read-only views, don't mutate."""

    t: float
    threat_pos: np.ndarray
    threat_vel: np.ndarray
    threat_status: np.ndarray
    threat_value: np.ndarray
    engaged: np.ndarray  # (n,) interceptors currently in the air on each threat
    eff_pos: np.ndarray
    eff_type: np.ndarray
    eff_ammo: np.ndarray
    eff_ready: np.ndarray  # (m,) bool, can fire right now
    types: list[EffectorType]


@dataclass
class Result:
    status: np.ndarray  # final threat status
    kill_time: np.ndarray  # nan if never killed
    shots: np.ndarray  # (m,) shots fired per effector
    cost: float
    t_end: float
    history: dict = field(default_factory=dict)  # only filled when record=True


class Engagement:
    def __init__(self, sc: Scenario, cfg: SimConfig, types: list[EffectorType], record=False):
        self.sc, self.cfg, self.types = sc, cfg, types
        self.rng = np.random.default_rng(sc.seed)
        n, m = sc.n_threats, sc.n_effectors
        self.t = 0.0

        self.t_pos = sc.threat_start.astype(float).copy()
        heading = -sc.threat_start / np.linalg.norm(sc.threat_start, axis=1, keepdims=True)
        self.t_vel = heading * sc.threat_speed[:, None]  # straight at the asset, constant speed
        self.status = np.full(n, PENDING)
        self.kill_time = np.full(n, np.nan)

        # per-effector stats pulled out of the type table so the hot loop is just array indexing
        col = lambda k: np.array([getattr(types[i], k) for i in sc.eff_type], dtype=float)  # noqa: E731
        self.e_range, self.e_speed, self.e_pk = col("range"), col("speed"), col("p_kill")
        self.e_cost, self.e_reload, self.e_flight = col("cost"), col("reload"), col("max_flight")
        self.ammo = col("magazine").astype(int)
        self.next_ready = np.zeros(m)
        self.shots = np.zeros(m, dtype=int)

        # interceptors: preallocate one slot per round in every magazine, never more can exist
        k = int(self.ammo.sum())
        self.i_pos = np.zeros((k, 2))
        self.i_head = np.zeros(k)
        self.i_speed = np.zeros(k)
        self.i_target = np.zeros(k, dtype=int)
        self.i_shooter = np.zeros(k, dtype=int)
        self.i_launch = np.zeros(k)
        self.i_active = np.zeros(k, dtype=bool)
        self.n_launched = 0

        self.record = record
        self.hist = {"t": [], "threat": [], "intercept": [], "kills": []} if record else {}

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.t_max or not np.isin(self.status, (PENDING, FLYING)).any()

    def observe(self) -> Obs:
        engaged = np.bincount(self.i_target[self.i_active], minlength=self.sc.n_threats)
        return Obs(
            t=self.t,
            threat_pos=self.t_pos,
            threat_vel=self.t_vel,
            threat_status=self.status,
            threat_value=self.sc.threat_value,
            engaged=engaged,
            eff_pos=self.sc.eff_pos,
            eff_type=self.sc.eff_type,
            eff_ammo=self.ammo,
            eff_ready=(self.ammo > 0) & (self.next_ready <= self.t),
            types=self.types,
        )

    def launch(self, pairs) -> int:
        """fire one interceptor per (effector, threat) pair. illegal shots get skipped, no crash,
        so a sloppy policy just wastes its turn. returns how many actually launched."""
        fired = 0
        for e, j in pairs:
            dist = np.linalg.norm(self.t_pos[j] - self.sc.eff_pos[e])
            ok = (
                self.status[j] == FLYING
                and self.ammo[e] > 0
                and self.next_ready[e] <= self.t
                and dist <= self.e_range[e]
            )
            if not ok:
                continue
            s = self.n_launched
            self.n_launched += 1
            d = self.t_pos[j] - self.sc.eff_pos[e]
            self.i_pos[s] = self.sc.eff_pos[e]
            self.i_head[s] = np.arctan2(
                d[1], d[0]
            )  # point at the threat, pro-nav sorts out the rest
            self.i_speed[s] = self.e_speed[e]
            self.i_target[s], self.i_shooter[s] = j, e
            self.i_launch[s] = self.t
            self.i_active[s] = True
            self.ammo[e] -= 1
            self.next_ready[e] = self.t + self.e_reload[e]
            self.shots[e] += 1
            fired += 1
        return fired

    def step(self) -> None:
        cfg, dt = self.cfg, self.cfg.dt
        self.status[(self.status == PENDING) & (self.sc.threat_launch <= self.t)] = FLYING
        flying = self.status == FLYING

        t_prev = self.t_pos.copy()
        self.t_pos[flying] += self.t_vel[flying] * dt

        # interceptors whose target is already gone just self-destruct, no retargeting
        a = np.flatnonzero(self.i_active)
        lost = self.status[self.i_target[a]] != FLYING
        timed_out = self.t - self.i_launch[a] >= self.e_flight[self.i_shooter[a]]
        self.i_active[a[lost | timed_out]] = False
        a = np.flatnonzero(self.i_active)

        if len(a):
            tgt = self.i_target[a]
            speed = self.i_speed[a]
            vel = speed[:, None] * np.stack([np.cos(self.i_head[a]), np.sin(self.i_head[a])], 1)
            w = pro_nav_turn_rate(
                self.i_pos[a], vel, t_prev[tgt], self.t_vel[tgt], cfg.nav_gain, cfg.max_accel, speed
            )
            i_prev = self.i_pos[a].copy()
            self.i_pos[a] += vel * dt
            self.i_head[a] += w * dt

            # intercept check. go in slot order so the p_kill rolls come out the same every run
            near = closest_approach(i_prev, self.i_pos[a], t_prev[tgt], self.t_pos[tgt])
            for s, j, dmin in zip(a, tgt, near, strict=True):
                if dmin > cfg.kill_radius or self.status[j] != FLYING:
                    continue
                self.i_active[s] = False  # one shot, hit or miss it's used up
                if self.rng.random() < self.e_pk[self.i_shooter[s]]:
                    self.status[j] = KILLED
                    self.kill_time[j] = self.t + dt
                    if self.record:
                        self.hist["kills"].append(self.t_pos[j].copy())

        leaked = (self.status == FLYING) & (np.linalg.norm(self.t_pos, axis=1) <= cfg.asset_radius)
        self.status[leaked] = LEAKED
        self.t += dt

        if self.record:
            tp = np.where((self.status == PENDING)[:, None], np.nan, self.t_pos)
            ip = np.where(self.i_active[:, None], self.i_pos, np.nan)
            self.hist["t"].append(self.t)
            self.hist["threat"].append(tp.copy())
            self.hist["intercept"].append(ip.copy())

    def result(self) -> Result:
        hist = {k: np.array(v) for k, v in self.hist.items()} if self.record else {}
        return Result(
            status=self.status.copy(),
            kill_time=self.kill_time.copy(),
            shots=self.shots.copy(),
            cost=float((self.shots * self.e_cost).sum()),
            t_end=self.t,
            history=hist,
        )


def run(sc: Scenario, cfg: SimConfig, types: list[EffectorType], policy, record=False) -> Result:
    """policy(obs) -> list of (effector_idx, threat_idx). called every cfg.replan_every seconds."""
    eng = Engagement(sc, cfg, types, record=record)
    every = max(1, round(cfg.replan_every / cfg.dt))
    tick = 0
    while not eng.done:
        if tick % every == 0:
            eng.launch(policy(eng.observe()))
        eng.step()
        tick += 1
    return eng.result()
