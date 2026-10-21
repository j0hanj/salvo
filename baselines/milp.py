"""exact static WTA for one snapshot, as a MILP. math in docs/FORMULATION.md.

unlike greedy/hungarian this can stack several shooters on one threat, and it scores that w the
real survival product, not a linear stand-in.
"""

import itertools

import numpy as np
from ortools.linear_solver import pywraplp

from baselines.common import cost, feasible
from baselines.hungarian import hungarian
from sim.engine import Obs


def objective(obs: Obs, pairs, cost_weight: float = 0.0) -> float:
    """true static-WTA objective of a set of shots. used by tests + to sanity check solves."""
    p = np.array([obs.types[k].p_kill for k in obs.eff_type])
    survive = np.ones(len(obs.threat_value))
    spent = 0.0
    c = cost(obs)
    for e, j in pairs:
        survive[j] *= 1 - p[e]
        spent += c[e]
    return float((obs.threat_value * (1 - survive)).sum() - cost_weight * spent)


def milp(obs: Obs, cost_weight: float = 0.0, time_limit: float = 5.0) -> list[tuple[int, int]]:
    ok = feasible(obs)
    if not ok.any():
        return []
    types = obs.eff_type
    pk = np.array([t.p_kill for t in obs.types])
    c = cost(obs)

    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit * 1000))
    x = {(e, j): s.BoolVar(f"x_{e}_{j}") for e, j in zip(*np.nonzero(ok), strict=True)}

    # each effector fires at most once this tick
    for e in np.flatnonzero(ok.any(1)):
        s.Add(sum(x[e, j] for j in np.flatnonzero(ok[e])) <= 1)

    obj = []
    for j in np.flatnonzero(ok.any(0)):
        shooters = np.flatnonzero(ok[:, j])
        ks = sorted(set(types[shooters]))
        caps = [int((types[shooters] == k).sum()) for k in ks]
        # one binary per "exactly this many shots of each type" outcome
        z = {}
        for a in itertools.product(*(range(cap + 1) for cap in caps)):
            z[a] = s.BoolVar(f"z_{j}_{'_'.join(map(str, a))}")
            kill = 1 - np.prod([(1 - pk[k]) ** n for k, n in zip(ks, a, strict=True)])
            obj.append(obs.threat_value[j] * kill * z[a])
        s.Add(sum(z.values()) == 1)
        for i, k in enumerate(ks):
            s.Add(
                sum(x[e, j] for e in shooters if types[e] == k)
                == sum(a[i] * v for a, v in z.items())
            )

    obj.append(-cost_weight * sum(c[e] * v for (e, _), v in x.items()))
    s.Maximize(sum(obj))
    status = s.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return hungarian(obs, cost_weight)  # solver gave up, fall back to something sane
    return [(int(e), int(j)) for (e, j), v in x.items() if v.solution_value() > 0.5]
