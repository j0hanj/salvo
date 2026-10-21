# the assignment problem

every `replan_every` seconds the policy gets a snapshot and decides who shoots at what right now.
this doc is the math for that one snapshot.

## setup

- ready effectors $e = 1..m$ (have ammo, not reloading). each has kill chance $p_e$ and shot cost $c_e$
- open threats $j = 1..n$ (in the air, nobody already shooting at them). each has value $V_j$
- $x_{ej} \in \{0, 1\}$: effector $e$ fires at threat $j$ this tick. only allowed if $j$ is in $e$'s range
- each effector fires at most once per tick: $\sum_j x_{ej} \le 1$

## the real objective (static WTA)

shots are independent, so threat $j$ survives everything fired at it w probability
$\prod_e (1 - p_e)^{x_{ej}}$. maximize expected value destroyed minus what you spent:

$$\max_x \; \sum_j V_j \Big(1 - \prod_e (1 - p_e)^{x_{ej}}\Big) \;-\; \lambda \sum_{e,j} c_e x_{ej}$$

$\lambda$ (`cost_weight`) is the exchange rate between threat value and shot cost. $\lambda = 0$ means
cost doesn't matter, only stopping stuff.

the product makes this nonlinear, and static WTA is NP-hard in general. two ways the baselines deal
with it:

## restriction: one shooter per threat (greedy-value, hungarian)

add $\sum_e x_{ej} \le 1$. now each threat gets at most one shot, the product collapses to a single
term, and the objective is linear: $\sum_{e,j} (V_j p_e - \lambda c_e)\, x_{ej}$. that's a plain
assignment problem. hungarian solves it exactly, greedy-value approximates it.

the cost: you can never double up on a high-value threat, even when one shot isn't enough.

## exact: count-based MILP (`baselines/milp.py`)

the trick: survival only depends on *how many shots of each type* hit a threat, not which specific
effectors fired them. w effector types $k = 1..K$ and $a_k$ shots of type $k$:

$$\text{survive}_j(a) = \prod_k (1 - p_k)^{a_k}$$

that's just a number for each count vector $a$, so precompute it. add a binary $z_{ja}$ = "threat $j$
gets exactly count vector $a$", w

$$\sum_a z_{ja} = 1, \qquad \sum_{e \in k} x_{ej} = \sum_a a_k\, z_{ja} \;\; \forall j, k$$

and the objective becomes linear:

$$\max \; \sum_{j,a} V_j \big(1 - \text{survive}(a)\big)\, z_{ja} \;-\; \lambda \sum_{e,j} c_e x_{ej}$$

this is exact, not an approximation. the catch is size: each threat gets $\prod_k (m_k + 1)$ count
vectors, where $m_k$ is the number of ready effectors of type $k$ that can reach it. fine at
`suite_small` scale, blows up at hundreds of effectors. that's the point: it's ground truth at small
N, and the learned policy has to get close to it at scales where it can't run.

## what none of these do

all of these are **myopic in time**. they solve the snapshot perfectly (or approximately) but don't
know a second wave is coming. a snapshot-optimal plan can dump the whole magazine on wave one and
have nothing left for wave two. "MILP is optimal" means optimal per snapshot, not over the whole
engagement. a planner that reasons over the future (dynamic WTA) is a different, harder problem, and
it's where a learned policy could in principle beat the snapshot MILP.
