# how the sim works

2D, meters + seconds, fixed timestep (`dt` = 0.1s). the thing being defended sits at the origin.

## pieces

- **threats** spawn somewhere on an arc 4-6km out, wait until their launch time, then fly in a straight
  line at constant speed straight at the asset. no dodging (yet). each one has a value = how bad it is
  if it gets through.
- **effector sites** sit in a ring a few hundred meters around the asset. each has a type (`short` or
  `long`) that sets range, interceptor speed, kill chance, cost per shot, magazine size, reload time.
  all numbers are made up and live in the suite yaml, not the code.
- **interceptors** launch pointed at the threat, then steer w proportional navigation (below). one
  interceptor = one shot. hit or miss, it's used up.
- **policy** = the thing we're actually studying. every `replan_every` seconds (1s) it sees the full
  state and returns a list of (site, threat) shots to take. illegal shots (out of range, no ammo,
  reloading, threat not in the air) just get skipped.

## one tick

1. threats whose launch time came up start flying
2. threats move
3. interceptors whose target is already dead, or that ran out of flight time, self-destruct
4. interceptors steer + move
5. intercept check: if an interceptor got within `kill_radius` (10m) of its target at any point
   during the tick, roll `p_kill`. hit = threat dead.
6. leak check: any threat still flying within `asset_radius` (50m) of the origin = leaked

hit is checked before leak, so if both happen in one tick the hit counts (it happened on the way in).

## what counts as "stopped"

a threat is **stopped** if an interceptor killed it before it reached the asset. **leaked** if it
reached the asset. if the run hits `t_max` w threats still flying they're neither, and they count as
not stopped. (w current suites `t_max` is long enough that this shouldn't happen, phase 2's harness
will flag it if it does.)

## pro-nav in one paragraph

if you and a target are on a collision course, the line of sight between you doesn't rotate, it just
gets shorter. so pro-nav steers to keep that line of sight still: turn acceleration = N x closing
speed x how fast the line of sight is rotating. N=3 is the standard choice. it's what real seekers
have used forever because it only needs relative position + velocity, no prediction of where the
target's going. turn acceleration is capped (`max_accel`) so interceptors can't turn on a dime.

## why check the whole tick for hits

an interceptor at 300 m/s moves 30m per tick, way more than the 10m kill radius. if you only check
distance at the end of the tick, fast stuff flies right through each other. so the check treats both
as moving in a straight line during the tick and finds the closest they got (`closest_approach`).

## determinism

every scenario gets its own seed spawned from the suite seed (`np.random.SeedSequence`). that seed
splits into two streams: one for the layout (where things spawn, when, how fast) and one for the
`p_kill` dice rolls. kill rolls happen in a fixed order (interceptor slot order). same seed in = byte
identical run out, tested in `tests/test_engine.py`.

## scenario knobs (`sim/scenario.py`)

| knob | what it does |
|---|---|
| `n_threats`, `n_effectors` | [lo, hi], picked uniformly per scenario |
| `type_mix` | fraction of sites of each type |
| `arrival` | `spread` = launches uniform over the window. `waves` = bunched into `n_waves` groups, everything shows up at once, harder |
| `value` | `uniform` = all worth 1. `lognormal` = a few threats matter way more, so who you shoot first matters |
| `sector_deg` | threats come from inside this arc (120 deg default) |
| `threat_speed`, `spawn_radius`, `defense_radius` | geometry |

## eval suites

`configs/suites/<name>_v<k>.yaml` expands into the same list of scenarios every time. the `.lock` next
to it holds a hash of the config + every scenario array. `make suites` rebuilds and checks it, and
`load_suite` refuses to run on a suite that doesn't match its lock. changing a suite means making a v2,
never editing v1, so old results always point at the exact scenarios they ran on.

`suite_small_v1`: 200 scenarios, 5-10 threats, 2-3 sites, waves, lognormal values. small enough that
the exact solver (phase 2) is still fast. ammo is deliberately tight. the first draft had deep
magazines and greedy stopped every threat in a quick sanity check, which makes every method look the
same.

## known simplifications (things someone will ask about)

- threats fly straight, no evasion, no altitude. 2D only.
- no sensor noise. the policy sees true positions. tracking/sensing is out of scope for now.
- an interceptor whose target died doesn't retarget, it's just wasted.
- the kill model is a coin flip at closest approach, not a physics/damage model.
