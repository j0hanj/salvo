# experiments

lab notebook. newest at the bottom. every entry: what i ran, what came out, what i think it means, what's next.
every number here comes from a script in the repo, named next to it, w the run dir in `results/`.

template:

```
## yyyy-mm-dd: short title
ran: script + config + commit
result: numbers w confidence intervals, plus the figure path
conclusion:
next:
```

## 2026-10-14: suite_small_v1 calibration
ran: greedy-nearest on a handful of suite_small scenarios as a sanity check (not a result, no CIs).
result: w the first draft's magazines (8 short / 4 long) greedy stopped every threat, so the suite
couldn't separate methods. cut magazines to 3 / 2, sites to 2-3, threat speed to 40-80 m/s, then
locked v1. real per-method numbers come from the phase 2 harness.
conclusion:
next: phase 2, baselines + eval harness on suite_small_v1.

## 2026-10-21: suite_small_v1 doesn't separate methods
ran: greedy-nearest, greedy-value, hungarian on v1 (quick checks, first 20-40 scenarios).
result: identical stopped / value / cost for all three. `scripts/decision_stats.py --suite
suite_small_v1 --limit 40`: 37 of 254 decision ticks had more than one legal shot, baselines
disagreed on 9. same script on v2: 152 of 331, disagree on 91. root cause: the generator bunched launch times, not
arrival times, so threats reached the defense one at a time. added `impact_waves`, then a quick
sweep of threat speed / reload / magazine on 40 scenarios to find a regime where the methods diverge.
locked that as suite_small_v2. v1 kept as-is.
conclusion:
next: full baseline comparison on v2.

## 2026-10-21: phase 2 baselines on suite_small_v2
ran: `make compare` (`scripts/compare.py`), 4 methods x 200 scenarios, run dir
`results/*-compare-suite_small_v2`, table + figure in the readme.
result: threats stopped is flat across methods: 79.2 [76.5, 81.8] (greedy-nearest) to 80.6
[78.2, 83.1] (hungarian), paired Δ vs milp within about a point either way. value defended
separates: greedy-nearest 79.3 [76.1, 82.5] vs 86.8-87.4 for the value-aware methods, paired Δ vs
milp -8.1 [-10.6, -5.7] pts. milp doesn't beat hungarian or greedy-value on value (Δ -0.3
[-1.4, 0.7] and -0.6 [-1.5, 0.2]). milp decision latency about 2.4 ms vs about 0.02 ms for the rest.
conclusion:
next: phase 3. bigger suites for the scaling study, MILP-labeled training data, GNN.

## 2026-10-29: phase 3, imitation-trained gnn + zero-shot scaling
ran: `make train` (`scripts/train.py` at `8bb2ef4`, run `results/20261029-213442-train`): bc on
milp episodes from suite_train_v1, then 3 dagger rounds, 7568 -> 30847 labeled ticks. then
`scripts/scaling.py` at `af0bf0b` (run `results/20261029-214947-scaling`): 5 methods x
suite_scale_n{10,25,50,100,250,500}_v1, 50 scenarios each.
result: at 10 threats the gnn matches milp on value (94.8 [91.3, 97.4] vs 94.8 [91.3, 97.4]). it
degrades w size: value 90.5 at 50, 85.0 at 100, 61.3 [59.9, 62.7] at 500, vs 97.2 [96.8, 97.6] for
hungarian. threats stopped falls to 31.4% at 500. `scripts/stacking.py` (first 5 scenarios): at
n=10 gnn and milp both put 3+ shooters on a targeted threat 1.7% of the time; at n=100 gnn 22.1% vs
milp 6.9%, mean shooters per targeted threat 1.98 vs 1.33. gnn latency 12.4 ms at 500 vs 272 ms for
milp. milp is slightly behind hungarian from 100 up (Δ value -1.0 [-1.2, -0.7] at 500).
conclusion:
next: test the aggregation hypothesis. candidates: add a sum / log-degree channel so the model can
see how many shooters compete for a threat, train on mixed sizes (5-50), or decode gnn scores w a
capacity-aware step instead of independent argmax per effector.

## 2026-10-30: is the over-stacking from aggregation or from decoding?
ran: `scripts/train.py --degree` at `cdf40a9` (run `results/20261030-002940-train`, same recipe as
the phase 3 model + a log-degree channel). then `scripts/ablation.py` at `cdf40a9` (run
`results/20261030-003930-ablation`): {plain, +degree} x {argmax, capped decode} plus hungarian, and
milp up to 100, on suite_scale_n{10,50,100,500}_v1, 50 scenarios each. table in the readme.
result: at 500 threats, value defended: gnn 61.3 [59.9, 62.7], gnn+deg 64.4 [62.4, 66.3], gnn
capped 87.6 [85.8, 89.3], gnn+deg capped 81.7 [79.1, 84.2], hungarian 97.2 [96.8, 97.6]. targeted
threats w 3+ shooters: gnn 48.6%, gnn+deg 36.5%, both capped variants 0. threats stopped w capped
decode at 500 is 65.2 [62.2, 68.3] vs 91.2 for hungarian. at 10 threats every variant is within the
ci of hungarian. a unit test (`tests/test_model.py`) confirms plain mean+max gives identical scores
when every edge is duplicated, the degree variant doesn't.
conclusion:
next: the gap left after capping. threats stopped drops w zero stacking, so effectors are holding
fire. first thing to check: how the model's confidence on legal edges shifts w graph size vs the
fixed 0.5 threshold.

## 2026-11-03: is the gap after capping the fire threshold?
ran: `scripts/confidence.py` at `ec2f8cd` (run `results/20261103-153508-confidence`) and
`scripts/threshold.py` at `5c5d500` (run `results/20261103-153806-threshold`), plain phase 3 model
(`results/20261029-213442-train`, trained at `8bb2ef4`), capped decode, suite_scale_n{10,50,100,500}_v1,
50 scenarios each. tables in the readme and `docs/tables/{confidence,threshold}.md`.
result: median best-edge score per free effector is 1.00 at 10 threats, 0.98 at 50, 0.61 at 100,
0.64 at 500. share of free effectors w no edge >= 0.5: 0.4% at 10, 42.6% [40.4, 44.9] at 500. idle
while a reachable threat is still uncovered: 0.4% at 10, 71.8% [70.3, 73.3] at 500. threshold
sweep at 500, value defended: 87.6 [85.8, 89.3] at 0.5, 90.8 at 0.3, 94.6 at 0.1, 95.4 [94.8, 96.1]
at 0.0; threats stopped 65.2 -> 84.9. hungarian 97.2, greedy-value 97.2 [96.8, 97.6]. gnn at 0.0 is
Δ -1.7 [-2.0, -1.4] vs hungarian and fires 1.16x hungarian's shots, greedy-value 1.00x. at 10
threats every threshold is within the ci of hungarian. threshold was swept on the eval suites.
conclusion:
next: the scores aren't calibrated across sizes, and even as a pure ranking the gnn loses to
greedy-value at scale. options: train on mixed sizes (5-50) so scores see bigger graphs, or train
w a ranking / listwise loss per effector instead of per-edge bce, then pick a threshold on a held
out suite rather than the eval ones.

## 2026-11-06: does training on bigger fights keep the scores calibrated?
ran: `scripts/train.py --suites suite_train_v1 suite_train_n25_v1 suite_train_n50_v1` at `e54b5de`
(run `results/20261106-141850-train`, same bc + 3 dagger recipe, 59095 labeled ticks by the last
round). then `scripts/threshold.py --tag mixed` at `e64056c` (run
`results/20261106-144508-threshold`) on suite_scale_n{10,50,100,500}_v1, 50 scenarios each. table
in the readme and `docs/tables/threshold_mixed.md`, compare to `docs/tables/threshold.md` (5-10
only model).
result: Δ value vs hungarian at threshold 0.5, mixed vs 5-10 only: 10 threats 0.2 vs 0.5, 50
threats -0.7 [-1.6, 0.2] vs -2.9 [-4.2, -1.6], 100 threats -2.3 [-3.1, -1.5] vs -5.4 [-7.3, -3.6],
500 threats -11.4 [-13.3, -9.6] vs -9.6 [-11.1, -8.1]. at threshold 0.0: 50 threats -0.5 vs -0.8,
100 -0.8 vs -0.9, 500 -1.9 [-2.3, -1.6] vs -1.7 [-2.0, -1.4]. at 50 threats every threshold is
within 0.3 of each other for the mixed model. greedy-value at 500 is 0.1 [-0.1, 0.2]. mixed model
fires 1.17x hungarian's shots at 500, threshold 0.0.
conclusion:
next: scores stay calibrated roughly where there's training data and drift past it, and even
perfectly thresholded the gnn ranks worse than greedy-value at 500. a per-effector ranking loss
(softmax over an effector's legal edges + a "hold" option) would make scores comparable across
sizes by construction instead of by coverage.

## 2026-11-09: does a per-effector ranking loss make scores size-independent?
ran: `scripts/train.py --loss listwise` at `3540f0e` (run `results/20261109-125935-train`,
suite_train_v1 only, bc + 3 dagger, same as the phase 3 model except the loss). loss = per
effector, softmax over its legal edge logits + a fixed hold logit 0, target = milp's edge or hold
(checked on the mixed run's data: milp never fires one effector twice in a tick). then
`scripts/threshold.py --tag listwise` at `3540f0e` (run `results/20261109-130741-threshold`),
suite_scale_n{10,50,100,500}_v1, 50 scenarios each. table in the readme and
`docs/tables/threshold_listwise.md`.
result: every threshold from 0.5 to 0.0 gives identical rows at every size. Δ value vs hungarian:
10 threats -7.1 [-10.4, -4.1], 50 -12.0 [-15.6, -8.7], 100 -10.7 [-12.9, -8.7], 500 -14.5
[-16.4, -12.7]. threats stopped at 500: 83.0 [80.9, 85.1] (hungarian 91.2). shots vs hungarian
1.07x at 10, 1.18x at 500. bce model at threshold 0.0 for comparison: -1.7 [-2.0, -1.4] at 500.
in training, val edge precision = positive rate and recall = 1.0, i.e. every legal edge scores
above hold.
conclusion:
next: the softmax is per effector, so logits are only comparable within one effector, but capped
decode sorts all edges together. either train a cross-effector ordering (e.g. pairwise loss on
milp's pick order, or softmax per threat too), or decode differently (each effector proposes its
argmax, threats resolve by value). bce + threshold 0 is still the best gnn so far.

## 2026-11-11: is the leftover overfire the survival floor?
ran: `scripts/threshold.py --thresholds 0 --floors 0.1 0.35 0.5 --tag floor` at `5cfa60f` (run
`results/20261111-224039-threshold`), 5-10 threat bce model (`results/20261029-213442-train`,
trained at `8bb2ef4`), capped decode, suite_scale_n{10,50,100,500}_v1, 50 scenarios each. table in
the readme and `docs/tables/threshold_floor.md`. 0.3 itself was skipped: one 0.7 shooter leaves
survival at 1 - 0.7 = 0.30000000000000004 in float, just above 0.3, so it acts like 0.1.
result: Δ value vs hungarian, floor 0.1 -> 0.35: 10 threats 0.4 -> 0.6 [-0.3, 1.6], 50 -0.8 -> 0.3
[-0.4, 1.0], 100 -0.9 -> -0.2 [-0.8, 0.3], 500 -1.7 [-2.0, -1.4] -> 0.0 [-0.2, 0.3]. shots vs
hungarian at 500: 1.16x -> 1.00x. threats stopped at 500: 84.9 -> 91.8 [90.2, 93.2] (hungarian
91.2). floor 0.5 gives the same rows as 0.35 at every size. greedy-value at 500: 0.1 [-0.1, 0.2].
both floors >= 0.35 mean at most one shooter per threat per tick for these two effector types
(p_kill 0.7 and 0.9). swept on the eval suites.
conclusion:
next: the gnn now ties the centralized baselines once decode forces one shooter per threat. that's
the setup to carry into phase 4 (decentralized, lossy comms), where hungarian / milp need a
central node and the gnn doesn't. before that: confirm on a held-out suite, and check whether a
floor that allows doubling up on high-value threats beats a flat one.

## 2026-11-14: does threshold 0 + floor 0.35 hold on suites it wasn't picked on?
ran: `scripts/threshold.py --thresholds 0.5 0 --floors 0.1 0.35 --suite-prefix suite_holdout
--tag holdout` at `da49305` (run `results/20261114-214338-threshold`), 5-10 threat bce model
(`results/20261029-213442-train`, trained at `8bb2ef4`), on suite_holdout_n{10,50,100,500}_v1
(added in `1f375ca`: copies of the scaling suites w fresh seeds), 50 scenarios
each. table in the readme and `docs/tables/threshold_holdout.md`.
result: Δ value vs hungarian, threshold 0 + floor 0.35: 10 threats -0.8 [-3.1, 0.7], 50 0.4
[-0.2, 1.2], 100 0.3 [-0.1, 0.8], 500 -0.2 [-0.6, 0.1]; shots 1.00x at every size. threats stopped
at 500: 92.6 [91.2, 93.9] vs hungarian 92.1. default decode (0.5 / 0.1) at 500: -9.4 [-10.7, -8.2].
floor 0.35 at threshold 0.5: -9.8 [-11.1, -8.7] at 500. threshold 0 at floor 0.1: -1.8 [-2.1,
-1.4] at 500, 1.18x shots. at 10 threats floor 0.1 has the higher point estimate (1.0 vs -0.8),
cis overlap. greedy-value at 500: -0.1 [-0.3, 0.1].
conclusion:
next: lock threshold 0 + floor 0.35 as the gnn's decode going into phase 4 (decentralized, lossy
comms), where hungarian / milp need a central node and the gnn doesn't.

## 2026-11-16: how does cbba degrade under independent message loss?
ran: `scripts/loss.py` at `e5509be` (run `results/20261116-001149-loss`). cbba w bundle length 1
(`baselines/cbba.py`), every message lost independently w p in {0, 0.5, 0.9, 0.99, 1.0}, rounds per
tick in {1, 3, 10}, then each effector fires on whatever it holds. hungarian (central, perfect
info) as reference. suite_small_v2 (200 scenarios) and suite_scale_n100_v1 (50). table in the
readme and `docs/tables/loss.md`.
result: suite_scale_n100_v1, Δ value vs hungarian: 10 rounds: 0.1 [-0.3, 0.4] at 0%, -0.0 [-0.4,
0.4] at 50%, -0.0 [-0.7, 0.5] at 90%, -4.8 [-5.7, -4.0] at 99%. 3 rounds: -0.9 at 0%, -2.3 [-3.3,
-1.5] at 90%, -10.5 at 99%. 1 round: -5.6 [-6.8, -4.5] at 0% (1.00 shots per threat vs 1.23 for
hungarian), -9.2 at 90%. 100% loss (no comms, every round count): -13.5 [-15.1, -12.0], 2.48
shooters per targeted threat, threats stopped 60.0 vs 92.1. suite_small_v2: every variant within
0.7 of hungarian, max 1.10 shooters per targeted threat. sanity check before the sweep (one-off,
first 5 scenarios of n100, not in a logged run): ~5.4 free effectors per tick w a legal shot, 82%
of those ticks have 2+, so there is contention to coordinate.
conclusion:
next: independent loss is the friendly case since max-consensus routes around dropped messages.
add bursty / correlated loss (gilbert-elliott per link, or whole nodes going dark for a few
ticks), then get the gnn running on local views so it can join the sweep.

## 2026-11-16: does bursty link loss hurt cbba more than independent loss?
ran: `scripts/burst.py` at `1bab701` (run `results/20261116-092031-burst`). cbba, 10 rounds per
tick, suite_scale_n100_v1 (50 scenarios). independent per-message loss vs gilbert-elliott links
(`sim/comms.py`, one chain per directed link, stepped once per 1 s replan) at long-run loss 50 /
90 / 95%, mean burst 10 / 20 / 50 ticks where feasible. table in the readme and
`docs/tables/burst.md`.
result: Δ value vs hungarian. 50%: independent -0.0 [-0.4, 0.4], bursty 0.0 to 0.1 at every
burst length. 90%: independent -0.0 [-0.7, 0.5], bursty -0.2 [-0.5, 0.2] (10 ticks), -0.4 [-0.9,
-0.0] (20), -0.5 [-1.0, -0.0] (50). 95%: independent -0.5 [-1.0, 0.0], 1.07 shooters per targeted
threat; bursty -2.9 [-3.7, -2.3] (20 ticks) and -3.0 [-3.7, -2.4] (50), 1.50 / 1.51 shooters,
threats stopped 81.8 / 81.5 vs 92.1 for hungarian.
note: a first run at `95e6498` used 90% loss w 1- and 5-tick bursts. the chain can't hit that
(mean good spell would be under a tick) and clipped p(good -> bad) to 1, so those rows actually
ran at ~50% / ~83% loss. that run was thrown out, `6948db4` makes such combos an error instead.
conclusion:
next: whole nodes going dark (an effector hears nothing and nobody hears it) and spatially
correlated jamming are still untested. and the gnn still needs a local-view version to join any
of these sweeps.

## 2026-11-17: node outages vs link loss at the same pairwise loss
ran: `scripts/outage.py` at `b1916ef` (run `results/20261117-150444-outage`). cbba, 10 rounds per
tick, suite_scale_n100_v1 (50 scenarios). node outages (`sim/comms.py` NodeOutage: a dark
effector hears nobody, is heard by nobody, still fires on its own pick) at 10 / 30 / 50 / 70% of
ticks, vs independent gilbert-elliott links at the matching pairwise loss 1 - (1 - down)^2 = 19 /
51 / 75 / 91%. mean spell 20 ticks for both. table in the readme and `docs/tables/outage.md`.
result: Δ value vs hungarian, node vs link: 10%: -0.4 [-0.8, 0.1] vs 0.1 [-0.3, 0.5]. 30%: -2.2
[-2.8, -1.7] vs 0.0 [-0.3, 0.4]. 50%: -5.0 [-6.0, -4.1] vs 0.0 [-0.4, 0.4]. 70%: -8.4 [-9.5, -7.4]
vs -0.7 [-1.1, -0.3]. shooters per targeted threat at 70%: 2.02 (nodes) vs 1.10 (links). no
comms at all: -13.5 [-15.1, -12.0], 2.48. threats stopped at 70% dark: 68.9 vs 92.1 for hungarian.
conclusion:
next: node outages are the regime where cbba degrades smoothly and a lot, so that's where the
decentralized gnn gets compared. needs: each effector builds its graph from what it can see /
hear, decodes alone, and a dark node falls back to its own view. spatially correlated jamming
(a jammer w a radius knocking out several nearby nodes together) is still untested.
