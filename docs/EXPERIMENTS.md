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
