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
