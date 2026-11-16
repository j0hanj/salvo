| loss model | threats stopped % | value defended % | Δ value vs hungarian | shooters per targeted threat |
|---|---|---|---|---|
| hungarian (central, no loss) | 92.1 [90.4, 93.7] | 97.2 [96.5, 97.8] | — | 1.00 [1.00, 1.00] |
| independent per message, 50% lost | 91.9 [90.2, 93.5] | 97.2 [96.5, 97.8] | -0.0 [-0.4, 0.4] | 1.00 [1.00, 1.00] |
| bursty links, 50% down, ~10 ticks | 92.0 [90.4, 93.5] | 97.2 [96.6, 97.8] | 0.0 [-0.3, 0.4] | 1.00 [1.00, 1.00] |
| bursty links, 50% down, ~20 ticks | 92.0 [90.4, 93.5] | 97.2 [96.6, 97.8] | 0.0 [-0.3, 0.4] | 1.00 [1.00, 1.00] |
| bursty links, 50% down, ~50 ticks | 92.0 [90.3, 93.5] | 97.3 [96.6, 97.8] | 0.1 [-0.3, 0.4] | 1.00 [1.00, 1.00] |
| independent per message, 90% lost | 91.8 [90.2, 93.4] | 97.2 [96.3, 97.9] | -0.0 [-0.7, 0.5] | 1.02 [1.01, 1.02] |
| bursty links, 90% down, ~10 ticks | 91.0 [89.1, 92.8] | 97.0 [96.3, 97.7] | -0.2 [-0.5, 0.2] | 1.08 [1.07, 1.09] |
| bursty links, 90% down, ~20 ticks | 91.1 [89.2, 92.9] | 96.8 [95.9, 97.5] | -0.4 [-0.9, -0.0] | 1.07 [1.07, 1.08] |
| bursty links, 90% down, ~50 ticks | 90.7 [88.8, 92.5] | 96.7 [96.0, 97.4] | -0.5 [-1.0, -0.0] | 1.08 [1.07, 1.09] |
| independent per message, 95% lost | 90.4 [88.5, 92.2] | 96.7 [96.0, 97.5] | -0.5 [-1.0, 0.0] | 1.07 [1.06, 1.08] |
| bursty links, 95% down, ~20 ticks | 81.8 [79.3, 84.2] | 94.3 [93.1, 95.3] | -2.9 [-3.7, -2.3] | 1.50 [1.47, 1.53] |
| bursty links, 95% down, ~50 ticks | 81.5 [79.3, 83.6] | 94.2 [93.2, 95.2] | -3.0 [-3.7, -2.4] | 1.51 [1.48, 1.53] |

`suite_scale_n100_v1`, 50 scenarios, same scenarios for every method. cbba w 10 rounds of message passing per tick. independent = every message lost w the given prob, fresh each round. bursty = each directed link is a gilbert-elliott chain stepped once per replan tick (1 s), down the given share of ticks in spells of the given mean length, and down for every round of a tick it's down in. 95% bootstrap cis, Δ paired per scenario. made by `scripts/burst.py` at `1bab701`.
