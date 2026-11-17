| comms | threats stopped % | value defended % | Δ value vs hungarian | shooters per targeted threat |
|---|---|---|---|---|
| hungarian (central, no loss) | 92.1 [90.4, 93.7] | 97.2 [96.5, 97.8] | — | 1.00 [1.00, 1.00] |
| nodes dark 10% of ticks | 89.9 [88.0, 91.8] | 96.8 [96.1, 97.5] | -0.4 [-0.8, 0.1] | 1.11 [1.10, 1.13] |
| links down 19% (same pairwise loss) | 92.1 [90.5, 93.6] | 97.3 [96.7, 97.9] | 0.1 [-0.3, 0.5] | 1.00 [1.00, 1.00] |
| nodes dark 30% of ticks | 84.4 [82.2, 86.7] | 95.0 [94.0, 95.9] | -2.2 [-2.8, -1.7] | 1.37 [1.34, 1.39] |
| links down 51% (same pairwise loss) | 92.0 [90.4, 93.5] | 97.2 [96.6, 97.8] | 0.0 [-0.3, 0.4] | 1.00 [1.00, 1.00] |
| nodes dark 50% of ticks | 77.0 [74.4, 79.5] | 92.2 [90.8, 93.4] | -5.0 [-6.0, -4.1] | 1.67 [1.64, 1.71] |
| links down 75% (same pairwise loss) | 92.0 [90.3, 93.5] | 97.2 [96.6, 97.8] | 0.0 [-0.4, 0.4] | 1.00 [1.00, 1.00] |
| nodes dark 70% of ticks | 68.9 [66.6, 71.0] | 88.8 [87.3, 90.2] | -8.4 [-9.5, -7.4] | 2.02 [1.97, 2.06] |
| links down 91% (same pairwise loss) | 90.5 [88.5, 92.4] | 96.5 [95.7, 97.3] | -0.7 [-1.1, -0.3] | 1.10 [1.09, 1.11] |
| no comms at all | 60.0 [57.6, 62.3] | 83.7 [81.7, 85.5] | -13.5 [-15.1, -12.0] | 2.48 [2.41, 2.56] |

`suite_scale_n100_v1`, 50 scenarios, same scenarios for every method. cbba w 10 rounds of message passing per tick. node rows: each effector is dark the given share of ticks (hears nobody, heard by nobody, still fires on its own pick). link rows: every directed link independently down 1 - (1 - down)^2 of ticks, the same pairwise loss. both bursty w a mean spell of 20 ticks (1 s each). 95% bootstrap cis, Δ paired per scenario. made by `scripts/outage.py` at `b1916ef`.
