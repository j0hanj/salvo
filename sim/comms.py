"""link models for decentralized methods. who can hear whom on a given replan tick.

independent per-message loss is the friendly case: drop a message and the same bid probably gets
through on the next round or via someone else. real jamming comes in bursts, so a link that's
down tends to stay down for a while.

GilbertElliott: every directed link is its own two-state markov chain, stepped once per replan
tick. good = everything gets through, bad = nothing does. two knobs:
  loss  = long-run fraction of ticks a link spends bad
  burst = mean length of a bad spell, in ticks
from those, p(bad -> good) = 1 / burst and p(good -> bad) = loss / (1 - loss) / burst, which
gives the right stationary loss. that second one has to be a probability, i.e. the mean good
spell burst * (1 - loss) / loss has to be at least one tick, so burst >= loss / (1 - loss): 90%
loss needs bursts of 9+ ticks. combos outside that are rejected instead of silently clipped (a
clip quietly changes the loss you asked for).
"""

import numpy as np


class GilbertElliott:
    def __init__(self, m: int, loss: float, burst: float, rng: np.random.Generator):
        if not 0 <= loss < 1:
            raise ValueError("loss must be in [0, 1)")
        if burst < 1:
            raise ValueError("burst is a mean length in ticks, >= 1")
        # small slack: 0.9 / 0.1 is 9.000000000000002 in float, and burst = 9 should be allowed
        if loss > 0 and burst < loss / (1 - loss) - 1e-9:
            raise ValueError(
                f"loss {loss} needs mean bursts >= {loss / (1 - loss):.3g} ticks, got {burst}"
            )
        self.rng = rng
        self.p_recover = 1.0 / burst
        self.p_fail = loss / (1 - loss) / burst
        # start each link from the stationary distribution, not all-good
        self.bad = rng.random((m, m)) < loss
        np.fill_diagonal(self.bad, False)

    def step(self) -> np.ndarray:
        """advance one tick. returns (m, m) bool, up[i, k] = i hears k this tick."""
        u = self.rng.random(self.bad.shape)
        self.bad = np.where(self.bad, u >= self.p_recover, u < self.p_fail)
        np.fill_diagonal(self.bad, False)  # you always hear yourself
        return ~self.bad
