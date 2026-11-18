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


def _rates(loss: float, burst: float) -> tuple[float, float]:
    """(p good -> bad, p bad -> good) per tick for a chain that's bad `loss` of the time in
    spells averaging `burst` ticks. raises if no valid chain does that."""
    if not 0 <= loss < 1:
        raise ValueError("loss must be in [0, 1)")
    if burst < 1:
        raise ValueError("burst is a mean length in ticks, >= 1")
    # small slack: 0.9 / 0.1 is 9.000000000000002 in float, and burst = 9 should be allowed
    if loss > 0 and burst < loss / (1 - loss) - 1e-9:
        raise ValueError(
            f"loss {loss} needs mean bursts >= {loss / (1 - loss):.3g} ticks, got {burst}"
        )
    return loss / (1 - loss) / burst, 1.0 / burst


class GilbertElliott:
    def __init__(self, m: int, loss: float, burst: float, rng: np.random.Generator):
        self.p_fail, self.p_recover = _rates(loss, burst)
        self.rng = rng
        # start each link from the stationary distribution, not all-good
        self.bad = rng.random((m, m)) < loss
        np.fill_diagonal(self.bad, False)

    def step(self) -> np.ndarray:
        """advance one tick. returns (m, m) bool, up[i, k] = i hears k this tick."""
        u = self.rng.random(self.bad.shape)
        self.bad = np.where(self.bad, u >= self.p_recover, u < self.p_fail)
        np.fill_diagonal(self.bad, False)  # you always hear yourself
        return ~self.bad


class NodeOutage:
    """whole effectors go dark: a jammed node hears nobody and nobody hears it, but it can still
    see and shoot, so it bids alone. one chain per node, same two knobs as GilbertElliott
    (`down` = share of ticks a node is dark, `burst` = mean dark spell in ticks).

    vs GilbertElliott at the same pairwise loss (1 - (1 - down)^2), the failures here are
    correlated across every link touching the node, which is closer to a jammer sitting on top of
    one launcher than to random fading."""

    def __init__(self, m: int, down: float, burst: float, rng: np.random.Generator):
        self.p_fail, self.p_recover = _rates(down, burst)
        self.rng = rng
        self.dark = rng.random(m) < down

    def step(self) -> np.ndarray:
        """advance one tick. returns (m, m) bool, up[i, k] = i hears k this tick."""
        u = self.rng.random(self.dark.shape)
        self.dark = np.where(self.dark, u >= self.p_recover, u < self.p_fail)
        lit = ~self.dark
        return (lit[:, None] & lit[None, :]) | np.eye(len(lit), dtype=bool)


def flood(masks) -> np.ndarray:
    """who knows about whom after relaying over a sequence of per-round masks. masks: iterable
    of (m, m) bool, mask[i, k] = i hears k that round. everyone starts knowing only themselves;
    each round you learn everything anyone you hear already knew. returns (m, m) bool,
    knows[i, j] = j's state reached i. this is plain flooding, the same reach cbba's max-
    consensus gets w the same rounds."""
    knows = None
    for h in masks:
        h = np.asarray(h, dtype=bool)
        if knows is None:
            knows = np.eye(len(h), dtype=bool)
        knows = knows | ((h.astype(np.int32) @ knows.astype(np.int32)) > 0)
    if knows is None:
        raise ValueError("need at least one round")
    return knows


class Channel:
    """one tick of comms for a decentralized method: `tick(m)` returns the per-round masks.

    kind="full": everyone hears everyone. "iid": every message lost w prob `loss`, fresh each
    round. "link": gilbert-elliott links, "node": NodeOutage; both step once per tick and hold
    for every round of it. built lazily on the first tick, and for link / node it draws from the
    rng in the same order as BurstyCBBA does, so the same seed gives both methods the same link
    states."""

    def __init__(self, kind: str = "full", loss: float = 0.0, burst: float = 1.0, rounds: int = 10):
        if kind not in ("full", "iid", "link", "node"):
            raise ValueError(f"unknown channel kind {kind!r}")
        self.kind, self.loss, self.burst, self.rounds = kind, loss, burst, rounds
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.model = None

    def tick(self, m: int) -> list[np.ndarray]:
        eye = np.eye(m, dtype=bool)
        if self.kind == "full":
            return [np.ones((m, m), dtype=bool)] * self.rounds
        if self.kind == "iid":
            return [(self.rng.random((m, m)) >= self.loss) | eye for _ in range(self.rounds)]
        if self.model is None:
            cls = GilbertElliott if self.kind == "link" else NodeOutage
            self.model = cls(m, self.loss, self.burst, self.rng)
        return [self.model.step()] * self.rounds
