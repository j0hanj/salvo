# the learned policy

a graph neural net that looks at one decision tick and scores every legal shot. trained to copy
the exact per-tick MILP, then run on fights way bigger than it ever saw.

## why learn anything if the MILP is exact

two reasons. the MILP's cost blows up w swarm size (see the scaling section of the readme), and it
needs one computer that sees everything. a GNN's cost grows roughly linearly w the number of legal
shots, and its message passing maps onto real messages between nodes, which is what phase 4 (each
defender runs its own copy, talking over a lossy network) needs.

## the graph (`learned/features.py`)

- one node per effector: kill chance, log cost, range, interceptor speed, ammo left, reload, ready
- one node per threat: log value, distance to the asset, time to impact, speed, how many
  interceptors are already on it
- one edge per legal shot (ready, in range, threat open): distance / range, expected value of the
  shot (value x kill chance), how long the interceptor takes to get there

no type ids, no node indices, nothing that only makes sense at one swarm size. that's the whole
bet behind zero-shot scaling: if every input is a physical quantity, the model can't overfit to
"there are always 3 effectors".

## the model (`learned/model.py`)

encode every node and edge w a small MLP, then 3 rounds of message passing:

1. each edge updates from itself + its two endpoints
2. each threat updates from the **mean and max** of its edges
3. each effector updates from the mean and max of its edges

then each edge gets a logit: "would the expert fire along this edge". mean + max, not sum, so a
threat w 50 possible shooters doesn't get numbers 10x bigger than one w 5. updates are residual so
stacking rounds doesn't wash out the inputs. tested to be permutation equivariant: shuffle the
threats and the scores shuffle the same way.

decoding (`learned/policy.py`): each effector fires along its highest-scoring edge if the model
puts it above 0.5. several effectors can pick the same threat, same as the MILP.

## training (`learned/train.py`, `scripts/train.py`)

**behavior cloning.** the MILP flies every scenario in `suite_train_v1` and at every tick w a legal
shot we record the graph + which edges it fired along. binary cross-entropy per edge, positives
upweighted since most legal shots aren't taken.

**the problem w plain BC.** the model only ever sees states the expert reached. once it makes one
different call, it's in a state the expert never visited, and errors compound.

**DAgger** fixes that: let the model fly the episodes, have the MILP label what it *would* have
done in the states the model actually reached, add those to the dataset, retrain. repeat. 3 rounds.

**no leakage.** train/val split is by scenario, not by tick, since ticks from one episode are
near-duplicates. and `suite_train_v1` is a different seed from every eval suite, so nothing the
model is scored on was ever trained on.

## things someone will poke at

- it's imitating a myopic expert, so the best it can do on quality is match per-tick MILP. beating
  it needs a signal that sees the whole raid (RL, or labels from a planner that looks ahead).
- labels aren't unique when two effectors are interchangeable, so some "mistakes" are equally good
  shots. edge-level precision / recall understates how close it is; episode metrics are the truth.
- latency is python + torch on one cpu thread, no batching across ticks. phase 5 moves inference
  to C++.

## what actually happened

it matches the MILP at training size and over-commits more and more as the swarm grows (numbers
in the readme and `docs/EXPERIMENTS.md`). three follow-ups narrowed down why:

- **aggregation** (`scripts/ablation.py`). mean + max can't count how many effectors line up on
  a threat, so a log-degree channel was added (`EdgeScorer(degree=True)`, `scripts/train.py
  --degree`). it barely helps. the aggregation wasn't the main problem.
- **decoding** (`learned/policy.py`, `decode="capped"`). the default decode lets every effector
  argmax on its own, so nothing stops twenty of them picking the same threat. capped decode walks
  edges most-confident-first and skips a threat once it's covered. stacking goes to zero and most
  of the loss comes back. this was the main problem.
- **the fire threshold** (`scripts/confidence.py`, `scripts/threshold.py`). an edge only fires if
  its score is >= 0.5, and the scores shrink on bigger graphs, so at scale most free effectors sit
  idle next to an uncovered threat. dropping the threshold closes most of what's left.

with both fixes it's close to hungarian at 500 threats but still behind greedy-value, a
hand-written ranking, while firing more shots. the per-edge bce loss gives scores that mean
something at the training size and drift as the graph grows. mixed-size training or a per-effector
ranking loss are the obvious next things to try.
