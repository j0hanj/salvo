# swarm-wta

who shoots what, when a bunch of cheap drones show up at once, and the radios are getting jammed.

## the problem

shooting million-dollar interceptors at drones that cost a few thousand doesn't scale. so you need to pick which defender goes after
which incoming threat (the weapon-target assignment problem, WTA), fast, cost-aware, and keep re-planning
as things change. on top of that, jamming kills the radio links, so anything that needs a central computer
or a perfect network falls apart.

**question:** can a learned, decentralized assignment policy (a graph neural net where each defender is its
own computer and messages are real network packets) keep stopping threats while the network is dropping
packets, and beat the standard decentralized method (CBBA, an auction algorithm)?

**main metric:** % of threats stopped vs % of messages lost, every method on the exact same scenarios.

not claiming a new invention. CBBA, GNNs for multi-robot coordination, and learned WTA solvers all exist.
the point is a careful, reproducible measurement on a real lossy network on cheap hardware, and an honest
look at where each method wins and loses.

## plan

0. repo + tooling
1. 2D simulator + scenario generator
2. baselines (greedy, hungarian, exact MILP, CBBA) + eval harness
3. GNN trained to copy the exact solver on small fights, tested on way bigger ones
4. decentralized version in sim, w simulated packet loss
5. on raspberry pis talking over real wifi
6. jamming experiments (`tc netem`) + write-up

## setup

```
brew install uv   # or see astral.sh/uv
make setup
make test
make lint
```

## abstraction note

everything here is generic. threats and defenders are just points w made-up range / speed / kill-probability
numbers. no real weapon systems, no real specs, nothing export-controlled. "intercept" means two points got
close enough. this is a resource-allocation study, not a weapons project.
