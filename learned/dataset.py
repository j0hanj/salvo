"""imitation data: play episodes, and at every tick w a legal shot, record the graph + which edges
the expert (milp) picked.

driver = the policy actually flying the episode, labeler = the expert we copy. driver == labeler is
plain behavior cloning. driver = the model, labeler = milp is DAgger: label the states the model
itself ends up in, which fixes its drift away from the expert's states.
"""

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from learned.features import featurize
from sim.engine import run


@dataclass
class Batch:
    """many graphs packed into flat arrays. *_ptr[i]:*_ptr[i+1] = graph i's rows."""

    eff_x: np.ndarray
    thr_x: np.ndarray
    edges: np.ndarray  # (2, E), indices local to each graph
    edge_x: np.ndarray
    y: np.ndarray  # (E,) 1 if the expert fired along this edge
    eff_ptr: np.ndarray
    thr_ptr: np.ndarray
    edge_ptr: np.ndarray
    scn: np.ndarray  # (G,) which scenario each graph came from, so train/val can split by scenario

    def __len__(self):
        return len(self.edge_ptr) - 1

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(**{k: f[k] for k in f.files})


def pack(samples) -> Batch:
    """samples: list of (scenario idx, Graph, y)."""

    def ptr(sizes):
        return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    gs = [g for _, g, _ in samples]
    return Batch(
        eff_x=np.concatenate([g.eff_x for g in gs]),
        thr_x=np.concatenate([g.thr_x for g in gs]),
        edges=np.concatenate([g.edges for g in gs], axis=1),
        edge_x=np.concatenate([g.edge_x for g in gs]),
        y=np.concatenate([y for _, _, y in samples]).astype(np.float32),
        eff_ptr=ptr([len(g.eff_x) for g in gs]),
        thr_ptr=ptr([len(g.thr_x) for g in gs]),
        edge_ptr=ptr([g.edges.shape[1] for g in gs]),
        scn=np.array([i for i, _, _ in samples], dtype=np.int64),
    )


def episode_samples(sc, sim_cfg, types, driver, labeler):
    samples = []

    def policy(obs):
        g = featurize(obs)
        if g.edges.shape[1]:
            picked = set(labeler(obs))
            y = np.array([(e, j) in picked for e, j in g.edges.T.tolist()], dtype=np.float32)
            samples.append((g, y))
        return driver(obs)

    run(sc, sim_cfg, types, policy)
    return samples


_ctx = None


def _init(suite, driver, labeler):
    global _ctx
    _ctx = (suite, driver, labeler)


def _work(i):
    suite, driver, labeler = _ctx
    return episode_samples(suite.scenarios[i], suite.cfg.sim, suite.cfg.types, driver, labeler)


def collect(suite, driver, labeler, limit=None, jobs=1) -> Batch:
    idx = range(len(suite.scenarios) if limit is None else min(limit, len(suite.scenarios)))
    if jobs == 1:
        _init(suite, driver, labeler)
        out = [_work(i) for i in idx]
    else:
        with Pool(jobs, initializer=_init, initargs=(suite, driver, labeler)) as pool:
            out = pool.map(_work, idx, chunksize=4)
    return pack([(i, g, y) for i, ep in zip(idx, out, strict=True) for g, y in ep])


def merge(a: Batch, b: Batch) -> Batch:
    """stack two datasets. graph-local edge indices don't change, only the pointers shift."""
    return Batch(
        eff_x=np.concatenate([a.eff_x, b.eff_x]),
        thr_x=np.concatenate([a.thr_x, b.thr_x]),
        edges=np.concatenate([a.edges, b.edges], axis=1),
        edge_x=np.concatenate([a.edge_x, b.edge_x]),
        y=np.concatenate([a.y, b.y]),
        eff_ptr=np.concatenate([a.eff_ptr, b.eff_ptr[1:] + a.eff_ptr[-1]]),
        thr_ptr=np.concatenate([a.thr_ptr, b.thr_ptr[1:] + a.thr_ptr[-1]]),
        edge_ptr=np.concatenate([a.edge_ptr, b.edge_ptr[1:] + a.edge_ptr[-1]]),
        scn=np.concatenate([a.scn, b.scn]),
    )
