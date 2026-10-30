import torch

from learned.features import EDGE_FEATURES, EFF_FEATURES, THR_FEATURES
from learned.model import EdgeScorer, scatter_mean_max


def random_graph(m, n, e, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(m, len(EFF_FEATURES), generator=g),
        torch.randn(n, len(THR_FEATURES), generator=g),
        torch.stack(
            [torch.randint(0, m, (e,), generator=g), torch.randint(0, n, (e,), generator=g)]
        ),
        torch.randn(e, len(EDGE_FEATURES), generator=g),
    )


def test_scatter_mean_max():
    src = torch.tensor([[1.0], [3.0], [5.0]])
    idx = torch.tensor([0, 0, 2])
    mean, mx = scatter_mean_max(src, idx, 3)
    assert mean.squeeze().tolist() == [2.0, 0.0, 5.0]
    assert mx.squeeze().tolist() == [3.0, 0.0, 5.0]


def test_permuting_threats_permutes_scores():
    torch.manual_seed(0)
    model = EdgeScorer(hidden=16, rounds=2).eval()
    eff_x, thr_x, edges, edge_x = random_graph(4, 6, 15)
    perm = torch.randperm(6)  # new threat i = old threat perm[i]
    inv = torch.argsort(perm)
    edges_p = torch.stack([edges[0], inv[edges[1]]])
    with torch.no_grad():
        a = model(eff_x, thr_x, edges, edge_x)
        b = model(eff_x, thr_x[perm], edges_p, edge_x)
    assert torch.allclose(a, b, atol=1e-5)


def test_runs_on_any_size_and_no_edges():
    model = EdgeScorer(hidden=16, rounds=2)
    for m, n, e in [(1, 1, 1), (3, 10, 20), (150, 500, 4000)]:
        assert model(*random_graph(m, n, e)).shape == (e,)
    eff_x, thr_x, _, _ = random_graph(2, 3, 1)
    empty = model(
        eff_x, thr_x, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, len(EDGE_FEATURES))
    )
    assert empty.shape == (0,)
