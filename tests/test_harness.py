import math

from eval.harness import evaluate, read_csv, run_one, write_csv
from eval.suites import load_suite


def test_row_is_consistent():
    suite = load_suite("suite_small_v1")
    r = run_one(suite, 0, "greedy-nearest")
    assert r["stopped"] + r["leaked"] + r["unresolved"] == r["n_threats"]
    assert r["unresolved"] == 0
    assert 0 <= r["stopped_frac"] <= 1 and 0 <= r["value_frac"] <= 1
    assert r["decisions"] > 0 and r["lat_p95_ms"] >= 0


def test_parallel_matches_serial():
    keys = ["scenario", "method", "stopped", "leaked", "shots", "cost"]
    methods = ["greedy-nearest", "hungarian"]
    serial = evaluate("suite_small_v1", methods, jobs=1, limit=6)
    par = evaluate("suite_small_v1", methods, jobs=2, limit=6)
    assert len(serial) == 12
    assert [[r[k] for k in keys] for r in serial] == [[r[k] for k in keys] for r in par]


def test_csv_roundtrip(tmp_path):
    suite = load_suite("suite_small_v1")
    rows = [run_one(suite, i, "greedy-nearest") for i in range(3)]
    write_csv(rows, tmp_path / "x.csv")
    back = read_csv(tmp_path / "x.csv")
    assert back[1]["stopped"] == rows[1]["stopped"]
    assert back[0]["method"] == "greedy-nearest"
    assert math.isclose(back[2]["cost"], rows[2]["cost"])


def test_gnn_checkpoint_loads_and_runs(tmp_path):
    import torch

    from learned.model import EdgeScorer

    model = EdgeScorer(hidden=8, rounds=1)
    path = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"hidden": 8, "rounds": 1}}, path)
    r = run_one(load_suite("suite_small_v2"), 0, f"gnn:{path}")
    assert r["stopped"] + r["leaked"] + r["unresolved"] == r["n_threats"]


def test_stacking_columns():
    suite = load_suite("suite_small_v2")
    h = run_one(suite, 0, "hungarian")  # hungarian never stacks
    assert h["stack_mean"] == 1.0 and h["stack3_frac"] == 0.0
    m = run_one(suite, 0, "milp")
    assert m["stack_mean"] >= 1.0


def test_gnn_threshold_override(tmp_path):
    import torch

    from eval.harness import resolve
    from learned.model import EdgeScorer

    path = tmp_path / "m.pt"
    model = EdgeScorer(hidden=8, rounds=1)
    torch.save({"state_dict": model.state_dict(), "config": {"hidden": 8, "rounds": 1}}, path)
    assert resolve(f"gnn:{path}").threshold == 0.5
    p = resolve(f"gnn-capped@0.2:{path}")
    assert p.threshold == 0.2 and p.decode == "capped" and p.survival_floor == 0.1
    assert resolve(f"gnn-capped@0/0.4:{path}").survival_floor == 0.4
    r = run_one(load_suite("suite_small_v2"), 0, f"gnn-capped@0:{path}")
    assert r["stopped"] + r["leaked"] + r["unresolved"] == r["n_threats"]


def test_cbba_row_matches_greedy_value():
    # fully connected cbba picks what greedy-value picks, so whole episodes come out the same
    suite = load_suite("suite_small_v2")
    keys = ["stopped", "leaked", "shots", "cost"]
    c, g = run_one(suite, 0, "cbba"), run_one(suite, 0, "greedy-value")
    assert [c[k] for k in keys] == [g[k] for k in keys]


def test_lossy_cbba_reproducible_and_lossless_case():
    suite = load_suite("suite_small_v2")
    keys = ["stopped", "leaked", "shots", "cost"]
    # no loss + enough rounds to converge = plain cbba
    a, b = run_one(suite, 0, "cbba@0/50"), run_one(suite, 0, "cbba")
    assert [a[k] for k in keys] == [b[k] for k in keys]
    # w loss, reseeding per scenario means order doesn't matter: serial == parallel
    m = ["cbba@0.5/5"]
    serial = evaluate("suite_small_v2", m, jobs=1, limit=4)
    par = evaluate("suite_small_v2", m, jobs=2, limit=4)
    assert [[r[k] for k in keys] for r in serial] == [[r[k] for k in keys] for r in par]


def test_bursty_cbba_lossless_case_and_reproducible():
    suite = load_suite("suite_small_v2")
    keys = ["stopped", "leaked", "shots", "cost"]
    a, b = run_one(suite, 0, "cbba-burst@0/5"), run_one(suite, 0, "cbba")
    assert [a[k] for k in keys] == [b[k] for k in keys]
    # link state is rebuilt per scenario, so a scenario's result doesn't depend on what ran first
    m = ["cbba-burst@0.7/5"]
    serial = evaluate("suite_small_v2", m, jobs=1, limit=4)
    par = evaluate("suite_small_v2", m, jobs=2, limit=4)
    assert [[r[k] for k in keys] for r in serial] == [[r[k] for k in keys] for r in par]
