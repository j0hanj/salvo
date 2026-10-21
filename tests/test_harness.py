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
