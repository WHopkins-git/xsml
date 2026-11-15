import csv

import numpy as np
import pytest

from xsml import metrics


def test_factors_known_values():
    t = np.zeros(4)
    assert metrics.rms_factor(t, np.full(4, 0.30103)) == pytest.approx(2.0, rel=1e-5)        # every point a factor of 2 high
    assert metrics.rms_factor(t, np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(10.0)
    assert metrics.median_factor(t, np.array([0.0, 0.1, 1.0, -2.0])) == pytest.approx(10 ** 0.55)
    assert metrics.rms_factor(t, t) == 1.0
    assert metrics.rms_factor(t, np.array([3.0, 4.0, 0.0, 0.0])) == pytest.approx(10 ** 2.5)  # sqrt((9 + 16) / 4)
    with pytest.raises(ValueError):
        metrics.rms_factor(t, np.array([np.nan, 0, 0, 0]))
    with pytest.raises(ValueError):
        metrics.rms_factor([], [])


def test_per_group():
    t = np.zeros(6)
    p = np.array([1.0, 1.0, 0.0, 0.0, 2.0, -2.0])
    g = np.array([1, 1, 2, 2, 3, 3])
    s = metrics.per_group(t, p, g)
    assert s.to_dict() == {1: 10.0, 2: 1.0, 3: 100.0}


def test_cluster_bootstrap_on_a_toy():
    # ten groups; five have residual 0 and five have residual 1 at every point: only the mix of groups varies
    g = np.repeat(np.arange(10), 20)
    p = np.where(g < 5, 0.0, 1.0)
    t = np.zeros_like(p)
    r = metrics.cluster_bootstrap(t, p, g, n_boot=500, seed=1)
    assert r["value"] == pytest.approx(10 ** np.sqrt(0.5)) and r["n_groups"] == 10 and r["n_points"] == 200
    assert r["low"] < r["value"] < r["high"]
    # treating points as independent would give a much narrower interval than resampling groups
    r_points = metrics.cluster_bootstrap(t, p, np.arange(200), n_boot=500, seed=1)
    assert (r["high"] - r["low"]) > 3 * (r_points["high"] - r_points["low"])
    # identical groups: no spread at all
    r0 = metrics.cluster_bootstrap(t, np.full(200, 0.5), g, n_boot=100)
    assert r0["low"] == pytest.approx(r0["high"]) == pytest.approx(10 ** 0.5)
    assert metrics.cluster_bootstrap(t, p, g, n_boot=200, seed=3) == metrics.cluster_bootstrap(t, p, g, n_boot=200, seed=3)
    with pytest.raises(ValueError, match="at least two groups"):
        metrics.cluster_bootstrap(t, p, np.zeros(200))


def test_paired_comparison():
    g = np.repeat(np.arange(8), 10)
    t = np.zeros(80)
    hard = np.where(g < 4, 0.2, 1.0)                    # group difficulty shared by both models
    a, b = hard, hard * 1.5                             # A is better in every group
    r = metrics.paired_comparison(t, a, b, g, n_boot=300)
    assert r["ratio"] < 1 and r["high"] < 1 and r["frac_a_better"] == 1.0
    same = metrics.paired_comparison(t, a, a, g, n_boot=100)
    assert same["ratio"] == 1 and same["low"] == 1 and same["high"] == 1


def test_log_score_appends_rows(tmp_path):
    path = tmp_path / "results" / "score_log.csv"
    row = metrics.log_score(path, model="tree", features=["Z", "A"], split="5 isotope folds", metric="rms_factor", value=2.5, ci_low=2.0, ci_high=3.1, n_groups=12)
    metrics.log_score(path, model="xgb", features="Z A N", split="5 isotope folds", metric="rms_factor", value=2.1, declared="protocol", notes="first run")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == metrics.LOG_COLUMNS and len(rows) == 2
    assert rows[0]["features"] == "Z A" and rows[0]["declared"] == "extra" and rows[1]["declared"] == "protocol"
    assert rows[0]["timestamp"].endswith("Z") and row["commit"]
    with pytest.raises(ValueError, match="missing"):
        metrics.log_score(path, model="x")
    with pytest.raises(ValueError, match="unknown"):
        metrics.log_score(path, model="x", features="a", split="s", metric="m", value=1, colour="red")


def test_shipped_score_log_has_the_header_only():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "results" / "score_log.csv").read_text().strip().splitlines()
    assert text[0].split(",") == metrics.LOG_COLUMNS
