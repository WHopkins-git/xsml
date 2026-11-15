"""Scores, intervals and the score log.

All scores are computed from residuals of log10(cross section): ``r = log10(predicted) - log10(measured)``.
A factor of 2 means "wrong by a factor of two in either direction".

Intervals come from a cluster bootstrap: whole groups (isotopes by default) are resampled with replacement,
because the points of one isotope are not independent observations. With few groups the interval is wide,
and with one group no interval exists; the functions say so instead of returning a number.
"""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["rms_factor", "median_factor", "per_group", "cluster_bootstrap", "paired_comparison", "LOG_COLUMNS", "log_score"]


def _resid(log_true, log_pred) -> np.ndarray:
    r = np.asarray(log_pred, dtype=float) - np.asarray(log_true, dtype=float)
    if r.size == 0:
        raise ValueError("no points to score")
    if not np.isfinite(r).all():
        raise ValueError("non-finite residuals: check for zero or negative cross sections and NaN predictions")
    return r


def rms_factor(log_true, log_pred) -> float:
    """``10 ** RMSE`` of log10 cross section. Arguments are log10 values."""
    return float(10 ** np.sqrt(np.mean(_resid(log_true, log_pred) ** 2)))


def median_factor(log_true, log_pred) -> float:
    """``10 ** median(|residual|)`` of log10 cross section. Arguments are log10 values."""
    return float(10 ** np.median(np.abs(_resid(log_true, log_pred))))


def per_group(log_true, log_pred, groups, metric=rms_factor) -> pd.Series:
    """The metric of every group (isotope by default), as a Series indexed by group label."""
    t, p, g = np.asarray(log_true, float), np.asarray(log_pred, float), np.asarray(groups)
    return pd.Series({k: metric(t[g == k], p[g == k]) for k in sorted(set(g.tolist()))}, name=getattr(metric, "__name__", "metric"))


def _blocks(groups) -> list[np.ndarray]:
    g = np.asarray(groups)
    return [np.flatnonzero(g == k) for k in sorted(set(g.tolist()))]


def cluster_bootstrap(log_true, log_pred, groups, metric=rms_factor, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """Point estimate and percentile interval of ``metric`` under resampling of whole groups.

    Returns ``{"value", "low", "high", "n_groups", "n_points", "level"}``. The point estimate pools all points.
    Raises ValueError with fewer than two groups: a single group gives no information on group-to-group spread.
    """
    t, p = np.asarray(log_true, float), np.asarray(log_pred, float)
    blocks = _blocks(groups)
    if len(blocks) < 2:
        raise ValueError("cluster bootstrap needs at least two groups; with one group report the score without an interval and say why")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        stats[b] = metric(t[idx], p[idx])
    a = (1 - level) / 2
    return {"value": metric(t, p), "low": float(np.quantile(stats, a)), "high": float(np.quantile(stats, 1 - a)),
            "n_groups": len(blocks), "n_points": int(t.size), "level": level}


def paired_comparison(log_true, log_pred_a, log_pred_b, groups, metric=rms_factor, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """Compare two sets of predictions of the same points: ratio ``metric(A) / metric(B)`` with a paired interval.

    Each bootstrap sample draws the same groups for A and for B, so that the difficulty of the drawn isotopes
    cancels. A ratio below 1 favours A (for a metric where lower is better). ``frac_a_better`` is the share of
    resamples with a ratio below 1. If the interval of the ratio contains 1, the data do not show a difference.
    """
    t, a, b = np.asarray(log_true, float), np.asarray(log_pred_a, float), np.asarray(log_pred_b, float)
    blocks = _blocks(groups)
    if len(blocks) < 2:
        raise ValueError("a paired comparison over groups needs at least two groups")
    rng = np.random.default_rng(seed)
    ratio = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        ratio[i] = metric(t[idx], a[idx]) / metric(t[idx], b[idx])
    q = (1 - level) / 2
    return {"metric_a": metric(t, a), "metric_b": metric(t, b), "ratio": metric(t, a) / metric(t, b), "low": float(np.quantile(ratio, q)),
            "high": float(np.quantile(ratio, 1 - q)), "frac_a_better": float(np.mean(ratio < 1)), "n_groups": len(blocks), "level": level}


#: Columns of results/score_log.csv, in order.
LOG_COLUMNS = ["timestamp", "commit", "dirty", "model", "features", "split", "metric", "value", "ci_low", "ci_high", "n_groups", "n_points", "declared", "notes"]


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def log_score(path: Path | str = "results/score_log.csv", **fields) -> dict:
    """Append one score to the score log and return the row written.

    Required fields: ``model``, ``features``, ``split``, ``metric``, ``value``. Optional: ``ci_low``, ``ci_high``,
    ``n_groups``, ``n_points``, ``declared`` (``"protocol"`` for the one score a protocol allows, ``"extra"`` for any
    other run; default ``"extra"``) and ``notes``. ``features`` may be a list of column names. The timestamp (UTC),
    the git commit of the working directory and whether it had uncommitted changes are added automatically.
    Every score that is computed is logged, including the ones that are not reported.
    """
    required = ["model", "features", "split", "metric", "value"]
    missing = [k for k in required if k not in fields]
    unknown = [k for k in fields if k not in LOG_COLUMNS or k in ("timestamp", "commit", "dirty")]
    if missing or unknown:
        raise ValueError(f"log_score: missing fields {missing}, unknown fields {unknown}; allowed: {LOG_COLUMNS[3:]}")
    path = Path(path)
    if isinstance(fields["features"], (list, tuple)):
        fields["features"] = " ".join(map(str, fields["features"]))
    fields.setdefault("declared", "extra")
    if fields["declared"] not in ("protocol", "extra"):
        raise ValueError("declared must be 'protocol' or 'extra'")
    cwd = path.resolve().parent
    row = {"timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "commit": _git(["rev-parse", "HEAD"], cwd) or "unknown",
           "dirty": {True: "yes", False: "no"}[bool(_git(["status", "--porcelain", "--untracked-files=no"], cwd))], **fields}
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        if header != LOG_COLUMNS:
            raise ValueError(f"{path} has another header than expected: {header}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, restval="", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    return row
