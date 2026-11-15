"""Feature importance with the isotope as the unit.

Three tools, which answer different questions:

* :func:`block_permutation`: how much does a trained model lean on a block of features? The block is swapped
  between isotopes (every isotope receives the values of another one), on held-out isotopes.
* :func:`drop_column`: should a block of features be given to the model at all? The model is retrained
  without it and compared on held-out isotopes, with a paired interval over isotopes.
* :func:`shap_values`: additive contributions of each feature to each prediction of a tree model.

Correlated features share importance in ways none of these tools can separate; group such features into one
block. Importance on training rows says what a model memorised, not what generalises.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

__all__ = ["block_permutation", "drop_column", "shap_values"]


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def block_permutation(model, X: pd.DataFrame, y, groups, block: list[str] | str, n_repeats: int = 20, seed: int = 0) -> dict:
    """Rise of the RMSE of ``y`` when the features in ``block`` are swapped between groups (isotopes).

    ``model`` is already fitted; ``X``, ``y`` and ``groups`` should be rows the model did not train on. In each
    repeat every group receives the block values of another group, so that the rows of a group stay together and
    the features of the block stay consistent with each other. The features of the block must be constant within
    a group (properties of the isotope); for a feature that varies from row to row, such as the energy, use
    ``sklearn.inspection.permutation_importance`` and read it with the caveats in the module docstring.

    Returns ``{"mean", "std", "values"}``: the rise per repeat (same unit as ``y``, log10 if ``y`` is log10 sigma).
    """
    block = [block] if isinstance(block, str) else list(block)
    g = pd.Series(np.asarray(groups), index=X.index)
    if g.nunique() < 2:
        raise ValueError("block permutation needs at least two groups")
    if (X[block].groupby(g).nunique(dropna=False) > 1).any().any():
        raise ValueError(f"features {block} are not constant within a group; block permutation over isotopes does not apply to them")
    values = X[block].groupby(g).first()
    base = _rmse(model.predict(X), y)
    rng = np.random.default_rng(seed)
    rise = []
    for _ in range(n_repeats):
        donor = dict(zip(values.index, rng.permutation(values.index)))
        swapped = X.copy()
        swapped[block] = values.loc[g.map(donor)].to_numpy()
        rise.append(_rmse(model.predict(swapped), y) - base)
    rise = np.array(rise)
    return {"mean": float(rise.mean()), "std": float(rise.std(ddof=1)) if n_repeats > 1 else float("nan"), "values": rise}


def drop_column(make_model: Callable, X: pd.DataFrame, y, groups, splits, blocks: dict[str, list[str]] | list[str], n_boot: int = 2000, level: float = 0.95,
                seed: int = 0) -> pd.DataFrame:
    """Retrain without each block of features and compare on held-out groups.

    ``make_model`` returns a new unfitted estimator; ``splits`` is a list of ``(train_positions, test_positions)``
    that keeps groups together (:func:`xsml.splits.fold_splits`); ``blocks`` maps a name to the columns to drop
    together (a list of column names drops them one at a time).

    For every group the RMSE of ``y`` is computed from out-of-fold predictions, with all features and without the
    block. The table gives, per block, the mean over groups of (RMSE without - RMSE with) and its percentile
    interval from resampling groups. A positive value means the block helps; an interval that contains 0 means
    the data do not show that it does.
    """
    if not isinstance(blocks, dict):
        blocks = {c: [c] for c in blocks}
    y = np.asarray(y, dtype=float)
    g = np.asarray(groups)

    def per_group_rmse(cols: list[str]) -> pd.Series:
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            m = make_model() if not hasattr(make_model, "fit") else clone(make_model)
            m.fit(X.iloc[tr][cols], y[tr])
            oof[te] = m.predict(X.iloc[te][cols])
        ok = ~np.isnan(oof)
        return (pd.Series((oof[ok] - y[ok]) ** 2).groupby(g[ok]).mean()) ** 0.5

    full = per_group_rmse(list(X.columns))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(full), (n_boot, len(full)))
    q = (1 - level) / 2
    rows = []
    for name, cols in blocks.items():
        rest = [c for c in X.columns if c not in cols]
        if not rest:
            raise ValueError(f"dropping {name} leaves no feature")
        d = (per_group_rmse(rest) - full).to_numpy()
        means = d[draws].mean(axis=1)
        rows.append({"block": name, "dropped": " ".join(cols), "delta_rmse": float(d.mean()), "low": float(np.quantile(means, q)),
                     "high": float(np.quantile(means, 1 - q)), "n_groups": len(d)})
    return pd.DataFrame(rows).sort_values("delta_rmse", ascending=False).reset_index(drop=True)


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values of a fitted tree model (scikit-learn tree or forest, XGBoost) for the rows of ``X``.

    Returns a DataFrame with one column per feature plus ``base``; the columns of a row add up to the model's
    prediction for that row. Uses ``shap.TreeExplainer`` (path-dependent, no background data). Explain held-out
    isotopes with a model that did not train on them if the question is about generalisation.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    out = pd.DataFrame(values, columns=list(X.columns), index=X.index)
    out["base"] = float(np.ravel(explainer.expected_value)[0])
    return out
