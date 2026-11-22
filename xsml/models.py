"""Model constructors and a tuning wrapper.

The constructors return ordinary scikit-learn and XGBoost estimators with fixed seeds; every keyword
argument is passed through, and nothing about the estimator is hidden.

:func:`tune` searches hyperparameters with cross-validation over isotope-grouped folds
(:func:`xsml.splits.fold_splits`). :func:`nested_cv` wraps it in an outer loop, so that the score of the
tuned model comes from isotopes that the tuning never saw.

A search space is a dictionary ``{parameter: specification}`` with specifications

    ("int", low, high)      integer, uniform
    ("float", low, high)    real, uniform
    ("log", low, high)      real, log-uniform
    ("logint", low, high)   integer, log-uniform
    ("cat", [a, b, ...])    one of the listed values
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .splits import fold_splits

__all__ = ["decision_tree", "random_forest", "xgboost", "log_weights", "cv_rmse", "tune", "nested_cv"]


def decision_tree(**kwargs) -> DecisionTreeRegressor:
    """``sklearn.tree.DecisionTreeRegressor`` with ``min_samples_leaf=5`` and ``random_state=0`` unless overridden."""
    return DecisionTreeRegressor(**{"min_samples_leaf": 5, "random_state": 0, **kwargs})


def random_forest(**kwargs) -> RandomForestRegressor:
    """``sklearn.ensemble.RandomForestRegressor``: 200 trees, ``min_samples_leaf=5``, a third of the features per split."""
    return RandomForestRegressor(**{"n_estimators": 200, "min_samples_leaf": 5, "max_features": 1 / 3, "n_jobs": -1, "random_state": 0, **kwargs})


def xgboost(**kwargs):
    """``xgboost.XGBRegressor``: 300 trees of depth 6, learning rate 0.05, row subsampling 0.8, histogram method."""
    from xgboost import XGBRegressor

    return XGBRegressor(**{"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "tree_method": "hist", "n_jobs": -1, "random_state": 0, **kwargs})


def log_weights(df: pd.DataFrame, floor: float = 0.02) -> np.ndarray:
    """Fitting weights for log10(sigma) from the reported uncertainties, one per row of ``df``, with mean 1.

    The uncertainty of log10(sigma) is taken as ``(unc_b / sigma_b) / ln(10)`` and the weight as its inverse
    square. The relative uncertainty is not allowed below ``floor`` (2 % by default): what an uncertainty includes
    (statistical only, or total) is not recorded in the table, about one row in six with an uncertainty quotes less
    than 2 %, and without a floor a few datasets that quote very small errors would carry the fit. Rows without a
    usable uncertainty (missing, zero or negative; about 11 % of the kept rows) receive the median weight of the
    other rows of their dataset, or, where the dataset has none, the median weight of the frame.

    Pass the result as ``sample_weight`` to :func:`cv_rmse`, :func:`tune` or an estimator's ``fit``. Weighting is a
    modelling choice to be declared; the uncertainty itself is not a model input (``xsml.data.check_inputs``).
    Needs the columns ``sigma_b``, ``unc_b`` and ``dataset``.
    """
    if not 0 < floor:
        raise ValueError("floor must be positive")
    relative = (df["unc_b"] / df["sigma_b"]).to_numpy(dtype=float)
    usable = np.isfinite(relative) & (relative > 0)
    weight = np.full(len(df), np.nan)
    weight[usable] = 1.0 / (np.maximum(relative[usable], floor) / np.log(10)) ** 2
    if not usable.any():
        return np.ones(len(df))
    w = pd.Series(weight, index=df.index)
    by_dataset = w.groupby(df["dataset"].astype(str), observed=True).transform("median")      # NaN where a dataset has no usable row
    w = w.fillna(by_dataset).fillna(float(np.median(weight[usable])))
    return (w / w.mean()).to_numpy()


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def cv_rmse(model, X, y, splits, sample_weight=None) -> tuple[float, np.ndarray]:
    """Mean over the folds of the RMSE of ``y`` on the held-out fold, and the out-of-fold predictions.

    ``splits`` is a list of ``(train_positions, test_positions)``. Rows in no test fold keep NaN in the predictions.
    """
    y = np.asarray(y, dtype=float)
    oof, scores = np.full(len(y), np.nan), []
    for tr, te in splits:
        m = clone(model)
        m.fit(_rows(X, tr), y[tr], **({} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)[tr]}))
        oof[te] = m.predict(_rows(X, te))
        scores.append(float(np.sqrt(np.mean((oof[te] - y[te]) ** 2))))
    return float(np.mean(scores)), oof


def _suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "logint":
            out[name] = trial.suggest_int(name, spec[1], spec[2], log=True)
        elif kind == "float":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "log":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "cat":
            out[name] = trial.suggest_categorical(name, list(spec[1]))
        else:
            raise ValueError(f"unknown specification {spec!r} for {name}")
    return out


def _distributions(space: dict) -> dict:
    from scipy import stats

    class _LogInt:
        def __init__(self, lo, hi):
            self.d = stats.loguniform(lo, hi)

        def rvs(self, *args, **kwargs):
            return np.round(self.d.rvs(*args, **kwargs)).astype(int)

    out = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[name] = stats.randint(spec[1], spec[2] + 1)
        elif kind == "logint":
            out[name] = _LogInt(spec[1], spec[2])
        elif kind == "float":
            out[name] = stats.uniform(spec[1], spec[2] - spec[1])
        elif kind == "log":
            out[name] = stats.loguniform(spec[1], spec[2])
        elif kind == "cat":
            out[name] = list(spec[1])
        else:
            raise ValueError(f"unknown specification {spec!r} for {name}")
    return out


def tune(make_model: Callable, space: dict, X, y, splits, method: str = "optuna", n_trials: int = 30, seed: int = 0, sample_weight=None, refit: bool = True) -> dict:
    """Hyperparameter search by cross-validation over the given folds, minimising the mean RMSE of ``y`` over the folds.

    Parameters
    ----------
    make_model : callable returning an unfitted estimator from keyword arguments, for example ``xsml.models.xgboost``.
    space : search space (see the module docstring).
    X, y : features (DataFrame or array) and target (log10 cross section).
    splits : list of ``(train_positions, test_positions)``, normally ``xsml.splits.fold_splits(df, n_folds)``.
    method : ``"optuna"`` (TPE sampler) or ``"random"`` (``sklearn.model_selection.RandomizedSearchCV``).
    refit : fit the best model on all rows of ``X`` and return it as ``"model"``.

    Returns a dictionary: ``best_params``, ``best_score`` (mean RMSE over the folds), ``trials`` (DataFrame of all
    trials), ``model`` (refitted, or None) and ``search`` (the optuna study or the RandomizedSearchCV object).

    The best score is optimistically biased as an estimate of performance: it was selected. Score the tuned
    model on rows that took no part in the search (:func:`nested_cv`, or a hold-out that is touched once).
    """
    y = np.asarray(y, dtype=float)
    if method == "optuna":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda trial: cv_rmse(make_model(**_suggest(trial, space)), X, y, splits, sample_weight)[0], n_trials=n_trials)
        best, score, search = dict(study.best_params), float(study.best_value), study
        trials = study.trials_dataframe(attrs=("number", "value", "params"))
    elif method == "random":
        from sklearn.model_selection import RandomizedSearchCV

        search = RandomizedSearchCV(make_model(), _distributions(space), n_iter=n_trials, cv=list(splits), scoring="neg_root_mean_squared_error",
                                    random_state=seed, refit=False)
        search.fit(X, y, **({} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)}))
        best, score = dict(search.best_params_), float(-search.best_score_)
        trials = pd.DataFrame(search.cv_results_["params"]).assign(value=-search.cv_results_["mean_test_score"])
    else:
        raise ValueError("method must be 'optuna' or 'random'")
    model = None
    if refit:
        model = make_model(**best)
        model.fit(X, y, **({} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)}))
    return {"best_params": best, "best_score": score, "trials": trials, "model": model, "search": search}


def nested_cv(make_model: Callable, space: dict, X, y, isotope_ids, n_outer: int = 5, n_inner: int = 4, **tune_kwargs) -> dict:
    """Nested cross-validation over isotopes.

    Outer loop: ``n_outer`` isotope-grouped folds. For each, the hyperparameters are tuned with :func:`tune` on
    ``n_inner`` isotope-grouped folds of the outer training isotopes only, the best model is refitted on those
    rows and predicts the outer test fold. Returns ``{"oof": out-of-fold predictions, "fold": outer fold of each
    row, "best_params": list, "inner_scores": list}``. Score ``oof`` against ``y`` with :mod:`xsml.metrics`.
    """
    iso = np.asarray(isotope_ids)
    y = np.asarray(y, dtype=float)
    outer = fold_splits(iso, n_outer)
    oof, fold, params, inner_scores = np.full(len(y), np.nan), np.full(len(y), -1), [], []
    for k, (tr, te) in enumerate(outer):
        inner = fold_splits(iso[tr], n_inner)
        res = tune(make_model, space, _rows(X, tr), y[tr], inner, refit=True, **tune_kwargs)
        oof[te], fold[te] = res["model"].predict(_rows(X, te)), k
        params.append(res["best_params"])
        inner_scores.append(res["best_score"])
    return {"oof": oof, "fold": fold, "best_params": params, "inner_scores": inner_scores}
