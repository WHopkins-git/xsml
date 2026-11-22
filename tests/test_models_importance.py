import numpy as np
import pandas as pd
import pytest

from xsml import features, importance, models, splits
from xsml.data import isotope_id

FEATS = ["log10_energy", "Z", "N", "N_minus_Z", "S_n"]


@pytest.fixture(scope="module")
def frame(toy):
    df = features.append(toy[toy.MT == 103].reset_index(drop=True))
    df["iso"] = isotope_id(df.Z.to_numpy(), df.A.to_numpy())
    return df


def test_constructors_return_plain_estimators():
    assert type(models.decision_tree(max_depth=3)).__name__ == "DecisionTreeRegressor"
    assert models.decision_tree(max_depth=3).max_depth == 3 and models.decision_tree().random_state == 0
    assert models.random_forest(n_estimators=7).n_estimators == 7
    assert type(models.xgboost(n_estimators=5)).__name__ == "XGBRegressor"


def test_cv_rmse_and_oof(frame):
    cv = splits.fold_splits(frame, 4)
    score, oof = models.cv_rmse(models.decision_tree(min_samples_leaf=2), frame[FEATS], frame.log10_sigma, cv)
    assert np.isfinite(oof).all() and 0 < score < 1


@pytest.mark.parametrize("method", ["optuna", "random"])
def test_tune(frame, method):
    cv = splits.fold_splits(frame, 4)
    space = {"max_depth": ("int", 1, 6), "min_samples_leaf": ("logint", 1, 20)}
    res = models.tune(models.decision_tree, space, frame[FEATS], frame.log10_sigma, cv, method=method, n_trials=8, seed=0)
    assert set(res["best_params"]) == {"max_depth", "min_samples_leaf"} and len(res["trials"]) == 8
    assert res["best_score"] == pytest.approx(res["trials"]["value"].min())
    assert res["model"].get_params()["max_depth"] == res["best_params"]["max_depth"]
    again = models.tune(models.decision_tree, space, frame[FEATS], frame.log10_sigma, cv, method=method, n_trials=8, seed=0, refit=False)
    assert again["best_params"] == res["best_params"] and again["model"] is None


def test_nested_cv(frame):
    space = {"max_depth": ("int", 1, 5)}
    res = models.nested_cv(models.decision_tree, space, frame[FEATS], frame.log10_sigma, frame.iso, n_outer=4, n_inner=3, n_trials=4, method="random")
    assert np.isfinite(res["oof"]).all() and len(res["best_params"]) == 4
    assert (res["fold"] == splits.isotope_folds(frame, 4)).all()


def _planted():
    """Forty isotopes; y depends on 'signal' only; 'noise' is an unrelated isotope property; 'x' varies within an isotope."""
    rng = np.random.default_rng(0)
    iso = np.repeat(np.arange(40), 12)
    signal, noise = rng.normal(size=40)[iso], rng.normal(size=40)[iso]
    x = np.tile(np.linspace(0, 1, 12), 40)
    X = pd.DataFrame({"signal": signal, "noise": noise, "x": x})
    y = 2 * signal + 0.5 * x + rng.normal(scale=0.05, size=len(iso))
    return X, y, iso


def test_block_permutation_finds_the_planted_feature():
    X, y, iso = _planted()
    tr, te = iso < 30, iso >= 30
    m = models.random_forest(n_estimators=50, min_samples_leaf=2, max_features=None, n_jobs=1).fit(X[tr], y[tr])
    s = importance.block_permutation(m, X[te], y[te], iso[te], "signal", n_repeats=10)
    n = importance.block_permutation(m, X[te], y[te], iso[te], ["noise"], n_repeats=10)
    assert s["mean"] > 0.5 and abs(n["mean"]) < 0.1 and len(s["values"]) == 10
    with pytest.raises(ValueError, match="not constant"):
        importance.block_permutation(m, X[te], y[te], iso[te], "x")


def test_drop_column_with_intervals():
    X, y, iso = _planted()
    cv = splits.fold_splits(iso, 5)
    make = lambda: models.random_forest(n_estimators=30, min_samples_leaf=2, max_features=None, n_jobs=1)  # noqa: E731
    t = importance.drop_column(make, X, y, iso, cv, {"signal": ["signal"], "noise": ["noise"]}, n_boot=300).set_index("block")
    assert t.loc["signal", "low"] > 0 and t.loc["signal", "delta_rmse"] > 0.5
    assert t.loc["noise", "low"] < 0.05 and abs(t.loc["noise", "delta_rmse"]) < 0.1
    assert (t.n_groups == 40).all() and (t.low <= t.delta_rmse).all() and (t.delta_rmse <= t.high).all()


@pytest.mark.parametrize("make", [lambda: models.decision_tree(max_depth=4), lambda: models.random_forest(n_estimators=10, n_jobs=1),
                                  lambda: models.xgboost(n_estimators=20, max_depth=3, n_jobs=1)])
def test_shap_values_add_up_to_the_prediction(make):
    X, y, iso = _planted()
    m = make().fit(X, y)
    sv = importance.shap_values(m, X.iloc[:25])
    assert list(sv.columns) == ["signal", "noise", "x", "base"]
    assert np.allclose(sv.sum(axis=1), m.predict(X.iloc[:25]), atol=1e-3)
    assert sv.signal.abs().mean() > sv.noise.abs().mean()


def test_log_weights_known_values_clipping_and_missing():
    df = pd.DataFrame({"dataset": ["a", "a", "a", "b", "b", "c"], "sigma_b": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                       "unc_b": [0.10, 0.10, np.nan, 0.001, 0.0, np.nan]})
    w = models.log_weights(df, floor=0.02)
    raw = np.array([(np.log(10) / 0.10) ** 2, (np.log(10) / 0.05) ** 2, np.nan, (np.log(10) / 0.02) ** 2, np.nan, np.nan])   # 0.1 % is raised to the 2 % floor
    raw[2] = np.median(raw[[0, 1]])                    # missing: median of the usable rows of dataset a
    raw[4] = raw[3]                                    # zero uncertainty: median of dataset b, which has one usable row
    raw[5] = np.median(raw[[0, 1, 3]])                 # dataset c has no usable row: median of the frame
    assert np.allclose(w, raw / raw.mean())
    assert w.mean() == pytest.approx(1.0) and (w > 0).all() and np.isfinite(w).all()
    assert w[1] / w[0] == pytest.approx(4.0)           # half the relative uncertainty, four times the weight
    assert w[3] / w[0] == pytest.approx(25.0)          # capped by the floor: (10 % / 2 %) squared, not (10 % / 0.1 %) squared
    assert np.allclose(models.log_weights(df, floor=0.5), np.ones(6))        # everything at the floor: equal weights
    nothing = models.log_weights(df.assign(unc_b=np.nan))
    assert np.allclose(nothing, 1.0)
    with pytest.raises(ValueError):
        models.log_weights(df, floor=0)


def test_log_weights_are_accepted_as_sample_weight(frame):
    cv = splits.fold_splits(frame, 4)
    w = models.log_weights(frame)
    assert len(w) == len(frame) and w.mean() == pytest.approx(1.0)
    score, oof = models.cv_rmse(models.decision_tree(min_samples_leaf=2), frame[FEATS], frame.log10_sigma, cv, sample_weight=w)
    assert np.isfinite(oof).all() and score > 0
