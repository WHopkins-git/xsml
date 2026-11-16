import numpy as np
import pytest

from xsml import data, features, libraries, metrics, models, splits


def test_shipped_library_curves():
    a = libraries.available()
    assert set(a.library) == {"ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"} and {"Cl35_np.csv", "Fe56_ninl.csv", "Sm152_ng.csv", "Sm152_ng_bins.csv"} <= set(a.file)
    assert set(a.library[a.file == "Sm152_ng.csv"]) == {"ENDF-B-VIII.0"} and len(a[a.file == "Fe56_ninl.csv"]) == 3
    assert set(a.form[a.file.str.endswith("_bins.csv")]) == {"bin averages"} and set(a.form[~a.file.str.endswith("_bins.csv")]) == {"pointwise"}
    a = a[a.file == "Cl35_np.csv"]
    c = libraries.curve("Cl-35", 103, "ENDF-B-VIII.0")
    assert c.energy_eV.is_monotonic_increasing and c.energy_eV.min() == 1e5 and c.energy_eV.max() == 2e7 and (c.sigma_b >= 0).all()
    assert len(libraries.curve("Cl-35", 103)) == a.points.sum()


def test_library_interpolation():
    c = libraries.curve("Cl-35", 103, "JENDL-5")
    c = c[c.sigma_b > 0]
    i = len(c) // 2
    e = c.energy_eV.to_numpy()
    assert libraries.at([e[i]], "JENDL-5")[0] == pytest.approx(c.sigma_b.iloc[i], rel=1e-6)
    mid = libraries.at([np.sqrt(e[i] * e[i + 1])], "JENDL-5")[0]
    assert min(c.sigma_b.iloc[i], c.sigma_b.iloc[i + 1]) <= mid <= max(c.sigma_b.iloc[i], c.sigma_b.iloc[i + 1])
    assert np.isnan(libraries.at([1.0, 3e7], "JENDL-5")).all()                 # outside 0.1 to 20 MeV


def test_fe56_inelastic_and_sm152_capture_curves():
    fe = libraries.curve("Fe-56", 4, "ENDF-B-VIII.0")
    assert fe.energy_eV.is_monotonic_increasing and fe.energy_eV.max() == 2e7 and 8.6e5 < fe.energy_eV[fe.sigma_b > 0].min() < 8.7e5
    assert np.isnan(libraries.at([5e5], "ENDF-B-VIII.0", "Fe-56", 4)[0])                      # below the inelastic threshold
    assert 1.0 < libraries.at([5e6], "ENDF-B-VIII.0", "Fe-56", 4)[0] < 2.0                    # barns, plateau of the curve
    sm = libraries.curve("Sm-152", 102)
    assert set(sm.library) == {"ENDF-B-VIII.0"}
    assert libraries.at([0.0253], "ENDF-B-VIII.0", "Sm-152", 102)[0] == pytest.approx(206.0, rel=0.01)   # thermal capture, barns
    with pytest.raises(KeyError):
        libraries.curve("Sm-152", 102, "JENDL-5")                                             # only bin averages are shipped for it


def test_bin_means():
    b = libraries.bin_means("Sm-152", 102)
    assert set(b.library) == {"ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"} and (b.e_high_eV > b.e_low_eV).all() and b.e_high_eV.max() == 2e7
    one = libraries.bin_means("Sm-152", 102, "JENDL-5")
    assert np.allclose(one.e_low_eV.to_numpy()[1:], one.e_high_eV.to_numpy()[:-1], rtol=1e-6)  # contiguous bins
    assert np.allclose(np.log10(one.e_high_eV / one.e_low_eV).to_numpy()[:-1], 0.25, atol=1e-6)
    with pytest.raises(FileNotFoundError):
        libraries.bin_means("Cl-35", 103)


def test_every_shipped_library_file_loads_and_is_well_formed():
    """Every file in data/libraries: readable through the loader, energies increasing, cross sections not negative."""
    import re
    from pathlib import Path

    from xsml.features import element_symbols

    number_of = {name: mt for mt, name in libraries.REACTION_NAMES.items()}
    z_of = {symbol: z for z, symbol in element_symbols().items()}
    files = sorted((Path(libraries.DATA_DIR) / "libraries").glob("*.csv"))
    assert len(files) >= 30
    listed = libraries.available()
    assert set(listed.file) == {f.name for f in files}
    for f in files:
        symbol, mass, reaction = re.fullmatch(r"([A-Z][a-z]?)(\d+)_([a-z0-9]+)(?:_bins)?\.csv", f.name).groups()
        target, mt = (z_of[symbol], int(mass)), number_of[reaction]
        if f.name.endswith("_bins.csv"):
            t = libraries.bin_means(target, mt)
            assert list(t.columns) == ["library", "e_low_eV", "e_high_eV", "sigma_b"], f.name
            for lib, g in t.groupby("library"):
                assert g.e_low_eV.is_monotonic_increasing and (g.e_high_eV > g.e_low_eV).all() and (g.sigma_b > 0).all(), (f.name, lib)
                assert np.isfinite(g[["e_low_eV", "e_high_eV", "sigma_b"]].to_numpy()).all() and g.e_high_eV.max() <= 2e7, (f.name, lib)
        else:
            t = libraries.curve(target, mt)
            assert list(t.columns) == ["library", "energy_eV", "sigma_b"], f.name
            for lib, g in t.groupby("library"):
                assert len(g) >= 10 and g.energy_eV.is_monotonic_increasing and (g.energy_eV > 0).all() and g.energy_eV.max() <= 2e7, (f.name, lib)
                assert (g.sigma_b >= 0).all() and (g.sigma_b > 0).any() and np.isfinite(g.sigma_b).all(), (f.name, lib)
                positive = g[g.sigma_b > 0]
                middle = np.sqrt(positive.energy_eV.iloc[len(positive) // 2] * positive.energy_eV.iloc[len(positive) // 2 + 1])
                assert libraries.at([middle], lib, target, mt)[0] > 0, (f.name, lib)
        assert set(t.library) <= {"ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"}


def test_library_not_shipped():
    with pytest.raises(FileNotFoundError, match="Cl35_np.csv"):
        libraries.curve("Pb-208", 102)
    with pytest.raises(KeyError):
        libraries.curve("Cl-35", 103, "TENDL-2023")


def test_quickstart_end_to_end(data_dir, tmp_path):
    """The README quickstart on the toy table: load, features, isotope folds, fit, score with an interval, log."""
    df = data.load(data_dir=data_dir)
    df = features.append(data.select(df, projectile="n", mt=103), path=data_dir / "mass_1.mas20.txt").reset_index(drop=True)
    X, y, iso = df[["log10_energy", "Z", "N", "S_n", "Q_reaction"]], df.log10_sigma, data.isotope_id(df.Z.to_numpy(), df.A.to_numpy())
    cv = splits.fold_splits(df, 4)
    for tr, te in cv:
        splits.assert_no_leakage(df.iloc[tr], df.iloc[te])
    _, oof = models.cv_rmse(models.decision_tree(min_samples_leaf=2), X, y, cv)
    score = metrics.cluster_bootstrap(y, oof, iso, n_boot=200)
    assert 1 < score["low"] <= score["value"] <= score["high"] < 10 and score["n_groups"] == 12
    row = metrics.log_score(tmp_path / "score_log.csv", model="decision_tree", features=list(X.columns), split="4 isotope folds", metric="rms_factor",
                            value=score["value"], ci_low=score["low"], ci_high=score["high"], n_groups=score["n_groups"], n_points=score["n_points"])
    assert row["value"] == score["value"] and (tmp_path / "score_log.csv").read_text().count("\n") == 2
