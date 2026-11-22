import numpy as np
import pytest

from xsml import data, features, libraries, metrics, models, splits


def test_shipped_library_curves():
    a = libraries.available()
    assert {"ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"} <= set(a.library) and {"Cl35_np.csv", "Fe56_ninl.csv", "Sm152_ng.csv", "Sm152_ng_bins.csv"} <= set(a.file)
    assert set(a.library[a.file == "Sm152_ng.csv"]) == {"ENDF-B-VIII.0"} and len(a[a.file == "Fe56_ninl.csv"]) == 3
    assert set(a.form[a.file.str.endswith("_bins.csv")]) == {"bin averages"} and set(a.form[~a.file.str.endswith("_bins.csv")]) == {"pointwise"}
    a = a[a.file == "Cl35_np.csv"]
    c = libraries.curve("Cl-35", 103, "ENDF-B-VIII.0")
    assert c.energy_eV.is_monotonic_increasing and c.energy_eV.min() == 1e-5 and c.energy_eV.max() == 2e7 and (c.sigma_b >= 0).all()
    assert len(libraries.curve("Cl-35", 103)) == a.points.sum()


def test_cl35_np_curves_cover_the_full_range_of_the_evaluations():
    """From 1e-5 eV (the lower limit of the files) to 20 MeV; 1/v below thermal; the points from 0.1 MeV are the NJOY ones."""
    a = libraries.available()
    a = a[a.file == "Cl35_np.csv"]
    assert set(a.library) == {"ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"} and (a.e_min_eV == 1e-5).all() and (a.e_max_eV == 2e7).all()
    from_100_keV = {"ENDF-B-VII.1": 15483, "ENDF-B-VIII.0": 15525, "JENDL-5": 22911}
    low = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.0253])
    for lib, n in from_100_keV.items():
        c = libraries.curve("Cl-35", 103, lib)
        assert (c.energy_eV >= 1e5).sum() == n and (c.sigma_b[c.energy_eV < 1e5] > 0).all()
        s = libraries.at(low, lib)
        assert s[-1] == pytest.approx(0.479, rel=2e-3)                          # barns at 0.0253 eV, as calculated from each file
        assert s * np.sqrt(low) == pytest.approx(s[-1] * np.sqrt(0.0253), rel=1e-3)   # 1/v: no threshold, positive Q-value
        assert libraries.at([397.8154], lib)[0] > 50                            # the resonance at 398 eV is resolved
        assert np.isnan(libraries.at([0.9e-5, 2.1e7], lib)).all()


def test_library_interpolation():
    c = libraries.curve("Cl-35", 103, "JENDL-5")
    c = c[c.sigma_b > 0]
    i = len(c) // 2
    e = c.energy_eV.to_numpy()
    assert libraries.at([e[i]], "JENDL-5")[0] == pytest.approx(c.sigma_b.iloc[i], rel=1e-6)
    mid = libraries.at([np.sqrt(e[i] * e[i + 1])], "JENDL-5")[0]
    assert min(c.sigma_b.iloc[i], c.sigma_b.iloc[i + 1]) <= mid <= max(c.sigma_b.iloc[i], c.sigma_b.iloc[i + 1])
    assert np.isnan(libraries.at([5e-6, 3e7], "JENDL-5")).all()                # outside 1e-5 eV to 20 MeV


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
    from pathlib import Path

    files = sorted((Path(libraries.DATA_DIR) / "libraries").glob("*.csv"))
    assert len(files) >= 90
    listed = libraries.available()
    assert set(listed.file) == {f.name for f in files}
    assert set(listed.library) == set(libraries.LIBRARIES)
    assert set(listed.basis[listed.library == "TENDL-2025"]) == {"model calculation"} and set(listed.basis[listed.library != "TENDL-2025"]) == {"evaluation"}
    for f in files:
        info = libraries.parse_file_name(f.name)
        target, mt, projectile = (info["Z"], info["A"]), info["MT"], info["projectile"]
        if info["binned"]:
            t = libraries.bin_means(target, mt, projectile=projectile)
            assert list(t.columns) == ["library", "e_low_eV", "e_high_eV", "sigma_b"], f.name
            for lib, g in t.groupby("library"):
                assert g.e_low_eV.is_monotonic_increasing and (g.e_high_eV > g.e_low_eV).all() and (g.sigma_b > 0).all(), (f.name, lib)
                assert np.isfinite(g[["e_low_eV", "e_high_eV", "sigma_b"]].to_numpy()).all() and g.e_high_eV.max() <= 2e7, (f.name, lib)
        else:
            t = libraries.curve(target, mt, projectile=projectile, state=info["state"])
            assert list(t.columns) == ["library", "energy_eV", "sigma_b"], f.name
            for lib, g in t.groupby("library"):
                top = 3e7 if lib == "TENDL-2025" else 2e7
                assert len(g) >= 5 and g.energy_eV.is_monotonic_increasing and (g.energy_eV > 0).all() and g.energy_eV.max() <= top, (f.name, lib)
                assert (g.sigma_b >= 0).all() and (g.sigma_b > 0).any() and np.isfinite(g.sigma_b).all(), (f.name, lib)
                positive = g[g.sigma_b > 0]
                middle = np.sqrt(positive.energy_eV.iloc[len(positive) // 2 - 1] * positive.energy_eV.iloc[len(positive) // 2])
                assert libraries.at([middle], lib, target, mt, projectile=projectile, state=info["state"])[0] > 0, (f.name, lib)
        assert set(t.library) <= set(libraries.LIBRARIES)
        # the neutron files hold the three evaluations only, the proton files the model library only
        assert (set(t.library) == {"TENDL-2025"}) == (projectile == "p"), f.name


def test_tendl_proton_curves_start_at_the_threshold_from_ame():
    """Each proton curve starts at the threshold that AME2020 gives for its reaction (isomer: above it, by the level energy)."""
    from xsml import features

    listed = libraries.available()
    listed = listed[listed.projectile == "p"]
    assert len(listed) >= 60 and set(listed.library) == {"TENDL-2025"} and set(listed.state) == {"", "g", "m"}
    masses = features.ame_table().set_index(["Z", "A"]).atomic_mass_u
    for row in listed.itertuples():
        info = libraries.parse_file_name(row.file)
        q = features.q_value(info["Z"], info["A"], "p", features.MT_EJECTILES[info["MT"]])
        c = libraries.curve((info["Z"], info["A"]), info["MT"], "TENDL-2025", projectile="p", state=info["state"])
        first = c.energy_eV.iloc[0]
        if q >= 0:
            assert first < 1e6, row.file                                                           # no threshold: the table starts low
            continue
        threshold = -q * 1e6 * (masses[(info["Z"], info["A"])] + 1.00727647) / masses[(info["Z"], info["A"])]
        assert c.sigma_b.iloc[0] == 0, row.file
        if row.state == "m":
            assert threshold + 1e3 < first < threshold + 1e6, (row.file, first, threshold)
        else:
            assert first == pytest.approx(threshold, rel=2e-4, abs=2e3), (row.file, first, threshold)


def test_tendl_ground_state_and_isomer_add_up_to_the_total():
    total = libraries.curve("Mo-100", 16, "TENDL-2025", projectile="p")
    ground = libraries.curve("Mo-100", 16, "TENDL-2025", projectile="p", state="g")
    isomer = libraries.curve("Mo-100", 16, "TENDL-2025", projectile="p", state="m")
    grid = total.energy_eV.to_numpy()
    summed = np.interp(grid, ground.energy_eV, ground.sigma_b, left=0) + np.interp(grid, isomer.energy_eV, isomer.sigma_b, left=0)
    assert np.allclose(summed, total.sigma_b, atol=1e-6)
    assert isomer.energy_eV.iloc[0] - ground.energy_eV.iloc[0] == pytest.approx(142.7e3 * 101 / 100, rel=0.01)     # Tc-99m lies 142.7 keV up
    assert libraries.parse_file_name("Mo100_p2n_m.csv") == {"Z": 42, "A": 100, "target": "Mo-100", "projectile": "p", "MT": 16, "state": "m", "binned": False}
    assert libraries.parse_file_name("Fe56_ng_bins.csv")["binned"] and libraries.parse_file_name("Cl35_np.csv")["projectile"] == "n"
    with pytest.raises(FileNotFoundError):
        libraries.curve("Mo-100", 16, projectile="p", state="m2")


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
