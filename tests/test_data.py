import numpy as np
import pytest

from xsml import data


def test_load_good_hash_returns_kept_rows_with_verdict_columns(data_dir, toy):
    df = data.load(data_dir=data_dir)
    assert len(df) == int(toy.keep.sum()) == len(toy) - 2
    assert df.keep.all()
    for c in ("ds_flag", "ds_offset_dex", "ds_weight", "pt_outlier", "entry_date", "reaction"):
        assert c in df.columns
    assert len(data.load(kept_only=False, data_dir=data_dir)) == len(toy)


def test_load_bad_hash_raises_with_instruction(data_dir):
    with open(data_dir / "exfor_linted_toy.parquet", "ab") as f:
        f.write(b"x")
    with pytest.raises(data.DataFileError, match="SHA-256 mismatch") as e:
        data.load(data_dir=data_dir)
    assert "Download" in str(e.value)


def test_load_missing_file_raises_with_instruction(data_dir):
    (data_dir / "exfor_linted_toy.parquet").unlink()
    with pytest.raises(data.DataFileError, match="not found") as e:
        data.load(data_dir=data_dir)
    assert "data/README.md" in str(e.value)


def test_manifest_of_repository_is_consistent():
    m = data.manifest()
    assert m["file"].startswith("exfor_linted_") and len(m["sha256"]) == 64 and m["build_date"] and m["source"]["frame"]


def test_isotope_id_roundtrip():
    assert data.isotope_id(17, 35) == 17035
    ids = data.isotope_id(np.array([17, 92]), np.array([35, 238]))
    assert ids.tolist() == [17035, 92238]
    z, a = data.split_isotope_id(ids)
    assert z.tolist() == [17, 92] and a.tolist() == [35, 238]


def test_select_by_target_and_reaction(toy):
    s = data.select(toy, projectile="n", mt=103, target="Cl-35")
    assert len(s) == 15 and set(s.Z) == {17} and set(s.A) == {35} and set(s.MT) == {103}
    assert len(data.select(toy, target=(17, 35))) == 30
    assert len(data.select(toy, mt=103, target="35Cl", e_min=1e6)) == (s.energy_eV >= 1e6).sum()
    assert s.index.isin(toy.index).all()
    assert len(data.select(toy, mt=103, plain=True)) == 12 * 15
    with pytest.raises(ValueError):
        data.select(toy, target="Xx-1")


def test_datasets_summary(toy):
    d = data.datasets(data.select(toy, mt=103))
    assert len(d) == 12 and (d.points == 15).all() and {"author", "year", "entry_date", "ds_flag"} <= set(d.columns)


def test_check_inputs_rejects_verdict_and_uncertainty_columns(toy):
    data.check_inputs(["log10_energy", "Z", "N", "A", "S_n", "Q_reaction"])
    data.check_inputs(toy[["Z", "A", "energy_eV"]])
    with pytest.raises(ValueError, match="ds_offset_dex") as e:
        data.check_inputs(["Z", "ds_offset_dex", "keep"])
    assert "keep" in str(e.value) and "conventions.md" in str(e.value)
    with pytest.raises(ValueError, match="unc_b"):
        data.check_inputs(["Z", "unc_b"])
    with pytest.raises(ValueError):
        data.check_inputs(toy)                                      # the whole table holds both kinds
    assert set(data.VERDICT_COLUMNS) <= set(toy.columns) and set(data.UNCERTAINTY_COLUMNS) <= set(toy.columns)
    assert {"keep", "reason", "ds_flag", "ds_offset_dex", "ds_weight", "pt_outlier"} <= set(data.VERDICT_COLUMNS)
