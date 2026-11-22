"""Loading the linted EXFOR table.

The table is not stored in git. It is downloaded once into ``data/`` and verified against the
SHA-256 recorded in ``data/MANIFEST.json`` every time it is loaded.

The only selection applied by default is ``keep == True`` (the rows retained by the linter).
``load(kept_only=False)`` returns every row. No other row is removed anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

__all__ = ["DATA_DIR", "VERDICT_COLUMNS", "UNCERTAINTY_COLUMNS", "DataFileError", "manifest", "sha256", "load", "isotope_id", "split_isotope_id", "select", "datasets", "check_inputs"]

#: Folder holding MANIFEST.json and the downloaded parquet. Override with the environment variable XSML_DATA_DIR.
DATA_DIR = Path(os.environ.get("XSML_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


#: Columns that hold a verdict on the data, not a property of the nucleus or of the measurement. They were computed
#: on the whole table, by comparing each dataset with the other datasets of the same reaction, so they carry
#: information about rows that may later be held out. They are not model inputs (see :func:`check_inputs`).
VERDICT_COLUMNS = ["keep", "reason", "ds_flag", "ds_offset_dex", "ds_weight", "pt_outlier", "missing_uncertainty", "relative_uncertainty_flag",
                   "uncertainty_unit_error", "critique_kind"]

#: The reported uncertainty and the flags on it. An uncertainty is a statement about a measured value, known only
#: where a measurement exists, so it is not a model input either; it can serve as a fitting weight
#: (:func:`xsml.models.log_weights`).
UNCERTAINTY_COLUMNS = ["unc_b", "missing_uncertainty", "relative_uncertainty_flag", "uncertainty_unit_error"]


class DataFileError(RuntimeError):
    """The data file is missing or does not match the manifest."""


def manifest(data_dir: Path | str | None = None) -> dict:
    """Contents of ``MANIFEST.json`` (file name, SHA-256, build date, source frames)."""
    path = Path(data_dir or DATA_DIR) / "MANIFEST.json"
    if not path.exists():
        raise DataFileError(f"{path} not found. The package must be installed from a clone of the repository (pip install -e .).")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path | str) -> str:
    """SHA-256 of a file, as a hexadecimal string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _checked_path(data_dir: Path | str | None, verify: bool) -> Path:
    data_dir = Path(data_dir or DATA_DIR)
    m = manifest(data_dir)
    path = data_dir / m["file"]
    how = (f"Download {m['file']} from the link in data/README.md ({m.get('download', 'link not set')}) "
           f"and place it at {path}.")
    if not path.exists():
        raise DataFileError(f"Data file not found: {path}\n{how}")
    if verify:
        got = sha256(path)
        if got != m["sha256"]:
            raise DataFileError(
                f"SHA-256 mismatch for {path}\n  expected {m['sha256']} (data/MANIFEST.json, version {m.get('version')})\n  found    {got}\n"
                f"The file is incomplete, corrupted or from another version. {how}")
    return path


def load(columns: list[str] | None = None, kept_only: bool = True, verify: bool = True, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the linted table as a DataFrame, one row per measured point.

    Parameters
    ----------
    columns : list of column names to read (default: all; see docs/data_dictionary.md).
    kept_only : if True (default), return only the rows with ``keep == True``, the rows retained by the
        linter. If False, return every row; the column ``reason`` then says why a row was not retained.
    verify : check the SHA-256 of the file against data/MANIFEST.json before reading (default True).
    data_dir : folder holding MANIFEST.json and the parquet (default: ``DATA_DIR``).

    The verdict columns (``ds_flag``, ``ds_offset_dex``, ``ds_weight``, ``pt_outlier`` and the flag columns)
    are always returned as they are; this function does not filter or weight on them.
    """
    path = _checked_path(data_dir, verify)
    filters = [("keep", "==", True)] if kept_only else None
    df = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    if "pt_outlier" in df.columns:
        df["pt_outlier"] = df["pt_outlier"].astype("boolean")  # True, False or <NA> (not assessed)
    return df.reset_index(drop=True)


def isotope_id(Z, A):
    """Integer label of a target isotope, ``1000 * Z + A`` (Cl-35 is 17035). Works on scalars, arrays and Series."""
    return np.asarray(Z).astype(np.int64) * 1000 + np.asarray(A).astype(np.int64) if not np.isscalar(Z) else int(Z) * 1000 + int(A)


def split_isotope_id(iso):
    """Inverse of :func:`isotope_id`: returns ``(Z, A)``."""
    iso = np.asarray(iso) if not np.isscalar(iso) else int(iso)
    return iso // 1000, iso % 1000


def _parse_target(target) -> tuple[int, int]:
    """``"Cl-35"``, ``"Cl35"``, ``"35Cl"`` or ``(17, 35)`` to ``(Z, A)``. Element symbols are read from the AME2020 table."""
    if isinstance(target, (tuple, list)) and len(target) == 2:
        return int(target[0]), int(target[1])
    import re

    from .features import element_symbols

    s = str(target).strip().replace(" ", "")
    m = re.fullmatch(r"([A-Za-z]{1,2})-?(\d{1,3})", s) or re.fullmatch(r"(\d{1,3})-?([A-Za-z]{1,2})", s)
    if not m:
        raise ValueError(f"cannot read target {target!r}; use for example 'Cl-35' or (17, 35)")
    a, b = m.groups()
    sym, mass = (a, b) if a.isalpha() else (b, a)
    z_of = {v.lower(): k for k, v in element_symbols().items()}
    if sym.lower() not in z_of:
        raise ValueError(f"unknown element symbol {sym!r}")
    return z_of[sym.lower()], int(mass)


def select(df: pd.DataFrame, projectile: str | None = None, mt: int | list[int] | None = None, target=None, Z: int | None = None,
           A: int | None = None, e_min: float | None = None, e_max: float | None = None, plain: bool = False) -> pd.DataFrame:
    """Rows of ``df`` for a projectile, reaction (ENDF MT number), target and incident-energy window (eV).

    Every argument is optional and the conditions are combined with AND. ``target`` accepts ``"Cl-35"`` or
    ``(17, 35)``. With ``plain=True`` only rows without a product-state qualifier and without the modifier
    fields SF5, SF8 and SF9 are returned (an unqualified cross section); this is off by default.
    The result is a copy with the original index, so that masks built on it can be aligned with ``df``.
    """
    m = pd.Series(True, index=df.index)
    if projectile is not None:
        m &= df["projectile"] == projectile
    if mt is not None:
        m &= df["MT"].isin(np.atleast_1d(mt))
    if target is not None:
        Z, A = _parse_target(target)
    if Z is not None:
        m &= df["Z"] == Z
    if A is not None:
        m &= df["A"] == A
    if e_min is not None:
        m &= df["energy_eV"] >= e_min
    if e_max is not None:
        m &= df["energy_eV"] <= e_max
    if plain:
        m &= (df["product_state"].astype(str) == "") & df["sf5"].isna() & df["sf8"].isna() & df["sf9"].isna()
    return df[m].copy()


def datasets(df: pd.DataFrame) -> pd.DataFrame:
    """One row per EXFOR dataset in ``df``: bibliography, dates, number of points, energy range and the per-dataset verdicts."""
    cols = [c for c in ["entry", "author", "year", "reference", "entry_date", "projectile", "Z", "A", "MT", "reaction", "ds_flag", "ds_offset_dex", "ds_weight"] if c in df.columns]
    g = df.groupby("dataset", observed=True)
    out = g[cols].first()
    out["points"] = g.size()
    out["e_min_eV"] = g["energy_eV"].min()
    out["e_max_eV"] = g["energy_eV"].max()
    return out.reset_index()


def check_inputs(columns) -> None:
    """Raise ValueError if a verdict column or an uncertainty column is among the model inputs ``columns``.

    ``columns`` is a list of column names or a DataFrame. The columns of :data:`VERDICT_COLUMNS` were computed with
    the whole table in view, including rows that are later held out, and the columns of :data:`UNCERTAINTY_COLUMNS`
    exist only where a measurement exists; a model that takes them as inputs cannot be used where it is needed and
    is scored with information from its test rows. The check is never applied automatically: call it on the list
    of features before fitting. Filtering or weighting the training rows by these columns is a different matter,
    a modelling choice to be declared (see docs/conventions.md).
    """
    names = list(columns.columns) if isinstance(columns, pd.DataFrame) else list(columns)
    verdict = [c for c in names if c in VERDICT_COLUMNS]
    uncertainty = [c for c in names if c in UNCERTAINTY_COLUMNS and c not in VERDICT_COLUMNS]
    if verdict or uncertainty:
        parts = []
        if verdict:
            parts.append(f"verdict columns {verdict} (computed on the whole table, hold-out rows included)")
        if uncertainty:
            parts.append(f"uncertainty columns {uncertainty} (known only where a measurement exists)")
        raise ValueError("not usable as model inputs: " + "; ".join(parts) + ". See docs/conventions.md, 'Verdict columns and leakage'.")
