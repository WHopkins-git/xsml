"""Train and test splits in which the isotope is the unit of observation.

Points of one isotope are not independent: they share a nucleus and often an experiment. A split that puts
some points of an isotope in training and others in test measures interpolation within that isotope.
Every function here keeps an isotope (or a larger group) on one side.

The fold assignment uses no random numbers: isotopes are sorted by Z and then A and dealt to the folds in
turn, so the folds are identical on every machine and for every version of numpy and scikit-learn.

All functions return boolean masks or integer positions aligned with the rows of ``df``; none of them
removes rows.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .data import isotope_id

__all__ = ["LeakageError", "isotope_folds", "fold_splits", "leave_one_isotope_out", "leave_one_element_out", "holdout_by_date", "assert_no_leakage"]


class LeakageError(AssertionError):
    """A group that is held out also appears in the training rows."""


def _iso(df_or_ids) -> np.ndarray:
    if isinstance(df_or_ids, pd.DataFrame):
        return np.asarray(isotope_id(df_or_ids["Z"].to_numpy(), df_or_ids["A"].to_numpy()))
    return np.asarray(df_or_ids).astype(np.int64)


def isotope_folds(df_or_ids, n_folds: int = 5, offset: int = 0) -> np.ndarray:
    """Fold number (0 to ``n_folds - 1``) of every row, with all rows of an isotope in the same fold.

    The distinct isotopes are sorted by Z and then A; the isotope at rank r goes to fold ``(r + offset) % n_folds``.
    ``df_or_ids`` is a DataFrame with columns ``Z`` and ``A``, or an array of :func:`xsml.data.isotope_id` values.
    ``offset`` gives a different, equally deterministic, assignment (for repeated cross-validation).
    """
    iso = _iso(df_or_ids)
    distinct = sorted(set(iso.tolist()))  # 1000 * Z + A sorts by Z, then A
    if len(distinct) < n_folds:
        raise ValueError(f"{len(distinct)} isotopes cannot fill {n_folds} folds")
    fold_of = {g: (r + offset) % n_folds for r, g in enumerate(distinct)}
    return np.array([fold_of[g] for g in iso.tolist()], dtype=np.int64)


def fold_splits(df_or_ids, n_folds: int = 5, offset: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(train_positions, test_positions), ...]`` for :func:`isotope_folds`; accepted as ``cv=`` by scikit-learn."""
    fold = isotope_folds(df_or_ids, n_folds, offset)
    pos = np.arange(len(fold))
    return [(pos[fold != k], pos[fold == k]) for k in range(n_folds)]


def leave_one_isotope_out(df: pd.DataFrame, Z: int, A: int) -> tuple[np.ndarray, np.ndarray]:
    """``(train_mask, test_mask)``: every row of the target (Z, A), in any reaction, is test; all other rows are training."""
    test = ((df["Z"] == Z) & (df["A"] == A)).to_numpy().copy()        # a copy: pandas 3 hands out read-only arrays
    if not test.any():
        raise ValueError(f"no rows for Z={Z}, A={A}")
    return ~test, test


def leave_one_element_out(df: pd.DataFrame, Z: int) -> tuple[np.ndarray, np.ndarray]:
    """``(train_mask, test_mask)``: every row of the element Z, of any isotope and reaction, is test."""
    test = (df["Z"] == Z).to_numpy().copy()
    if not test.any():
        raise ValueError(f"no rows for Z={Z}")
    return ~test, test


def holdout_by_date(df: pd.DataFrame, cutoff: str, where=None, date_col: str = "entry_date") -> tuple[np.ndarray, np.ndarray]:
    """``(train_mask, test_mask)`` by date of compilation into EXFOR.

    Test rows are those with ``df[date_col] >= cutoff`` (ISO date string, for example ``"2019-01-01"``), restricted
    to the boolean mask ``where`` if one is given (for example the rows of one reaction). Training rows are all
    other rows. ``entry_date`` is the date an entry was first compiled into EXFOR; it is not the year of publication.

    The training mask is only the complement of the test mask. Whether the remaining rows of the same isotope,
    entry or experiment may be used for training is a decision for the protocol; check it with
    :func:`assert_no_leakage`.
    """
    d = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    if d.isna().any():
        raise ValueError(f"{int(d.isna().sum())} rows have no valid {date_col}")
    test = (d >= pd.Timestamp(cutoff)).to_numpy().copy()
    if where is not None:
        test = test & np.asarray(where, dtype=bool)
    return ~test, test


def _groups(df: pd.DataFrame, level: str) -> set:
    if level == "isotope":
        return set(_iso(df).tolist())
    if level == "element":
        return set(df["Z"].tolist())
    if level in ("dataset", "entry"):
        return set(df[level].astype(str).tolist())
    raise ValueError("level must be 'isotope', 'element', 'dataset' or 'entry'")


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, level: str = "isotope") -> None:
    """Raise :class:`LeakageError` if a group of ``test`` also occurs in ``train``.

    ``level`` is ``"isotope"`` (default: target Z and A, any reaction), ``"element"`` (Z), ``"dataset"`` or
    ``"entry"`` (EXFOR identifiers). Call it on the frames actually passed to ``fit`` and to scoring.
    """
    shared = _groups(train, level) & _groups(test, level)
    if shared:
        some = sorted(shared)[:8]
        raise LeakageError(f"{len(shared)} {level} group(s) of the test rows are also in the training rows, for example {some}")
