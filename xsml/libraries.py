"""Curves of evaluated nuclear data libraries.

Only what is shipped in ``data/libraries/`` is available: the repository does not contain the evaluated
libraries themselves and does not process ENDF files. :func:`available` lists what is shipped. Two forms exist:

* pointwise curves (:func:`curve`, :func:`at`), file ``<Element><A>_<reaction>.csv`` with the columns
  ``library, energy_eV, sigma_b``;
* averages over bins of a quarter of a decade in energy (:func:`bin_means`), file ``..._bins.csv`` with the columns
  ``library, e_low_eV, e_high_eV, sigma_b``. A bin average is not a curve: in a resonance region it is dominated
  by the resonance peaks.

Not every library is present for every reaction, and some pointwise curves of resonance reactions are thinned
(they do not resolve every resonance). The sources and the checks are in ``scripts/build_library_curves.py``;
a reaction is added there with one line.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, _parse_target
from .features import element_symbols

__all__ = ["REACTION_NAMES", "available", "curve", "at", "bin_means"]

#: File-name suffix for the MT numbers that have a short name.
REACTION_NAMES = {1: "tot", 2: "el", 4: "ninl", 16: "n2n", 18: "f", 102: "ng", 103: "np", 107: "na"}


def _folder(data_dir=None) -> Path:
    return Path(data_dir or DATA_DIR) / "libraries"


def _file(target, mt: int, data_dir=None) -> Path:
    z, a = _parse_target(target)
    return _folder(data_dir) / f"{element_symbols()[z]}{a}_{REACTION_NAMES.get(mt, f'mt{mt}')}.csv"


def _curve_files(data_dir=None) -> list:
    return [f for f in sorted(_folder(data_dir).glob("*.csv")) if not f.name.endswith("_bins.csv")]


def available(data_dir=None) -> pd.DataFrame:
    """What is shipped: one row per file and library, with the form (``pointwise`` or ``bin averages``), the number of
    points or bins and the energy range."""
    rows = []
    for f in sorted(_folder(data_dir).glob("*.csv")):
        t = pd.read_csv(f)
        binned = f.name.endswith("_bins.csv")
        for lib, g in t.groupby("library"):
            rows.append({"file": f.name, "form": "bin averages" if binned else "pointwise", "library": lib, "points": len(g),
                         "e_min_eV": g.e_low_eV.min() if binned else g.energy_eV.min(), "e_max_eV": g.e_high_eV.max() if binned else g.energy_eV.max()})
    return pd.DataFrame(rows)


def curve(target="Cl-35", mt: int = 103, library: str | None = None, data_dir=None) -> pd.DataFrame:
    """Pointwise curve(s) for a neutron-induced reaction: columns ``library``, ``energy_eV``, ``sigma_b``.

    ``mt`` is the ENDF number: 103 for (n,p), 4 for inelastic scattering, 102 for (n,g).
    ``library`` is ``"ENDF-B-VII.1"``, ``"ENDF-B-VIII.0"`` or ``"JENDL-5"``; None returns all that the file holds.
    Raises FileNotFoundError, naming what is available, if the reaction is not shipped.
    """
    path = _file(target, mt, data_dir)
    if not path.exists():
        have = ", ".join(p.name for p in _curve_files(data_dir)) or "none"
        raise FileNotFoundError(f"no library curve for {target} MT {mt} ({path.name}). Shipped curves: {have}. See xsml/libraries.py for the file format.")
    t = pd.read_csv(path)
    if library is not None:
        if library not in set(t.library):
            raise KeyError(f"library {library!r} not in {path.name}; available: {sorted(set(t.library))}")
        t = t[t.library == library]
    return t.reset_index(drop=True)


def at(energy_eV, library: str, target="Cl-35", mt: int = 103, data_dir=None) -> np.ndarray:
    """The library cross section (barns) at the given incident energies, by linear interpolation in log-log.

    NaN outside the energy range of the curve and where the curve is zero (below a threshold). Interpolating a
    resolved-resonance region at a few energies does not represent it; compare over an energy range instead.
    """
    c = curve(target, mt, library, data_dir)
    e, s = c.energy_eV.to_numpy(), c.sigma_b.to_numpy()
    ok = (e > 0) & (s > 0)
    x = np.asarray(energy_eV, dtype=float)
    out = 10 ** np.interp(np.log10(x), np.log10(e[ok]), np.log10(s[ok]), left=np.nan, right=np.nan)
    lo = np.interp(x, e, s, left=np.nan, right=np.nan)
    return np.where(np.isnan(lo) | (lo <= 0), np.nan, out)


def bin_means(target, mt: int, library: str | None = None, data_dir=None) -> pd.DataFrame:
    """Library cross section averaged over energy bins of a quarter of a decade: columns ``library``, ``e_low_eV``,
    ``e_high_eV``, ``sigma_b`` (linear energy average of the cross section over the bin).

    Use it where no pointwise curve is shipped, and draw it as steps. It is not comparable point by point with
    measurements in a resonance region. Raises FileNotFoundError if the reaction is not shipped.
    """
    path = _file(target, mt, data_dir)
    path = path.with_name(path.stem + "_bins.csv")
    if not path.exists():
        have = ", ".join(p.name for p in sorted(_folder(data_dir).glob("*_bins.csv"))) or "none"
        raise FileNotFoundError(f"no bin averages for {target} MT {mt} ({path.name}). Shipped: {have}.")
    t = pd.read_csv(path)
    if library is not None:
        if library not in set(t.library):
            raise KeyError(f"library {library!r} not in {path.name}; available: {sorted(set(t.library))}")
        t = t[t.library == library]
    return t.reset_index(drop=True)
