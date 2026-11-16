"""Curves of evaluated nuclear data libraries.

Only what is shipped in ``data/libraries/`` is available: the repository does not contain the libraries
themselves and does not process ENDF files. :func:`available` lists what is shipped. Two forms exist:

* pointwise curves (:func:`curve`, :func:`at`), file ``<Element><A>_<reaction>[_<state>].csv`` with the columns
  ``library, energy_eV, sigma_b``;
* averages over bins of a quarter of a decade in energy (:func:`bin_means`), file ``..._bins.csv`` with the columns
  ``library, e_low_eV, e_high_eV, sigma_b``. A bin average is not a curve: in a resonance region it is dominated
  by the resonance peaks.

Libraries (:data:`LIBRARIES`): ENDF/B-VII.1, ENDF/B-VIII.0 and JENDL-5 are evaluations of neutron-induced
reactions. TENDL-2025 is of another kind (:data:`MODEL_BASED`): its curves are calculated with the nuclear
model code TALYS, not fitted to the measurements of the reaction concerned, and are shipped here for
proton-induced reactions up to 30 MeV, the range in which TENDL tabulates the channels one by one.

Not every library is present for every reaction, and some pointwise curves of resonance reactions are thinned
(they do not resolve every resonance). The sources and the checks are in ``scripts/build_library_curves.py``;
a reaction is added there with one line.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, _parse_target
from .features import element_symbols

__all__ = ["LIBRARIES", "MODEL_BASED", "REACTION_NAMES", "OUTGOING", "available", "curve", "at", "bin_means", "parse_file_name"]

#: Library names as they appear in the ``library`` column.
LIBRARIES = ["ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5", "TENDL-2025"]

#: Libraries whose curves are model calculations (TALYS), not evaluations of the measurements.
MODEL_BASED = ["TENDL-2025"]

#: Reaction part of the file name for neutron-induced reactions, by MT number.
REACTION_NAMES = {1: "tot", 2: "el", 4: "ninl", 16: "n2n", 18: "f", 102: "ng", 103: "np", 107: "na"}

#: Outgoing particles by MT number, used in the file name for other projectiles ("p" + "2n" gives ``p2n``).
OUTGOING = {4: "n", 16: "2n", 17: "3n", 22: "na", 28: "np", 37: "4n", 102: "g", 103: "p", 104: "d", 105: "t", 106: "he3", 107: "a", 111: "2p", 112: "pa"}

_NAME = re.compile(r"([A-Z][a-z]?)(\d+)_([a-z0-9]+?)(?:_(g|m\d?))?(_bins)?\.csv")


def _folder(data_dir=None) -> Path:
    return Path(data_dir or DATA_DIR) / "libraries"


def _reaction_name(projectile: str, mt: int) -> str:
    if projectile == "n":
        return REACTION_NAMES.get(mt, f"mt{mt}")
    return projectile + OUTGOING[mt] if mt in OUTGOING else f"{projectile}mt{mt}"


def _file(target, mt: int, data_dir=None, projectile: str = "n", state: str | None = None) -> Path:
    z, a = _parse_target(target)
    tail = f"_{state}" if state else ""
    return _folder(data_dir) / f"{element_symbols()[z]}{a}_{_reaction_name(projectile, mt)}{tail}.csv"


def parse_file_name(name: str) -> dict:
    """Target, projectile, MT number, product state and form of a file of ``data/libraries/``, from its name.

    ``state`` is None for the cross section of the reaction as a whole, ``"g"`` for the part that leaves the
    product in its ground state and ``"m"`` for the part that leaves it in an isomeric state.
    """
    m = _NAME.fullmatch(name)
    if not m:
        raise ValueError(f"{name} is not a library file name")
    symbol, mass, reaction, state, binned = m.groups()
    z = {s: k for k, s in element_symbols().items()}[symbol]
    for projectile in ("n", "p", "d", "t", "he3", "a", "g"):
        for mt in sorted(set(REACTION_NAMES) | set(OUTGOING)):
            if _reaction_name(projectile, mt) == reaction:
                return {"Z": z, "A": int(mass), "target": f"{symbol}-{mass}", "projectile": projectile, "MT": mt, "state": state, "binned": bool(binned)}
    raise ValueError(f"reaction {reaction!r} of {name} is not known")


def _curve_files(data_dir=None) -> list:
    return [f for f in sorted(_folder(data_dir).glob("*.csv")) if not f.name.endswith("_bins.csv")]


def available(data_dir=None) -> pd.DataFrame:
    """What is shipped: one row per file and library.

    Columns: ``file``, ``target``, ``projectile``, ``MT``, ``state`` (empty, ``g`` or ``m``; see
    :func:`parse_file_name`), ``form`` (``pointwise`` or ``bin averages``), ``library``, ``basis`` (``evaluation``, or
    ``model calculation`` for the libraries in :data:`MODEL_BASED`), ``points`` and the energy range.
    """
    rows = []
    for f in sorted(_folder(data_dir).glob("*.csv")):
        t = pd.read_csv(f)
        info = parse_file_name(f.name)
        binned = info["binned"]
        for lib, g in t.groupby("library"):
            rows.append({"file": f.name, "target": info["target"], "projectile": info["projectile"], "MT": info["MT"], "state": info["state"] or "",
                         "form": "bin averages" if binned else "pointwise", "library": lib, "basis": "model calculation" if lib in MODEL_BASED else "evaluation",
                         "points": len(g), "e_min_eV": g.e_low_eV.min() if binned else g.energy_eV.min(), "e_max_eV": g.e_high_eV.max() if binned else g.energy_eV.max()})
    return pd.DataFrame(rows)


def curve(target="Cl-35", mt: int = 103, library: str | None = None, data_dir=None, projectile: str = "n", state: str | None = None) -> pd.DataFrame:
    """Pointwise curve(s) of a reaction: columns ``library``, ``energy_eV``, ``sigma_b``.

    ``mt`` is the ENDF number, which describes the outgoing particles: 103 one proton, 4 one neutron (inelastic
    scattering for an incident neutron, (p,n) for an incident proton), 16 two neutrons, 102 photon only.
    ``projectile`` is ``"n"`` (default) or ``"p"``. ``state`` selects the part of the reaction that leaves the
    product in its ground state (``"g"``) or in an isomeric state (``"m"``), where the library gives them
    separately; None is the reaction as a whole.
    ``library`` is one of :data:`LIBRARIES`; None returns all that the file holds.
    Raises FileNotFoundError, naming what is available, if the reaction is not shipped.
    """
    path = _file(target, mt, data_dir, projectile, state)
    if not path.exists():
        have = ", ".join(p.name for p in _curve_files(data_dir)) or "none"
        raise FileNotFoundError(f"no library curve for {target} MT {mt} ({path.name}). Shipped curves: {have}. See xsml/libraries.py for the file format.")
    t = pd.read_csv(path)
    if library is not None:
        if library not in set(t.library):
            raise KeyError(f"library {library!r} not in {path.name}; available: {sorted(set(t.library))}")
        t = t[t.library == library]
    return t.reset_index(drop=True)


def at(energy_eV, library: str, target="Cl-35", mt: int = 103, data_dir=None, projectile: str = "n", state: str | None = None) -> np.ndarray:
    """The library cross section (barns) at the given incident energies, by linear interpolation in log-log.

    NaN outside the energy range of the curve and where the curve is zero (below a threshold). Interpolating a
    resolved-resonance region at a few energies does not represent it; compare over an energy range instead.
    The libraries tabulate these curves for linear interpolation; between two widely spaced points (the TENDL
    tables have steps of 1 to 2 MeV) the log-log value differs slightly from the linear one.
    """
    c = curve(target, mt, library, data_dir, projectile, state)
    e, s = c.energy_eV.to_numpy(), c.sigma_b.to_numpy()
    ok = (e > 0) & (s > 0)
    x = np.asarray(energy_eV, dtype=float)
    out = 10 ** np.interp(np.log10(x), np.log10(e[ok]), np.log10(s[ok]), left=np.nan, right=np.nan)
    lo = np.interp(x, e, s, left=np.nan, right=np.nan)
    return np.where(np.isnan(lo) | (lo <= 0), np.nan, out)


def bin_means(target, mt: int, library: str | None = None, data_dir=None, projectile: str = "n") -> pd.DataFrame:
    """Library cross section averaged over energy bins of a quarter of a decade: columns ``library``, ``e_low_eV``,
    ``e_high_eV``, ``sigma_b`` (linear energy average of the cross section over the bin).

    Use it where no pointwise curve is shipped, and draw it as steps. It is not comparable point by point with
    measurements in a resonance region. Raises FileNotFoundError if the reaction is not shipped.
    """
    path = _file(target, mt, data_dir, projectile)
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
