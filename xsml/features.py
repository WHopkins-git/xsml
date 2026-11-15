"""Nuclear properties from the AME2020 atomic mass evaluation (``data/mass_1.mas20.txt``).

Two tables are provided, both with one row per nuclide:

* :func:`ame_table`: what the mass table gives, and what follows from it by mass differences
  (binding energy, separation energies, Q-values).
* :func:`nuclide_features`: the same, plus quantities generated from Z and N alone
  (parity flags, pairing sign, distances to magic numbers, and so on).

:func:`append` joins either table to the rows of a data frame and adds the Q-value of each row's reaction.
There is no preset feature set: which columns to give a model is the user's decision.

Energies are in MeV. A quantity is NaN where a partner nuclide is not in the mass table.
Mass excesses marked ``#`` in AME2020 are extrapolated, not measured; the column ``estimated`` flags them.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR

__all__ = ["MAGIC", "LIGHT", "MT_EJECTILES", "ame_table", "nuclide_features", "element_symbols", "q_value", "reaction_q", "append"]

#: Magic numbers used for the shell-distance features.
MAGIC = (2, 8, 20, 28, 50, 82, 126)

#: Light particles as (Z, A); the photon is (0, 0).
LIGHT = {"g": (0, 0), "n": (0, 1), "p": (1, 1), "d": (1, 2), "t": (1, 3), "he3": (2, 3), "a": (2, 4)}

#: Emitted light particles for ENDF MT numbers with a fixed set of outgoing particles.
#: MT 2 (elastic) and MT 4 for incident neutrons (inelastic) are handled in :func:`reaction_q`.
MT_EJECTILES = {
    4: ["n"], 11: ["n", "n", "d"], 16: ["n", "n"], 17: ["n", "n", "n"], 22: ["n", "a"], 23: ["n", "a", "a", "a"], 24: ["n", "n", "a"],
    25: ["n", "n", "n", "a"], 28: ["n", "p"], 29: ["n", "a", "a"], 32: ["n", "d"], 33: ["n", "t"], 34: ["n", "he3"], 37: ["n", "n", "n", "n"],
    41: ["n", "n", "p"], 42: ["n", "n", "n", "p"], 44: ["n", "p", "p"], 45: ["n", "p", "a"],
    102: [], 103: ["p"], 104: ["d"], 105: ["t"], 106: ["he3"], 107: ["a"], 108: ["a", "a"], 111: ["p", "p"], 112: ["p", "a"], 113: ["t", "a", "a"],
}


@lru_cache(maxsize=4)
def _read_ame(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()
    for line in lines[36:]:
        try:
            n, z, a = int(line[4:9]), int(line[9:14]), int(line[14:19])
            me = line[28:42]
            mass_u = int(line[106:109]) + float(line[110:123].replace("#", ".")) * 1e-6
            rows.append((z, n, a, line[20:23].strip(), float(me.replace("#", ".")), float(line[42:54].replace("#", ".")), "#" in me, mass_u))
        except ValueError:
            continue
    t = pd.DataFrame(rows, columns=["Z", "N", "A", "element", "mass_excess", "mass_excess_unc", "estimated", "atomic_mass_u"])
    if len(t) < 3000 or not (t.A == t.Z + t.N).all():
        raise ValueError(f"{path} was not read as an AME2020 mass table")
    t[["mass_excess", "mass_excess_unc"]] /= 1e3  # keV to MeV
    return t.sort_values(["Z", "A"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _mass_excess_cached(path: str) -> dict:
    t = _read_ame(path)
    me = dict(zip(zip(t.Z.tolist(), t.A.tolist()), t.mass_excess.tolist()))
    me[(0, 0)] = 0.0  # photon
    return me


def _mass_excess(path) -> dict:
    """Mass excess (MeV) by (Z, A); the neutron is (0, 1) and the photon (0, 0)."""
    return _mass_excess_cached(str(path or DATA_DIR / "mass_1.mas20.txt"))


def element_symbols(path: Path | str | None = None) -> dict:
    """Element symbol by Z, as written in the AME2020 table."""
    t = _read_ame(str(path or DATA_DIR / "mass_1.mas20.txt"))
    return dict(zip(t.Z, t.element))


def q_value(Z: int, A: int, projectile: str, ejectiles: list[str], path: Path | str | None = None) -> float:
    """Q-value (MeV) of target(projectile, ejectiles)residual from AME2020 mass excesses, ground state to ground state.

    ``projectile`` and each ejectile are keys of :data:`LIGHT`. Positive Q: energy is released; negative Q: the
    reaction has a threshold. NaN if the target or the residual nucleus is not in the mass table.
    """
    me = _mass_excess(path)
    zp, ap = LIGHT[projectile]
    zr, ar, out = Z + zp, A + ap, 0.0
    for e in ejectiles:
        ze, ae = LIGHT[e]
        zr, ar, out = zr - ze, ar - ae, out + me[(ze, ae)]
    if (Z, A) not in me or (zr, ar) not in me:
        return float("nan")
    return me[(Z, A)] + me[(zp, ap)] - me[(zr, ar)] - out


def ame_table(path: Path | str | None = None) -> pd.DataFrame:
    """AME2020, one row per nuclide (Z >= 1, A >= 2), with the quantities that follow from mass differences.

    Columns (MeV): ``Z``, ``N``, ``A``, ``element``, ``mass_excess``, ``mass_excess_unc``, ``estimated``,
    ``atomic_mass_u`` (atomic mass in u), ``binding_energy``, ``B_per_A``, ``S_n``, ``S_p``, ``S_2n``, ``S_2p`` (separation energies),
    ``Q_alpha``, ``Q_beta_minus`` (decay energies) and, for an incident neutron,
    ``Q_ng``, ``Q_np``, ``Q_na``, ``Q_n2n``, ``Q_nd``, ``Q_nt``.
    For other reactions use :func:`q_value` or :func:`reaction_q`.
    """
    t = _read_ame(str(path or DATA_DIR / "mass_1.mas20.txt"))
    me = _mass_excess(path)
    g = lambda z, a: me.get((z, a), np.nan)  # noqa: E731
    n, h, he4 = me[(0, 1)], me[(1, 1)], me[(2, 4)]
    out = t[(t.Z >= 1) & (t.A >= 2)].copy()
    za = list(zip(out.Z, out.A))
    out["binding_energy"] = [z * h + (a - z) * n - g(z, a) for z, a in za]
    out["B_per_A"] = out.binding_energy / out.A
    out["S_n"] = [g(z, a - 1) + n - g(z, a) for z, a in za]
    out["S_p"] = [g(z - 1, a - 1) + h - g(z, a) for z, a in za]
    out["S_2n"] = [g(z, a - 2) + 2 * n - g(z, a) for z, a in za]
    out["S_2p"] = [g(z - 2, a - 2) + 2 * h - g(z, a) for z, a in za]
    out["Q_alpha"] = [g(z, a) - g(z - 2, a - 4) - he4 for z, a in za]
    out["Q_beta_minus"] = [g(z, a) - g(z + 1, a) for z, a in za]
    for name, ej in (("Q_ng", []), ("Q_np", ["p"]), ("Q_na", ["a"]), ("Q_n2n", ["n", "n"]), ("Q_nd", ["d"]), ("Q_nt", ["t"])):
        out[name] = [q_value(z, a, "n", ej, path) for z, a in za]
    return out.reset_index(drop=True)


def nuclide_features(path: Path | str | None = None) -> pd.DataFrame:
    """:func:`ame_table` plus properties generated from Z and N, one row per nuclide.

    Added columns: ``N_minus_Z``, ``asymmetry`` ((N - Z) / A), ``Z_odd``, ``N_odd``, ``A_odd`` (0 or 1),
    ``pairing`` (+1 even-even, 0 odd A, -1 odd-odd), ``Z_magic_dist``, ``N_magic_dist`` (distance to the nearest
    magic number), ``Z_magic``, ``N_magic`` (1 if magic), ``P_factor`` (Np Nn / (Np + Nn) with the two distances,
    0 if both are 0), ``radius_fm`` (1.25 A^(1/3)) and ``A_third`` (A^(1/3)).
    """
    f = ame_table(path)
    magic = np.array(MAGIC)
    f["N_minus_Z"] = f.N - f.Z
    f["asymmetry"] = (f.N - f.Z) / f.A
    f["Z_odd"], f["N_odd"], f["A_odd"] = f.Z % 2, f.N % 2, f.A % 2
    f["pairing"] = np.where(f.A_odd == 1, 0, np.where(f.Z_odd == 0, 1, -1))
    f["Z_magic_dist"] = np.abs(f.Z.to_numpy()[:, None] - magic).min(axis=1)
    f["N_magic_dist"] = np.abs(f.N.to_numpy()[:, None] - magic).min(axis=1)
    f["Z_magic"], f["N_magic"] = (f.Z_magic_dist == 0).astype(int), (f.N_magic_dist == 0).astype(int)
    s = f.Z_magic_dist + f.N_magic_dist
    f["P_factor"] = np.where(s > 0, f.Z_magic_dist * f.N_magic_dist / s.where(s > 0, 1), 0.0)
    f["A_third"] = f.A ** (1 / 3)
    f["radius_fm"] = 1.25 * f.A_third
    return f


def reaction_q(df: pd.DataFrame, path: Path | str | None = None) -> pd.Series:
    """Ground-state Q-value (MeV) of the reaction of every row of ``df`` (columns ``projectile``, ``Z``, ``A``, ``MT``).

    The outgoing particles are taken from :data:`MT_EJECTILES`. MT 2 (elastic) and MT 4 for an incident neutron
    (inelastic) give 0. NaN for MT numbers without a fixed set of outgoing particles (total, fission, production
    cross sections and others) and where a mass is missing.
    """
    keys = df[["projectile", "Z", "A", "MT"]].astype({"projectile": str}).drop_duplicates()
    q = {}
    for p, z, a, mt in keys.itertuples(index=False):
        if mt == 2 or (mt == 4 and p == "n"):
            q[(p, z, a, mt)] = 0.0 if (z, a) in _mass_excess(path) else np.nan
        elif mt in MT_EJECTILES and p in LIGHT:
            q[(p, z, a, mt)] = q_value(int(z), int(a), p, MT_EJECTILES[mt], path)
        else:
            q[(p, z, a, mt)] = np.nan
    idx = pd.MultiIndex.from_frame(df[["projectile", "Z", "A", "MT"]].astype({"projectile": str}))
    return pd.Series(pd.Series(q).reindex(idx).to_numpy(), index=df.index, name="Q_reaction")


def append(df: pd.DataFrame, table: pd.DataFrame | None = None, path: Path | str | None = None) -> pd.DataFrame:
    """A copy of ``df`` with the columns of ``table`` (default :func:`nuclide_features`) joined on the target (Z, A),
    and with ``N``, ``log10_energy``, ``log10_sigma`` and ``Q_reaction`` (see :func:`reaction_q`) added.

    Rows whose target is not in the mass table keep NaN in the joined columns; no row is dropped.
    """
    table = nuclide_features(path) if table is None else table
    add = table.drop(columns=[c for c in table.columns if c in df.columns and c not in ("Z", "A")])
    out = df.merge(add, on=["Z", "A"], how="left", validate="many_to_one")
    out.index = df.index
    out["N"] = out.A - out.Z
    out["log10_energy"] = np.log10(out.energy_eV)
    out["log10_sigma"] = np.log10(out.sigma_b)
    out["Q_reaction"] = reaction_q(out, path)
    return out
