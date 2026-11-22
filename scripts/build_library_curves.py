#!/usr/bin/env python
"""Write the evaluated-library files of ``data/libraries/`` (run by the supervisor, not by students).

    python scripts/build_library_curves.py --source <path to the EXFOR-ML working copy>

Adding a reaction is one line in ``REACTIONS`` below. Everything is copied or parsed from files on disk; nothing
is downloaded, interpolated or computed beyond reading, with one exception that is stated under "njoy" (the
Cl-35 (n,p) curves of the two ENDF/B libraries below 0.1 MeV). Three kinds of file are written, named by
``xsml.libraries`` (``<Element><A>_<reaction>.csv`` and ``..._bins.csv``):

"mf3"   pointwise curve of ENDF/B-VII.1, ENDF/B-VIII.0 and JENDL-5: section MF 3 of the ENDF-6 files in
        ``data/external/2026-10-08/`` (the library archives as distributed), as tabulated (0 K, no processing),
        up to 20 MeV. Right only where the resonance parameters add nothing to the reaction (threshold reactions).
        This is checked: the bin averages of the curve must agree with the NJOY-processed bin averages
        (``docs/paper/data/library_bins_njoy.json``) within ``TOLERANCE_DEX``; a library that fails is left out.
"nndc"  pointwise curve of ENDF/B-VIII.0 from the cache ``<El>-<A>_MT<mt>_endfb8.0.npz`` (``--nndc-cache``), a
        table retrieved earlier from the IAEA-NDS ENDF service with the resonances reconstructed by that service.
        Checked against the NJOY bin averages in the same way. ``thin`` keeps every n-th point (and all points
        above 100 keV) to limit the file size; a thinned curve does not resolve every resonance.
"bins"  averages over bins of a quarter of a decade in energy, the three libraries, copied from
        ``docs/paper/data/library_bins_njoy.json`` (NJOY2016, RECONR and BROADR at 293.6 K; linear energy average of
        the cross section over the part of each bin where it is positive). The only form available here for
        reactions with resonances (capture, fission, total, elastic) in ENDF/B-VII.1 and JENDL-5.
"njoy"  Cl-35 (n,p) only. From 0.1 to 20 MeV: pointwise NJOY2016 curves of
        ``docs/paper/data/cl35_library_curves.json`` (RECONR and BROADR, 293.6 K), which hold nothing below 0.1 MeV.
        Below 0.1 MeV, down to the lower limit of the evaluations (1e-5 eV), at 0 K, from the ENDF-6 files:
        JENDL-5 gives the curve in MF 3 / MT 103 and it is copied as filed. ENDF/B-VII.1 and ENDF/B-VIII.0 give
        only a placeholder (1e-20 b) in MF 3 below 1 MeV; their cross section is in the resonance parameters
        (MF 2, R-matrix limited format LRF 7, Reich-Moore approximation, with a proton channel). NJOY is not
        available where this script runs, so that part is calculated here (``rml_cross_section``, ``linearise``) on a
        grid on which linear interpolation holds to 0.1 %. Checks, printed in the report: the same calculation
        with the JENDL-5 parameters against the JENDL-5 MF 3 curve below 0.1 MeV, the calculation against the
        NJOY curve of the same library from 0.1 to 1 MeV (medians; the peaks differ because NJOY broadened them
        to 293.6 K), the value at the join, the value at 0.0253 eV and sigma * sqrt(E) from 1e-5 to 0.0253 eV.

Proton-induced reactions (``TENDL_PROTON``) come from TENDL-2025, a library calculated with the model code TALYS:
section MF 3 of ``data/external/2026-10-11/tendl2025/p/<Z>_<A>.endf`` as tabulated, up to 30 MeV (above 30 MeV
TENDL gives no separate channels). Where the file splits the product into ground state and isomer (MF 10), those
parts are written as separate files ``..._g.csv`` and ``..._m.csv``; they are copied, never summed. Checked for
every entry: target and projectile of the file, linear interpolation, the Q-value of the section against AME2020,
the first tabulated energy against the threshold from that Q-value, and ground state plus isomer against the total.

Values are written with seven significant figures.
"""
import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from xsml import libraries as xl  # noqa: E402
from xsml.features import element_symbols  # noqa: E402

LIBRARIES = ["ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"]
TOLERANCE_DEX = 0.05

# (Z, A, MT, kinds, options). One line per reaction.
REACTIONS = [
    (17, 35, 103, ["njoy"], {}),                      # Cl-35 (n,p)
    (26, 56, 4, ["mf3"], {}),                         # Fe-56 inelastic
    (62, 152, 102, ["nndc", "bins"], {}),             # Sm-152 (n,g): Sm-153
    # medical
    (71, 176, 102, ["bins"], {}),                     # Lu-176 (n,g): Lu-177
    (70, 176, 102, ["bins"], {}),                     # Yb-176 (n,g): Yb-177, Lu-177
    (42, 98, 102, ["bins"], {}),                      # Mo-98 (n,g): Mo-99
    (42, 100, 16, ["mf3", "bins"], {}),                 # Mo-100 (n,2n): Mo-99
    (67, 165, 102, ["nndc", "bins"], {"thin": 4}),    # Ho-165 (n,g): Ho-166
    (30, 64, 103, ["mf3", "bins"], {}),                 # Zn-64 (n,p): Cu-64
    (30, 67, 103, ["mf3", "bins"], {}),                 # Zn-67 (n,p): Cu-67
    (30, 68, 103, ["mf3"], {}),                       # Zn-68 (n,p)
    (22, 47, 103, ["mf3", "bins"], {}),                 # Ti-47 (n,p): Sc-47
    # industry
    (92, 238, 102, ["bins"], {}),                     # U-238 (n,g)
    (92, 235, 18, ["nndc", "bins"], {"thin": 8}),     # U-235 fission
    (94, 239, 18, ["bins"], {}),                      # Pu-239 fission
    (27, 59, 102, ["nndc", "bins"], {"thin": 2}),     # Co-59 (n,g)
    (28, 58, 103, ["mf3", "bins"], {}),                 # Ni-58 (n,p)
    (26, 54, 103, ["mf3", "bins"], {}),                 # Fe-54 (n,p)
    (26, 56, 102, ["bins"], {}),                      # Fe-56 (n,g)
    (25, 55, 102, ["bins"], {}),                      # Mn-55 (n,g)
    (40, 90, 16, ["mf3", "bins"], {}),                  # Zr-90 (n,2n)
    (79, 197, 102, ["nndc", "bins"], {"thin": 4}),    # Au-197 (n,g)
    (26, 56, 1, ["bins"], {}),                        # Fe-56 total
    (26, 56, 2, ["bins"], {}),                        # Fe-56 elastic
]

# (Z, A, MT): proton-induced, TENDL-2025. One line per reaction; the comment names the product.
TENDL_PROTON = [
    (42, 100, 16),   # Mo-100 (p,2n) Tc-99, isomer Tc-99m
    (42, 100, 4),    # Mo-100 (p,n) Tc-100
    (42, 94, 4),     # Mo-94 (p,n) Tc-94, isomer Tc-94m
    (52, 124, 4),    # Te-124 (p,n) I-124
    (52, 124, 16),   # Te-124 (p,2n) I-123
    (54, 124, 16),   # Xe-124 (p,2n) Cs-123, decays to I-123
    (54, 124, 28),   # Xe-124 (p,np) Xe-123, decays to I-123
    (48, 112, 16),   # Cd-112 (p,2n) In-111
    (32, 70, 16),    # Ge-70 (p,2n) As-69
    (20, 44, 4),     # Ca-44 (p,n) Sc-44, isomer Sc-44m
    (20, 48, 16),    # Ca-48 (p,2n) Sc-47
    (21, 45, 4),     # Sc-45 (p,n) Ti-45
    (24, 52, 4),     # Cr-52 (p,n) Mn-52, isomer Mn-52m
    (28, 58, 107),   # Ni-58 (p,a) Co-55
    (28, 60, 4),     # Ni-60 (p,n) Cu-60
    (28, 62, 4),     # Ni-62 (p,n) Cu-62
    (30, 66, 4),     # Zn-66 (p,n) Ga-66
    (30, 70, 107),   # Zn-70 (p,a) Cu-67
    (33, 75, 17),    # As-75 (p,3n) Se-73
    (34, 77, 4),     # Se-77 (p,n) Br-77
    (34, 78, 16),    # Se-78 (p,2n) Br-77
    (36, 82, 16),    # Kr-82 (p,2n) Rb-81
    (38, 86, 4),     # Sr-86 (p,n) Y-86
    (40, 90, 4),     # Zr-90 (p,n) Nb-90
    (7, 14, 107),    # N-14 (p,a) C-11
    (8, 16, 107),    # O-16 (p,a) N-13
    (64, 155, 4),    # Gd-155 (p,n) Tb-155
    (64, 156, 16),   # Gd-156 (p,2n) Tb-155
    (66, 162, 16),   # Dy-162 (p,2n) Ho-161
    (67, 165, 4),    # Ho-165 (p,n) Er-165
    (69, 169, 4),    # Tm-169 (p,n) Yb-169
    (79, 197, 4),    # Au-197 (p,n) Hg-197
    (88, 226, 16),   # Ra-226 (p,2n) Ac-225
]

ARCHIVES = {"ENDF-B-VII.1": "ENDF-B-VII.1-neutrons.zip", "ENDF-B-VIII.0": "ENDF-B-VIII.0_neutrons.zip", "JENDL-5": "jendl5-n.tar.gz"}


def member_name(lib: str, z: int, a: int) -> str:
    sym = element_symbols()[z]
    return f"n_{z:03d}-{sym}-{a:03d}.dat" if lib == "JENDL-5" else f"n-{z:03d}_{sym}_{a:03d}.endf"


def read_members(archive: Path, wanted: set) -> dict:
    """{member file name: text} for the wanted members of a zip or tar.gz archive (one pass)."""
    out = {}
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = name.split("/")[-1]
                if base in wanted:
                    out[base] = z.read(name).decode("latin-1")
    else:
        with tarfile.open(archive, "r|gz") as tf:
            for m in tf:
                base = m.name.split("/")[-1]
                if base in wanted:
                    out[base] = tf.extractfile(m).read().decode("latin-1")
                    if len(out) == len(wanted):
                        break
    return out


def path_of(z: int, a: int, mt: int, bins: bool = False) -> Path:
    p = xl._file((z, a), mt, REPO / "data")
    return p.with_name(p.stem + "_bins.csv") if bins else p


def write(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.7g", lineterminator="\n")


def check(bin_means, reference: dict, e: np.ndarray, s: np.ndarray) -> float:
    """Largest difference (dex) between the bin averages of a curve and the NJOY bin averages, over the bins both have."""
    # the table is linear between its points; a fine grid makes the average of the first bin above a threshold comparable
    fine = np.unique(np.concatenate([e, np.logspace(np.log10(e[e > 0][0]), np.log10(e[-1]), 20000)]))
    mine = bin_means(fine, np.interp(fine, e, s))
    d = [abs(mine[k] - reference[str(k)]) for k in mine if str(k) in reference and np.isfinite(mine[k]) and np.isfinite(reference[str(k)])]
    return max(d) if d else float("nan")


CC = 2.196771e-3                # k (1e12 cm^-1) = CC * sqrt(mu * E), mu in neutron masses, E in eV (centre of mass)
ALPHA = 7.2973525693e-3         # fine-structure constant
NEUTRON_EV = 939.56542052e6     # neutron mass, eV
_COULOMB = {}


def _neutral_penetrability(L: int, rho: np.ndarray) -> np.ndarray:
    r2 = rho * rho
    return [rho, rho * r2 / (1 + r2), rho * r2 * r2 / (9 + 3 * r2 + r2 * r2), rho * r2 ** 3 / (225 + 45 * r2 + 6 * r2 * r2 + r2 ** 3)][L]


class _Channel:
    """A particle channel of a spin group of an LRF 7 range: its penetrability as a function of the entrance energy."""

    def __init__(self, pair, row, e_cm_low: float, e_cm_high: float):
        self.ma, self.mb, self.zz, self.q, self.mt = pair[0], pair[1], pair[2] * pair[3], pair[6], int(round(pair[9]))
        self.L, self.radius = int(round(row[1])), row[5]                           # the true radius APT enters the penetrability
        self.mu = self.ma * self.mb / (self.ma + self.mb)
        if self.zz:
            # Coulomb penetrability rho / (F^2 + G^2), exact (mpmath) at 200 nodes; a cubic spline of its logarithm between them
            from scipy.interpolate import CubicSpline
            low, high = np.floor((e_cm_low + self.q) / 1e5) * 1e5, np.ceil((e_cm_high + self.q) / 1e5) * 1e5
            if low <= 0:
                raise ValueError("charged-particle channel at or below its threshold")
            key = (self.L, self.radius, self.mu, self.zz, low, high)
            if key not in _COULOMB:
                nodes = np.linspace(low, high, 200)
                _COULOMB[key] = CubicSpline(nodes, np.log([self._coulomb(v) for v in nodes]))
            self._spline = _COULOMB[key]

    def _coulomb(self, e: float) -> float:
        import mpmath
        rho = CC * np.sqrt(self.mu * e) * self.radius
        eta = self.zz * ALPHA * np.sqrt(self.mu * NEUTRON_EV / (2 * e))
        f, g = mpmath.coulombf(self.L, eta, rho), mpmath.coulombg(self.L, eta, rho)
        return float(rho / (f * f + g * g))

    def penetrability(self, e_cm_entrance) -> np.ndarray:
        e = np.asarray(e_cm_entrance, dtype=float) + self.q
        return np.exp(self._spline(e)) if self.zz else _neutral_penetrability(self.L, CC * np.sqrt(self.mu * np.abs(e)) * self.radius)


def rml_cross_section(r, energy_eV, mt_out: int = 600) -> np.ndarray:
    """Cross section (barn, 0 K) from the neutron channels to the channels of ``mt_out``, from a resolved range in the
    R-matrix limited format (LRF 7) in the Reich-Moore approximation (KRM 3), without shift factors.

    ``r`` is a range of ``exforlint.data.endf6.mf2_ranges``. For every spin group: R = sum over the resonances of
    g g^T / (E_r - E - i Gamma_gamma / 2), with the reduced amplitudes g = sign(Gamma) sqrt(|Gamma| / (2 P(E_r))), and
    sigma = pi / k^2 * g_J * 4 P_n P_p |[(1 - i R P)^-1 R]_np|^2 (ENDF-6 manual, appendix D).
    """
    p = r.params
    if not (r.lrf == 7 and p["krm"] == 3 and all(q[8] <= 0 for q in p["pairs"]) and all(g["kbk"] == 0 and g["kps"] == 0 for g in p["sg"])):
        raise ValueError("only LRF 7 with KRM 3, no shift, no background R-matrix and no tabulated phase shifts")
    pairs = p["pairs"]
    neutron = [q for q in pairs if int(round(q[9])) == 2][0]
    to_cm = neutron[1] / (neutron[0] + neutron[1])
    E = np.asarray(energy_eV, dtype=float)
    ecm = E * to_cm
    k2 = CC ** 2 * neutron[0] * neutron[1] / (neutron[0] + neutron[1]) * ecm
    sigma = np.zeros_like(E)
    for g in p["sg"]:
        g_j = (2 * abs(g["aj"]) + 1) / ((2 * neutron[4] + 1) * (2 * neutron[5] + 1))
        rows, res = g["chan"], g["res"]
        mts = [int(round(pairs[int(round(row[0])) - 1][9])) for row in rows]
        if mts.count(102) != 1:
            raise ValueError("one eliminated capture channel per spin group is expected")
        part = [i for i, mt in enumerate(mts) if mt != 102]
        if not any(np.any(res[:, 1 + i]) for i in part if mts[i] == mt_out):
            continue                                                                # no width in the outgoing channel
        ch = []
        for i in part:
            need = np.concatenate([[ecm.min(), ecm.max()], res[res[:, 1 + i] != 0, 0] * to_cm])
            ch.append(_Channel(pairs[int(round(rows[i][0])) - 1], rows[i], need.min(), need.max()))
        e_r, gamma_g = res[:, 0], np.abs(res[:, 1 + mts.index(102)])
        amplitude = np.zeros((len(e_r), len(ch)))
        for j, (i, c) in enumerate(zip(part, ch)):
            w = res[:, 1 + i]
            ok = w != 0
            amplitude[ok, j] = np.sign(w[ok]) * np.sqrt(np.abs(w[ok]) / (2 * c.penetrability(e_r[ok] * to_cm)))
        P = np.stack([c.penetrability(ecm) for c in ch], axis=1)
        R = np.einsum("li,lj,el->eij", amplitude, amplitude, 1.0 / (e_r[None, :] - E[:, None] - 0.5j * gamma_g[None, :]))
        X = np.linalg.solve(np.eye(len(ch))[None] - 1j * R * P[:, None, :], R)
        for a, c_in in enumerate(ch):
            for b, c_out in enumerate(ch):
                if c_in.mt == 2 and c_out.mt == mt_out:
                    sigma += np.pi / k2 * g_j * 4 * P[:, a] * P[:, b] * np.abs(X[:, a, b]) ** 2
    return sigma


def linearise(function, seeds: np.ndarray, tolerance: float = 1e-3) -> tuple:
    """(E, sigma) on a grid, refined from ``seeds`` until the value in the middle of every interval is given by
    linear interpolation between its ends to ``tolerance`` (relative), as RECONR does.

    The energies are rounded to the seven significant figures with which they are written, so the values belong to
    the energies in the file; an interval that seven figures cannot divide is left as it is.
    """
    seven = lambda v: np.array([float(f"{x:.7g}") for x in v])
    e = np.unique(seven(seeds))
    s = function(e)
    for _ in range(60):
        mid = seven(0.5 * (e[:-1] + e[1:]))
        inside = (mid > e[:-1]) & (mid < e[1:])
        mid = np.where(inside, mid, 0.5 * (e[:-1] + e[1:]))
        s_mid = function(mid)
        linear = s[:-1] + (s[1:] - s[:-1]) * (mid - e[:-1]) / (e[1:] - e[:-1])
        bad = (np.abs(linear - s_mid) > tolerance * s_mid) & inside
        if not bad.any():
            return e, s
        e, s = np.concatenate([e, mid[bad]]), np.concatenate([s, s_mid[bad]])
        order = np.argsort(e)
        e, s = e[order], s[order]
    raise RuntimeError("linearisation did not converge")


def cl35_np_below(endf6, texts: dict, njoy: dict, join: float = 1e5) -> tuple:
    """{library: (E, sigma)} of Cl-35 (n,p) below ``join`` (0 K), and the notes of the checks; see "njoy" in the module docstring."""
    out, notes = {}, {}
    filed, ranges = {}, {}
    for lib in LIBRARIES:
        sections = endf6.split_sections(texts[lib])
        filed[lib] = endf6.mf3_section(sections, 103)
        (ranges[lib],) = endf6.mf2_ranges(sections)[1]
    thermal = np.array([1e-5, 1e-3, 0.0253])
    for lib in LIBRARIES:
        rec, r = filed[lib], ranges[lib]
        low = rec["x"] < join
        if r.el != rec["x"][0] or r.eh < join:
            raise SystemExit(f"{lib}: the resolved range {r.el} to {r.eh} eV does not cover {rec['x'][0]} to {join} eV")
        function = lambda e, r=r: rml_cross_section(r, e)
        if rec["y"][low].max() < 1e-15:                                             # placeholder in MF 3: the curve is in MF 2
            widths = np.concatenate([np.abs(g["res"][:, 1:]).sum(axis=1) for g in r.params["sg"]])
            centres = np.concatenate([g["res"][:, 0] for g in r.params["sg"]])
            steps = np.array([0.0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
            seeds = np.concatenate([np.logspace(np.log10(r.el), np.log10(join), 201), [0.0253]] + [c + sign * steps * w for c, w in zip(centres, widths) for sign in (-1, 1)])
            seeds = np.concatenate([[r.el, join], seeds[(seeds > r.el) & (seeds < join)]])
            e, s = linearise(function, seeds)
            s = s + endf6.interpolate(rec, e)                                       # the section of MF 3 is added to the resonance part
            at_join = s[-1] / njoy[lib]["S"][0] - 1
            if e[-1] != join or njoy[lib]["E"][0] != join or abs(at_join) > 1e-3:
                raise SystemExit(f"{lib}: the calculated curve and the NJOY curve differ by {at_join:.2e} at {join} eV")
            e, s = e[:-1], s[:-1]                                                   # the point at the join is the NJOY point
            x, y = np.asarray(njoy[lib]["E"]), np.asarray(njoy[lib]["S"])
            band = (x < 1e6) & (y > 0)
            against_njoy = float(np.median(np.abs(function(x[band]) / y[band] - 1)))
            if against_njoy > 1e-3:
                raise SystemExit(f"{lib}: the calculated curve differs from the NJOY curve by {against_njoy:.2e} (median, 0.1 to 1 MeV)")
            notes[lib] = f"below 0.1 MeV calculated from MF 2 (LRF 7), {len(e)} points; against NJOY: {at_join:+.1e} at the join, median {against_njoy:.1e} from 0.1 to 1 MeV"
        else:                                                                       # the curve is in MF 3: copied as filed
            if rec["int"][0] != 2 or rec["nbt"][0] < low.sum() or not (np.diff(rec["x"][low]) >= 0).all():
                raise SystemExit(f"{lib}: MF 3 / MT 103 below {join} eV is not a table for linear interpolation")
            e, s = rec["x"][low], rec["y"][low]
            against_filed = np.abs(function(e) / s - 1)
            if np.median(against_filed) > 1e-3:
                raise SystemExit(f"{lib}: the calculation from MF 2 differs from MF 3 as filed by {np.median(against_filed):.2e} (median)")
            notes[lib] = (f"below 0.1 MeV MF 3 / MT 103 as filed, {len(e)} points; the calculation from MF 2 of this file agrees with it to "
                          f"{np.median(against_filed):.1e} (median), {np.quantile(against_filed, 0.99):.1e} (99 % of the points), {against_filed.max():.1e} (largest)")
            notes[lib] += f"; {int((np.diff(e) == 0).sum())} energies are given twice in the file (steps), kept"
        t = np.interp(thermal, e, s)
        if e[0] != 1e-5 or (s <= 0).any() or abs(t[0] * np.sqrt(thermal[0]) / (t[2] * np.sqrt(thermal[2])) - 1) > 2e-3:
            raise SystemExit(f"{lib}: first energy {e[0]} eV, or the curve is not 1/v below 0.0253 eV")
        notes[lib] += f"; {t[2]:.4f} b at 0.0253 eV; sigma * sqrt(E) at 1e-5, 1e-3 and 0.0253 eV: {t[0] * np.sqrt(thermal[0]):.5f}, {t[1] * np.sqrt(thermal[1]):.5f}, {t[2] * np.sqrt(thermal[2]):.5f}"
        out[lib] = (e, s)
    return out, notes


def tendl(source: Path, endf6, report: list) -> None:
    """The proton-induced TENDL-2025 curves of ``TENDL_PROTON``, with the checks of the module docstring."""
    from xsml import features

    folder = source / "data" / "external" / "2026-10-11" / "tendl2025" / "p"
    for z, n, mt in TENDL_PROTON:
        label = f"{element_symbols()[z]}-{n} (p) MT {mt}"
        f = folder / f"{z}_{n}.endf"
        if not f.exists():
            report.append((label, "pointwise (MF 3)", "TENDL-2025", 0, "LEFT OUT: target not on disk"))
            continue
        text = f.read_text(encoding="latin-1")
        sections = endf6.split_sections(text)
        za, awr, lis = endf6.target_za(sections)
        nsub = int(round(endf6.parse_float(endf6._fields(text.splitlines()[3])[4])))        # 10010: incident proton
        awi = endf6.parse_float(endf6._fields(text.splitlines()[3])[0])                     # mass of the projectile in neutron masses
        rec = endf6.mf3_section(sections, mt)
        if rec is None or za != 1000 * z + n or lis != 0 or nsub != 10010 or not text.startswith(" TENDL-2025"):
            report.append((label, "pointwise (MF 3)", "TENDL-2025", 0, f"LEFT OUT: section missing or header mismatch (ZA {za}, LIS {lis}, NSUB {nsub})"))
            continue
        q_ame = features.q_value(z, n, "p", features.MT_EJECTILES[mt])
        threshold = max(-rec["c1"], 0.0) * (awr + awi) / awr
        parts = [("", rec, "")]
        lines = sections.get((10, mt))
        if lines:
            states, i = [], 1
            for _ in range(int(round(endf6.parse_float(endf6._fields(lines[0])[4])))):
                r, i = endf6.tab1(lines, i)
                states.append(r)
            states.sort(key=lambda r: r["l2"])
            isomers = [r for r in states if r["l2"] > 0]
            grid = rec["x"][rec["x"] < 3e7]
            total = sum(np.interp(grid, r["x"], r["y"], left=0, right=0) for r in states)
            worst = float(np.abs(total - np.interp(grid, rec["x"], rec["y"])).max())
            for r in states:
                tag = "g" if r["l2"] == 0 else ("m" if len(isomers) == 1 else f"m{isomers.index(r) + 1}")
                parts.append((tag, r, f"level {r['l2']}, {(r['c1'] - r['c2']) / 1e3:.1f} keV above the ground state; states sum to the total within {worst:.1e} b"))
        for tag, r, note in parts:
            if set(r["int"].tolist()) != {2} or abs(rec["c1"] / 1e6 - q_ame) > 1e-3:
                report.append((label + " " + tag, "pointwise (MF 3)", "TENDL-2025", 0, f"LEFT OUT: interpolation {set(r['int'].tolist())} or Q {rec['c1'] / 1e6:.4f} against AME {q_ame:.4f}"))
                continue
            keep = np.flatnonzero(r["x"] <= 3e7)
            keep = keep[np.concatenate([[True], np.diff(r["x"][keep]) > 0])]                     # the table repeats 30 MeV with a zero: keep the first
            e, sig = r["x"][keep], r["y"][keep]
            own_threshold = max(-r["c2"], 0.0) * (awr + awi) / awr
            if q_ame < 0 and abs(e[0] - own_threshold) > 2e3 + 1e-4 * own_threshold:
                report.append((label + " " + tag, "pointwise", "TENDL-2025", 0, f"LEFT OUT: first energy {e[0]:.6g} eV against threshold {own_threshold:.6g} eV"))
                continue
            path = xl._file((z, n), mt, REPO / "data", "p", tag or None)
            write(pd.DataFrame({"library": "TENDL-2025", "energy_eV": e, "sigma_b": sig}), path)
            kind = "MF 3" if not tag else "MF 10"
            report.append((label + (" " + tag if tag else ""), f"pointwise ({kind})", "TENDL-2025", len(e),
                           f"Q {rec['c1'] / 1e6:.5f} MeV (AME2020 {q_ame:.5f}); first energy {e[0] / 1e6:.5f} MeV, threshold {threshold / 1e6:.5f}; max {sig.max():.4f} b" + ("; " + note if note else "")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--nndc-cache", default=str(Path.home() / ".nucml_next" / "nndc_cache"))
    a = ap.parse_args()
    source = Path(a.source).resolve()
    sys.path.insert(0, str(source))
    from exforlint.data import endf6, library_curves

    bins_json = json.load(open(source / "docs" / "paper" / "data" / "library_bins_njoy.json"))
    need = {lib: {member_name(lib, z, n) for z, n, mt, kinds, opt in REACTIONS if "mf3" in kinds or "njoy" in kinds} for lib in LIBRARIES}
    texts = {lib: read_members(source / "data" / "external" / "2026-10-08" / ARCHIVES[lib], need[lib]) for lib in LIBRARIES}
    report = []
    for z, n, mt, kinds, opt in REACTIONS:
        gid = f"{z}_{n}_{mt}"
        label = f"{element_symbols()[z]}-{n} MT {mt}"
        for kind in kinds:
            if kind == "njoy":
                curves = json.load(open(source / "docs" / "paper" / "data" / "cl35_library_curves.json"))
                below, notes = cl35_np_below(endf6, {lib: texts[lib][member_name(lib, z, n)] for lib in LIBRARIES}, curves)
                t = pd.concat([pd.DataFrame({"library": lib, "energy_eV": np.concatenate([below[lib][0], curves[lib]["E"]]), "sigma_b": np.concatenate([below[lib][1], curves[lib]["S"]])})
                               for lib in LIBRARIES], ignore_index=True)
                write(t, path_of(z, n, mt))
                for lib in LIBRARIES:
                    report.append((label, "pointwise (NJOY from 0.1 MeV)", lib, int((t.library == lib).sum()), notes[lib]))
            elif kind == "mf3":
                parts = []
                for lib in LIBRARIES:
                    text = texts[lib].get(member_name(lib, z, n))
                    rec = endf6.mf3_section(endf6.split_sections(text), mt) if text else None
                    if rec is None:
                        report.append((label, "pointwise (MF 3)", lib, 0, "LEFT OUT: no file or no section"))
                        continue
                    if set(rec["int"].tolist()) != {2}:
                        report.append((label, "pointwise (MF 3)", lib, 0, f"LEFT OUT: interpolation laws {sorted(set(rec['int'].tolist()))}, not linear"))
                        continue
                    keep = rec["x"] <= 2e7
                    e, s = rec["x"][keep], rec["y"][keep]
                    ref = bins_json[lib].get(gid, {}).get("bins")
                    worst = check(library_curves.bin_means, ref, e, s) if ref else float("nan")
                    if not ref:
                        # no NJOY bins to compare with: accept only if the resonance formalisms of the file cannot hold this channel
                        formalisms = sorted({r.lrf for r in endf6.mf2_ranges(endf6.split_sections(text))[1] if r.lru == 1})
                        if not set(formalisms) <= {1, 2, 3}:
                            report.append((label, "pointwise (MF 3)", lib, 0, f"LEFT OUT: no NJOY bins and resonance formalism LRF {formalisms}"))
                            continue
                    if ref and not worst <= TOLERANCE_DEX:
                        report.append((label, "pointwise (MF 3)", lib, 0, f"LEFT OUT: differs from the NJOY bin averages by {worst:.3f} dex"))
                        continue
                    parts.append(pd.DataFrame({"library": lib, "energy_eV": e, "sigma_b": s}))
                    report.append((label, "pointwise (MF 3)", lib, len(e), f"max {worst:.3f} dex from NJOY bins" if ref else f"no NJOY bins; resolved resonances LRF {formalisms} (neutron, photon and fission widths only)"))
                if parts:
                    write(pd.concat(parts, ignore_index=True), path_of(z, n, mt))
            elif kind == "nndc":
                f = Path(a.nndc_cache) / f"{element_symbols()[z]}-{n}_MT{mt}_endfb8.0.npz"
                if not f.exists():
                    report.append((label, "pointwise (IAEA service)", "ENDF-B-VIII.0", 0, "LEFT OUT: not in the cache"))
                    continue
                c = np.load(f)
                keep = c["energies"] <= 2e7
                e, s = c["energies"][keep], c["xs"][keep]
                worst = check(library_curves.bin_means, bins_json["ENDF-B-VIII.0"][gid]["bins"], e, s)
                if not worst <= TOLERANCE_DEX:
                    report.append((label, "pointwise (IAEA service)", "ENDF-B-VIII.0", 0, f"LEFT OUT: differs from the NJOY bin averages by {worst:.3f} dex"))
                    continue
                note = f"max {worst:.3f} dex from NJOY bins (full table)"
                step = opt.get("thin", 1)
                if step > 1:
                    sel = (np.arange(len(e)) % step == 0) | (e > 1e5)
                    sel[-1] = True
                    note += f"; thinned from {len(e)} points (every {step}th below 100 keV)"
                    e, s = e[sel], s[sel]
                write(pd.DataFrame({"library": "ENDF-B-VIII.0", "energy_eV": e, "sigma_b": s}), path_of(z, n, mt))
                report.append((label, "pointwise (IAEA service)", "ENDF-B-VIII.0", len(e), note))
            elif kind == "bins":
                parts = []
                for lib in LIBRARIES:
                    b = bins_json[lib].get(gid, {}).get("bins")
                    if not b:
                        report.append((label, "bin averages", lib, 0, "LEFT OUT: not in the NJOY bin file"))
                        continue
                    k = np.array(sorted(int(x) for x in b))
                    value = np.array([b[str(x)] for x in k], dtype=float)
                    ok = np.isfinite(value)
                    # the curves end at 20 MeV: the last bin is averaged over its part below 20 MeV, and its upper edge is written as 20 MeV
                    parts.append(pd.DataFrame({"library": lib, "e_low_eV": 10 ** (0.25 * k[ok]), "e_high_eV": np.minimum(10 ** (0.25 * (k[ok] + 1)), 2e7), "sigma_b": 10 ** value[ok]}))
                    report.append((label, "bin averages", lib, int(ok.sum()), ""))
                if parts:
                    write(pd.concat(parts, ignore_index=True), path_of(z, n, mt, bins=True))
            else:
                raise SystemExit(f"unknown kind {kind}")
    tendl(source, endf6, report)
    print(pd.DataFrame(report, columns=["reaction", "form", "library", "points", "note"]).to_string(index=False))
    total = sum(f.stat().st_size for f in (REPO / "data" / "libraries").glob("*.csv"))
    print(f"data/libraries: {len(list((REPO / 'data' / 'libraries').glob('*.csv')))} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
