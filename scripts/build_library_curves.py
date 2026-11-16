#!/usr/bin/env python
"""Write the evaluated-library files of ``data/libraries/`` (run by the supervisor, not by students).

    python scripts/build_library_curves.py --source <path to the EXFOR-ML working copy>

Adding a reaction is one line in ``REACTIONS`` below. Everything is copied or parsed from files on disk; nothing
is downloaded, interpolated or computed beyond reading. Three kinds of file are written, named by
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
"njoy"  Cl-35 (n,p) only: pointwise NJOY2016 curves of ``docs/paper/data/cl35_library_curves.json``.

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--nndc-cache", default=str(Path.home() / ".nucml_next" / "nndc_cache"))
    a = ap.parse_args()
    source = Path(a.source).resolve()
    sys.path.insert(0, str(source))
    from exforlint.data import endf6, library_curves

    bins_json = json.load(open(source / "docs" / "paper" / "data" / "library_bins_njoy.json"))
    need = {lib: {member_name(lib, z, n) for z, n, mt, kinds, opt in REACTIONS if "mf3" in kinds} for lib in LIBRARIES}
    texts = {lib: read_members(source / "data" / "external" / "2026-10-08" / ARCHIVES[lib], need[lib]) for lib in LIBRARIES}
    report = []
    for z, n, mt, kinds, opt in REACTIONS:
        gid = f"{z}_{n}_{mt}"
        label = f"{element_symbols()[z]}-{n} MT {mt}"
        for kind in kinds:
            if kind == "njoy":
                curves = json.load(open(source / "docs" / "paper" / "data" / "cl35_library_curves.json"))
                t = pd.concat([pd.DataFrame({"library": lib, "energy_eV": curves[lib]["E"], "sigma_b": curves[lib]["S"]}) for lib in LIBRARIES], ignore_index=True)
                write(t, path_of(z, n, mt))
                report.append((label, "pointwise (NJOY)", "all three", len(t), ""))
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
    print(pd.DataFrame(report, columns=["reaction", "form", "library", "points", "note"]).to_string(index=False))
    total = sum(f.stat().st_size for f in (REPO / "data" / "libraries").glob("*.csv"))
    print(f"data/libraries: {len(list((REPO / 'data' / 'libraries').glob('*.csv')))} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
