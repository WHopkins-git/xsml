#!/usr/bin/env python
"""Write the evaluated-library files of ``data/libraries/`` (run by the supervisor, not by students).

    python scripts/build_library_curves.py --source <path to the EXFOR-ML working copy>

Everything is copied or parsed from files on disk; nothing is downloaded and nothing is computed beyond
reading. Sources, relative to ``--source`` unless stated:

Cl35_np.csv        pointwise Cl-35 (n,p), MT 103, 0.1 to 20 MeV, ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5:
                   ``docs/paper/data/cl35_library_curves.json`` (NJOY2016, RECONR and BROADR at 293.6 K).
Fe56_ninl.csv      pointwise Fe-56 (n,n'), MT 4, threshold to 20 MeV, the same three libraries: section MF 3 / MT 4
                   of the ENDF-6 files in ``data/external/2026-10-08/`` (the library archives as distributed), read
                   with ``exforlint.data.endf6`` as tabulated (0 K, no processing). The inelastic threshold lies
                   above the resolved-resonance range, so the section is the complete cross section.
Sm152_ng.csv       pointwise Sm-152 (n,g), MT 102, 1e-5 eV to 20 MeV, ENDF/B-VIII.0 only: the cache file
                   ``Sm-152_MT102_endfb8.0.npz`` (``--nndc-cache``), a tabulated section retrieved earlier from the
                   IAEA-NDS ENDF service by ``exforlint.visualization.endf_reader`` (resonances reconstructed by
                   that service).
Sm152_ng_bins.csv  Sm-152 (n,g) of the three libraries as averages over bins of a quarter of a decade in energy:
                   ``docs/paper/data/library_bins_njoy.json`` (NJOY2016, RECONR and BROADR at 293.6 K, linear
                   energy average of the cross section over the part of each bin where it is positive).

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
LIBRARIES = ["ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"]
ARCHIVES = {"ENDF-B-VII.1": ("ENDF-B-VII.1-neutrons.zip", "n-026_Fe_056.endf"), "ENDF-B-VIII.0": ("ENDF-B-VIII.0_neutrons.zip", "n-026_Fe_056.endf"),
            "JENDL-5": ("jendl5-n.tar.gz", "n_026-Fe-056.dat")}


def write(table: pd.DataFrame, name: str) -> None:
    dest = REPO / "data" / "libraries"
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / name, index=False, float_format="%.7g", lineterminator="\n")
    print(name, len(table), "rows;", table.groupby("library").size().to_dict())


def member_text(archive: Path, suffix: str) -> str:
    """Text of the archive member whose name ends with ``suffix``."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.endswith(suffix):
                    return z.read(name).decode("latin-1")
    else:
        with tarfile.open(archive, "r|gz") as tf:
            for m in tf:
                if m.name.endswith(suffix):
                    return tf.extractfile(m).read().decode("latin-1")
    raise FileNotFoundError(f"{suffix} not in {archive}")


def cl35(source: Path) -> None:
    curves = json.load(open(source / "docs" / "paper" / "data" / "cl35_library_curves.json"))
    write(pd.concat([pd.DataFrame({"library": lib, "energy_eV": curves[lib]["E"], "sigma_b": curves[lib]["S"]}) for lib in LIBRARIES], ignore_index=True), "Cl35_np.csv")


def fe56(source: Path) -> None:
    sys.path.insert(0, str(source))
    from exforlint.data import endf6

    parts = []
    for lib in LIBRARIES:
        archive, suffix = ARCHIVES[lib]
        sections = endf6.split_sections(member_text(source / "data" / "external" / "2026-10-08" / archive, suffix))
        rec = endf6.mf3_section(sections, 4)
        if set(rec["int"].tolist()) != {2}:
            raise SystemExit(f"{lib}: MF3/MT4 is not lin-lin throughout; a pointwise copy would misrepresent it")
        keep = rec["x"] <= 2e7
        parts.append(pd.DataFrame({"library": lib, "energy_eV": rec["x"][keep], "sigma_b": rec["y"][keep]}))
    write(pd.concat(parts, ignore_index=True), "Fe56_ninl.csv")


def sm152(cache: Path) -> None:
    z = np.load(cache / "Sm-152_MT102_endfb8.0.npz")
    keep = z["energies"] <= 2e7
    write(pd.DataFrame({"library": "ENDF-B-VIII.0", "energy_eV": z["energies"][keep], "sigma_b": z["xs"][keep]}), "Sm152_ng.csv")


def bins(source: Path, gid: str, name: str) -> None:
    b = json.load(open(source / "docs" / "paper" / "data" / "library_bins_njoy.json"))
    parts = []
    for lib in LIBRARIES:
        k = np.array(sorted(int(x) for x in b[lib][gid]["bins"]))
        value = np.array([b[lib][gid]["bins"][str(x)] for x in k], dtype=float)
        ok = np.isfinite(value)
        # the curves end at 20 MeV: the last bin is averaged over its part below 20 MeV, and its upper edge is written as 20 MeV
        parts.append(pd.DataFrame({"library": lib, "e_low_eV": 10 ** (0.25 * k[ok]), "e_high_eV": np.minimum(10 ** (0.25 * (k[ok] + 1)), 2e7), "sigma_b": 10 ** value[ok]}))
    write(pd.concat(parts, ignore_index=True), name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--nndc-cache", default=str(Path.home() / ".nucml_next" / "nndc_cache"))
    a = ap.parse_args()
    source = Path(a.source).resolve()
    cl35(source)
    fe56(source)
    sm152(Path(a.nndc_cache))
    bins(source, "62_152_102", "Sm152_ng_bins.csv")


if __name__ == "__main__":
    main()
