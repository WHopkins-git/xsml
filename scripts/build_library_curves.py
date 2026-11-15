#!/usr/bin/env python
"""Write ``data/libraries/Cl35_np.csv`` from the EXFOR-ML working copy (run by the supervisor, not by students).

    python scripts/build_library_curves.py --source <path to the EXFOR-ML working copy>

Source: ``docs/paper/data/cl35_library_curves.json``, the pointwise Cl-35 (n,p) cross section (MT 103) of
several evaluated libraries between 0.1 and 20 MeV, reconstructed with NJOY2016 (RECONR and BROADR at 293.6 K).
Three libraries are written: ENDF/B-VII.1, ENDF/B-VIII.0 and JENDL-5. Values are copied, rounded to seven
significant figures; points with a zero cross section (below threshold) are kept.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LIBRARIES = ["ENDF-B-VII.1", "ENDF-B-VIII.0", "JENDL-5"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    curves = json.load(open(Path(a.source) / "docs" / "paper" / "data" / "cl35_library_curves.json"))
    parts = [pd.DataFrame({"library": lib, "energy_eV": curves[lib]["E"], "sigma_b": curves[lib]["S"]}) for lib in LIBRARIES]
    out = pd.concat(parts, ignore_index=True)
    dest = REPO / "data" / "libraries"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "Cl35_np.csv", index=False, float_format="%.7g", lineterminator="\n")
    print(out.groupby("library").agg(points=("energy_eV", "size"), e_min=("energy_eV", "min"), e_max=("energy_eV", "max"), s_min=("sigma_b", "min"), s_max=("sigma_b", "max")))


if __name__ == "__main__":
    main()
