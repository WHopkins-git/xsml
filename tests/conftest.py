"""A tiny synthetic table with the columns of the real one, written to a temporary data folder with its manifest."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
ISOTOPES = [(17, 35), (17, 37), (19, 39), (20, 40), (22, 46), (22, 47), (26, 54), (26, 56), (28, 58), (28, 60), (29, 63), (30, 64)]


def toy_table() -> pd.DataFrame:
    """Twelve isotopes, (n,p) and (n,g), 15 points each; sigma depends smoothly on Z, A and energy. No randomness."""
    rows = []
    for i, (z, a) in enumerate(ISOTOPES):
        for mt in (103, 102):
            for j in range(15):
                e = 10 ** (5.5 + 0.1 * j)
                log_s = -1.0 - 0.03 * (a - 2 * z) + 0.2 * np.sin(j / 3 + i) + (0.5 if mt == 102 else 0.0) - 0.1 * (np.log10(e) - 6) ** 2
                dropped = i == 3 and j == 0
                rows.append({"dataset": f"9{i:04d}{mt:03d}", "entry": f"9{i:04d}", "subentry": f"9{i:04d}{mt:03d}", "author": f"Author{i}", "year": 1960 + 5 * i,
                             "reference": "J,TOY,1,1,1960", "doi": None, "entry_date": f"{1970 + 5 * i}-01-01", "projectile": "n", "Z": z, "A": a, "MT": mt, "MF": 3,
                             "reaction": f"{z}-XX-{a}({'N,P' if mt == 103 else 'N,G'}),,SIG", "product_state": "", "sf5": None, "sf8": None, "sf9": None, "energy_eV": e, "sigma_b": 10 ** log_s,
                             "unc_b": 0.05 * 10 ** log_s, "keep": not dropped, "reason": "class rule" if dropped else "kept",
                             "ds_flag": "ok", "ds_offset_dex": 0.0, "ds_weight": 1.0, "pt_outlier": False, "missing_uncertainty": False,
                             "relative_uncertainty_flag": False, "uncertainty_unit_error": False, "critique_kind": "", "energy_frame": ""})
    t = pd.DataFrame(rows)
    return t.astype({"Z": "int16", "A": "int16", "MT": "int16", "MF": "int16", "year": "Int16", "ds_offset_dex": "float32", "ds_weight": "float32"})


@pytest.fixture(scope="session")
def toy() -> pd.DataFrame:
    return toy_table()


@pytest.fixture()
def data_dir(tmp_path, toy) -> Path:
    """A data folder with the toy parquet, a matching MANIFEST.json and the AME table."""
    d = tmp_path / "data"
    d.mkdir()
    toy.to_parquet(d / "exfor_linted_toy.parquet", index=False)
    digest = hashlib.sha256((d / "exfor_linted_toy.parquet").read_bytes()).hexdigest()
    (d / "MANIFEST.json").write_text(json.dumps({"file": "exfor_linted_toy.parquet", "version": "toy", "sha256": digest, "download": "nowhere"}))
    shutil.copy(REPO / "data" / "mass_1.mas20.txt", d / "mass_1.mas20.txt")
    return d
