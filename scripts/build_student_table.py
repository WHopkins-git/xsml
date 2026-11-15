#!/usr/bin/env python
"""Build the student table ``dist/exfor_linted_<version>.parquet`` and ``data/MANIFEST.json``.

This script is run by the supervisor, not by students: it needs the EXFOR-ML working copy
(the linter's output frames and the X4Pro SQLite database), which students do not have.

    python scripts/build_student_table.py --source <path to the EXFOR-ML working copy>

Inputs, all read-only, relative to ``--source``:

  data/linter_db/released_frame.parquet             one row per measured point of the linter's candidate set
                                                    (energies after the linter's frame corrections)
  data/linter_db/released_point_verdicts.parquet    row-aligned with the frame: ``keep`` and ``reason``
  data/linter_db/queue_v2_refit_2_datasets.parquet  per dataset: consensus ``verdict`` and ``offset`` (dex)
  data/exfor_consensus_pure.parquet                 earlier consensus table (2026-10-27): ``ds_weight`` per dataset,
                                                    ``pt_outlier`` per point
  data/x4sqlite1.db                                 X4Pro: first author, publication year, reference, DOI

Nothing is removed: every row of the frame is written, with the linter's ``keep`` column, so that the
default selection of ``xsml.data.load`` (``keep == True``) is a documented filter and not a hidden one.
The output is sorted (projectile, Z, A, MT, dataset, energy) and compressed with zstd.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent

FRAME = "data/linter_db/released_frame.parquet"
VERDICTS = "data/linter_db/released_point_verdicts.parquet"
CONSENSUS = "data/linter_db/queue_v2_refit_2_datasets.parquet"
EARLIER = "data/exfor_consensus_pure.parquet"
X4DB = "data/x4sqlite1.db"

COLUMNS = [
    "dataset", "entry", "subentry", "author", "year", "reference", "doi", "entry_date",
    "projectile", "Z", "A", "MT", "MF", "reaction", "product_state", "sf5", "sf8", "sf9",
    "energy_eV", "sigma_b", "unc_b",
    "keep", "reason", "ds_flag", "ds_offset_dex", "ds_weight", "pt_outlier",
    "missing_uncertainty", "relative_uncertainty_flag", "uncertainty_unit_error", "critique_kind", "energy_frame",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def blank_to_none(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    return s.where(s.notna() & (s.astype(str) != ""), None)


def build(source: Path) -> tuple[pd.DataFrame, dict]:
    f = pd.read_parquet(source / FRAME)
    v = pd.read_parquet(source / VERDICTS, columns=["keep", "reason"])
    if len(f) != len(v):
        raise SystemExit("frame and point verdicts are not row-aligned")
    out = pd.DataFrame({
        "dataset": f.Entry.astype(str),
        "projectile": f.Projectile.astype(str),
        "Z": f.Z.astype("int16"), "A": f.A.astype("int16"), "MT": f.MT.astype("int16"), "MF": f.MF.astype("int16"),
        "reaction": f.reaction.astype(str),
        "product_state": f.ProductState.fillna("").astype(str),
        "sf5": blank_to_none(f.sf5), "sf8": blank_to_none(f.sf8), "sf9": blank_to_none(f.sf9),
        "energy_eV": f.Energy.astype("float64"), "sigma_b": f.CrossSection.astype("float64"), "unc_b": f.Uncertainty.astype("float64"),
        "entry_date": f.entry_date.astype(str),
        "keep": v.keep.to_numpy(bool), "reason": v.reason.astype(str).to_numpy(),
        "missing_uncertainty": f.missing_uncertainty.to_numpy(bool),
        "relative_uncertainty_flag": f.relative_uncertainty_flag.to_numpy(bool),
        "uncertainty_unit_error": f.uncertainty_unit_error.to_numpy(bool),
        "critique_kind": f.critique_kind.fillna("").astype(str),
        "energy_frame": f.energy_frame.fillna("").astype(str),
    })
    out["entry"] = out.dataset.str[:5]
    out["subentry"] = out.dataset.str[:8]
    checks = {"rows": int(len(out))}

    # consensus of the release: one verdict and one offset per dataset
    q = pd.read_parquet(source / CONSENSUS).drop_duplicates("ds").set_index("ds")
    out["ds_flag"] = out.dataset.map(q.verdict)
    out["ds_offset_dex"] = out.dataset.map(q.offset).astype("float32")
    checks["kept_rows_with_ds_flag"] = int(out.ds_flag[out.keep].notna().sum())

    # earlier consensus table: weight per dataset, spike flag per point (matched on dataset, energy and value)
    e = pd.read_parquet(source / EARLIER, columns=["Entry", "Energy", "CrossSection", "ds_weight", "pt_outlier"])
    e["Entry"] = e.Entry.astype(str)
    w = e.drop_duplicates("Entry").set_index("Entry").ds_weight
    out["ds_weight"] = out.dataset.map(w).astype("float32")
    key = e.groupby(["Entry", "Energy", "CrossSection"], sort=False).pt_outlier.max().rename("pt_outlier").reset_index()
    m = out[["dataset", "energy_eV", "sigma_b"]].merge(key, left_on=["dataset", "energy_eV", "sigma_b"], right_on=["Entry", "Energy", "CrossSection"], how="left")
    if len(m) != len(out):
        raise SystemExit("point match of pt_outlier changed the row count")
    out["pt_outlier"] = m.pt_outlier.astype("boolean").to_numpy()
    checks["kept_rows_with_ds_weight"] = int(out.ds_weight[out.keep].notna().sum())
    checks["kept_rows_with_pt_outlier"] = int(out.pt_outlier[out.keep].notna().sum())
    del e, key, m

    # bibliography from X4Pro
    con = sqlite3.connect(f"file:{(source / X4DB).as_posix()}?mode=ro", uri=True)
    ds = pd.read_sql("select DatasetID, author1, year1 from x4pro_ds", con).drop_duplicates("DatasetID").set_index("DatasetID")
    en = pd.read_sql("select Entry, Reference1, DOI, DateDebut from ENTRY", con).drop_duplicates("Entry").set_index("Entry")
    con.close()
    out["author"] = out.dataset.map(ds.author1)
    out["year"] = pd.to_numeric(out.dataset.map(ds.year1), errors="coerce").astype("Int16")
    out["reference"] = out.entry.map(en.Reference1)
    out["doi"] = out.entry.map(en.DOI)
    checks["rows_with_author"] = int(out.author.notna().sum())
    checks["rows_with_year"] = int(out.year.notna().sum())
    checks["rows_with_reference"] = int(out.reference.notna().sum())
    checks["entry_date_equals_x4pro_DateDebut"] = bool((out.entry_date == out.entry.map(en.DateDebut).astype(str).str[:10]).all())

    out = out.sort_values(["projectile", "Z", "A", "MT", "dataset", "energy_eV"], kind="stable").reset_index(drop=True)
    for c in ["dataset", "entry", "subentry", "author", "reference", "doi", "entry_date", "projectile", "reaction", "product_state", "sf5", "sf8", "sf9", "reason",
              "ds_flag", "critique_kind", "energy_frame"]:
        out[c] = out[c].astype("category")
    checks["kept_rows"] = int(out.keep.sum())
    checks["kept_datasets"] = int(out.dataset[out.keep].nunique())
    return out[COLUMNS], checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="EXFOR-ML working copy holding the linter's frames and the X4Pro database")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out", default=str(REPO / "dist"))
    ap.add_argument("--manifest", default=str(REPO / "data" / "MANIFEST.json"))
    a = ap.parse_args()
    source = Path(a.source).resolve()

    table, checks = build(source)
    name = f"exfor_linted_{a.version}.parquet"
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path, compression="zstd", compression_level=19, row_group_size=1 << 20)

    manifest = {
        "file": name,
        "version": a.version,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": int(len(table)),
        "kept_rows": checks["kept_rows"],
        "columns": list(table.columns),
        "build_date": dt.date.today().isoformat(),
        "built_by": "scripts/build_student_table.py",
        "source": {
            "frame": FRAME, "point_verdicts": VERDICTS, "consensus": CONSENSUS, "earlier_consensus": EARLIER, "bibliography": X4DB,
            "source_commit": git_head(source),
            "sha256": {p: sha256(source / p) for p in (FRAME, VERDICTS, CONSENSUS, EARLIER)},
        },
        "checks": checks,
        "download": "TODO(supervisor): OneDrive link to the parquet",
    }
    Path(a.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("file", "sha256", "bytes", "rows", "kept_rows", "build_date")}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
