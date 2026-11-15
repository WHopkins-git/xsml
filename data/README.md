# Data

## Files

| File | In git | What it is |
|---|---|---|
| `exfor_linted_v1.parquet` | no (download) | the linted EXFOR table, one row per measured point; columns in `docs/data_dictionary.md` |
| `MANIFEST.json` | yes | file name, SHA-256, size, row counts, build date and the source frames of the table |
| `mass_1.mas20.txt` | yes | AME2020 atomic mass table, unmodified, read by `xsml.features` |
| `libraries/Cl35_np.csv` | yes | pointwise Cl-35 (n,p) cross section of three evaluated libraries |

## Download

**TODO(supervisor): insert the OneDrive link to `exfor_linted_v1.parquet` here and in the `download` field of `MANIFEST.json`.**

Place the file in this folder. `xsml.data.load()` verifies it against `MANIFEST.json`:

```
file    exfor_linted_v1.parquet
sha256  see MANIFEST.json
```

A mismatch means the download is incomplete or the file belongs to another version of the manifest.
Download it again; do not edit the manifest.

## The table

Built on the date given in `MANIFEST.json` by `scripts/build_student_table.py` from the output of the
supervisor's EXFOR linter (release frame of 2026-11-01), which reads the X4Pro database of EXFOR. It holds the
linter's candidate set: cross sections (EXFOR quantity `SIG`) for incident neutrons, protons, deuterons,
tritons, helium-3, alpha particles and photons on isotopic targets, with positive values. Every candidate
row is present, with the linter's verdict in the column `keep`; nothing was removed when the table was built.
The script, the names and SHA-256 of its input frames and the commit of the source repository are recorded
in `MANIFEST.json`. The script cannot be re-run without those inputs, which are not distributed.

## The mass table

`mass_1.mas20.txt` is the file `mass.mas20` of the Atomic Mass Evaluation 2020:
W.J. Huang, M. Wang, F.G. Kondev, G. Audi and S. Naimi, Chinese Physics C 45, 030002 (2021);
M. Wang, W.J. Huang, F.G. Kondev, G. Audi and S. Naimi, Chinese Physics C 45, 030003 (2021).

## The library curves

`libraries/Cl35_np.csv` has the columns `library`, `energy_eV`, `sigma_b`. It holds the Cl-35 (n,p) cross
section (ENDF reaction number MT 103) of ENDF/B-VII.1, ENDF/B-VIII.0 and JENDL-5 between 0.1 and 20 MeV, as
reconstructed on a pointwise grid with NJOY2016 (modules RECONR and BROADR, 293.6 K) in the supervisor's
earlier work, copied by `scripts/build_library_curves.py`. No other reaction and no other library is shipped.

## Licence of the data

The MIT licence of this repository covers the code only.

* The table is derived from EXFOR, the experimental nuclear reaction database maintained by the International
  Network of Nuclear Reaction Data Centres under the auspices of the IAEA Nuclear Data Section, read through
  the X4Pro database. Each row carries its EXFOR dataset identifier, first author, year and reference so that
  the original work can be cited.
* **TODO(supervisor): the terms under which EXFOR-derived data, the AME2020 file and the library-derived curves
  may be redistributed have not been checked. Check them before the download link or this repository is made
  public, and state them here.** Until then the table is shared with the project students only and must not be
  passed on.

## Citing

When results from the table are reported, cite EXFOR and the original publications of the datasets used
(columns `author`, `year`, `reference`, `doi`), and AME2020 for the mass-derived quantities.
