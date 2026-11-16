# Data

## Files

| File | In git | What it is |
|---|---|---|
| `exfor_linted_v1.parquet` | no (download) | the linted EXFOR table, one row per measured point; columns in `docs/data_dictionary.md` |
| `MANIFEST.json` | yes | file name, SHA-256, size, row counts, build date and the source frames of the table |
| `mass_1.mas20.txt` | yes | AME2020 atomic mass table, unmodified, read by `xsml.features` |
| `libraries/*.csv` | yes | evaluated-library curves for three reactions (see below) |

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

All files were written by `scripts/build_library_curves.py` from files on the supervisor's disk; nothing was
computed beyond reading. Pointwise files have the columns `library`, `energy_eV`, `sigma_b`.

| File | Content | Source and resolution |
|---|---|---|
| `libraries/Cl35_np.csv` | Cl-35 (n,p), MT 103, 0.1 to 20 MeV; ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5 | pointwise reconstruction with NJOY2016 (RECONR and BROADR, 293.6 K) from the supervisor's earlier work; 15,483 to 22,911 points per library |
| `libraries/Fe56_ninl.csv` | Fe-56 inelastic scattering, MT 4, threshold to 20 MeV; ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5 | section MF 3 / MT 4 of the ENDF-6 files as distributed (linear interpolation between points, 0 K, not Doppler broadened); 769, 2,925 and 7,504 points |
| `libraries/Sm152_ng.csv` | Sm-152 (n,g), MT 102, 1e-5 eV to 20 MeV; ENDF/B-VIII.0 only | pointwise table retrieved earlier from the IAEA-NDS ENDF service, with the resonances reconstructed by that service; 21,413 points |
| `libraries/Sm152_ng_bins.csv` | Sm-152 (n,g) averaged over bins of a quarter of a decade in energy; ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5 | NJOY2016 (RECONR and BROADR, 293.6 K), linear energy average of the cross section in each bin; columns `library`, `e_low_eV`, `e_high_eV`, `sigma_b`; the last bin ends at 20 MeV |

Checks made when the files were built:

* Fe-56: in each of the three files MT 4 equals the sum of the level and continuum sections MT 51 to 91 to better
  than 0.05 %, and its bin averages agree with the NJOY-processed ones to 0.02 dex. The inelastic threshold
  (0.862 MeV) lies above the resolved-resonance range of the files, so MF 3 is the complete cross section. The
  JENDL-5 section has small non-zero values below the threshold of the first level, from 0.105 MeV.
* Sm-152: the bin averages of the pointwise ENDF/B-VIII.0 curve agree with the NJOY-processed ones to 0.001 dex.

Caveats:

* A bin average in the resolved-resonance region (for Sm-152 from a few eV to a few keV) is dominated by the
  resonance peaks and is not comparable point by point with a measurement or with a smooth model curve.
* The Fe-56 curves fluctuate strongly up to a few MeV (resolved structure in the evaluation); measured points
  taken with a broad energy resolution average over that structure.
* No other reaction and no other library is shipped.

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
