# Data

## Files

| File | In git | What it is |
|---|---|---|
| `exfor_linted_v1.parquet` | no (download) | the linted EXFOR table, one row per measured point; columns in `docs/data_dictionary.md` |
| `MANIFEST.json` | yes | file name, SHA-256, size, row counts, build date and the source frames of the table |
| `mass_1.mas20.txt` | yes | AME2020 atomic mass table, unmodified, read by `xsml.features` |
| `libraries/*.csv` | yes | evaluated-library curves for a selection of neutron-induced reactions (see below) |

## Download



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
computed beyond reading, except the part of the Cl-35 (n,p) curves of ENDF/B below 0.1 MeV (see the checks). Pointwise files have the columns `library`, `energy_eV`, `sigma_b`.

| File | Content | Source and resolution |
|---|---|---|
| `libraries/Cl35_np.csv` | Cl-35 (n,p), MT 103, 1e-5 eV to 20 MeV (the reaction has a positive Q-value and no threshold); ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5 | from 0.1 MeV: pointwise reconstruction with NJOY2016 (RECONR and BROADR, 293.6 K) from the supervisor's earlier work, 15,483 to 22,911 points per library. Below 0.1 MeV, at 0 K: JENDL-5 section MF 3 / MT 103 as filed (10,658 points); ENDF/B-VII.1 and ENDF/B-VIII.0 calculated from the resonance parameters of the files (5,182 points each), because their MF 3 holds only a placeholder there |
| `libraries/Fe56_ninl.csv` | Fe-56 inelastic scattering, MT 4, threshold to 20 MeV; ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5 | section MF 3 / MT 4 of the ENDF-6 files as distributed (linear interpolation between points, 0 K, not Doppler broadened); 769, 2,925 and 7,504 points |
| `libraries/Sm152_ng.csv` | Sm-152 (n,g), MT 102, 1e-5 eV to 20 MeV; ENDF/B-VIII.0 only | pointwise table retrieved earlier from the IAEA-NDS ENDF service, with the resonances reconstructed by that service; 21,413 points |
| `libraries/Sm152_ng_bins.csv` | Sm-152 (n,g) averaged over bins of a quarter of a decade in energy; ENDF/B-VII.1, ENDF/B-VIII.0, JENDL-5 | NJOY2016 (RECONR and BROADR, 293.6 K), linear energy average of the cross section in each bin; columns `library`, `e_low_eV`, `e_high_eV`, `sigma_b`; the last bin ends at 20 MeV |

Checks made when the files were built:

* Fe-56: in each of the three files MT 4 equals the sum of the level and continuum sections MT 51 to 91 to better
  than 0.05 %, and its bin averages agree with the NJOY-processed ones to 0.02 dex. The inelastic threshold
  (0.862 MeV) lies above the resolved-resonance range of the files, so MF 3 is the complete cross section. The
  JENDL-5 section has small non-zero values below the threshold of the first level, from 0.105 MeV.
* Cl-35 (n,p) below 0.1 MeV. The three files hold the same table of resonances up to 1.2 MeV (MF 2, R-matrix
  limited format, Reich-Moore approximation, with a proton channel). JENDL-5 also gives the resulting curve in
  MF 3 / MT 103; the two ENDF/B files give 1e-20 b there, and NJOY, which made the curves from 0.1 MeV, was not
  available. The ENDF/B part was therefore calculated by the build script from the parameters (0 K, grid for
  linear interpolation to 0.1 %). The same calculation with the JENDL-5 file reproduces the JENDL-5 MF 3 curve
  below 0.1 MeV to 0.004 % (median; 0.5 % or better for 99 % of the points, up to 3 % next to the narrowest
  resonances, where the seven-figure energies of the file do not resolve the peak), and the calculation agrees
  with the NJOY curves of ENDF/B from 0.1 to 1 MeV to 0.01 % (median; the peaks differ because NJOY broadened
  them to 293.6 K) and to 0.02 % at the join. At 0.0253 eV the three curves give 0.479 b; the comment sections
  of the files state 0.480 b (ENDF/B, 0 K) and 0.47917 b (JENDL-5, 300 K). Below 0.0253 eV the curves follow
  1/v to 0.03 %. The curves are not Doppler broadened below 0.1 MeV: the resonance at 398 eV is narrower and
  higher than at room temperature. Fourteen energies of the JENDL-5 section are given twice (steps); they are kept.
* Sm-152: the bin averages of the pointwise ENDF/B-VIII.0 curve agree with the NJOY-processed ones to 0.001 dex.

Caveats:

* A bin average in the resolved-resonance region (for Sm-152 from a few eV to a few keV) is dominated by the
  resonance peaks and is not comparable point by point with a measurement or with a smooth model curve.
* The Fe-56 curves fluctuate strongly up to a few MeV (resolved structure in the evaluation); measured points
  taken with a broad energy resolution average over that structure.
* Further reactions were added with the same three kinds of source (MF 3 as filed for threshold reactions, the IAEA-NDS service table for ENDF/B-VIII.0, NJOY bin averages). `xsml.libraries.available()` lists every file; the list of reactions, the sources and the checks applied to each are in `scripts/build_library_curves.py`. Some pointwise ENDF/B-VIII.0 curves of resonance reactions are thinned and do not resolve every resonance. For proton-induced reactions the files hold TENDL-2025 (library name `TENDL-2025`), a library calculated with the model code TALYS and not an evaluation of the measurements: section MF 3 as tabulated, up to 30 MeV, and the ground-state and isomer parts (`_g`, `_m`) where the file gives them.

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
