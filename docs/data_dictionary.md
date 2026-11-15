# Data dictionary: `exfor_linted_v1.parquet`

One row per measured point. 5,085,941 rows in 53,613 datasets from 10,771 EXFOR entries; 3,383,322 rows in
25,996 datasets have `keep == True` and are what `xsml.data.load()` returns by default (421 target isotopes).
Counts are from the build recorded in `data/MANIFEST.json`.

The table holds cross sections (EXFOR quantity `SIG`) for incident n, p, d, t, he3, alpha and photons on
isotopic targets, with positive values. Natural-element targets are not in the table.

Values are as compiled in EXFOR and converted by the X4Pro database, except the energy, which carries the
linter's frame corrections (column `energy_frame`). The table is not cleaned beyond the `keep` verdict:
expect extreme values (kept energies range from 6e-10 eV to 4e11 eV, kept cross sections from 1e-32 b to
2e8 b), zero or negative uncertainties in a few thousand rows, and bibliographic oddities (the smallest
`year` is 1900).

## Identification and bibliography

| Column | Type | Meaning |
|---|---|---|
| `dataset` | text | EXFOR dataset identifier: subentry (8 characters) plus a pointer character where a subentry holds several data columns (9 characters). One dataset is one table of points for one reaction. |
| `entry` | text | EXFOR entry, the first 5 characters of `dataset`. |
| `subentry` | text | EXFOR subentry, the first 8 characters of `dataset`. |
| `author` | text | Family name of the first author, from X4Pro. |
| `year` | integer | Year of the publication (first author's reference), from X4Pro. Not the date of compilation. |
| `reference` | text | First reference of the entry in EXFOR coding, for example `J,PR/C,99,044612,2019`. |
| `doi` | text | DOI of that reference where EXFOR gives one (about three quarters of the rows), otherwise null. |
| `entry_date` | text, `YYYY-MM-DD` | Date on which the entry was first compiled into EXFOR. From 1966-02-21 to 2025-09-02. |

## Reaction

| Column | Type | Meaning |
|---|---|---|
| `projectile` | text | `n`, `p`, `d`, `t`, `he3`, `a` (alpha) or `g` (photon). |
| `Z`, `A` | integer | Proton number and mass number of the target. |
| `MT` | integer | Reaction number assigned by the X4Pro database, copied unchanged. A convenience label: the `reaction` string is the authority. See "MT and projectile" below. |
| `MF` | integer | ENDF file number assigned by X4Pro (3 for almost all kept rows). |
| `reaction` | text | Full EXFOR reaction string, including the product nucleus, for example `17-CL-35(N,P)16-S-35,,SIG`. This string defines what was measured. |
| `product_state` | text | Qualifier of the product: empty for the unqualified reaction, otherwise for example `G` (ground state), `M` (isomer), or a marker of another restriction. Rows with different product states of one reaction are different quantities. |
| `sf5`, `sf8`, `sf9` | text or null | Modifier fields of the EXFOR reaction string (branch, modifier, data type). Null means the field is empty. All kept rows have an empty `sf5`. |

## MT and projectile

`MT` is not computed by the linter or by this repository. It is the number that X4Pro attaches to each dataset
and is passed through unchanged. Its meaning was checked against the reaction strings of all kept rows:

* For the standard numbers, `MT` describes the **outgoing particles**, whatever the projectile. MT 4 is "one
  neutron out": inelastic scattering `(N,INL)` for a neutron, but `(P,N)` for a proton and `(A,N)` for an alpha.
  MT 103 is "one proton out": `(N,P)` for a neutron, `(D,P)` for a deuteron, and inelastic scattering `(P,INL)` for
  a proton. MT 102 is "photon only": `(N,G)`, `(P,G)`.
* MT 1 (total), 2 (elastic), 3 (nonelastic or absorption), 18 (fission) and 101 (neutron absorption) are sums
  over channels and have no single product.
* MT 9000 collects quantities without an ENDF number: for charged particles and photons almost always a
  production cross section `(z,X)` of a named residual product, for neutrons mostly `(N,SCT)`. **Rows with
  MT 9000 on one target are many different reactions.** They are told apart only by the product in `reaction`.
* A few datasets carry a number that does not match their string (12 kept rows of `(D,INL)` are filed under
  MT 103).
* Rows to which X4Pro gave no number (MT 0; 188,536 rows) are all among the rows the linter did not keep.

For the 438,511 kept rows whose reaction string fixes the outgoing particles, the product nucleus named in
the string equals the residual nucleus implied by the projectile and those particles in every row, and the MT
label implies the same particles in all but the 12 rows above.

Kept rows by projectile and MT, with the main EXFOR process code that the label stands for (the larger labels
only; cells with fewer than 150 rows are left empty):

| MT | n | p | d | a | he3 | t | g |
|---|---|---|---|---|---|---|---|
| 1 | `N,TOT` 2,314,557 | | | | | | `G,TOT` 2,243 |
| 2 | `N,EL` 7,755 | | | | | | `G,EL` 660 |
| 3 | `N,NON` 356 | `P,NON` 775 | | `A,NON` 157 | | | `G,ABS` 2,740 |
| 4 | `N,INL` 37,892 | `P,N` 24,679 | `D,N` 3,001 | `A,N` 21,101 | `HE3,N` 1,180 | | `G,N` 3,209 |
| 16 | `N,2N` 10,827 | `P,2N` 3,171 | `D,2N` 2,767 | `A,2N` 4,204 | `HE3,2N` 947 | | `G,2N` 1,542 |
| 17 | `N,3N` 541 | `P,3N` 2,650 | `D,3N` 559 | `A,3N` 2,440 | `HE3,3N` 671 | | `G,3N` 535 |
| 18 | `N,F` 540,566 | `P,F` 4,261 | `D,F` 654 | `A,F` 1,352 | `HE3,F` 212 | | `G,F` 2,120 |
| 37 | `N,4N` 155 | `P,4N` 1,270 | `D,4N` 422 | `A,4N` 919 | `HE3,4N` 321 | | |
| 101 | `N,ABS` 7,611 | | | | | | |
| 102 | `N,G` 244,162 | `P,G` 6,145 | `D,G` 243 | `A,G` 1,249 | | | |
| 103 | `N,P` 13,385 | `P,INL` 372 | `D,P` 4,123 (and `D,INL` 12) | `A,P` 1,370 | `HE3,P` 268 | `T,P` 1,292 | `G,P` 673 |
| 105 | `N,T` 2,146 | | `D,T` 251 | | | | |
| 107 | `N,A` 16,651 | `P,A` 3,253 | `D,A` 1,142 | | `HE3,A` 227 | `T,A` 593 | |
| 9000 | `N,SCT` 21,331, `N,X` 1,634 | `P,X` 15,470 | `D,X` 3,620 | `A,X` 8,612 | `HE3,X` 1,902 | `T,X` 469 | `G,X` 12,280, `G,SCT` 1,066 |

Consequences for use:

* Select a reaction by projectile, target and `MT` for the standard channels, then **check the `reaction`
  strings of what was selected** (`df.reaction.value_counts()`).
* For a production cross section of a particular nuclide (for example a medical isotope made with protons),
  select on the `reaction` string and on `product_state`; `MT` alone is not enough.
* `features.reaction_q` reads the outgoing particles from the `reaction` string when that column is present
  and returns NaN where the string does not fix them (totals, fission, `(z,X)`, scattering sums).

## Measurement

| Column | Type | Meaning |
|---|---|---|
| `energy_eV` | float | Incident energy of the projectile in eV, laboratory frame. |
| `sigma_b` | float | Cross section in barns for kept rows. For rows with `keep == False` the number may be another quantity or in another unit. |
| `unc_b` | float | Uncertainty of the cross section as given by X4Pro, in barns; NaN where none is available (about 11 % of kept rows). What the uncertainty includes (statistical, systematic, total) differs between datasets and is not recorded here. |

## Verdicts of the linter

The linter is the supervisor's tool for checking EXFOR cross sections. These columns say what it concluded,
not how. They are outputs of a model of the data, not ground truth: a flagged dataset can be right and its
neighbours wrong.

| Column | Type | Meaning |
|---|---|---|
| `keep` | bool | True if the linter retained the row as a usable cross section in barns against an incident energy. |
| `reason` | text | `kept`, or the linter's label for why the row was not retained (`class rule`, `modifier rule`, `unit rule`, `superseded rule`, `duplicate rule`, `duplicate entry`, `limit rule`, `outdated, ...`, `known issue: ...`). |
| `ds_flag` | text | Verdict on the dataset from a comparison with the other datasets of the same reaction: `ok`, `normalisation` (the dataset is offset from the others), `shape` (its energy dependence differs), `normalisation+shape`, or `unassessable` (too few other datasets overlap it for a judgement; this means "cannot judge", not "fine"). Constant within a dataset. Present for every kept row; null for most rows that were not kept. |
| `ds_offset_dex` | float | Offset of the dataset from the others, in log10 units; positive means the dataset is higher. NaN where no offset could be formed. Constant within a dataset. |
| `ds_weight` | float | A suggested fitting weight for the dataset, between 0 and 1: lower for datasets that would need a larger added uncertainty to agree with the others, 1 for datasets that could not be judged. Constant within a dataset. See the note below. |
| `pt_outlier` | bool or missing | True if the single point was flagged as a spike relative to the other data of its reaction. See the note below. |
| `missing_uncertainty` | bool | True if the dataset has no uncertainty in its original table. |
| `relative_uncertainty_flag` | bool | True if the typical relative uncertainty of the dataset is implausibly small or large. A flag on the uncertainty, not on the values. |
| `uncertainty_unit_error` | bool | True if the uncertainty column appears to be in another unit than the values. No kept row has it. |
| `critique_kind` | text | Empty, or the kind of criticism recorded in the EXFOR entry itself: `data wrong (factor or direction)` or `general`. The rows are kept; the flag reports what the compilers wrote. |
| `energy_frame` | text | Empty if the energy is as compiled. Otherwise the linter converted it to the laboratory frame: `corrected`, `corrected, inverse kinematics` or `corrected from excitation energy`. |

**Note on `ds_weight` and `pt_outlier`.** `ds_flag` and `ds_offset_dex` come from the linter release that
produced `keep` (frames of 2026-11-01). `ds_weight` and `pt_outlier` come from an earlier run of the same
comparison (table of 2026-10-27) on a slightly different selection of data, joined by dataset and, for
`pt_outlier`, by energy and value. The two runs can disagree for a dataset (for example `ds_flag == "ok"` with
a low `ds_weight`). 3,352,465 kept rows have a `ds_weight` and 3,356,488 a `pt_outlier` (16,068 of them
True); the rest are missing because the dataset or point was not in the earlier run.

## Columns added by the toolkit

`features.append` adds `N`, `log10_energy`, `log10_sigma`, `Q_reaction` and the columns of
`features.nuclide_features()` (documented in the docstrings of `xsml/features.py`). `data.isotope_id(Z, A)`
gives the isotope label `1000 * Z + A`.
