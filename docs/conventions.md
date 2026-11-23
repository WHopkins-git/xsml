# Conventions

## Units

| Quantity | Unit | Column or function |
|---|---|---|
| incident energy | eV, laboratory frame of the projectile | `energy_eV` |
| cross section and its uncertainty | barn | `sigma_b`, `unc_b` |
| masses, binding energies, separation energies, Q-values | MeV | `xsml.features` |
| atomic mass | u | `atomic_mass_u` |
| offsets and residuals | dex (log10 units; 0.30 dex is a factor of 2) | `ds_offset_dex`, `xsml.metrics` |

Models are fitted to `log10(sigma_b)`. `features.append` adds `log10_energy` and `log10_sigma`.

## Identifiers

* `entry`: EXFOR entry, five characters (one publication or one experiment). Example: `14840`.
* `subentry`: entry plus a three-digit subentry number, eight characters. Example: `14840006`.
* `dataset`: the subentry, followed by a pointer character where one subentry holds several data columns. Example: `145560031`. One dataset is one table of points for one reaction. This is the identifier to keep through every step.
* An isotope is labelled `1000 * Z + A` (`data.isotope_id`); Cl-35 is `17035`.
* `reaction` holds the full EXFOR reaction string, including the product nucleus. It is the authority on what a row is.
* `MT` is a convenience label taken from the X4Pro database: 1 total, 2 elastic, 4 one neutron out, 16 two neutrons out, 18 fission, 102 photon only, 103 one proton out, 107 one alpha out, and so on. The number describes the outgoing particles whatever the projectile, so MT 4 is `(N,INL)` for a neutron and `(P,N)` for a proton, and MT 103 is `(N,P)` for a neutron and `(P,INL)` for a proton. MT 9000 holds production cross sections `(z,X)` of many different products. The table by projectile is in `docs/data_dictionary.md`.

## Dates

* `year`: year of the publication, as given by EXFOR for the first author's reference.
* `entry_date`: date on which the entry was first compiled into EXFOR. An old measurement can have a recent `entry_date`, and a recent publication can be compiled years after it appeared.

A hold-out "by date" in this repository is by `entry_date` unless stated otherwise: it asks what the database
contained at a given moment, which is what a model trained at that moment could have seen.

## Splits

* The isotope (target Z and A, all reactions) is the unit. Leaving out an isotope means leaving out every row of that target.
* Folds: the distinct isotopes are sorted by Z and then A and dealt to the folds in turn (`splits.isotope_folds`). No random numbers are used, so the folds do not depend on the machine or on library versions.
* `splits.assert_no_leakage(train, test, level)` is called on the frames that are fitted and scored. Levels: `isotope`, `element`, `dataset`, `entry`.

## Comparison columns and leakage

The table carries two kinds of output of the linter, both computed once, on the whole table
(`xsml.data.VERDICT_COLUMNS` lists them together).

* **Removals:** `keep` and `reason`. They record which rows the linter did not retain as usable cross sections,
  and why.
* **Comparison labels:** `ds_flag`, `ds_offset_dex`, `ds_weight`, `pt_outlier`, the three uncertainty flags and
  `critique_kind`. `ds_flag`, `ds_offset_dex` and `ds_weight` compare each dataset with the other datasets of the
  same target and reaction; `pt_outlier` asks whether a point is corroborated by another dataset of the same
  reaction. Nothing is removed on their account.

A flag says a dataset disagrees with the others, not that it is wrong; whether to trust it is a hypothesis to
test, with scores on the full target.

Two consequences follow for both kinds.

* **They are not model inputs.** A label exists only where a measurement exists, and it was formed with every
  dataset of the reaction in view, including any that are later held out. A model that takes such a column as a
  feature is scored with information from its test rows and cannot be used for a nucleus without data.
  `xsml.data.check_inputs(features)` raises an error if such a column is in a feature list; it is not called
  automatically. The same holds for the reported uncertainty `unc_b` (`xsml.data.UNCERTAINTY_COLUMNS`).
* **Filtering or weighting by them is a modelling choice, to be declared.** This includes the default
  `keep == True` of `data.load()`. There is a mild caveat: the label of a training dataset of a reaction was
  formed with the held-out datasets of that same reaction among its comparators. Datasets of other targets and
  reactions are not affected. Where the held-out datasets are the only neighbours of each other, their own flags
  say only that they disagree with one another. The test rows are never selected by these columns: a test set
  from which the flagged datasets have been taken out is an easier test, not a cleaner one. Score the whole
  target, and look at the score dataset by dataset.

Columns that are safe as inputs are those that describe the nucleus and the conditions of the reaction, known
without any measurement: `Z`, `A`, `N`, the projectile and the reaction (`MT`), the incident energy, and the
quantities derived from the AME2020 mass table by `xsml.features` (masses, binding and separation energies,
Q-values, parity and shell quantities). Bibliographic columns (`dataset`, `entry`, `author`, `year`,
`entry_date`) identify the experiment and are not inputs either.

## Scores

With residual `r = log10(predicted) - log10(measured)`:

* RMS factor: `10 ** sqrt(mean(r**2))`
* median factor: `10 ** median(|r|)`

A factor of 1 is perfect; a factor of 2 means wrong by a factor of two in either direction. Intervals are
95 % percentile intervals from a cluster bootstrap that resamples whole isotopes (`metrics.cluster_bootstrap`).
Two models are compared on the same resampled isotopes (`metrics.paired_comparison`). A score from a single
isotope or a single experiment has no such interval; that is stated, not hidden.

A score in another metric (for example one used by a paper under comparison) is reported in addition to the
factors, not instead of them, and its definition is written out.

## The score log

`results/score_log.csv` has one row per score ever computed on test data, written by `metrics.log_score`:
timestamp (UTC), git commit, whether the working tree had uncommitted changes, model, features, split,
metric, value, interval, number of groups and points, `declared` (`protocol` or `extra`) and notes. The log is
committed with the work. A result whose row has `dirty = yes` cannot be reproduced from the commit alone.

## Randomness

Every constructor in `xsml.models` fixes `random_state=0`; the bootstrap functions take a `seed`. Seeds are
recorded in the protocol. Changing a seed after seeing a score is a new run and is logged as one.

## Writing

Reports state what was done, on which rows, with which commit. Numbers quoted in text are produced by code
in the repository and can be regenerated. Claims that were not checked are marked as such.
