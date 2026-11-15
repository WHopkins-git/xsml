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
