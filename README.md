# xsml

A small toolkit for machine learning on experimental nuclear cross sections.

It provides a linted table of EXFOR cross sections, the AME2020 mass table with derived nuclear properties,
splits in which the isotope is the unit of observation, scores with intervals, thin wrappers around tree
models, and three feature-importance tools. It is a toolkit, not a course. The notebooks in `examples/` show the functions in use; they are not a method.

## Contents

| Path | What it is |
|---|---|
| `xsml/data.py` | `load()` with a SHA-256 check, `select()`, `isotope_id()`, `datasets()` |
| `xsml/features.py` | `ame_table()`, `nuclide_features()`, `q_value()`, `reaction_q()`, `append()` |
| `xsml/splits.py` | `isotope_folds()`, `fold_splits()`, `leave_one_isotope_out()`, `leave_one_element_out()`, `holdout_by_date()`, `assert_no_leakage()` |
| `xsml/metrics.py` | `rms_factor()`, `median_factor()`, `per_group()`, `cluster_bootstrap()`, `paired_comparison()`, `log_score()` |
| `xsml/models.py` | `decision_tree()`, `random_forest()`, `xgboost()`, `cv_rmse()`, `tune()`, `nested_cv()` |
| `xsml/importance.py` | `block_permutation()`, `drop_column()`, `shap_values()` |
| `xsml/libraries.py` | `curve()`, `at()`, `bin_means()`, `available()`: the evaluated-library curves that are shipped |
| `examples/` | two executed notebooks showing the functions in use on other reactions |
| `docs/data_dictionary.md` | every column of the table |
| `docs/conventions.md` | units, identifiers, dates, metrics, the fold rule |
| `results/score_log.csv` | the score log: every score computed is appended here |
| `data/` | `MANIFEST.json`, the AME2020 mass table, the library curves; the parquet is downloaded here |
| `scripts/` | how the table and the library curves were built (needs sources that are not in this repository) |

Every function has a docstring: `help(xsml.splits.holdout_by_date)`.

## Install

Python 3.10 or later.

```
git clone <repository URL>
cd xsml
python -m venv .venv            # then activate it
pip install -e .
pip install pytest matplotlib   # optional: tests and plots
```

The install must be editable (`-e`): the package reads its data files from the `data/` folder of the clone.

## Data download

The table is about 49 MB and is not stored in git.

1. Download `exfor_linted_v1.parquet` from the link in `data/README.md`.
2. Place it in `data/`.
3. `xsml.data.load()` checks the SHA-256 of the file against `data/MANIFEST.json` on every call and stops with an explanation if the file is missing, incomplete or from another version.

Then run `pytest` from the repository root; all tests should pass. `pytest -m notebooks` also executes the example notebooks (about two minutes).

## Quickstart

```python
from xsml import data, features, splits, models, metrics

df = data.load()                                                       # rows kept by the linter, hash-checked
d = data.select(df, projectile="n", mt=102, e_min=1e5, e_max=2e7, plain=True)   # neutron capture, 0.1 to 20 MeV
d = features.append(d).reset_index(drop=True)                          # AME2020 and generated nuclear properties
X, y = d[["log10_energy", "Z", "N", "S_n"]], d["log10_sigma"]
iso = data.isotope_id(d.Z, d.A)
cv = splits.fold_splits(d, n_folds=5)                                  # an isotope is never split between train and test
_, oof = models.cv_rmse(models.decision_tree(), X, y, cv)              # out-of-fold predictions
score = metrics.cluster_bootstrap(y, oof, iso)                         # RMS factor with an interval over isotopes
print(score)
metrics.log_score("results/score_log.csv", model="decision_tree", features=list(X.columns), split="5 isotope folds",
                  metric="rms_factor", value=score["value"], ci_low=score["low"], ci_high=score["high"],
                  n_groups=score["n_groups"], n_points=score["n_points"])
```

The quickstart uses neutron capture so that it does not touch the reaction of the first brief.

## What the loader does and does not do

* `data.load()` returns the rows with `keep == True`: the rows the linter retained. This is the only default selection.
* The linter's verdict columns (`ds_flag`, `ds_offset_dex`, `ds_weight`, `pt_outlier` and the uncertainty flags) are returned untouched. Whether to filter on them, weight by them or ignore them is a modelling decision, to be stated and tested.
* `data.load(kept_only=False)` returns every row, with the reason for which a row was not retained. Rows that were not retained include values that are not cross sections in barns; they are for inspection, not for training without thought.
* Natural-element targets are not in the table. Every row has an isotopic target.
* How the linter reaches its verdicts is not documented here. It will be demonstrated separately.

## Evaluated libraries

The repository does not contain evaluated libraries and does not process ENDF files. It ships curves for a
selection of neutron-induced reactions in `data/libraries/`, from ENDF/B-VII.1, ENDF/B-VIII.0 and JENDL-5:
pointwise curves (`libraries.curve`, `libraries.at`) and, for reactions with resonances, averages over
quarter-decade energy bins (`libraries.bin_means`). For proton-induced reactions it ships curves of TENDL-2025 up to 30 MeV; TENDL is calculated with a nuclear model code and is not an evaluation of the measurements. `libraries.available()` lists everything, with a column `basis`. Not every library is
present for every reaction. Sources and caveats are in `data/README.md` and `scripts/build_library_curves.py`.
For another reaction the curve has to be obtained from the library's own distribution and saved in the same format.

## Rules of the game

1. **The isotope is the unit of observation.** Points of one isotope are not independent. Splits, scores and intervals are made over isotopes, not over points.
2. **Report intervals.** A score without an interval is not a result. Where no interval can be given (one group), say so and say why.
3. **No tuning on the hold-out.** Hyperparameters, features, cleaning rules and model choice are fixed without the hold-out. The hold-out is scored once.
4. **Write the protocol first, and commit it.** Split, features, model families, metric and the criterion of success are written down before any model sees the test data.
5. **One score per protocol.** Nothing is re-run until it looks good. Every score that is computed goes into `results/score_log.csv` with `metrics.log_score`; runs outside the protocol are logged as `declared="extra"` and declared in the report.
6. **Check for leakage with code, not by eye.** Call `splits.assert_no_leakage` on the frames passed to `fit` and to scoring.
7. **Every row traces to its source.** Keep the `dataset` identifier through every step.

## Licence

The code is under the MIT licence (`LICENSE`). The data are not: see `data/README.md`.
