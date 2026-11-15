"""xsml: a small toolkit for machine learning on experimental nuclear cross sections (EXFOR).

Modules
-------
data        load the linted table (with a SHA-256 check), select reactions and targets
features    AME2020 mass table and nuclear properties derived from it
splits      isotope-grouped folds, leave-one-out splits, hold-out by compilation date, leakage guard
metrics     RMS and median factor, per-isotope scores, cluster bootstrap, paired comparison, score log
models      constructors for tree models and a tuning wrapper over isotope-grouped folds
importance  block permutation, drop-column and SHAP for tree models
libraries   pointwise evaluated-library curves shipped with the repository
"""
from . import data, features, importance, libraries, metrics, models, splits  # noqa: F401

__version__ = "0.1.0"
