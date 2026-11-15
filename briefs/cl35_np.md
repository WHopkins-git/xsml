# Brief 1: the Cl-35 (n,p) cross section

Monday 16 November to Friday 27 November 2026. Nothing in this brief is marked.

## The problem

Predict the cross section of the reaction Cl-35 (n,p) S-35 as a function of incident neutron energy with a
model trained on experimental data, and compare the result with:

P. Vicente-Valdez, L. Bernstein and M. Fratoni, "Nuclear data evaluation augmented by machine learning",
Annals of Nuclear Energy 163, 108596 (2021) ("NucML" below), together with the dissertation and the public
code and data that accompany it.

The toolkit in this repository supplies the data, the nuclear properties, the splits and the scores. It does
not supply a method. Reading the paper, deciding what to build and defending the decision are the work.

## The hold-out set

The test data are fixed in advance:

> every row of the table with projectile `n`, `Z == 17`, `A == 35`, `MT == 103` and `keep == True`
> whose `entry_date` is on or after **2019-01-01**.

`entry_date` is the date on which an entry was compiled into EXFOR, not the year of publication. The cutoff is
set by the supervisor and is not changed. At the build of table v1 the hold-out is 380 rows in 4 datasets
from 2 EXFOR entries. `splits.holdout_by_date` selects it.

Before the protocol is committed, the hold-out rows may be counted and their energies listed. Their cross
sections are not plotted, tabulated, compared with anything or shown to a model. These measurements are
published; anything already known about them from the literature is declared in the report.

## What is required

1. **Protocol.** A file `protocol.md`, at most two pages, committed before any model is fitted to or scored on
   any Cl-35 row. It states:
   * the training rows and the split used for development;
   * the features;
   * the model families and how hyperparameters are chosen;
   * the metric and how its uncertainty is obtained;
   * how leakage from the hold-out into training is avoided, and how that is verified in code;
   * what result would count as better than NucML, what as worse, and what as undecided.

   The commit hash of the protocol is quoted in the report.
2. **A NucML-style model.** A model trained on the feature set chosen by NucML, assembled from the columns of
   this toolkit.
3. **An own model.** A curated feature set, with the evidence for each choice, and tuned hyperparameters.
4. **Comparison with the paper**, in three parts:
   * (a) a table of the methodological differences between this work and NucML: split, features, data
     cleaning, metric, hyperparameters;
   * (b) a reproduction of the spirit of the paper's headline result, in the paper's own metric;
   * (c) the models against the evaluated libraries, in the factor metric (`xsml.libraries`, `xsml.metrics`).
5. **Report.** At most four pages including figures, with a section headed "Limits". Every number in it can be
   regenerated from the committed code, and the commit is stated.

## Rules

* Intervals are reported with every score. Where none can be given, that is said, with the reason.
* The isotope is the unit of observation: for splitting, for scoring during development and for intervals.
* Nothing is tuned on the hold-out: not hyperparameters, not features, not cleaning rules, not the choice between models.
* One score per protocol. The hold-out is scored once for each model named in the protocol. Nothing is re-run until it looks good.
* Every score computed on test data is appended to `results/score_log.csv` with `metrics.log_score`. Runs outside the protocol are logged with `declared="extra"` and declared in the report.
* The linter's verdict columns may be used for filtering or weighting the training data if the protocol says how. They are not applied to the hold-out.

## What this exercise can and cannot show

The hold-out is small in the sense that matters: its rows come from very few experiments, and over much of its
energy range from one. Rows of one experiment share a normalisation, so they are far from independent, and a
score on them has no honest interval over experiments. It is therefore possible that the question "is this
model better than NucML on Cl-35 (n,p)?" cannot be decided with these data. A report that reaches that
conclusion, and shows why, is a good report. A report that claims a win on the strength of a point estimate is
not.

The supervisor's own work reports results on Cl-35 (n,p), obtained with the same linter that produced this
table, and the reaction was chosen for this brief with knowledge of them. The reaction is therefore not an
unseen test of the method. Whatever comes out of these two weeks is an exercise in doing a comparison
properly; it is not evidence about the method.

## Checkpoints

| Date | What |
|---|---|
| Mon 16 Nov | start; repository installed, tests pass, paper read |
| Fri 20 Nov | protocol defence: `protocol.md` committed and defended in ten minutes |
| Fri 27 Nov | report (four pages) and `results/score_log.csv` committed |

## Deliverables

`protocol.md`, the code that produces every number, `results/score_log.csv`, and the report as a PDF.
