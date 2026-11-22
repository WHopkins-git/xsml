# Choosing a split

A split is a statement about what the model is for. The rows placed in the test set stand for the cases the
model will meet in use, and the rows left in training stand for what will be known at that moment. A split is
therefore right or wrong only with respect to a question. This note sets out which question each common
design answers, and in particular when holding out an entire isotope is the appropriate test and when it is not.

## When to hold out a whole isotope

Holding out every row of a target nucleus (`splits.leave_one_isotope_out`, or the isotope-grouped folds of
`splits.isotope_folds` and `splits.fold_splits`) is the appropriate test in three situations.

* **The use is prediction for a nucleus with no measurements.** Examples are an unmeasured or exotic isotope,
  or a reaction channel of an evaluation for which nothing is known experimentally. In use, the model will
  have no point of that nucleus; the test must not give it any.
* **The question is whether the model has learned systematics.** If the claim is that the model has captured
  how a cross section varies across the chart of nuclides (with Z, N, separation energies, Q-values and so
  on), the evidence is its behaviour on a nucleus it has not seen.
* **The points of one isotope are strongly correlated.** They share a nucleus, a smooth excitation function
  and often an experiment. A random split by points places near-duplicates of the test points in training,
  so the model is tested on what it has in effect already seen, and the result is optimistic. Grouping by
  isotope removes this route.

## When not to hold out a whole isotope

If the use is interpolation or gap-filling within the curve of an isotope that has been measured, holding out
the whole isotope tests the wrong thing. A model that fills a gap needs the measurements on either side of
it, and in use it will have them. The honest test reproduces that situation: an energy window, or a whole
dataset (one experiment), of the isotope is held out, and the remaining rows of the same isotope stay in
training.

Holding out the isotope in this case also penalises exactly the information the model is meant to use. A
model whose strength is isotope-specific structure (the position of a threshold, the level of a plateau, a
normalisation fixed by neighbouring points) is prevented from showing it, and a model that ignores such
structure is favoured. The result then says little about performance in the intended use.

## Other designs

| Design | Question it answers | Support in `xsml.splits` |
|---|---|---|
| Random split by points | How well does the model reproduce points like those it has seen? Optimistic when points are correlated. | none, by design |
| Leave one dataset (experiment) out | Can the model predict a new measurement of an isotope that already has others? | no function; a mask on `dataset` or `entry`, checked with `assert_no_leakage(level="dataset")` or `level="entry"` |
| Leave an energy window out | Can the model fill a gap, or extend a curve, within a measured isotope? | no function; a mask on `energy_eV` for the isotope concerned |
| Leave one isotope out | Can the model predict a nucleus with no measurements, given its neighbours? | `leave_one_isotope_out`, `isotope_folds`, `fold_splits` |
| Leave one element out | Can the model predict a nucleus when its isotopic neighbours are unknown as well? | `leave_one_element_out` |
| Hold-out by compilation date | What would a model trained at a given moment have predicted for what was compiled later? | `holdout_by_date` |

Some remarks on the rows of the table.

* **Leave one dataset out** sits between the random split and the isotope split. It removes the correlation
  within an experiment (common normalisation, common method) but keeps other experiments on the same isotope
  in training. Where one isotope and energy range is covered by a single experiment, it approaches a gap test.
* **Leave an energy window out** has two forms. A window inside the measured range tests interpolation; a
  window at the upper or lower end tests extrapolation in energy, which is a harder and different question.
* **Leave one element out** is stricter than leaving out one isotope. Neighbouring isotopes of one element
  are look-alikes: similar structure, often measured in the same experiment with the same method. A model
  can do well on a held-out isotope by copying its neighbours; holding out the element shows how much of the
  result depended on them.
* **Hold-out by date** is a prospective test. The training mask returned by `holdout_by_date` is only the
  complement of the test mask; whether earlier rows of the same isotope or experiment may stay in training is
  a decision that depends on the question, and should be stated.
* **Random split by points** is not wrong in itself. It answers one narrow question, and it is the right
  design only if that question is the one being asked.

These designs form a hierarchy of leakage: random split, leave one isotope out, leave one element out. Each
step removes a closer relative of the test rows from training, and the task becomes harder at each step. The
hierarchy continues beyond the element: a test nucleus in a well-measured region of the chart is surrounded
by similar nuclei, whereas one at the edge of the measured region, or in a region with few measurements, is
an extrapolation. Results by chart region (light, medium and heavy nuclei, near and far from closed shells,
near and far from stability) show where the model interpolates between neighbours and where it extrapolates.

## Scores from different splits are not comparable

A number obtained with a random split and a number obtained with an isotope split measure different things,
even with the same model, data and metric. Neither is the "true" performance; each is the performance on
one question. A lower error under an easier split is not an improvement, and two models can be compared
only under the same split.

For this reason the split, the metric and the use case belong together and should be stated together. A
report should say, in one sentence, which question its split answers, for example "prediction for a nucleus
with no measurements, neighbouring isotopes known" or "interpolation across a gap of half a decade in energy
within a measured isotope". If the intended use changes, the split changes with it, and so does the meaning
of the result. Whatever the design, the grouping that is claimed should be checked on the frames that are
fitted and scored (`splits.assert_no_leakage`).
