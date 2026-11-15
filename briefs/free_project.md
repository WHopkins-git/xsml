# Brief 2: free project

From Monday 30 November 2026 to the deadline of Friday 5 March 2027.

## The task

Choose a target isotope and reaction for which a prediction would be useful, and find out, with honest
uncertainty, how well a model trained on experimental data predicts it. The question, the test set and the
method are the student's. The rules of the game in `README.md` and the conventions in `docs/conventions.md`
apply throughout: the isotope is the unit, the test set is frozen before modelling, the protocol is committed
first, every score is logged.

## Milestones

| Date | Deliverable |
|---|---|
| Fri 4 Dec 2026 | Target isotope and reaction: half a page. Why it matters, what data exist (datasets, points, energy range, dates), what the evaluated libraries say, what could go wrong. |
| Fri 11 Dec 2026 | Plan: three pages, with the test set frozen. Question, test rows and how they were fixed, training rows, features to be considered, model families, metric, criterion of success, risks. The frozen test set is committed as a list of dataset identifiers. |
| Fri 18 Dec 2026 | The pipeline runs end to end, from `data.load()` to a logged score on development folds, on a clean clone. No result is expected to be good yet. |
| Mid January 2027 | First result with intervals, on development data. |
| 6 Feb 2027 | Analysis frozen. The test set is scored according to the protocol. No new analysis after this date; only checks and writing. |
| 20 Feb 2027 | Full draft of the report. |
| Fri 5 Mar 2027 | Deadline. |

## Notes

* A change to the frozen test set after 11 December is a change of protocol: it is dated, justified and committed before any score is computed on the new set.
* Evaluated-library curves are shipped for Cl-35 (n,p) only. For another reaction, obtaining the curve and documenting where it came from is part of the project.
* A negative or undecided result, properly supported, is a complete project.
