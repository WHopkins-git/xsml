import numpy as np
import pytest

from xsml import splits
from xsml.data import isotope_id


def test_folds_keep_isotopes_together_and_are_deterministic(toy):
    fold = splits.isotope_folds(toy, 4)
    iso = isotope_id(toy.Z.to_numpy(), toy.A.to_numpy())
    assert all(len(set(fold[iso == g])) == 1 for g in set(iso.tolist()))
    assert sorted(set(fold.tolist())) == [0, 1, 2, 3]
    # the rule: isotopes sorted by Z then A, dealt in turn
    expected = {g: r % 4 for r, g in enumerate(sorted(set(iso.tolist())))}
    assert all(fold[i] == expected[g] for i, g in enumerate(iso.tolist()))
    # independent of row order
    shuffled = toy.iloc[::-1]
    assert (splits.isotope_folds(shuffled, 4) == fold[::-1]).all()
    assert (splits.isotope_folds(iso, 4) == fold).all()
    assert (splits.isotope_folds(toy, 4, offset=1) == (fold + 1) % 4).all()


def test_fold_splits_cover_every_row_once_without_leakage(toy):
    seen = np.zeros(len(toy), int)
    for tr, te in splits.fold_splits(toy, 5):
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(toy)
        splits.assert_no_leakage(toy.iloc[tr], toy.iloc[te])
        seen[te] += 1
    assert (seen == 1).all()


def test_too_many_folds(toy):
    with pytest.raises(ValueError):
        splits.isotope_folds(toy, 13)


def test_leave_one_isotope_and_element_out(toy):
    tr, te = splits.leave_one_isotope_out(toy, 17, 35)
    assert te.sum() == 30 and (tr ^ te).all()
    splits.assert_no_leakage(toy[tr], toy[te])
    with pytest.raises(splits.LeakageError):
        splits.assert_no_leakage(toy[tr], toy[te], level="element")       # Cl-37 is still in training
    tr, te = splits.leave_one_element_out(toy, 17)
    assert te.sum() == 60
    splits.assert_no_leakage(toy[tr], toy[te], level="element")
    with pytest.raises(ValueError):
        splits.leave_one_isotope_out(toy, 1, 1)


def test_holdout_by_date(toy):
    tr, te = splits.holdout_by_date(toy, "2000-01-01")
    assert (toy.entry_date[te] >= "2000-01-01").all() and (toy.entry_date[tr] < "2000-01-01").all() and te.sum() == 6 * 30
    where = (toy.MT == 103).to_numpy()
    tr, te = splits.holdout_by_date(toy, "2000-01-01", where=where)
    assert te.sum() == 6 * 15 and (tr ^ te).all()
    # the complement still holds the other reaction of the held-out isotopes: the guard must say so
    with pytest.raises(splits.LeakageError):
        splits.assert_no_leakage(toy[tr], toy[te])
    with pytest.raises(splits.LeakageError):
        splits.assert_no_leakage(toy[tr], toy[te], level="entry")


def test_guard_raises_when_a_held_out_isotope_is_in_training(toy):
    test = toy[(toy.Z == 17) & (toy.A == 35) & (toy.MT == 103)]
    train = toy.drop(test.index)                                           # Cl-35 (n,g) rows remain
    with pytest.raises(splits.LeakageError, match="17035"):
        splits.assert_no_leakage(train, test)
    splits.assert_no_leakage(train[~((train.Z == 17) & (train.A == 35))], test)
    splits.assert_no_leakage(train, test, level="dataset")
