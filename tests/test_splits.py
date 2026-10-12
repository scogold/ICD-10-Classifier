import pandas as pd

from icd10.data.splits import SPLITS, compute_subject_splits


def test_no_patient_in_two_splits(synthetic_dataset):
    splits = compute_subject_splits(synthetic_dataset, seed=0, val_fraction=0.1, test_fraction=0.1)
    subjects = {name: set(splits.loc[splits["split"] == name, "subject_id"]) for name in SPLITS}
    assert not subjects["train"] & subjects["val"]
    assert not subjects["train"] & subjects["test"]
    assert not subjects["val"] & subjects["test"]
    assert len(splits) == len(synthetic_dataset)  # every admission is assigned


def test_patients_with_many_admissions_stay_together(synthetic_dataset):
    splits = compute_subject_splits(synthetic_dataset, seed=1, val_fraction=0.2, test_fraction=0.2)
    assert (splits.groupby("subject_id")["split"].nunique() == 1).all()


def test_splits_are_deterministic_given_seed(synthetic_dataset):
    a = compute_subject_splits(synthetic_dataset, seed=7, val_fraction=0.1, test_fraction=0.1)
    b = compute_subject_splits(synthetic_dataset, seed=7, val_fraction=0.1, test_fraction=0.1)
    c = compute_subject_splits(synthetic_dataset, seed=8, val_fraction=0.1, test_fraction=0.1)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])


def test_split_fractions_are_by_patient():
    df = pd.DataFrame({"subject_id": [i // 2 for i in range(200)], "hadm_id": range(200)})
    splits = compute_subject_splits(df, seed=0, val_fraction=0.1, test_fraction=0.2)
    patients = splits.groupby("split")["subject_id"].nunique()
    assert patients.to_dict() == {"test": 20, "train": 70, "val": 10}

