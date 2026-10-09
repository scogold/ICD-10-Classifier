import pandas as pd
import pytest

from icd10.data.splits import (
    SPLITS,
    check_no_subject_leakage,
    compute_subject_splits,
    load_edin_splits,
    load_or_create_splits,
    split_file_path,
)


def test_no_patient_in_two_splits(synthetic_dataset):
    splits = compute_subject_splits(synthetic_dataset, seed=0, val_fraction=0.1, test_fraction=0.1)
    subjects = {name: set(splits.loc[splits["split"] == name, "subject_id"]) for name in SPLITS}
    assert not subjects["train"] & subjects["val"]
    assert not subjects["train"] & subjects["test"]
    assert not subjects["val"] & subjects["test"]
    check_no_subject_leakage(splits)
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


def test_leakage_check_raises_on_shared_patient():
    splits = pd.DataFrame({"hadm_id": [1, 2, 3], "subject_id": [10, 10, 11], "split": ["train", "test", "val"]})
    with pytest.raises(ValueError, match="1 patients"):
        check_no_subject_leakage(splits)


def test_load_edin_splits(tmp_path):
    df = pd.DataFrame({"subject_id": [10, 10, 11, 12], "hadm_id": [1, 2, 3, 4]})
    path = tmp_path / "split.csv"
    pd.DataFrame({"_id": [1, 2, 3, 99], "split": ["train", "train", "valid", "test"]}).to_csv(path, index=False)
    splits = load_edin_splits(path, df).set_index("hadm_id")
    assert splits["split"].to_dict() == {1: "train", 2: "train", 3: "val"}
    assert splits["subject_id"].to_dict() == {1: 10, 2: 10, 3: 11}


def test_a_saved_split_file_with_leakage_is_rejected(tmp_path):
    config = {"seed": 0, "paths": {"data_dir": str(tmp_path)}, "data": {"split_file": None}}
    dataset_path = tmp_path / "processed" / "mimiciv_icd10.parquet"
    path = split_file_path(config, dataset_path)
    path.parent.mkdir(parents=True)
    pd.DataFrame({"hadm_id": [1, 2], "subject_id": [10, 10], "split": ["train", "test"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="1 patients"):
        load_or_create_splits(config, pd.DataFrame(), dataset_path)
