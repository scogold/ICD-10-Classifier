"""Patient-level train/val/test splits.

Splits are by subject_id so that no patient appears in more than one split. A split is saved
as a CSV of IDs only (hadm_id, subject_id, split) under data_dir/splits.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from icd10.config import resolve_path

SPLITS = ("train", "val", "test")


def compute_subject_splits(
    df: pd.DataFrame, seed: int, val_fraction: float, test_fraction: float
) -> pd.DataFrame:
    """Randomly assign whole patients to train/val/test with a fixed seed."""
    subjects = np.sort(df["subject_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = round(len(subjects) * test_fraction)
    n_val = round(len(subjects) * val_fraction)
    n_train = len(subjects) - n_test - n_val
    split_of_subject = dict(zip(subjects, ["test"] * n_test + ["val"] * n_val + ["train"] * n_train))
    return df[["hadm_id", "subject_id"]].assign(split=df["subject_id"].map(split_of_subject))


def split_file_path(config: dict[str, Any], dataset_path: Path) -> Path:
    name = f"{dataset_path.stem}_subject_seed{config['seed']}.csv"
    return resolve_path(config["paths"]["data_dir"]) / "splits" / name


def load_or_create_splits(config: dict[str, Any], df: pd.DataFrame, dataset_path: Path) -> pd.DataFrame:
    path = split_file_path(config, dataset_path)
    if path.exists():
        return pd.read_csv(path)
    data = config["data"]
    splits = compute_subject_splits(df, config["seed"], data["val_fraction"], data["test_fraction"])
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    return splits
