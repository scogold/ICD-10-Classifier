"""Load MIMIC-IV notes and ICD-10 labels and join them into one admission-level table.

Output columns: subject_id, hadm_id, text, icd_codes (list of dot-less ICD-10 codes,
in seq_num order). Raw MIMIC files are only read, never modified.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from icd10.config import resolve_path


def mimic_file(mimic_dir: Path, folder: str, table: str) -> Path:
    """Real MIMIC ships `<table>.csv.gz`; the synthetic fixture uses plain `<table>.csv` so it is readable."""
    for suffix in (".csv.gz", ".csv"):
        path = mimic_dir / folder / f"{table}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No {folder}/{table}.csv.gz or {folder}/{table}.csv under {mimic_dir}")


def load_notes(mimic_dir: Path) -> pd.DataFrame:
    """One row per admission. If an admission has several discharge notes, join them in note_seq order."""
    notes = pd.read_csv(
        mimic_file(mimic_dir, "note", "discharge"),
        usecols=["subject_id", "hadm_id", "note_seq", "text"],
        dtype={"text": str},
    )
    notes = notes.dropna(subset=["hadm_id", "text"]).sort_values(["hadm_id", "note_seq"])
    notes = notes.groupby(["hadm_id", "subject_id"], as_index=False).agg(text=("text", " ".join))  # sorted by hadm_id
    return notes[["subject_id", "hadm_id", "text"]]


def load_icd10_codes(mimic_dir: Path) -> pd.DataFrame:
    """One row per admission with its ICD-10 codes (deduplicated, in seq_num order)."""
    diagnoses = pd.read_csv(
        mimic_file(mimic_dir, "hosp", "diagnoses_icd"),
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    diagnoses = diagnoses[diagnoses["icd_version"] == 10].dropna(subset=["icd_code"])
    diagnoses["icd_code"] = diagnoses["icd_code"].str.strip()
    diagnoses = diagnoses.sort_values(["hadm_id", "seq_num"])
    codes = diagnoses.groupby("hadm_id")["icd_code"].agg(lambda c: list(dict.fromkeys(c)))
    return codes.rename("icd_codes").reset_index()


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) into single spaces.

    This is the only text preprocessing. De-identification placeholders (`___`) are left untouched.
    """
    return " ".join(text.split())


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Join notes to ICD-10 codes on hadm_id; drop admissions without a note or without codes.

    The result is cached as parquet under data_dir; later calls read the cache instead of MIMIC.
    """
    path = resolve_path(config["paths"]["data_dir"]) / "processed" / "mimiciv_icd10.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["icd_codes"] = df["icd_codes"].map(list)  # parquet returns numpy arrays
        return df

    if config["paths"]["mimic_dir"] is None:
        raise ValueError("Set paths.mimic_dir in src/icd10/config.py to the MIMIC-IV folder (contains note/ and hosp/).")
    mimic_dir = resolve_path(config["paths"]["mimic_dir"])
    notes = load_notes(mimic_dir)
    codes = load_icd10_codes(mimic_dir)
    df = notes.merge(codes, on="hadm_id", how="inner")  # keeps the notes' hadm_id order
    df["text"] = df["text"].map(normalize_whitespace)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
