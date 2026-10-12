import pandas as pd

from icd10.data.dataset import load_dataset, normalize_whitespace
from icd10.data.prepare import load_or_tokenize
from conftest import SYNTHETIC_MIMIC_DIR


def _raw_icd10_diagnoses():
    raw = pd.read_csv(SYNTHETIC_MIMIC_DIR / "hosp" / "diagnoses_icd.csv", dtype={"icd_code": str})
    return raw[raw["icd_version"] == 10]


def test_only_icd10_codes_are_kept(synthetic_dataset):
    icd10 = _raw_icd10_diagnoses()
    icd10_pairs = set(zip(icd10["hadm_id"], icd10["icd_code"]))
    for hadm_id, codes in zip(synthetic_dataset["hadm_id"], synthetic_dataset["icd_codes"]):
        assert codes, "admissions without ICD-10 codes must be dropped"
        assert all((hadm_id, code) in icd10_pairs for code in codes)


def test_admissions_without_note_or_codes_are_dropped(synthetic_dataset):
    notes = pd.read_csv(SYNTHETIC_MIMIC_DIR / "note" / "discharge.csv", usecols=["hadm_id"])
    expected = set(notes["hadm_id"]) & set(_raw_icd10_diagnoses()["hadm_id"])
    assert set(synthetic_dataset["hadm_id"]) == expected
    assert synthetic_dataset["hadm_id"].is_unique


def test_codes_have_no_dots_and_text_is_normalized(synthetic_dataset):
    assert not any("." in code for codes in synthetic_dataset["icd_codes"] for code in codes)
    assert not synthetic_dataset["text"].str.contains(r"\s{2,}|\n").any()


def test_normalize_whitespace_keeps_placeholders():
    assert normalize_whitespace("  Name:  ___\n\n  Age:\t70 ") == "Name: ___ Age: 70"


def test_several_notes_are_joined_in_note_seq_order_and_codes_stay_strings(tmp_path):
    # Written as .csv.gz like real MIMIC; the shared fixture covers plain .csv.
    (tmp_path / "note").mkdir()
    (tmp_path / "hosp").mkdir()
    pd.DataFrame({
        "note_id": ["a", "b", "c"], "subject_id": [1, 1, 2], "hadm_id": [10, 10, 20],
        "note_seq": [2, 1, 1], "text": ["second part", "first part", "other"],
    }).to_csv(tmp_path / "note" / "discharge.csv.gz", index=False)
    pd.DataFrame({
        "subject_id": [1, 1, 1, 2], "hadm_id": [10, 10, 10, 20], "seq_num": [2, 1, 3, 1],
        "icd_code": ["E119", "I10", "4019", "0389"], "icd_version": [10, 10, 9, 9],
    }).to_csv(tmp_path / "hosp" / "diagnoses_icd.csv.gz", index=False)
    df = load_dataset({"paths": {"mimic_dir": tmp_path, "data_dir": tmp_path / "data"}})
    assert df["hadm_id"].tolist() == [10]  # admission 20 only has ICD-9 codes
    assert df["text"].iloc[0] == "first part second part"
    assert df["icd_codes"].iloc[0] == ["I10", "E119"]


class FakeTokenizer:
    """One token per word; counts calls so the test can tell a cache hit from re-tokenizing."""

    name_or_path = "org/model"

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, truncation, max_length):
        self.calls += 1
        input_ids = [list(range(len(text.split())))[:max_length] for text in texts]
        return {"input_ids": input_ids, "attention_mask": [[1] * len(ids) for ids in input_ids]}


def test_tokenized_cache_is_reused_only_for_the_same_text_and_max_length(tmp_path):
    tokenizer = FakeTokenizer()
    old_df = pd.DataFrame({"hadm_id": [1], "text": ["old text"]})
    first = load_or_tokenize(old_df, tokenizer, 512, tmp_path)
    assert first["length"] == [2]
    load_or_tokenize(old_df.copy(), tokenizer, 512, tmp_path)
    assert tokenizer.calls == 1  # second call was a cache hit
    load_or_tokenize(pd.DataFrame({"hadm_id": [1], "text": ["rebuilt text here"]}), tokenizer, 512, tmp_path)
    load_or_tokenize(old_df, tokenizer, 1, tmp_path)
    assert tokenizer.calls == 3
    cache_dirs = sorted(p.name for p in (tmp_path / "tokenized").glob("*/*"))
    assert cache_dirs == ["org__model_len1", "org__model_len512", "org__model_len512"]