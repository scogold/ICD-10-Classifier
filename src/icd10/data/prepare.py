"""Glue that turns a config into ready-to-train datasets: build, split, label, tokenize."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from icd10.config import resolve_path
from icd10.data.dataset import load_dataset
from icd10.data.labels import build_label_vocab, codes_to_indices, count_codes, multi_hot
from icd10.data.splits import SPLITS, load_or_create_splits


def text_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of hadm_id + text: changes whenever the text that gets tokenized changes."""
    row_hashes = pd.util.hash_pandas_object(df[["hadm_id", "text"]], index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def load_or_tokenize(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int, data_dir: Path
) -> Dataset:
    """Return a Dataset with hadm_id, input_ids, attention_mask and length (tokens kept).

    Cached under data_dir/tokenized/<text fingerprint>/<tokenizer>_len<max_length>. Only token IDs
    are cached; labels are attached later because they depend on the label setting and split.
    """
    safe_tokenizer = tokenizer.name_or_path.replace("/", "__").replace("\\", "__").replace(":", "")
    cache_dir = data_dir / "tokenized" / text_fingerprint(df) / f"{safe_tokenizer}_len{max_length}"
    if cache_dir.exists():
        return load_from_disk(str(cache_dir))

    def tokenize(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["length"] = [len(ids) for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_pandas(df[["hadm_id", "text"]], preserve_index=False)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))
    return dataset


@dataclass
class PreparedData:
    datasets: dict[str, Dataset]  # split -> hadm_id, input_ids, attention_mask, length, label_ids
    vocab: list[str]
    train_counts: list[int]  # train admissions per vocab code, aligned with vocab


def prepare_data(config: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> PreparedData:
    df = load_dataset(config)
    # Tokenize the whole processed dataset so one cache serves every split and label setting.
    max_length = config["data"]["max_length"]
    data_dir = resolve_path(config["paths"]["data_dir"])
    tokenized = load_or_tokenize(df, tokenizer, max_length, data_dir)
    row_of_hadm = {hadm_id: i for i, hadm_id in enumerate(tokenized["hadm_id"])}

    splits = load_or_create_splits(config, df)
    df = df.merge(splits[["hadm_id", "split"]], on="hadm_id", how="inner")

    train_counts = count_codes(df.loc[df["split"] == "train", "icd_codes"])
    vocab = build_label_vocab(train_counts, config["data"]["label_setting"])
    code_to_index = {code: i for i, code in enumerate(vocab)}
    df["label_ids"] = [codes_to_indices(codes, code_to_index) for codes in df["icd_codes"]]
    # Admissions with no code in the vocabulary have nothing to predict (as in the top-50 literature).
    df = df[df["label_ids"].map(len) > 0]

    datasets = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        dataset = tokenized.select([row_of_hadm[h] for h in part["hadm_id"]])
        datasets[split] = dataset.add_column("label_ids", list(part["label_ids"]))
    return PreparedData(datasets=datasets, vocab=vocab, train_counts=[train_counts[code] for code in vocab])


@dataclass
class MultiLabelCollator:
    """Pads token IDs and turns each row's label indices into a multi-hot float vector."""

    tokenizer: PreTrainedTokenizerBase
    num_labels: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            return_tensors="pt",
        )
        label_ids = [feature["label_ids"] for feature in features]
        batch["labels"] = torch.from_numpy(multi_hot(label_ids, self.num_labels)).float()
        return batch
