"""Glue that turns a config into ready-to-train datasets: build, split, label, tokenize."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from icd10.config import resolve_path
from icd10.data.dataset import load_dataset, processed_dataset_path
from icd10.data.labels import build_label_vocab, codes_to_indices, count_codes, multi_hot
from icd10.data.splits import SPLITS, load_or_create_splits


def tokenized_cache_dir(data_dir: Path, dataset_path: Path, tokenizer_name: str, max_length: int) -> Path:
    """Cache key: processed dataset version, tokenizer name and max_length.

    Only token IDs are cached; labels are attached later because they depend on the label
    setting and split.
    """
    # Size and mtime change whenever the processed file is rebuilt, so stale token IDs are never reused.
    stat = dataset_path.stat()
    version = f"{stat.st_size}-{stat.st_mtime_ns}"
    safe_tokenizer = tokenizer_name.replace("/", "__").replace("\\", "__")
    return data_dir / "tokenized" / f"{dataset_path.stem}-{version}" / f"{safe_tokenizer}_len{max_length}"


def load_or_tokenize(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int, cache_dir: Path
) -> Dataset:
    """Return a Dataset with hadm_id, input_ids, attention_mask and length (tokens kept)."""
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
    dataset_path = processed_dataset_path(config)
    # Tokenize the whole processed dataset so one cache serves every split and label setting.
    max_length = config["data"]["max_length"]
    data_dir = resolve_path(config["paths"]["data_dir"])
    cache_dir = tokenized_cache_dir(data_dir, dataset_path, config["model"]["checkpoint"], max_length)
    tokenized = load_or_tokenize(df, tokenizer, max_length, cache_dir)
    row_of_hadm = {hadm_id: i for i, hadm_id in enumerate(tokenized["hadm_id"])}

    splits = load_or_create_splits(config, df, dataset_path)
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
