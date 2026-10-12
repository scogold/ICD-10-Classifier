"""Evaluate a finished run from its saved logits, without re-running the model.

Run directory format (written by train.py):

    config.yaml            resolved config used for the run
    run_metadata.json      seed, library versions, GPU, precision
    label_vocab.json       {"label_setting", "codes": [...], "train_counts": [...]}
                           codes are dot-less ICD-10; train_counts[i] = train admissions with codes[i]
    {val,test}_logits.npy  float32 (n_admissions, n_labels), raw logits (apply a sigmoid for probabilities)
    {val,test}_labels.npy  uint8   (n_admissions, n_labels), multi-hot gold labels, columns match codes
    {val,test}_hadm_ids.npy int64  (n_admissions,), row order of the two arrays above
    metrics.json           written by this script

Thresholds (one global, and one per label) are tuned on val only, then applied to test.

Usage:
    python -m icd10.evaluate outputs/top50-base/20260101-120000
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from icd10.metrics import (
    bucket_metrics,
    compute_metrics,
    sigmoid,
    tune_global_threshold,
    tune_per_label_thresholds,
)


def load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (gold labels, probabilities) for one split."""
    labels = np.load(run_dir / f"{split}_labels.npy")
    probs = sigmoid(np.load(run_dir / f"{split}_logits.npy"))
    return labels, probs


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    vocab = json.loads((run_dir / "label_vocab.json").read_text(encoding="utf-8"))

    val_y, val_p = load_split(run_dir, "val")
    test_y, test_p = load_split(run_dir, "test")

    threshold = tune_global_threshold(val_y, val_p)
    per_label = tune_per_label_thresholds(val_y, val_p, fallback=threshold)
    results: dict[str, Any] = {
        "global_threshold": threshold,
        "val": compute_metrics(val_y, val_p, threshold),
        "test": compute_metrics(test_y, test_p, threshold),
        "test_per_label_thresholds": compute_metrics(test_y, test_p, per_label),
    }

    if vocab["label_setting"] == "full":
        results["test_buckets"] = bucket_metrics(test_y, test_p, threshold, np.asarray(vocab["train_counts"]))

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    results = evaluate_run(args.run_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
