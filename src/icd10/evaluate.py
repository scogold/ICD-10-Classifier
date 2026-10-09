"""Evaluate a finished run from its saved logits, without re-running the model.

Run directory format (written by train.py):

    config.yaml            resolved config used for the run
    run_metadata.json      seed, library versions, GPU, precision
    label_vocab.json       {"label_setting", "codes": [...], "train_counts": [...]}
                           codes are dot-less ICD-10; train_counts[i] = train admissions with codes[i]
    {val,test}_logits.npy  float32 (n_admissions, n_labels), raw logits (apply a sigmoid for probabilities)
    {val,test}_labels.npy  uint8   (n_admissions, n_labels), multi-hot gold labels, columns match codes
    {val,test}_hadm_ids.npy int64  (n_admissions,), row order of the two arrays above    metrics.json           written by this script

The threshold is tuned on val only, then applied to test.

Usage:
    python -m icd10.evaluate outputs/top50-base/20260101-120000 [--per-label-thresholds]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from icd10.metrics import (
    compute_metrics,
    frequency_buckets,
    macro_auc,
    macro_f1,
    micro_f1,
    sigmoid,
    tune_global_threshold,
    tune_per_label_thresholds,
)


def load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (gold labels, probabilities) for one split."""
    labels = np.load(run_dir / f"{split}_labels.npy")
    probs = sigmoid(np.load(run_dir / f"{split}_logits.npy"))
    return labels, probs


def bucket_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float | np.ndarray, buckets: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    results = {}
    for name, columns in buckets.items():
        if len(columns) == 0:
            continue
        column_threshold = threshold[columns] if isinstance(threshold, np.ndarray) else threshold
        y, p = y_true[:, columns], probs[:, columns]
        y_pred = p >= column_threshold
        results[name] = {
            "n_labels": int(len(columns)),
            "micro_f1": micro_f1(y, y_pred),
            "macro_f1": macro_f1(y, y_pred),
            "macro_auc": macro_auc(y, p),
        }
    return results


def evaluate_run(run_dir: Path, per_label_thresholds: bool | None = None) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    vocab = json.loads((run_dir / "label_vocab.json").read_text(encoding="utf-8"))
    eval_config = config["eval"]
    if per_label_thresholds is None:
        per_label_thresholds = eval_config["per_label_thresholds"]

    val_y, val_p = load_split(run_dir, "val")
    test_y, test_p = load_split(run_dir, "test")

    threshold = tune_global_threshold(val_y, val_p)
    results: dict[str, Any] = {
        "global_threshold": threshold,
        "val": compute_metrics(val_y, val_p, threshold),
        "test": compute_metrics(test_y, test_p, threshold),
    }
    thresholds: float | np.ndarray = threshold
    if per_label_thresholds:
        thresholds = tune_per_label_thresholds(val_y, val_p, fallback=threshold)
        results["test_per_label_thresholds"] = compute_metrics(test_y, test_p, thresholds)

    if vocab["label_setting"] == "full":
        buckets = frequency_buckets(np.asarray(vocab["train_counts"]), eval_config["bucket_edges"])
        results["test_buckets"] = bucket_metrics(test_y, test_p, thresholds, buckets)

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--per-label-thresholds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also tune one threshold per label on val (default: eval.per_label_thresholds in the run config).",
    )
    args = parser.parse_args(argv)
    results = evaluate_run(args.run_dir, args.per_label_thresholds)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
