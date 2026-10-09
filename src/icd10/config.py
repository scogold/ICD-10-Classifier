"""All experiment settings. Edit the values here to change a run.

BASE holds every setting with its default. Each experiment lists only what it changes and is
picked by name on the command line:

    python -m icd10.train top50_base

Relative paths are relative to the repo root.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE: dict[str, Any] = {
    "run_name": "base",  # outputs go to <output_dir>/<run_name>/<timestamp>/
    "seed": 42,  # seeds random, numpy, torch, the Trainer and the patient-level split
    "paths": {
        "mimic_dir": "../MIMIC4_Dataset",  # raw MIMIC-IV root (contains note/ and hosp/); None = use the MIMIC_DIR env var
        "data_dir": "data",  # processed data, splits, tokenized caches (gitignored)
        "output_dir": "outputs",  # checkpoints, logits, metrics, TensorBoard logs (gitignored)
    },
    "data": {
        "label_setting": "top50",  # top50 = 50 most frequent train codes; full = every code seen in train
        "max_length": 4096,  # tokens per note (model max 8192); check truncation with scripts/token_lengths.py
        "split_file": None,  # path to Edin et al. (2023) mimiciv_icd10_split.feather; None = compute splits
        "val_fraction": 0.1,  # val_fraction and test_fraction apply only when computing splits
        "test_fraction": 0.1,
    },
    "model": {
        "checkpoint": "thomas-sounack/BioClinical-ModernBERT-base",
    },
    "train": {
        # Optimization
        "lr": 5e-5,
        "batch_size": 8,  # per device; effective batch = batch_size * grad_accum
        "grad_accum": 4,
        "eval_batch_size": 8,
        "epochs": 5,
        "max_steps": -1,  # > 0 overrides epochs (useful for quick tests)
        "warmup_steps": 0.1,  # below 1 = fraction of total steps; 1 or more = number of steps
        "weight_decay": 0.01,
        "scheduler": "linear",  # any transformers scheduler: linear, cosine, constant_with_warmup, ...
        # Memory / speed
        "precision": "auto",  # auto (bf16 if supported, else fp16; fp32 on CPU) | bf16 | fp16 | fp32
        "gradient_checkpointing": True,
        "group_by_length": True,  # batch notes of similar length together to waste less padding
        # Evaluation / checkpointing: once per epoch on val; the best epoch is kept at the end
        "save_total_limit": 2,
        "metric_for_best_model": "micro_f1_tuned",  # val micro F1 at the best global threshold; or micro_auc, ...
        "logging_steps": 50,
    },
    "loss": {
        "type": "bce",  # bce | pos_weight_bce | focal
        "pos_weight_max": 50.0,  # pos_weight_bce: cap on negatives/positives per label
        "focal_gamma": 2.0,  # focal: focusing parameter
        "focal_alpha": None,  # focal: optional positive-class weight in [0, 1]
    },
    "eval": {
        "per_label_thresholds": False,  # also tune one threshold per label on val (global is always tuned)
        "bucket_edges": [10, 100],  # full setting: tail < 10 train admissions <= medium < 100 <= head
    },
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Top-50 codes, BioClinical ModernBERT base. Start here.
TOP50_BASE = deep_merge(BASE, {"run_name": "top50-base"})

# Every code seen in train (several thousand labels), BioClinical ModernBERT base.
FULL_BASE = deep_merge(BASE, {
    "run_name": "full-base",
    "data": {"label_setting": "full"},
    "train": {"epochs": 5},  # rare codes need more passes
    "loss": {"type": "pos_weight_bce"},  # helps the long tail; compare against bce on val
})

# Top-50 codes, BioClinical ModernBERT large.
TOP50_LARGE = deep_merge(BASE, {
    "run_name": "top50-large",
    "model": {"checkpoint": "thomas-sounack/BioClinical-ModernBERT-large"},
    "train": {
        "lr": 2e-5,  # large models prefer a smaller learning rate
        "batch_size": 2,
        "grad_accum": 8,  # keeps the effective batch at 16
        "eval_batch_size": 4,
    },
})

# End-to-end pipeline check on the synthetic fixture. Runs in seconds with a tiny random model.
# Processed data goes to data/smoke so it never mixes with real MIMIC artifacts.
SMOKE = deep_merge(BASE, {
    "run_name": "smoke",
    "paths": {"mimic_dir": "tests/fixtures/synthetic_mimic", "data_dir": "data/smoke"},
    "data": {"max_length": 512},  # the tiny model's position limit
    "model": {"checkpoint": "hf-tiny-v2/tiny-random-ModernBertModel"},
    "train": {
        "lr": 1e-3,
        "batch_size": 16,
        "grad_accum": 2,
        "eval_batch_size": 32,
        "epochs": 2,
        "gradient_checkpointing": False,
        "logging_steps": 5,
    },
})

# The real model and the BASE training settings, on the synthetic fixture instead of MIMIC.
# Checks that a real training run works end to end (model download, memory at max_length,
# gradient checkpointing, precision) before spending hours on MIMIC. Metrics are meaningless.
SYNTHETIC = deep_merge(BASE, {
    "run_name": "synthetic",
    "paths": {"mimic_dir": "tests/fixtures/synthetic_mimic", "data_dir": "data/smoke"},
})

EXPERIMENTS = {
    "top50_base": TOP50_BASE,
    "full_base": FULL_BASE,
    "top50_large": TOP50_LARGE,
    "smoke": SMOKE,
    "synthetic": SYNTHETIC,
}


def get_config(experiment: str) -> dict[str, Any]:
    """A fresh copy of one experiment's settings, so a run can't change the module-level dicts."""
    return copy.deepcopy(EXPERIMENTS[experiment])


def resolve_path(path: str | Path) -> Path:
    """Relative paths in the config are relative to the repo root, not the current directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
