"""Multi-label metrics computed from scores (logits or probabilities) and gold labels.

Conventions: `y_true` is an (n_samples, n_labels) 0/1 array and `probs` the matching
sigmoid probabilities. A threshold can be a single float or one value per label.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
PRECISION_AT_K = (5, 8, 15)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return expit(np.asarray(logits, dtype=np.float64))  # no overflow warnings for large logits


# --- F1 -----------------------------------------------------------------------------------


def precision_recall_f1(tp, fp, fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero where undefined (sklearn's zero_division=0). Works on scalars and arrays."""
    return _divide_or_zero(tp, tp + fp), _divide_or_zero(tp, tp + fn), _divide_or_zero(2 * tp, 2 * tp + fp + fn)


def _counts(y_true: np.ndarray, y_pred: np.ndarray, axis=None):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = (y_true & y_pred).sum(axis=axis)
    fp = (~y_true & y_pred).sum(axis=axis)
    fn = (y_true & ~y_pred).sum(axis=axis)
    return tp, fp, fn


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_recall_f1(*_counts(y_true, y_pred))[2])


def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return precision_recall_f1(*_counts(y_true, y_pred, axis=0))[2]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of per-label F1 over all labels."""
    return float(per_label_f1(y_true, y_pred).mean())


# --- AUC-ROC --------------------------------------------------------------------------------


def micro_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC over all (sample, label) pairs pooled together."""
    return float(roc_auc_score(y_true.ravel(), scores.ravel()))


def macro_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mean per-label AUC over labels with at least one positive and one negative in y_true.

    AUC is undefined for the other labels. NaN if no label qualifies.
    """
    positives = y_true.sum(axis=0)
    defined = np.flatnonzero((positives > 0) & (positives < len(y_true)))
    if len(defined) == 0:
        return float("nan")
    return float(np.mean([roc_auc_score(y_true[:, j], scores[:, j]) for j in defined]))


# --- Precision@k ----------------------------------------------------------------------------


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of each sample's k highest-scoring labels that are correct, averaged over samples.

    k is capped at the number of labels. Ties at the k-th score are broken arbitrarily.
    """
    k = min(k, scores.shape[1])
    top_k = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return float(np.take_along_axis(y_true, top_k, axis=1).sum(axis=1).mean() / k)


# --- Thresholds (tune on validation only) ---------------------------------------------------


def _tp_and_predicted(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True positives and predictions per (threshold, label) for `probs >= threshold`.

    Instead of thresholding the whole matrix once per grid value, count how many grid values
    each probability clears (grid sorted ascending) and accumulate those counts per label.
    """
    n_labels, n_bins = probs.shape[1], len(grid) + 1
    cleared = np.searchsorted(grid, np.asarray(probs, dtype=np.float64), side="right")
    flat = (cleared + n_bins * np.arange(n_labels)).ravel()
    predicted_hist = np.bincount(flat, minlength=n_labels * n_bins).reshape(n_labels, n_bins)
    tp_hist = np.bincount(flat[y_true.astype(bool).ravel()], minlength=n_labels * n_bins).reshape(n_labels, n_bins)

    def at_threshold(hist):  # probabilities clearing grid[t] are those that cleared more than t values
        return np.cumsum(hist[:, ::-1], axis=1)[:, ::-1][:, 1:].T  # (grid, labels)

    return at_threshold(tp_hist), at_threshold(predicted_hist)


def tune_global_threshold(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> float:
    """The single threshold that maximizes micro F1 (the lowest one if tied). Use on validation data only."""
    tp, predicted = (counts.sum(axis=1) for counts in _tp_and_predicted(y_true, probs, grid))
    f1 = precision_recall_f1(tp, predicted - tp, y_true.sum() - tp)[2]
    return float(grid[int(np.argmax(f1))])


def tune_per_label_thresholds(
    y_true: np.ndarray, probs: np.ndarray, fallback: float, grid: np.ndarray = THRESHOLD_GRID
) -> np.ndarray:
    """Per-label thresholds maximizing each label's F1. Labels with no positives use `fallback`."""
    tp, predicted = _tp_and_predicted(y_true, probs, grid)
    positives = y_true.sum(axis=0)
    f1 = precision_recall_f1(tp, predicted - tp, positives - tp)[2]  # (grid, labels)
    thresholds = np.asarray(grid, dtype=float)[np.argmax(f1, axis=0)]
    thresholds[positives == 0] = fallback
    return thresholds


# --- Summary --------------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float | np.ndarray, ks: tuple[int, ...] = PRECISION_AT_K
) -> dict[str, float]:
    y_pred = probs >= threshold
    results = {
        "micro_f1": micro_f1(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_auc": micro_auc(y_true, probs),
        "macro_auc": macro_auc(y_true, probs),
    }
    for k in ks:
        results[f"precision_at_{k}"] = precision_at_k(y_true, probs, k)
    return results


def frequency_buckets(train_counts: np.ndarray, edges: list[int]) -> dict[str, np.ndarray]:
    """Split label indices into tail / medium / head by train frequency.

    With edges [a, b]: tail has fewer than a train admissions, medium a to b-1, head b or more.
    """
    low, high = edges
    return {
        "head": np.flatnonzero(train_counts >= high),
        "medium": np.flatnonzero((train_counts >= low) & (train_counts < high)),
        "tail": np.flatnonzero(train_counts < low),
    }


def _divide_or_zero(numerator, denominator) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.zeros(numerator.shape), where=np.asarray(denominator) > 0)
