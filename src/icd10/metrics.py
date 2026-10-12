"""Multi-label metrics computed from scores (logits or probabilities) and gold labels.

Conventions: `y_true` is an (n_samples, n_labels) 0/1 array and `probs` the matching
sigmoid probabilities. A threshold can be a single float or one value per label.
"""

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
PRECISION_AT_K = (5, 8, 15)
BUCKET_EDGES = (10, 100)  # full setting: tail < 10 train admissions <= medium < 100 <= head


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return expit(np.asarray(logits, dtype=np.float64))  # no overflow warnings for large logits


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
    f1 = _f1(tp, predicted - tp, y_true.sum() - tp)
    return float(grid[int(np.argmax(f1))])


def tune_per_label_thresholds(
    y_true: np.ndarray, probs: np.ndarray, fallback: float, grid: np.ndarray = THRESHOLD_GRID
) -> np.ndarray:
    """Per-label thresholds maximizing each label's F1. Labels with no positives use `fallback`."""
    tp, predicted = _tp_and_predicted(y_true, probs, grid)
    positives = y_true.sum(axis=0)
    f1 = _f1(tp, predicted - tp, positives - tp)  # (grid, labels)
    thresholds = np.asarray(grid, dtype=float)[np.argmax(f1, axis=0)]
    thresholds[positives == 0] = fallback
    return thresholds


# --- Summary --------------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float | np.ndarray, ks: tuple[int, ...] = PRECISION_AT_K
) -> dict[str, float]:
    """Micro and macro F1 at `threshold`, micro and macro AUC-ROC, and precision@k for each k in `ks`.

    Macro F1 averages over all labels; F1 is zero where undefined (sklearn's zero_division=0).
    AUC is undefined without both positives and negatives, so macro AUC averages only over labels
    that have both, and either AUC is NaN if nothing qualifies.
    """
    y_true = y_true.astype(bool)
    y_pred = probs >= threshold
    tp = (y_true & y_pred).sum(axis=0)
    fp = (~y_true & y_pred).sum(axis=0)
    fn = (y_true & ~y_pred).sum(axis=0)
    positives = y_true.sum(axis=0)
    auc_labels = np.flatnonzero((positives > 0) & (positives < len(y_true)))
    results = {
        "micro_f1": float(_f1(tp.sum(), fp.sum(), fn.sum())),
        "macro_f1": float(_f1(tp, fp, fn).mean()),
        "micro_auc": (
            float(roc_auc_score(y_true.ravel(), probs.ravel())) if 0 < positives.sum() < y_true.size else float("nan")
        ),
        "macro_auc": (
            float(np.mean([roc_auc_score(y_true[:, j], probs[:, j]) for j in auc_labels]))
            if len(auc_labels) > 0
            else float("nan")
        ),
    }
    for k in ks:
        results[f"precision_at_{k}"] = precision_at_k(y_true, probs, k)
    return results


def frequency_buckets(train_counts: np.ndarray, edges: tuple[int, int] = BUCKET_EDGES) -> dict[str, np.ndarray]:
    """Split label indices into tail / medium / head by train frequency.

    With edges [a, b]: tail has fewer than a train admissions, medium a to b-1, head b or more.
    """
    low, high = edges
    return {
        "head": np.flatnonzero(train_counts >= high),
        "medium": np.flatnonzero((train_counts >= low) & (train_counts < high)),
        "tail": np.flatnonzero(train_counts < low),
    }


def bucket_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float, train_counts: np.ndarray
) -> dict[str, dict[str, float]]:
    """compute_metrics on each frequency bucket's labels. Empty buckets are skipped.

    Precision@k is left out: it isn't meaningful on a subset of labels.
    """
    return {
        name: {"n_labels": len(columns), **compute_metrics(y_true[:, columns], probs[:, columns], threshold, ks=())}
        for name, columns in frequency_buckets(train_counts).items()
        if len(columns) > 0
    }


def _f1(tp, fp, fn) -> np.ndarray:
    """F1 from counts, zero where undefined (sklearn's zero_division=0). Works on scalars and arrays."""
    numerator = np.asarray(2 * tp, dtype=np.float64)
    denominator = np.asarray(2 * tp + fp + fn)
    return np.divide(numerator, denominator, out=np.zeros(numerator.shape), where=denominator > 0)
