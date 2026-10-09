import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from icd10.evaluate import bucket_metrics
from icd10.metrics import (
    sigmoid,
    compute_metrics,
    frequency_buckets,
    macro_auc,
    macro_f1,
    micro_auc,
    micro_f1,
    precision_at_k,
    tune_global_threshold,
    tune_per_label_thresholds,
)


@pytest.fixture
def random_problem():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 12)) < 0.2).astype(np.uint8)
    y[:, 11] = 0  # a label with no positives
    probs = np.clip(0.6 * y + 0.5 * rng.random(y.shape), 0, 1)
    return y, probs


def test_f1_matches_sklearn(random_problem):
    y, probs = random_problem
    pred = probs >= 0.5
    assert micro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="micro", zero_division=0))
    assert macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro", zero_division=0))


def test_f1_hand_computed():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    pred = np.array([[1, 1, 0], [0, 1, 0]])
    # tp=2, fp=1, fn=1 -> micro F1 = 4/6; per-label F1 = 1, 2/3, 0
    assert micro_f1(y, pred) == pytest.approx(2 / 3)
    assert macro_f1(y, pred) == pytest.approx((1 + 2 / 3 + 0) / 3)


def test_auc_matches_sklearn_and_skips_labels_without_positives(random_problem):
    y, probs = random_problem
    assert micro_auc(y, probs) == pytest.approx(roc_auc_score(y.ravel(), probs.ravel()))
    expected_macro = np.mean([roc_auc_score(y[:, j], probs[:, j]) for j in range(11)])
    assert macro_auc(y, probs) == pytest.approx(expected_macro)


def test_precision_at_k_hand_computed():
    y = np.array([[1, 0, 1, 0], [0, 0, 0, 1]])
    scores = np.array([[0.9, 0.8, 0.7, 0.1], [0.4, 0.3, 0.2, 0.1]])
    assert precision_at_k(y, scores, 1) == pytest.approx((1 + 0) / 2)
    assert precision_at_k(y, scores, 2) == pytest.approx((1 / 2 + 0) / 2)
    assert precision_at_k(y, scores, 4) == pytest.approx((2 / 4 + 1 / 4) / 2)
    assert precision_at_k(y, scores, 10) == precision_at_k(y, scores, 4)  # k is capped at n_labels


def test_global_threshold_separates_separable_scores():
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    probs = np.where(y == 1, 0.35, 0.25)
    threshold = tune_global_threshold(y, probs)
    assert 0.25 < threshold <= 0.35
    assert micro_f1(y, probs >= threshold) == 1.0


def test_per_label_thresholds_differ_per_label_and_fall_back():
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    probs = np.array([[0.8, 0.1, 0.5], [0.6, 0.3, 0.5], [0.9, 0.2, 0.5], [0.5, 0.4, 0.5]])
    thresholds = tune_per_label_thresholds(y, probs, fallback=0.42)
    assert 0.6 < thresholds[0] <= 0.8
    assert 0.2 < thresholds[1] <= 0.3
    assert thresholds[2] == 0.42  # no positives -> fallback


def test_compute_metrics_reports_all_metrics(random_problem):
    y, probs = random_problem
    assert set(compute_metrics(y, probs, 0.5)) == {
        "micro_f1", "macro_f1", "micro_auc", "macro_auc", "precision_at_5", "precision_at_8", "precision_at_15",
    }


def test_frequency_buckets():
    buckets = frequency_buckets(np.array([500, 50, 5, 100, 10, 9]), [10, 100])
    assert buckets["head"].tolist() == [0, 3]
    assert buckets["medium"].tolist() == [1, 4]
    assert buckets["tail"].tolist() == [2, 5]


def _brute_force_thresholds(y, probs, grid):
    """Reference: threshold the whole matrix at every grid value."""
    micro = [micro_f1(y, probs >= t) for t in grid]
    per_label = np.stack([[f1_score(y[:, j], probs[:, j] >= t, zero_division=0) for j in range(y.shape[1])] for t in grid])
    return grid[int(np.argmax(micro))], grid[np.argmax(per_label, axis=0)]


def test_threshold_tuning_matches_brute_force_including_probs_on_grid_values():
    rng = np.random.default_rng(3)
    y = (rng.random((150, 7)) < 0.25).astype(np.uint8)
    probs = np.round(np.clip(0.4 * y + 0.6 * rng.random(y.shape), 0, 1), 2)  # many values sit exactly on the grid
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    expected_global, expected_per_label = _brute_force_thresholds(y, probs, grid)
    assert tune_global_threshold(y, probs, grid) == expected_global
    np.testing.assert_array_equal(tune_per_label_thresholds(y, probs, fallback=0.5, grid=grid), expected_per_label)
    # float32 probabilities (as loaded from saved logits) give the same answer
    assert tune_global_threshold(y, probs.astype(np.float32), grid) == tune_global_threshold(y, probs, grid)


def test_auc_handles_ties_and_undefined_labels():
    rng = np.random.default_rng(4)
    y = (rng.random((80, 5)) < 0.3).astype(np.uint8)
    y[:, 3] = 0  # no positives
    y[:, 4] = 1  # no negatives
    probs = np.round(rng.random(y.shape), 1)  # many ties
    assert micro_auc(y, probs) == pytest.approx(roc_auc_score(y.ravel(), probs.ravel()))
    assert macro_auc(y, probs) == pytest.approx(np.mean([roc_auc_score(y[:, j], probs[:, j]) for j in range(3)]))
    assert np.isnan(macro_auc(np.zeros((4, 2)), rng.random((4, 2))))


def test_sigmoid_is_stable_for_large_logits():
    with np.errstate(all="raise"):
        np.testing.assert_allclose(sigmoid(np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)), [0.0, 0.5, 1.0])


def test_empty_predictions_and_all_zero_labels_give_zero_f1():
    y = np.array([[1, 0], [0, 0]])
    nothing = np.zeros_like(y, dtype=bool)
    assert micro_f1(y, nothing) == 0.0 and macro_f1(y, nothing) == 0.0
    assert micro_f1(np.zeros_like(y), nothing) == 0.0


def test_bucket_metrics_use_each_buckets_own_columns_and_thresholds():
    y = np.array([[1, 0, 1], [0, 1, 1]])
    probs = np.array([[0.9, 0.2, 0.6], [0.1, 0.7, 0.4]])
    buckets = {"head": np.array([0, 2]), "medium": np.array([], dtype=int), "tail": np.array([1])}
    results = bucket_metrics(y, probs, np.array([0.5, 0.5, 0.3]), buckets)
    assert set(results) == {"head", "tail"}  # empty buckets are skipped
    assert results["head"]["n_labels"] == 2 and results["head"]["micro_f1"] == 1.0
    assert results["tail"]["micro_f1"] == 1.0
