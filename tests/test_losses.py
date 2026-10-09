import numpy as np
import torch
import torch.nn.functional as F

from icd10.losses import compute_pos_weight, make_loss_fn

LOSS_CONFIG = {"type": "bce", "pos_weight_max": 50.0, "focal_gamma": 2.0, "focal_alpha": None}


def _batch():
    torch.manual_seed(0)
    labels = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float32)
    return torch.randn(4, 3), labels


def test_pos_weight_is_negatives_over_positives_and_clipped():
    weights = compute_pos_weight(np.array([10, 1, 0]), n_samples=100, max_weight=50.0)
    assert weights.tolist() == [9.0, 50.0, 50.0]


def test_focal_with_gamma_zero_equals_bce():
    logits, labels = _batch()
    focal = make_loss_fn({**LOSS_CONFIG, "type": "focal", "focal_gamma": 0.0}, np.ones(3), 4)
    assert torch.allclose(focal(logits, labels), F.binary_cross_entropy_with_logits(logits, labels))


def test_pos_weight_bce_upweights_positives():
    logits, labels = _batch()
    bce = make_loss_fn(LOSS_CONFIG, np.ones(3), 4)
    weighted = make_loss_fn({**LOSS_CONFIG, "type": "pos_weight_bce"}, np.ones(3), 4)
    assert weighted(logits, labels) > bce(logits, labels)
