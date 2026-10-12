"""Multi-label losses. All of them treat each label as an independent sigmoid."""

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def compute_pos_weight(label_counts: np.ndarray, n_samples: int, max_weight: float) -> torch.Tensor:
    """negatives / positives per label, clipped so that very rare labels don't dominate."""
    positives = np.maximum(label_counts, 1)
    return torch.tensor(np.minimum((n_samples - positives) / positives, max_weight), dtype=torch.float32)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017): down-weights easy, well-classified labels."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_true = torch.exp(-bce)  # bce = -log(probability given to the true class)
    return (bce * (1 - p_true) ** gamma).mean()


def make_loss_fn(loss_config: dict[str, Any], train_label_counts: np.ndarray, n_train: int) -> LossFn:
    """Build the loss from the `loss` config section. Returns fn(logits, labels) -> scalar."""
    loss_type = loss_config["type"]
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits
    if loss_type == "pos_weight_bce":
        pos_weight = compute_pos_weight(train_label_counts, n_train, loss_config["pos_weight_max"])

        def weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight.to(logits.device))

        return weighted_bce
    if loss_type == "focal":
        return partial(focal_loss, gamma=loss_config["focal_gamma"])
    raise ValueError(f"Unknown loss type {loss_type!r}; use bce, pos_weight_bce or focal.")
