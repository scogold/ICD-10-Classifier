"""Label vocabulary (built from the train split only) and multi-hot encoding."""

from collections import Counter
from collections.abc import Iterable

import numpy as np

TOP_K = 50  # the "top50" setting


def count_codes(code_lists: Iterable[list[str]]) -> Counter:
    """Number of admissions each code appears in."""
    return Counter(code for codes in code_lists for code in set(codes))


def build_label_vocab(train_counts: Counter, setting: str) -> list[str]:
    """Codes ordered by train frequency (ties broken alphabetically), from `count_codes` on the train split.

    `top50` keeps the 50 most frequent codes; `full` keeps every code seen in train.
    """
    ordered = sorted(train_counts, key=lambda code: (-train_counts[code], code))
    if setting == "top50":
        return ordered[:TOP_K]
    if setting == "full":
        return ordered
    raise ValueError(f"Unknown label setting {setting!r}; use 'top50' or 'full'.")


def codes_to_indices(codes: list[str], code_to_index: dict[str, int]) -> list[int]:
    """Indices of the codes that are in the vocabulary; unknown codes are dropped."""
    return sorted({code_to_index[c] for c in codes if c in code_to_index})


def multi_hot(index_lists: Iterable[list[int]], num_labels: int) -> np.ndarray:
    index_lists = list(index_lists)
    matrix = np.zeros((len(index_lists), num_labels), dtype=np.uint8)
    for row, indices in enumerate(index_lists):
        matrix[row, indices] = 1
    return matrix
