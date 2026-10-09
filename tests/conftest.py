"""Shared fixtures. Tests use only synthetic data (tests/fixtures/synthetic_mimic or in-memory frames)."""

from pathlib import Path

import pytest

from icd10.data.dataset import build_dataset

SYNTHETIC_MIMIC_DIR = Path(__file__).parent / "fixtures" / "synthetic_mimic"


@pytest.fixture(scope="session")
def synthetic_dataset():
    if not (SYNTHETIC_MIMIC_DIR / "note" / "discharge.csv").exists():
        pytest.skip("Synthetic MIMIC fixture not generated yet (scripts/make_synthetic_mimic.py).")
    return build_dataset(SYNTHETIC_MIMIC_DIR)
