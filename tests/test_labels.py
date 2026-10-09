import numpy as np
import pytest

from icd10.data.labels import TOP_K, build_label_vocab, codes_to_indices, multi_hot


def test_vocab_is_ordered_by_frequency_then_code():
    train = [["I10", "E119"], ["I10"], ["E119", "I10"], ["N179"], ["A419"]]
    assert build_label_vocab(train, "full") == ["I10", "E119", "A419", "N179"]


def test_duplicate_codes_in_one_admission_count_once():
    assert build_label_vocab([["E119", "E119", "E119"], ["I10"], ["I10"]], "full") == ["I10", "E119"]


def test_top50_keeps_the_50_most_frequent_codes():
    # Code C0nn appears in nn+1 admissions, so higher numbers are more frequent.
    train = [[f"C{i:03d}"] for i in range(80) for _ in range(i + 1)]
    vocab = build_label_vocab(train, "top50")
    assert len(vocab) == TOP_K
    assert vocab[0] == "C079" and vocab[-1] == "C030"


def test_vocab_only_contains_codes_from_the_given_split():
    vocab = build_label_vocab([["I10"]], "full")
    assert vocab == ["I10"]  # codes seen only in val/test never enter the vocabulary


def test_unknown_label_setting_raises():
    with pytest.raises(ValueError):
        build_label_vocab([["I10"]], "top10")


def test_multi_hot_encoding_drops_unknown_codes():
    code_to_index = {"I10": 0, "E119": 1, "N179": 2}
    indices = [codes_to_indices(codes, code_to_index) for codes in [["N179", "I10"], ["Z999"], ["E119", "E119"]]]
    encoded = multi_hot(indices, len(code_to_index))
    expected = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(encoded, expected)
