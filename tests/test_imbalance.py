from __future__ import annotations

from math import isclose

from src.training.imbalance import (
    build_sample_weights,
    compute_binary_pos_weight,
)


def test_compute_binary_pos_weight_matches_neg_over_pos() -> None:
    targets = [0, 0, 0, 1, 1]
    assert isclose(compute_binary_pos_weight(targets), 3 / 2)


def test_build_sample_weights_uses_inverse_class_frequency() -> None:
    targets = [0, 0, 0, 1]
    weights = build_sample_weights(targets)
    assert weights == [1 / 3, 1 / 3, 1 / 3, 1.0]
