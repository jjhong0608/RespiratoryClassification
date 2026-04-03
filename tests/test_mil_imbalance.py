from __future__ import annotations

import pytest
from src.training.imbalance import compute_binary_pos_weight, resolve_imbalance


def test_compute_binary_pos_weight_uses_negative_over_positive_ratio() -> None:
    value = compute_binary_pos_weight([0, 0, 0, 1, 1])

    assert value == pytest.approx(1.5)


def test_resolve_imbalance_computes_auto_pos_weight() -> None:
    resolved = resolve_imbalance(
        targets=[0, 0, 0, 1, 1],
        pos_weight=None,
        auto_pos_weight=True,
        weighted_random=True,
    )

    assert resolved.pos_weight == pytest.approx(1.5)
    assert resolved.weighted_random is True
    assert dict(resolved.class_counts) == {0: 3, 1: 2}


def test_resolve_imbalance_rejects_conflicting_pos_weight_settings() -> None:
    with pytest.raises(
        ValueError,
        match="Set only one of `train.loss.auto_pos_weight` or `train.loss.pos_weight`",
    ):
        resolve_imbalance(
            targets=[0, 1],
            pos_weight=2.0,
            auto_pos_weight=True,
            weighted_random=False,
        )
