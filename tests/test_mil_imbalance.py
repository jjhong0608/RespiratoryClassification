from __future__ import annotations

import pytest
import torch
from src.training.imbalance import (
    build_main_index_to_binary_target,
    compute_binary_pos_weight,
    compute_class_counts_by_index,
    compute_sqrt_inverse_class_weights,
    compute_sqrt_normal_over_abnormal_pos_weight,
    resolve_imbalance,
)


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


def test_sqrt_inverse_class_weights_are_mean_one_and_ranked() -> None:
    counts = torch.tensor([3108, 554, 514, 235], dtype=torch.float32)

    weights = compute_sqrt_inverse_class_weights(counts)

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1]
    assert weights[1].item() == pytest.approx(weights[2].item(), rel=0.05)
    assert weights[2] < weights[3]


def test_compute_class_counts_by_index_returns_contiguous_counts() -> None:
    counts = compute_class_counts_by_index([0, 1, 1, 3], num_classes=4)

    assert torch.equal(counts, torch.tensor([1.0, 2.0, 0.0, 1.0]))


def test_branch_binary_mapping_and_pos_weight() -> None:
    mapping = build_main_index_to_binary_target(
        label_to_index={"normal": 0, "crackle": 1, "wheeze": 2, "rhonchi": 3},
        binary_label_to_index={
            "normal": 0,
            "crackle": 1,
            "wheeze": 1,
            "rhonchi": 1,
        },
    )
    pos_weight = compute_sqrt_normal_over_abnormal_pos_weight(
        {0: 3108, 1: 554 + 514 + 235}
    )

    assert mapping == (0, 1, 1, 1)
    assert pos_weight == pytest.approx((3108 / (554 + 514 + 235)) ** 0.5)
