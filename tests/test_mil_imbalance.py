from __future__ import annotations

import pytest
import torch
from src.training.imbalance import (
    build_main_index_to_binary_target,
    build_sqrt_inverse_class_sampler,
    compute_binary_pos_weight,
    compute_class_counts_by_index,
    compute_power_inverse_class_weights,
    compute_sqrt_inverse_class_weights,
    compute_sqrt_inverse_sample_weights,
    compute_sqrt_normal_over_abnormal_pos_weight,
    resolve_imbalance,
    resolve_loss_weights,
    resolve_sampler_num_samples,
)
from src.utils.config import ClassWeightingConfig, LossConfig, SamplerConfig


def test_compute_binary_pos_weight_uses_negative_over_positive_ratio() -> None:
    value = compute_binary_pos_weight([0, 0, 0, 1, 1])

    assert value == pytest.approx(1.5)


def test_resolve_imbalance_computes_auto_pos_weight() -> None:
    resolved = resolve_imbalance(
        targets=[0, 0, 0, 1, 1],
        loss_type="bce",
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
            loss_type="bce",
            pos_weight=2.0,
            auto_pos_weight=True,
            weighted_random=False,
        )


def test_resolve_imbalance_cross_entropy_ignores_binary_pos_weighting() -> None:
    resolved = resolve_imbalance(
        targets=[0, 0, 1, 1],
        num_classes=2,
        loss_type="cross_entropy",
        pos_weight=None,
        auto_pos_weight=False,
        weighted_random=False,
    )

    assert resolved.pos_weight is None
    assert dict(resolved.class_counts) == {0: 2, 1: 2}


@pytest.mark.parametrize(
    ("field", "kwargs", "error"),
    [
        ("auto_pos_weight", {"auto_pos_weight": True}, "auto_pos_weight.*one-logit"),
        ("pos_weight", {"pos_weight": 2.0}, "pos_weight.*one-logit"),
    ],
)
def test_resolve_imbalance_cross_entropy_rejects_binary_pos_weighting(
    field: str,
    kwargs: dict[str, object],
    error: str,
) -> None:
    base_kwargs = {
        "pos_weight": None,
        "auto_pos_weight": False,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=error):
        resolve_imbalance(
            targets=[0, 1],
            num_classes=2,
            loss_type="cross_entropy",
            weighted_random=False,
            **base_kwargs,
        )


def test_sqrt_inverse_class_weights_are_mean_one_and_ranked() -> None:
    counts = torch.tensor([3108, 554, 514, 235], dtype=torch.float32)

    weights = compute_sqrt_inverse_class_weights(counts)

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1]
    assert weights[1].item() == pytest.approx(weights[2].item(), rel=0.05)
    assert weights[2] < weights[3]


def test_power_inverse_class_weights_are_mean_one_and_ranked() -> None:
    counts = torch.tensor([658, 252, 156], dtype=torch.float32)

    weights = compute_power_inverse_class_weights(counts, power=0.75)

    expected = counts.pow(-0.75)
    expected = expected / expected.mean()
    assert torch.allclose(weights, expected)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]


def test_power_half_matches_sqrt_inverse_class_weights() -> None:
    counts = torch.tensor([3108, 554, 514, 235], dtype=torch.float32)

    power_weights = compute_power_inverse_class_weights(counts, power=0.5)
    sqrt_weights = compute_sqrt_inverse_class_weights(counts)

    assert torch.allclose(power_weights, sqrt_weights)


def test_resolve_loss_weights_uses_power_inverse_class_weighting() -> None:
    resolved = resolve_loss_weights(
        targets=[0, 0, 0, 1, 2],
        num_classes=3,
        label_to_index={"normal": 0, "crackle": 1, "wheeze": 2},
        loss_cfg=LossConfig(
            type="cross_entropy",
            class_weighting=ClassWeightingConfig(
                enabled=True,
                type="power_inverse_frequency",
                power=0.75,
            ),
        ),
    )

    expected = compute_power_inverse_class_weights(
        torch.tensor([3.0, 1.0, 1.0]),
        power=0.75,
    )
    assert resolved.class_weights is not None
    assert resolved.class_weights == pytest.approx(
        tuple(float(value.item()) for value in expected)
    )


def test_sqrt_inverse_sample_weights_match_class_counts() -> None:
    targets = [0, 0, 0, 0, 1, 1, 2]

    sample_weights, counts, class_weights, expected_probabilities = (
        compute_sqrt_inverse_sample_weights(targets, num_classes=3)
    )

    assert torch.equal(counts, torch.tensor([4.0, 2.0, 1.0]))
    assert torch.allclose(
        class_weights,
        torch.tensor([0.5, 1.0 / (2.0**0.5), 1.0], dtype=torch.float32),
    )
    assert torch.allclose(
        sample_weights,
        torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 1.0 / (2.0**0.5), 1.0 / (2.0**0.5), 1.0],
            dtype=torch.float64,
        ),
    )
    expected_mass = counts * class_weights
    assert torch.allclose(
        expected_probabilities,
        expected_mass / expected_mass.sum(),
    )


def test_sqrt_inverse_sample_weights_reject_zero_count_class() -> None:
    with pytest.raises(ValueError, match="zero samples"):
        compute_sqrt_inverse_sample_weights([0, 0, 1], num_classes=3)


def test_build_sqrt_inverse_class_sampler_resolves_dataset_size() -> None:
    sampler, summary = build_sqrt_inverse_class_sampler(
        [0, 0, 1, 1],
        num_classes=2,
        cfg=SamplerConfig(
            enabled=True,
            type="sqrt_inverse_class",
            replacement=True,
            num_samples="dataset_size",
            source="train",
        ),
    )

    assert sampler.num_samples == 4
    assert sampler.replacement is True
    assert summary.num_samples == 4
    assert summary.class_counts == (2, 2)


def test_resolve_sampler_num_samples_accepts_positive_integer() -> None:
    assert resolve_sampler_num_samples(12, dataset_size=4) == 12


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
