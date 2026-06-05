from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import WeightedRandomSampler

from src.data.dataset import RespiratoryClipDataset
from src.utils.config import LossConfig, SamplerConfig


@dataclass(frozen=True)
class ResolvedImbalance:
    pos_weight: float | None
    weighted_random: bool
    class_counts: Mapping[int, int]


@dataclass(frozen=True)
class ResolvedLossWeights:
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...] | None
    main_index_to_binary_target: tuple[int, ...] | None
    binary_counts: Mapping[int, int] | None
    branch_binary_pos_weight: float | None


@dataclass(frozen=True)
class SqrtInverseSamplerSummary:
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...]
    expected_class_probabilities: tuple[float, ...]
    num_samples: int


def collect_targets(dataset: RespiratoryClipDataset) -> list[int]:
    return dataset.targets


def class_counts(targets: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(target) for target in targets).items()))


def compute_binary_pos_weight(targets: Sequence[int], positive: int = 1) -> float:
    counts = class_counts(targets)
    positives = counts.get(positive, 0)
    negatives = sum(count for cls, count in counts.items() if cls != positive)
    if positives <= 0:
        raise ValueError("Cannot compute pos_weight with zero positive samples")
    if negatives <= 0:
        raise ValueError("Cannot compute pos_weight with zero negative samples")
    return float(negatives) / float(positives)


def compute_class_counts_by_index(
    targets: Sequence[int],
    *,
    num_classes: int,
) -> torch.Tensor:
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for target in targets:
        target_index = int(target)
        if target_index < 0 or target_index >= num_classes:
            raise ValueError(
                f"target index {target_index} is outside 0..{num_classes - 1}"
            )
        counts[target_index] += 1.0
    return counts


def _validate_class_counts_for_weights(class_counts: torch.Tensor) -> None:
    if class_counts.ndim != 1:
        raise ValueError(
            f"class_counts must be a 1D tensor, got {tuple(class_counts.shape)}"
        )
    if class_counts.numel() == 0:
        raise ValueError("class_counts must not be empty")
    if torch.any(class_counts < 0):
        raise ValueError("class_counts must be non-negative")


def compute_power_inverse_class_weights(
    class_counts: torch.Tensor,
    *,
    power: float,
) -> torch.Tensor:
    _validate_class_counts_for_weights(class_counts)
    if power <= 0:
        raise ValueError("power must be greater than zero")
    counts = class_counts.to(dtype=torch.float32).clamp_min(1.0)
    weights = counts.pow(-float(power))
    return weights / weights.mean().clamp_min(1e-12)


def compute_sqrt_inverse_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    return compute_power_inverse_class_weights(class_counts, power=0.5)


def build_main_index_to_binary_target(
    *,
    label_to_index: Mapping[str, int],
    binary_label_to_index: Mapping[str, int],
) -> tuple[int, ...]:
    if set(label_to_index.keys()) != set(binary_label_to_index.keys()):
        missing = sorted(set(label_to_index.keys()) - set(binary_label_to_index.keys()))
        extra = sorted(set(binary_label_to_index.keys()) - set(label_to_index.keys()))
        raise ValueError(
            "branch binary label map must contain exactly the main labels; "
            f"missing={missing} extra={extra}"
        )
    mapping = [0] * len(label_to_index)
    for label_name, class_index in label_to_index.items():
        binary_target = int(binary_label_to_index[label_name])
        if binary_target not in {0, 1}:
            raise ValueError("branch binary label targets must be 0 or 1")
        mapping[int(class_index)] = binary_target
    if all(value == 0 for value in mapping):
        raise ValueError("branch binary label map must include class 1")
    if all(value == 1 for value in mapping):
        raise ValueError("branch binary label map must include class 0")
    return tuple(mapping)


def compute_binary_counts_from_targets(
    targets: Sequence[int],
    *,
    main_index_to_binary_target: Sequence[int],
) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for target in targets:
        class_index = int(target)
        if class_index < 0 or class_index >= len(main_index_to_binary_target):
            raise ValueError(
                "target index is outside the branch binary mapping range: "
                f"{class_index}"
            )
        binary_target = int(main_index_to_binary_target[class_index])
        if binary_target not in {0, 1}:
            raise ValueError("branch binary mapping values must be 0 or 1")
        counts[binary_target] += 1
    return counts


def compute_sqrt_normal_over_abnormal_pos_weight(
    binary_counts: Mapping[int, int],
) -> float:
    normal_count = int(binary_counts.get(0, 0))
    abnormal_count = int(binary_counts.get(1, 0))
    if normal_count <= 0:
        raise ValueError(
            "Cannot compute branch binary pos_weight with zero normal samples"
        )
    if abnormal_count <= 0:
        raise ValueError(
            "Cannot compute branch binary pos_weight with zero abnormal samples"
        )
    return float(math.sqrt(float(normal_count) / float(abnormal_count)))


def resolve_loss_weights(
    *,
    targets: Sequence[int],
    num_classes: int,
    label_to_index: Mapping[str, int],
    loss_cfg: LossConfig,
) -> ResolvedLossWeights:
    class_counts_tensor = compute_class_counts_by_index(
        targets,
        num_classes=num_classes,
    )
    class_counts_tuple = tuple(int(value.item()) for value in class_counts_tensor)
    class_weights: tuple[float, ...] | None = None
    if loss_cfg.class_weighting.enabled:
        if loss_cfg.class_weighting.type == "sqrt_inverse_frequency":
            class_weights_tensor = compute_sqrt_inverse_class_weights(
                class_counts_tensor
            )
        elif loss_cfg.class_weighting.type == "power_inverse_frequency":
            class_weights_tensor = compute_power_inverse_class_weights(
                class_counts_tensor,
                power=float(loss_cfg.class_weighting.power),
            )
        else:
            raise ValueError(
                f"Unsupported class_weighting.type: {loss_cfg.class_weighting.type}"
            )
        class_weights = tuple(float(value.item()) for value in class_weights_tensor)

    main_index_to_binary_target: tuple[int, ...] | None = None
    binary_counts: dict[int, int] | None = None
    branch_binary_pos_weight: float | None = None
    if loss_cfg.branch_binary_auxiliary.enabled:
        main_index_to_binary_target = build_main_index_to_binary_target(
            label_to_index=label_to_index,
            binary_label_to_index=loss_cfg.branch_binary_auxiliary.label_to_index,
        )
        binary_counts = compute_binary_counts_from_targets(
            targets,
            main_index_to_binary_target=main_index_to_binary_target,
        )
        if loss_cfg.branch_binary_auxiliary.pos_weight.enabled:
            branch_binary_pos_weight = compute_sqrt_normal_over_abnormal_pos_weight(
                binary_counts
            )
        elif binary_counts.get(0, 0) <= 0 or binary_counts.get(1, 0) <= 0:
            raise ValueError(
                "branch_binary_auxiliary requires both normal and abnormal train "
                "examples"
            )

    return ResolvedLossWeights(
        class_counts=class_counts_tuple,
        class_weights=class_weights,
        main_index_to_binary_target=main_index_to_binary_target,
        binary_counts=binary_counts,
        branch_binary_pos_weight=branch_binary_pos_weight,
    )


def build_weighted_sampler(
    targets: Sequence[int],
    *,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    counts = class_counts(targets)
    weights = [1.0 / float(counts[int(target)]) for target in targets]
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def compute_sqrt_inverse_sample_weights(
    targets: Sequence[int],
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_list = [int(target) for target in targets]
    if not target_list:
        raise ValueError("Cannot build sqrt_inverse_class sampler for empty dataset")
    targets_tensor = torch.tensor(target_list, dtype=torch.long)
    counts = compute_class_counts_by_index(target_list, num_classes=num_classes)
    missing = torch.where(counts <= 0)[0].tolist()
    if missing:
        raise ValueError(
            "Cannot build sqrt_inverse_class sampler because classes have zero "
            f"samples: {missing}"
        )
    class_weights = torch.rsqrt(counts)
    sample_weights = class_weights[targets_tensor].to(dtype=torch.double)
    expected_mass = counts * class_weights
    expected_probabilities = expected_mass / expected_mass.sum().clamp_min(1e-12)
    return sample_weights, counts, class_weights, expected_probabilities


def resolve_sampler_num_samples(
    num_samples: str | int,
    *,
    dataset_size: int,
) -> int:
    if num_samples == "dataset_size":
        return int(dataset_size)
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError(
            "train.sampler.num_samples must be 'dataset_size' or a positive integer"
        )
    if num_samples <= 0:
        raise ValueError(
            "train.sampler.num_samples must be 'dataset_size' or a positive integer"
        )
    return int(num_samples)


def build_sqrt_inverse_class_sampler(
    targets: Sequence[int],
    *,
    num_classes: int,
    cfg: SamplerConfig,
    generator: torch.Generator | None = None,
) -> tuple[WeightedRandomSampler, SqrtInverseSamplerSummary]:
    sample_weights, counts, class_weights, expected_probabilities = (
        compute_sqrt_inverse_sample_weights(targets, num_classes=num_classes)
    )
    num_samples = resolve_sampler_num_samples(
        cfg.num_samples,
        dataset_size=len(sample_weights),
    )
    sampler = WeightedRandomSampler(
        [float(value) for value in sample_weights.tolist()],
        num_samples=num_samples,
        replacement=cfg.replacement,
        generator=generator,
    )
    summary = SqrtInverseSamplerSummary(
        class_counts=tuple(int(value.item()) for value in counts),
        class_weights=tuple(float(value.item()) for value in class_weights),
        expected_class_probabilities=tuple(
            float(value.item()) for value in expected_probabilities
        ),
        num_samples=num_samples,
    )
    return sampler, summary


def resolve_imbalance(
    *,
    targets: Sequence[int],
    num_classes: int = 2,
    loss_type: str = "bce",
    pos_weight: float | None,
    auto_pos_weight: bool,
    weighted_random: bool,
) -> ResolvedImbalance:
    if loss_type == "cross_entropy":
        if auto_pos_weight:
            raise ValueError(
                "train.loss.auto_pos_weight is only supported for one-logit binary "
                "bce/focal runs"
            )
        if pos_weight is not None:
            raise ValueError(
                "train.loss.pos_weight is only supported for one-logit binary "
                "bce/focal runs"
            )
        return ResolvedImbalance(
            pos_weight=None,
            weighted_random=weighted_random,
            class_counts=class_counts(targets),
        )
    if loss_type not in {"bce", "focal"}:
        raise ValueError(
            "train.loss.type must be one of 'bce', 'focal', or 'cross_entropy'"
        )
    if num_classes != 2:
        raise ValueError(
            "bce/focal imbalance resolution is supported only for two-label runs"
        )
    if auto_pos_weight and pos_weight is not None:
        raise ValueError(
            "Set only one of `train.loss.auto_pos_weight` or `train.loss.pos_weight`"
        )
    resolved_pos_weight: float | None
    if auto_pos_weight:
        resolved_pos_weight = compute_binary_pos_weight(targets)
    else:
        resolved_pos_weight = None if pos_weight is None else float(pos_weight)
    if resolved_pos_weight is not None and resolved_pos_weight <= 0:
        raise ValueError("train.loss.pos_weight must be greater than zero")
    return ResolvedImbalance(
        pos_weight=resolved_pos_weight,
        weighted_random=weighted_random,
        class_counts=class_counts(targets),
    )
