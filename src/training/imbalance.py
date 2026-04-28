from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import WeightedRandomSampler

from src.data.dataset import RespiratoryClipDataset


@dataclass(frozen=True)
class ResolvedImbalance:
    pos_weight: float | None
    weighted_random: bool
    class_counts: Mapping[int, int]


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


def resolve_imbalance(
    *,
    targets: Sequence[int],
    num_classes: int = 2,
    pos_weight: float | None,
    auto_pos_weight: bool,
    weighted_random: bool,
) -> ResolvedImbalance:
    if num_classes > 2:
        if auto_pos_weight:
            raise ValueError(
                "train.loss.auto_pos_weight is only supported for binary classification"
            )
        if pos_weight is not None:
            raise ValueError(
                "train.loss.pos_weight is only supported for binary classification"
            )
        return ResolvedImbalance(
            pos_weight=None,
            weighted_random=weighted_random,
            class_counts=class_counts(targets),
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
