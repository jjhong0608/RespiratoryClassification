from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch.utils.data import WeightedRandomSampler

from src.data.dataset import RespiratoryBagDataset


@dataclass(frozen=True)
class ResolvedImbalance:
    pos_weight: float | None
    weighted_random: bool
    class_counts: Mapping[int, int]


def collect_targets(dataset: RespiratoryBagDataset) -> list[int]:
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


def build_weighted_sampler(targets: Sequence[int]) -> WeightedRandomSampler:
    counts = class_counts(targets)
    weights = [1.0 / float(counts[int(target)]) for target in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def resolve_imbalance(
    *,
    targets: Sequence[int],
    pos_weight: float | None,
    weighted_random: bool,
) -> ResolvedImbalance:
    resolved_pos_weight = None if pos_weight is None else float(pos_weight)
    if resolved_pos_weight is not None and resolved_pos_weight <= 0:
        raise ValueError("train.loss.pos_weight must be greater than zero")
    return ResolvedImbalance(
        pos_weight=resolved_pos_weight,
        weighted_random=weighted_random,
        class_counts=class_counts(targets),
    )
