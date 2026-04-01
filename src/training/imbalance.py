from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from src.data.dataset import RespiratorySoundDataset
from src.utils.config import ImbalanceConfig


@dataclass(frozen=True)
class ResolvedImbalance:
    pos_weight: float | None
    sampler: str
    class_counts: Mapping[int, int]


def collect_targets(dataset: Dataset[tuple[Tensor, int]]) -> list[int]:
    if isinstance(dataset, RespiratorySoundDataset):
        return dataset.targets
    if isinstance(dataset, ConcatDataset):
        targets: list[int] = []
        for child in dataset.datasets:
            targets.extend(collect_targets(child))
        return targets
    raise TypeError(
        f"Unsupported dataset type for target extraction: {type(dataset)!r}"
    )


def class_counts(targets: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(t) for t in targets).items()))


def compute_binary_pos_weight(targets: Sequence[int], positive: int = 1) -> float:
    counts = class_counts(targets)
    positives = counts.get(positive, 0)
    negatives = sum(count for cls, count in counts.items() if cls != positive)
    if positives <= 0:
        raise ValueError("Cannot compute pos_weight with zero positive samples.")
    if negatives <= 0:
        raise ValueError("Cannot compute pos_weight with zero negative samples.")
    return float(negatives) / float(positives)


def build_sample_weights(targets: Sequence[int]) -> list[float]:
    counts = class_counts(targets)
    if not counts:
        raise ValueError("Cannot build sample weights from an empty target list.")
    return [1.0 / float(counts[int(target)]) for target in targets]


def build_weighted_sampler(targets: Sequence[int]) -> WeightedRandomSampler:
    weights = build_sample_weights(targets)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def resolve_imbalance(
    cfg: ImbalanceConfig,
    targets: Sequence[int],
    *,
    num_classes: int,
) -> ResolvedImbalance:
    if cfg.auto_pos_weight and cfg.pos_weight is not None:
        raise ValueError("Set only one of `auto_pos_weight` or `pos_weight`.")
    if num_classes != 2 and (
        cfg.auto_pos_weight or cfg.pos_weight is not None or cfg.sampler != "none"
    ):
        raise ValueError(
            "Imbalance options are currently supported only for binary tasks."
        )

    pos_weight: float | None = None
    if cfg.auto_pos_weight:
        pos_weight = compute_binary_pos_weight(targets)
    elif cfg.pos_weight is not None:
        pos_weight = float(cfg.pos_weight)
        if pos_weight <= 0:
            raise ValueError("`pos_weight` must be greater than zero.")

    return ResolvedImbalance(
        pos_weight=pos_weight,
        sampler=cfg.sampler,
        class_counts=class_counts(targets),
    )
