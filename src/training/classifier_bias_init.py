from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.models.model import ClassifierBiasInitConfig


def _tensor_stats(values: Tensor) -> dict[str, float]:
    values = values.detach().cpu().to(dtype=torch.float32)
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "max": float(values.max().item()),
    }


def _class_examples(
    *,
    class_names: Sequence[str] | None,
    positive_counts: Tensor,
    negative_counts: Tensor,
    pos_weight: Tensor,
    bias: Tensor,
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for class_index in range(min(limit, int(bias.numel()))):
        row: dict[str, Any] = {
            "class_index": class_index,
            "positive_count": float(positive_counts[class_index].item()),
            "negative_count": float(negative_counts[class_index].item()),
            "pos_weight": float(pos_weight[class_index].item()),
            "initialized_bias": float(bias[class_index].item()),
        }
        if class_names is not None and class_index < len(class_names):
            row["class_name"] = class_names[class_index]
        examples.append(row)
    return examples


def compute_classifier_bias_init(
    *,
    train_targets: Tensor,
    pos_weight: Tensor,
    cfg: ClassifierBiasInitConfig,
    class_names: Sequence[str] | None = None,
    example_count: int = 5,
) -> tuple[Tensor, dict[str, Any]]:
    if train_targets.ndim != 2:
        raise ValueError(
            f"train_targets must have shape (N, C), got {tuple(train_targets.shape)}"
        )
    targets = train_targets.detach().cpu().to(dtype=torch.float32)
    weights = pos_weight.detach().cpu().to(dtype=torch.float32)
    if weights.ndim != 1:
        raise ValueError(f"pos_weight must have shape (C,), got {tuple(weights.shape)}")
    if int(weights.numel()) != int(targets.shape[1]):
        raise ValueError(
            "pos_weight length must match train_targets class dimension; "
            f"got pos_weight={int(weights.numel())} classes={int(targets.shape[1])}"
        )
    if cfg.type not in {"prior", "weighted_prior"}:
        raise ValueError(
            "classifier bias initialization requires type 'prior' or "
            f"'weighted_prior', got {cfg.type!r}"
        )

    positive_counts = targets.sum(dim=0)
    negative_counts = float(targets.shape[0]) - positive_counts
    if cfg.type == "prior":
        numerator = positive_counts + float(cfg.eps)
    else:
        numerator = weights * positive_counts + float(cfg.eps)
    denominator = negative_counts + float(cfg.eps)
    unclamped_bias = torch.log(numerator / denominator)
    bias = torch.clamp(
        unclamped_bias,
        min=float(cfg.clamp_min),
        max=float(cfg.clamp_max),
    )
    metadata: dict[str, Any] = {
        "enabled": cfg.enabled,
        "type": cfg.type,
        "source": cfg.source,
        "eps": float(cfg.eps),
        "clamp_min": float(cfg.clamp_min),
        "clamp_max": float(cfg.clamp_max),
        "positive_counts": positive_counts.tolist(),
        "negative_counts": negative_counts.tolist(),
        "positive_count_stats": _tensor_stats(positive_counts),
        "negative_count_stats": _tensor_stats(negative_counts),
        "pos_weight_stats": _tensor_stats(weights),
        "bias_stats": _tensor_stats(bias),
        "num_clamped_min": int((bias == float(cfg.clamp_min)).sum().item()),
        "num_clamped_max": int((bias == float(cfg.clamp_max)).sum().item()),
        "num_zero_positive_classes": int((positive_counts == 0).sum().item()),
        "class_examples": _class_examples(
            class_names=class_names,
            positive_counts=positive_counts,
            negative_counts=negative_counts,
            pos_weight=weights,
            bias=bias,
            limit=example_count,
        ),
        "values": bias.tolist(),
    }
    return bias, metadata


def find_final_classifier_linear(
    classifier: nn.Module,
    *,
    num_classes: int,
) -> tuple[str, nn.Linear]:
    candidates: list[tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in classifier.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == num_classes
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one final classifier Linear layer with "
            f"out_features={num_classes}; found {len(candidates)}"
        )
    return candidates[0]


def apply_classifier_bias_init(
    classifier: nn.Module,
    *,
    bias: Tensor,
    num_classes: int,
) -> str:
    if bias.ndim != 1 or int(bias.numel()) != num_classes:
        raise ValueError(
            "bias must have shape (num_classes,), got "
            f"{tuple(bias.shape)} for num_classes={num_classes}"
        )
    module_name, final_linear = find_final_classifier_linear(
        classifier,
        num_classes=num_classes,
    )
    if final_linear.bias is None:
        raise ValueError("Final classifier Linear layer has no bias parameter")
    with torch.no_grad():
        final_linear.bias.copy_(
            bias.to(device=final_linear.bias.device, dtype=final_linear.bias.dtype)
        )
    return module_name
