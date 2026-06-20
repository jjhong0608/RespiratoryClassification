from __future__ import annotations

import torch
from torch import Tensor


def binary_positive_scores(logits: Tensor) -> Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits.squeeze(1))
    raise ValueError(
        f"Binary logits must have shape (B,) or (B, 1), got {tuple(logits.shape)}"
    )


def probabilities_and_predictions(
    logits: Tensor,
    *,
    num_classes: int,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    if num_classes == 2:
        probabilities = binary_positive_scores(logits.detach())
        predictions = (probabilities >= threshold).to(torch.long)
        return probabilities, predictions
    probabilities = torch.softmax(logits.detach(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return probabilities, predictions


def class_probabilities_and_predictions(
    logits: Tensor,
    *,
    num_classes: int,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    probabilities, predictions = probabilities_and_predictions(
        logits,
        num_classes=num_classes,
        threshold=threshold,
    )
    if num_classes == 2:
        positive = probabilities
        return torch.stack([1.0 - positive, positive], dim=1), predictions
    return probabilities, predictions
