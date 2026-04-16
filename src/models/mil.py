from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    if scores.shape != mask.shape:
        raise ValueError(
            "scores and mask must have the same shape for masked_softmax; "
            f"got {tuple(scores.shape)} and {tuple(mask.shape)}"
        )
    valid = mask.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~valid, -1e9)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    normalizer = weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return weights / normalizer


class GatedAttentionMil(nn.Module):
    def __init__(self, in_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.value = nn.Linear(in_dim, attention_dim)
        self.gate = nn.Linear(in_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        dropped = self.dropout(instance_embeddings)
        gated = torch.tanh(self.value(dropped)) * torch.sigmoid(self.gate(dropped))
        attention_scores = self.score(gated).squeeze(-1)
        attention_weights = masked_softmax(attention_scores, instance_mask, dim=1)
        bag_embedding = torch.sum(
            attention_weights.unsqueeze(-1) * instance_embeddings,
            dim=1,
        )
        return bag_embedding, attention_weights


class LinearSoftmaxMil(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, instance_probabilities: Tensor, instance_mask: Tensor) -> Tensor:
        mask = instance_mask.to(dtype=instance_probabilities.dtype)
        if instance_probabilities.ndim == 2:
            numerator = torch.sum(instance_probabilities.square() * mask, dim=1)
            denominator = torch.sum(instance_probabilities * mask, dim=1).clamp_min(
                self.eps
            )
            return (numerator / denominator).clamp(self.eps, 1.0 - self.eps)

        if instance_probabilities.ndim != 3:
            raise ValueError(
                "instance_probabilities must have shape (B, M) or (B, M, C), "
                f"got {tuple(instance_probabilities.shape)}"
            )

        expanded_mask = mask.unsqueeze(-1)
        numerator = torch.sum(instance_probabilities.square() * expanded_mask, dim=1)
        denominator = torch.sum(
            instance_probabilities * expanded_mask, dim=1
        ).clamp_min(self.eps)
        pooled = (numerator / denominator).clamp_min(self.eps)
        pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return pooled.clamp_min(self.eps)
