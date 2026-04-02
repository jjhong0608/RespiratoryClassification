from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MILAggregatorOutput:
    bag_logits: Tensor
    attention_weights: Tensor | None = None
    topk_indices: Tensor | None = None


@dataclass(frozen=True)
class TopKMILConfig:
    k: int = 1


@dataclass(frozen=True)
class AttentionMILConfig:
    input_dim: int
    hidden_dim: int = 128
    dropout: float = 0.0
    gated: bool = True


@dataclass(frozen=True)
class TemperatureMILConfig:
    temperature: float = 1.0


@dataclass(frozen=True)
class NoisyOrMILConfig:
    clamp_eps: float = 1e-6


class BaseMILAggregator(nn.Module):
    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        raise NotImplementedError


def _masked_fill_invalid(logits: Tensor, mask: Tensor, fill_value: float) -> Tensor:
    return logits.masked_fill(~mask, fill_value)


def _validate_mask(mask: Tensor) -> None:
    if mask.ndim != 2:
        raise ValueError(
            f"instance_mask must have shape (B, M); got {tuple(mask.shape)}"
        )
    valid_counts = mask.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("Each bag must contain at least one valid instance")


class MaxMILAggregator(BaseMILAggregator):
    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        del instance_embeddings
        _validate_mask(instance_mask)
        bag_logits = (
            _masked_fill_invalid(instance_logits, instance_mask, float("-inf"))
            .max(dim=1)
            .values
        )
        return MILAggregatorOutput(bag_logits=bag_logits)


class MeanMILAggregator(BaseMILAggregator):
    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        del instance_embeddings
        _validate_mask(instance_mask)
        masked_logits = instance_logits * instance_mask.to(dtype=instance_logits.dtype)
        counts = instance_mask.sum(dim=1).clamp_min(1).to(dtype=instance_logits.dtype)
        return MILAggregatorOutput(bag_logits=masked_logits.sum(dim=1) / counts)


class TopKMILAggregator(BaseMILAggregator):
    def __init__(self, cfg: TopKMILConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        del instance_embeddings
        _validate_mask(instance_mask)
        batch_size = instance_logits.shape[0]
        topk_indices = torch.full(
            (batch_size, self.cfg.k),
            fill_value=-1,
            dtype=torch.long,
            device=instance_logits.device,
        )
        bag_logits = torch.empty(batch_size, device=instance_logits.device)
        for bag_index in range(batch_size):
            valid_indices = torch.nonzero(
                instance_mask[bag_index], as_tuple=False
            ).view(-1)
            valid_logits = instance_logits[bag_index, valid_indices]
            current_k = min(self.cfg.k, int(valid_indices.numel()))
            values, order = torch.topk(valid_logits, k=current_k)
            selected = valid_indices[order]
            bag_logits[bag_index] = values.mean()
            topk_indices[bag_index, :current_k] = selected
        return MILAggregatorOutput(bag_logits=bag_logits, topk_indices=topk_indices)


class AttentionMILAggregator(BaseMILAggregator):
    def __init__(self, cfg: AttentionMILConfig):
        super().__init__()
        self.gated = cfg.gated
        self.dropout = nn.Dropout(cfg.dropout)
        self.value = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.query = nn.Linear(cfg.hidden_dim, 1)
        self.gate: nn.Linear | None = None
        if cfg.gated:
            self.gate = nn.Linear(cfg.input_dim, cfg.hidden_dim)

    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        _validate_mask(instance_mask)
        hidden = torch.tanh(self.value(self.dropout(instance_embeddings)))
        if self.gated:
            if self.gate is None:
                raise RuntimeError("Expected gated attention layer")
            hidden = hidden * torch.sigmoid(self.gate(instance_embeddings))
        raw_scores = self.query(hidden).squeeze(-1)
        raw_scores = _masked_fill_invalid(raw_scores, instance_mask, float("-inf"))
        weights = torch.softmax(raw_scores, dim=1)
        weights = weights * instance_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        bag_logits = (weights * instance_logits).sum(dim=1)
        return MILAggregatorOutput(
            bag_logits=bag_logits,
            attention_weights=weights,
        )


class LogSumExpMILAggregator(BaseMILAggregator):
    def __init__(self, cfg: TemperatureMILConfig):
        super().__init__()
        self.temperature = cfg.temperature

    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        del instance_embeddings
        _validate_mask(instance_mask)
        scaled = _masked_fill_invalid(
            instance_logits / self.temperature,
            instance_mask,
            float("-inf"),
        )
        return MILAggregatorOutput(
            bag_logits=self.temperature * torch.logsumexp(scaled, dim=1)
        )


class SoftmaxWeightedMILAggregator(BaseMILAggregator):
    def __init__(self, cfg: TemperatureMILConfig):
        super().__init__()
        self.temperature = cfg.temperature

    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        del instance_embeddings
        _validate_mask(instance_mask)
        weights = torch.softmax(
            _masked_fill_invalid(
                instance_logits / self.temperature,
                instance_mask,
                float("-inf"),
            ),
            dim=1,
        )
        weights = weights * instance_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return MILAggregatorOutput(bag_logits=(weights * instance_logits).sum(dim=1))


class NoisyOrMILAggregator(BaseMILAggregator):
    def __init__(self, cfg: NoisyOrMILConfig):
        super().__init__()
        self.clamp_eps = cfg.clamp_eps

    def forward(
        self,
        *,
        instance_logits: Tensor,
        instance_embeddings: Tensor,
        instance_mask: Tensor,
    ) -> MILAggregatorOutput:
        del instance_embeddings
        _validate_mask(instance_mask)
        probs = torch.sigmoid(instance_logits).clamp(
            min=self.clamp_eps,
            max=1.0 - self.clamp_eps,
        )
        probs = torch.where(instance_mask, probs, torch.zeros_like(probs))
        log_not_prob = torch.log1p(-probs.clamp(max=1.0 - self.clamp_eps))
        bag_neg_log = (log_not_prob * instance_mask.to(dtype=log_not_prob.dtype)).sum(
            dim=1
        )
        bag_prob = 1.0 - torch.exp(bag_neg_log)
        bag_prob = bag_prob.clamp(self.clamp_eps, 1.0 - self.clamp_eps)
        return MILAggregatorOutput(bag_logits=torch.logit(bag_prob, eps=self.clamp_eps))


def build_mil_aggregator(
    aggregator: Literal[
        "max",
        "mean",
        "topk",
        "attention",
        "logsumexp",
        "softmax_weighted",
        "noisy_or",
    ],
    *,
    input_dim: int,
    topk_k: int,
    attention_hidden_dim: int,
    attention_dropout: float,
    attention_gated: bool,
    logsumexp_temperature: float,
    softmax_weighted_temperature: float,
    noisy_or_clamp_eps: float,
) -> BaseMILAggregator:
    if aggregator == "max":
        return MaxMILAggregator()
    if aggregator == "mean":
        return MeanMILAggregator()
    if aggregator == "topk":
        return TopKMILAggregator(TopKMILConfig(k=topk_k))
    if aggregator == "attention":
        return AttentionMILAggregator(
            AttentionMILConfig(
                input_dim=input_dim,
                hidden_dim=attention_hidden_dim,
                dropout=attention_dropout,
                gated=attention_gated,
            )
        )
    if aggregator == "logsumexp":
        return LogSumExpMILAggregator(
            TemperatureMILConfig(temperature=logsumexp_temperature)
        )
    if aggregator == "softmax_weighted":
        return SoftmaxWeightedMILAggregator(
            TemperatureMILConfig(temperature=softmax_weighted_temperature)
        )
    if aggregator == "noisy_or":
        return NoisyOrMILAggregator(NoisyOrMILConfig(clamp_eps=noisy_or_clamp_eps))
    raise ValueError(f"Unsupported MIL aggregator: {aggregator}")
