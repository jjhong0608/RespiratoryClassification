from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from src.models.instance_head import InstanceHeadConfig, build_instance_head
from src.models.mil_aggregators import build_mil_aggregator
from src.models.segment_encoder import WhisperSegmentEncoder
from src.models.whisper_encoder import WhisperEncoderDims


@dataclass(frozen=True)
class MILModelConfig:
    encoder: WhisperEncoderDims
    instance_head_type: Literal["linear", "mlp"] = "linear"
    instance_hidden_dim: int = 256
    instance_dropout: float = 0.0
    aggregator: Literal[
        "max",
        "mean",
        "topk",
        "attention",
        "logsumexp",
        "softmax_weighted",
        "noisy_or",
    ] = "max"
    topk_k: int = 1
    attention_hidden_dim: int = 128
    attention_dropout: float = 0.0
    attention_gated: bool = True
    logsumexp_temperature: float = 1.0
    softmax_weighted_temperature: float = 1.0
    noisy_or_clamp_eps: float = 1e-6


@dataclass(frozen=True)
class MILModelOutput:
    bag_logits: Tensor
    instance_logits: Tensor
    instance_embeddings: Tensor
    attention_weights: Tensor | None = None
    topk_indices: Tensor | None = None


class RespiratoryMILModel(nn.Module):
    def __init__(self, cfg: MILModelConfig):
        super().__init__()
        self.cfg = cfg
        self.segment_encoder = WhisperSegmentEncoder(cfg.encoder)
        self.instance_head = build_instance_head(
            InstanceHeadConfig(
                input_dim=cfg.encoder.n_audio_state,
                head_type=cfg.instance_head_type,
                hidden_dim=cfg.instance_hidden_dim,
                dropout=cfg.instance_dropout,
            )
        )
        self.aggregator = build_mil_aggregator(
            cfg.aggregator,
            input_dim=cfg.encoder.n_audio_state,
            topk_k=cfg.topk_k,
            attention_hidden_dim=cfg.attention_hidden_dim,
            attention_dropout=cfg.attention_dropout,
            attention_gated=cfg.attention_gated,
            logsumexp_temperature=cfg.logsumexp_temperature,
            softmax_weighted_temperature=cfg.softmax_weighted_temperature,
            noisy_or_clamp_eps=cfg.noisy_or_clamp_eps,
        )

    @property
    def encoder(self) -> nn.Module:
        return self.segment_encoder.encoder

    def forward(self, segments: Tensor, instance_mask: Tensor) -> MILModelOutput:
        if segments.ndim != 4:
            raise ValueError(
                f"segments must have shape (B, M, n_mels, n_frames), got {tuple(segments.shape)}"
            )
        if instance_mask.ndim != 2:
            raise ValueError(
                f"instance_mask must have shape (B, M), got {tuple(instance_mask.shape)}"
            )
        batch_size, num_instances, _, _ = segments.shape
        flat_segments = segments.view(
            batch_size * num_instances,
            segments.shape[2],
            segments.shape[3],
        )
        encoded = self.segment_encoder(flat_segments)
        instance_embeddings = encoded.embeddings.view(
            batch_size,
            num_instances,
            self.cfg.encoder.n_audio_state,
        )
        instance_logits = self.instance_head(instance_embeddings)
        aggregated = self.aggregator(
            instance_logits=instance_logits,
            instance_embeddings=instance_embeddings,
            instance_mask=instance_mask,
        )
        return MILModelOutput(
            bag_logits=aggregated.bag_logits,
            instance_logits=instance_logits,
            instance_embeddings=instance_embeddings,
            attention_weights=aggregated.attention_weights,
            topk_indices=aggregated.topk_indices,
        )
