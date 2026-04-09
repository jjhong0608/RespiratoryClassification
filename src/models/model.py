from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor, nn

from src.models.instance_head import (
    InstanceHeadConfig as RuntimeInstanceHeadConfig,
    build_instance_head,
)
from src.models.mil_aggregators import build_mil_aggregator
from src.models.segment_encoder import (
    SegmentEncoderPoolingConfig,
    WhisperSegmentEncoder,
    WhisperSegmentEncoderConfig,
)
from src.models.whisper_encoder import WhisperEncoderDims


@dataclass(frozen=True)
class SegmentEncoderAdaptationConfig:
    mode: Literal["frozen", "partial", "full"] = "partial"
    num_layers: int = 1


@dataclass(frozen=True)
class SegmentEncoderConfig:
    dims: WhisperEncoderDims
    type: Literal["whisper"] = "whisper"
    backbone: str = "custom"
    pretrained_name_or_path: str | None = None
    strict: bool = True
    download_root: str | None = None
    pooling: SegmentEncoderPoolingConfig = field(
        default_factory=SegmentEncoderPoolingConfig
    )
    adaptation: SegmentEncoderAdaptationConfig = field(
        default_factory=SegmentEncoderAdaptationConfig
    )


@dataclass(frozen=True)
class InstanceHeadConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class InterAttentionConfig:
    hidden_dim: int = 128
    dropout: float = 0.0
    gated: bool = True


@dataclass(frozen=True)
class TopKConfig:
    k: int = 1


@dataclass(frozen=True)
class MILConfig:
    aggregator: Literal["attention", "max", "topk"] = "attention"
    attention: InterAttentionConfig = field(default_factory=InterAttentionConfig)
    topk: TopKConfig = field(default_factory=TopKConfig)


@dataclass(frozen=True)
class MILModelConfig:
    segment_encoder: SegmentEncoderConfig
    instance_head: InstanceHeadConfig
    mil: MILConfig = field(default_factory=MILConfig)


@dataclass(frozen=True)
class MILModelOutput:
    bag_logits: Tensor
    instance_logits: Tensor
    instance_embeddings: Tensor
    intra_attention_weights: Tensor
    inter_attention_weights: Tensor | None = None
    topk_indices: Tensor | None = None


class RespiratoryMILModel(nn.Module):
    def __init__(self, cfg: MILModelConfig):
        super().__init__()
        self.cfg = cfg
        self.segment_encoder = WhisperSegmentEncoder(
            WhisperSegmentEncoderConfig(
                dims=cfg.segment_encoder.dims,
                pooling=cfg.segment_encoder.pooling,
            )
        )
        self.instance_head = build_instance_head(
            RuntimeInstanceHeadConfig(
                input_dim=cfg.segment_encoder.dims.n_audio_state,
                head_type=cfg.instance_head.type,
                hidden_dim=cfg.instance_head.hidden_dim,
                dropout=cfg.instance_head.dropout,
            )
        )
        self.aggregator = build_mil_aggregator(
            cfg.mil.aggregator,
            input_dim=cfg.segment_encoder.dims.n_audio_state,
            topk_k=cfg.mil.topk.k,
            attention_hidden_dim=cfg.mil.attention.hidden_dim,
            attention_dropout=cfg.mil.attention.dropout,
            attention_gated=cfg.mil.attention.gated,
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
        flat_segments = segments.reshape(
            batch_size * num_instances,
            segments.shape[2],
            segments.shape[3],
        )
        encoded = self.segment_encoder(flat_segments)
        instance_embeddings = encoded.embeddings.reshape(
            batch_size,
            num_instances,
            self.cfg.segment_encoder.dims.n_audio_state,
        )
        intra_attention_weights = encoded.attention_weights.reshape(
            batch_size,
            num_instances,
            encoded.attention_weights.shape[1],
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
            intra_attention_weights=intra_attention_weights,
            inter_attention_weights=aggregated.inter_attention_weights,
            topk_indices=aggregated.topk_indices,
        )
