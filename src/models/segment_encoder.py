from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from src.models.whisper_encoder import AudioEncoder, WhisperEncoderDims


@dataclass(frozen=True)
class SegmentEncoderPoolingConfig:
    type: Literal["attention"] = "attention"
    hidden_dim: int = 128
    dropout: float = 0.0
    gated: bool = True


@dataclass(frozen=True)
class WhisperSegmentEncoderConfig:
    dims: WhisperEncoderDims
    pooling: SegmentEncoderPoolingConfig


@dataclass(frozen=True)
class SegmentEncoderOutput:
    embeddings: Tensor
    token_states: Tensor
    attention_weights: Tensor


class TokenAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, cfg: SegmentEncoderPoolingConfig):
        super().__init__()
        self.gated = cfg.gated
        self.dropout = nn.Dropout(cfg.dropout)
        self.value = nn.Linear(input_dim, cfg.hidden_dim)
        self.query = nn.Linear(cfg.hidden_dim, 1)
        self.gate: nn.Linear | None = None
        if cfg.gated:
            self.gate = nn.Linear(input_dim, cfg.hidden_dim)

    def forward(self, token_states: Tensor) -> tuple[Tensor, Tensor]:
        hidden = torch.tanh(self.value(self.dropout(token_states)))
        if self.gated:
            if self.gate is None:
                raise RuntimeError("Expected gated intra-segment attention layer")
            hidden = hidden * torch.sigmoid(self.gate(token_states))
        raw_scores = self.query(hidden).squeeze(-1)
        weights = torch.softmax(raw_scores, dim=1)
        embeddings = torch.sum(weights.unsqueeze(-1) * token_states, dim=1)
        return embeddings, weights


class WhisperSegmentEncoder(nn.Module):
    def __init__(self, cfg: WhisperSegmentEncoderConfig):
        super().__init__()
        if cfg.pooling.type != "attention":
            raise ValueError(
                f"Unsupported segment pooling type: {cfg.pooling.type!r}"
            )
        self.cfg = cfg
        self.encoder = AudioEncoder(cfg.dims)
        self.pooling = TokenAttentionPooling(cfg.dims.n_audio_state, cfg.pooling)

    def forward(self, segments: Tensor) -> SegmentEncoderOutput:
        encoded = self.encoder(segments)
        token_states = encoded.last_hidden_state
        embeddings, attention_weights = self.pooling(token_states)
        return SegmentEncoderOutput(
            embeddings=embeddings,
            token_states=token_states,
            attention_weights=attention_weights,
        )
