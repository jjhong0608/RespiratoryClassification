from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn


@dataclass(frozen=True)
class InstanceHeadConfig:
    input_dim: int
    head_type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


class BaseInstanceHead(nn.Module):
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError


class LinearInstanceHead(BaseInstanceHead):
    def __init__(self, cfg: InstanceHeadConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.input_dim, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.proj(embeddings).squeeze(-1)


class MLPInstanceHead(BaseInstanceHead):
    def __init__(self, cfg: InstanceHeadConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.net(embeddings).squeeze(-1)


def build_instance_head(cfg: InstanceHeadConfig) -> BaseInstanceHead:
    if cfg.head_type == "linear":
        return LinearInstanceHead(cfg)
    if cfg.head_type == "mlp":
        return MLPInstanceHead(cfg)
    raise ValueError(f"Unsupported instance head type: {cfg.head_type}")
