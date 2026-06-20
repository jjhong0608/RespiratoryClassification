from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ClassifierDims:
    in_dim: int
    num_classes: int
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class HuggingFaceClassifierDims:
    in_dim: int
    num_classes: int
    num_hidden_layers: int
    pooling: str = "mean"
    classifier_proj_size: int = 256
    use_weighted_layer_sum: bool = False


class LinearClassifier(nn.Module):
    def __init__(self, dims: ClassifierDims):
        super().__init__()
        self.proj = nn.Linear(dims.in_dim, dims.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class MlpClassifier(nn.Module):
    def __init__(self, dims: ClassifierDims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims.in_dim, dims.hidden_dim),
            nn.GELU(),
            nn.Dropout(dims.dropout),
            nn.Linear(dims.hidden_dim, dims.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HuggingFaceClassifier(nn.Module):
    def __init__(self, dims: HuggingFaceClassifierDims):
        super().__init__()
        if dims.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling for HF classifier: {dims.pooling}")
        self.pooling = dims.pooling
        self.use_weighted_layer_sum = dims.use_weighted_layer_sum
        self.num_layer_states = dims.num_hidden_layers + 1
        self.layer_weights: nn.Parameter | None = None
        self.cls_token: nn.Parameter | None = None
        self.cls_positional_embedding: nn.Parameter | None = None
        if dims.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(
                torch.ones(self.num_layer_states) / float(self.num_layer_states)
            )
        if dims.pooling == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, dims.in_dim))
            self.cls_positional_embedding = nn.Parameter(torch.empty(1, 1, dims.in_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.cls_positional_embedding, std=0.02)
        self.projector = nn.Linear(dims.in_dim, dims.classifier_proj_size)
        self.classifier = nn.Linear(dims.classifier_proj_size, dims.num_classes)

    def build_prefix_tokens(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.pooling != "cls":
            return None
        if self.cls_token is None or self.cls_positional_embedding is None:
            raise RuntimeError("CLS parameters are not initialized")
        token = self.cls_token + self.cls_positional_embedding
        return token.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def project_sequence(
        self,
        last_hidden_state: Tensor,
        hidden_states: tuple[Tensor, ...] | None = None,
    ) -> Tensor:
        if self.use_weighted_layer_sum:
            if hidden_states is None:
                raise ValueError(
                    "hidden_states are required when use_weighted_layer_sum=True"
                )
            if self.layer_weights is None:
                raise RuntimeError("layer_weights are not initialized")
            if len(hidden_states) != self.num_layer_states:
                raise ValueError(
                    "Expected hidden_states length "
                    f"{self.num_layer_states}, got {len(hidden_states)}"
                )
            stacked = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            features = (
                stacked * norm_weights.view(1, self.num_layer_states, 1, 1)
            ).sum(dim=1)
        else:
            features = last_hidden_state
        return self.projector(features)

    def pool_projected(self, features: Tensor) -> Tensor:
        if self.pooling == "cls":
            return features[:, 0, :]
        return features.mean(dim=1)

    def forward(
        self,
        last_hidden_state: Tensor,
        hidden_states: tuple[Tensor, ...] | None = None,
    ) -> Tensor:
        features = self.project_sequence(last_hidden_state, hidden_states)
        pooled_output = self.pool_projected(features)
        return self.classifier(pooled_output)
