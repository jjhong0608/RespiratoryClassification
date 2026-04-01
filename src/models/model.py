from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from src.models.classifier import (
    ClassifierDims,
    HuggingFaceClassifier,
    HuggingFaceClassifierDims,
    LinearClassifier,
    MlpClassifier,
)
from src.models.whisper_encoder import AudioEncoder, WhisperEncoderDims


@dataclass(frozen=True)
class WhisperClassifierConfig:
    encoder: WhisperEncoderDims
    num_classes: int
    head_type: Literal["hf", "linear", "mlp"] = "hf"
    pooling: Literal["mean", "cls"] = "mean"
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    hidden_dim: int = 256
    dropout: float = 0.0


class WhisperEncoderClassifier(nn.Module):
    def __init__(self, cfg: WhisperClassifierConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.head_type != "hf" and cfg.pooling != "mean":
            raise ValueError("Legacy linear/mlp heads only support pooling='mean'")
        self.encoder = AudioEncoder(cfg.encoder)
        if cfg.head_type == "hf":
            self.classifier: nn.Module = HuggingFaceClassifier(
                HuggingFaceClassifierDims(
                    in_dim=cfg.encoder.n_audio_state,
                    num_classes=cfg.num_classes,
                    num_hidden_layers=cfg.encoder.n_audio_layer,
                    pooling=cfg.pooling,
                    classifier_proj_size=cfg.classifier_proj_size,
                    use_weighted_layer_sum=cfg.use_weighted_layer_sum,
                )
            )
        else:
            dims = ClassifierDims(
                in_dim=cfg.encoder.n_audio_state,
                num_classes=cfg.num_classes,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
            )
            if cfg.head_type == "linear":
                self.classifier = LinearClassifier(dims)
            else:
                self.classifier = MlpClassifier(dims)

    def forward(self, x: Tensor) -> Tensor:
        need_hidden_states = (
            self.cfg.head_type == "hf" and self.cfg.use_weighted_layer_sum
        )
        if self.cfg.head_type == "hf":
            if not isinstance(self.classifier, HuggingFaceClassifier):
                raise RuntimeError("Expected HuggingFaceClassifier for head_type='hf'")
            prefix_tokens = self.classifier.build_prefix_tokens(
                x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )
            features = self.encoder(
                x,
                output_hidden_states=need_hidden_states,
                prefix_tokens=prefix_tokens,
            )
            return self.classifier(
                features.last_hidden_state,
                features.hidden_states,
            )
        features = self.encoder(x, output_hidden_states=need_hidden_states)
        if self.cfg.pooling == "mean":
            pooled = features.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.cfg.pooling}")
        return self.classifier(pooled)
