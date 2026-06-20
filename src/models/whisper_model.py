from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor, nn

from src.models.classifier import (
    ClassifierDims,
    HuggingFaceClassifier,
    HuggingFaceClassifierDims,
    LinearClassifier,
    MlpClassifier,
)
from src.models.model import EncoderAdaptationConfig
from src.models.outputs import RespiratoryModelOutput
from src.models.whisper_encoder import AudioEncoder, WhisperEncoderDims


@dataclass(frozen=True)
class WhisperModelConfig:
    encoder: WhisperEncoderDims
    num_classes: int
    adaptation: EncoderAdaptationConfig = field(
        default_factory=lambda: EncoderAdaptationConfig(mode="frozen", num_layers=1)
    )
    head_type: Literal["hf", "linear", "mlp"] = "hf"
    pooling: Literal["mean", "cls"] = "mean"
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    hidden_dim: int = 256
    dropout: float = 0.0


class RespiratoryWhisperModel(nn.Module):
    def __init__(self, cfg: WhisperModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.head_type != "hf" and cfg.pooling != "mean":
            raise ValueError(
                "Legacy linear/mlp Whisper heads only support pooling='mean'"
            )
        self.encoder = AudioEncoder(cfg.encoder)
        output_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
        if cfg.head_type == "hf":
            self.classifier: nn.Module = HuggingFaceClassifier(
                HuggingFaceClassifierDims(
                    in_dim=cfg.encoder.n_audio_state,
                    num_classes=output_dim,
                    num_hidden_layers=cfg.encoder.n_audio_layer,
                    pooling=cfg.pooling,
                    classifier_proj_size=cfg.classifier_proj_size,
                    use_weighted_layer_sum=cfg.use_weighted_layer_sum,
                )
            )
        else:
            dims = ClassifierDims(
                in_dim=cfg.encoder.n_audio_state,
                num_classes=output_dim,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
            )
            if cfg.head_type == "linear":
                self.classifier = LinearClassifier(dims)
            else:
                self.classifier = MlpClassifier(dims)

    def forward(self, input_values: Tensor) -> RespiratoryModelOutput:
        if input_values.ndim != 3:
            raise ValueError(
                "input_values must have shape (B, n_mels, n_frames), "
                f"got {tuple(input_values.shape)}"
            )
        need_hidden_states = (
            self.cfg.head_type == "hf" and self.cfg.use_weighted_layer_sum
        )
        if self.cfg.head_type == "hf":
            if not isinstance(self.classifier, HuggingFaceClassifier):
                raise RuntimeError("Expected HuggingFaceClassifier for head_type='hf'")
            prefix_tokens = self.classifier.build_prefix_tokens(
                input_values.shape[0],
                device=input_values.device,
                dtype=input_values.dtype,
            )
            features = self.encoder(
                input_values,
                output_hidden_states=need_hidden_states,
                prefix_tokens=prefix_tokens,
            )
            projected = self.classifier.project_sequence(
                features.last_hidden_state,
                features.hidden_states,
            )
            pooled_embedding = self.classifier.pool_projected(projected)
            logits = self.classifier.classifier(pooled_embedding)
        else:
            features = self.encoder(input_values, output_hidden_states=False)
            pooled_embedding = features.last_hidden_state.mean(dim=1)
            logits = self.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return RespiratoryModelOutput(logits=logits, pooled_embedding=pooled_embedding)
