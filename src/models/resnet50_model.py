from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models.classifier import ClassifierDims, LinearClassifier, MlpClassifier
from src.models.outputs import RespiratoryModelOutput


@dataclass(frozen=True)
class ResNet50AdaptationConfig:
    mode: Literal["frozen", "partial", "full"] = "frozen"
    num_layers: int = 1


@dataclass(frozen=True)
class ResNet50EncoderRuntimeConfig:
    type: Literal["resnet50"] = "resnet50"
    weights: Literal["imagenet", "none"] = "imagenet"
    adaptation: ResNet50AdaptationConfig = field(
        default_factory=ResNet50AdaptationConfig
    )
    input_channels: int = 3
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ResNet50ClassifierConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class ResNet50ModelConfig:
    encoder: ResNet50EncoderRuntimeConfig
    classifier: ResNet50ClassifierConfig
    num_classes: int


@dataclass(frozen=True)
class ResNet50PretrainedInfo:
    source: str
    weights: str
    embedding_dim: int
    input_channels: int
    image_size: int


class RespiratoryResNet50Model(nn.Module):
    embedding_dim = 2048

    def __init__(self, cfg: ResNet50ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_encoder(cfg.encoder)
        output_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
        dims = ClassifierDims(
            in_dim=self.embedding_dim,
            num_classes=output_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )
        if cfg.classifier.type == "linear":
            self.classifier: nn.Module = LinearClassifier(dims)
        elif cfg.classifier.type == "mlp":
            self.classifier = MlpClassifier(dims)
        else:
            raise ValueError(f"Unsupported classifier type: {cfg.classifier.type}")

    @staticmethod
    def _resolve_weights(
        weights: Literal["imagenet", "none"],
    ) -> ResNet50_Weights | None:
        if weights == "imagenet":
            return ResNet50_Weights.IMAGENET1K_V2
        if weights == "none":
            return None
        raise ValueError(f"Unsupported ResNet50 weights: {weights}")

    def _build_encoder(self, cfg: ResNet50EncoderRuntimeConfig) -> nn.Module:
        if cfg.input_channels != 3:
            raise ValueError("RespiratoryResNet50Model requires input_channels=3")
        encoder = resnet50(weights=self._resolve_weights(cfg.weights))
        encoder.fc = nn.Identity()
        return cast(nn.Module, encoder)

    def forward(self, input_values: Tensor) -> RespiratoryModelOutput:
        if input_values.ndim != 4:
            raise ValueError(
                "input_values must have shape (B, 3, image_size, image_size), "
                f"got {tuple(input_values.shape)}"
            )
        expected = (
            input_values.shape[0],
            self.cfg.encoder.input_channels,
            self.cfg.encoder.image_size,
            self.cfg.encoder.image_size,
        )
        if tuple(input_values.shape) != expected:
            raise ValueError(
                "input_values must have shape "
                f"{expected}, got {tuple(input_values.shape)}"
            )
        pooled_embedding = self.encoder(input_values)
        logits = self.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return RespiratoryModelOutput(logits=logits, pooled_embedding=pooled_embedding)
