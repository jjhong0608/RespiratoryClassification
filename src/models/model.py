from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor, nn
from transformers import ASTConfig, ASTModel

from src.models.classifier import ClassifierDims, LinearClassifier, MlpClassifier
from src.models.outputs import RespiratoryModelOutput


@dataclass(frozen=True)
class AstFeatureDims:
    num_mel_bins: int
    max_length: int


@dataclass(frozen=True)
class EncoderAdaptationConfig:
    mode: Literal["frozen", "partial", "full"] = "partial"
    num_layers: int = 1


@dataclass(frozen=True)
class AstArchitectureConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    frequency_stride: int = 10
    time_stride: int = 10
    patch_size: int = 16
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


@dataclass(frozen=True)
class AstEncoderConfig:
    feature_dims: AstFeatureDims
    type: Literal["ast"] = "ast"
    pretrained_name_or_path: str | None = None
    cache_dir: str | None = None
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: AstArchitectureConfig = field(default_factory=AstArchitectureConfig)


@dataclass(frozen=True)
class ClassifierConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0
    pooling: Literal["cls", "mean"] = "cls"


@dataclass(frozen=True)
class AstModelConfig:
    encoder: AstEncoderConfig
    classifier: ClassifierConfig
    num_classes: int


AstModelOutput = RespiratoryModelOutput


class RespiratoryAstModel(nn.Module):
    def __init__(self, cfg: AstModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_encoder(cfg.encoder)
        self.classifier: nn.Module
        hidden_size = int(self.encoder.config.hidden_size)
        output_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
        classifier_dims = ClassifierDims(
            in_dim=hidden_size,
            num_classes=output_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )
        if cfg.classifier.type == "linear":
            self.classifier = LinearClassifier(classifier_dims)
        elif cfg.classifier.type == "mlp":
            self.classifier = MlpClassifier(classifier_dims)
        else:
            raise ValueError(f"Unsupported classifier type: {cfg.classifier.type}")

    @staticmethod
    def _validate_pretrained_feature_dims(
        pretrained_cfg: ASTConfig,
        feature_dims: AstFeatureDims,
        *,
        name_or_path: str,
    ) -> None:
        actual_bins = int(pretrained_cfg.num_mel_bins)
        actual_length = int(pretrained_cfg.max_length)
        if (
            actual_bins == feature_dims.num_mel_bins
            and actual_length == feature_dims.max_length
        ):
            return
        raise ValueError(
            "Pretrained AST encoder input dims do not match model feature dims.\n"
            f"- encoder: num_mel_bins={actual_bins}, max_length={actual_length}\n"
            f"- model:   num_mel_bins={feature_dims.num_mel_bins}, "
            f"max_length={feature_dims.max_length}\n"
            f"- name_or_path: {name_or_path}"
        )

    @staticmethod
    def _build_scratch_config(cfg: AstEncoderConfig) -> ASTConfig:
        architecture = cfg.architecture
        ast_config = ASTConfig()
        ast_config.hidden_size = architecture.hidden_size
        ast_config.num_hidden_layers = architecture.num_hidden_layers
        ast_config.num_attention_heads = architecture.num_attention_heads
        ast_config.intermediate_size = architecture.intermediate_size
        ast_config.hidden_dropout_prob = architecture.hidden_dropout_prob
        ast_config.attention_probs_dropout_prob = (
            architecture.attention_probs_dropout_prob
        )
        ast_config.frequency_stride = architecture.frequency_stride
        ast_config.time_stride = architecture.time_stride
        ast_config.patch_size = architecture.patch_size
        ast_config.qkv_bias = architecture.qkv_bias
        ast_config.layer_norm_eps = architecture.layer_norm_eps
        ast_config.initializer_range = architecture.initializer_range
        ast_config.num_mel_bins = cfg.feature_dims.num_mel_bins
        ast_config.max_length = cfg.feature_dims.max_length
        return ast_config

    def _build_encoder(self, cfg: AstEncoderConfig) -> ASTModel:
        if cfg.pretrained_name_or_path is not None:
            pretrained_cfg = ASTConfig.from_pretrained(
                cfg.pretrained_name_or_path,
                cache_dir=cfg.cache_dir,
            )
            self._validate_pretrained_feature_dims(
                pretrained_cfg,
                cfg.feature_dims,
                name_or_path=cfg.pretrained_name_or_path,
            )
            return ASTModel.from_pretrained(
                cfg.pretrained_name_or_path,
                cache_dir=cfg.cache_dir,
            )
        return ASTModel(self._build_scratch_config(cfg))

    def _pool(self, hidden_states: Tensor) -> Tensor:
        if self.cfg.classifier.pooling == "cls":
            return hidden_states[:, 0, :]
        return hidden_states.mean(dim=1)

    def forward(self, input_values: Tensor) -> RespiratoryModelOutput:
        if input_values.ndim != 3:
            raise ValueError(
                "input_values must have shape (B, max_length, num_mel_bins), "
                f"got {tuple(input_values.shape)}"
            )
        outputs = self.encoder(input_values=input_values, return_dict=True)
        pooled_embedding = self._pool(outputs.last_hidden_state)
        logits = self.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return RespiratoryModelOutput(logits=logits, pooled_embedding=pooled_embedding)
