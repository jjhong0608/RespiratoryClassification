from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from transformers import ASTConfig, ASTModel

from src.models.classifier import ClassifierDims, LinearClassifier, MlpClassifier
from src.models.mil import GatedAttentionMil, LinearSoftmaxMil


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
    pooling: Literal["cls", "mean"] = "cls"
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: AstArchitectureConfig = field(default_factory=AstArchitectureConfig)


@dataclass(frozen=True)
class InstanceHeadConfig:
    projection_dim: int | None = None
    dropout: float = 0.0
    normalize: bool = False


@dataclass(frozen=True)
class GatedAttentionMilConfig:
    attention_dim: int = 128
    dropout: float = 0.0


@dataclass(frozen=True)
class LinearSoftmaxMilConfig:
    eps: float = 1e-6


@dataclass(frozen=True)
class MilConfig:
    type: Literal["gated_attention", "linear_softmax"] = "gated_attention"
    gated_attention: GatedAttentionMilConfig = field(
        default_factory=GatedAttentionMilConfig
    )
    linear_softmax: LinearSoftmaxMilConfig = field(
        default_factory=LinearSoftmaxMilConfig
    )


@dataclass(frozen=True)
class ClassifierConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class AstMilModelConfig:
    encoder: AstEncoderConfig
    instance_head: InstanceHeadConfig
    mil: MilConfig
    classifier: ClassifierConfig
    num_classes: int


@dataclass(frozen=True)
class AstMilOutput:
    bag_logits: Tensor
    bag_probabilities: Tensor
    instance_logits: Tensor
    instance_probabilities: Tensor
    instance_embeddings: Tensor
    bag_embedding: Tensor
    instance_mask: Tensor
    attention_weights: Tensor | None = None


class InstanceHead(nn.Module):
    def __init__(self, in_dim: int, cfg: InstanceHeadConfig):
        super().__init__()
        out_dim = cfg.projection_dim or in_dim
        layers: list[nn.Module] = []
        if cfg.projection_dim is not None:
            layers.append(nn.Linear(in_dim, out_dim))
        if cfg.normalize:
            layers.append(nn.LayerNorm(out_dim))
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Identity() if not layers else nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RespiratoryAstMilModel(nn.Module):
    def __init__(self, cfg: AstMilModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_encoder(cfg.encoder)
        hidden_size = int(self.encoder.config.hidden_size)
        self.instance_head = InstanceHead(hidden_size, cfg.instance_head)
        self.embedding_dim = self.instance_head.out_dim
        output_dim = 1 if cfg.num_classes == 2 else cfg.num_classes
        classifier_dims = ClassifierDims(
            in_dim=self.embedding_dim,
            num_classes=output_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )
        if cfg.classifier.type == "linear":
            self.classifier: nn.Module = LinearClassifier(classifier_dims)
        elif cfg.classifier.type == "mlp":
            self.classifier = MlpClassifier(classifier_dims)
        else:
            raise ValueError(f"Unsupported classifier type: {cfg.classifier.type}")
        if cfg.mil.type == "gated_attention":
            self.mil_pool: nn.Module = GatedAttentionMil(
                self.embedding_dim,
                cfg.mil.gated_attention.attention_dim,
                cfg.mil.gated_attention.dropout,
            )
        elif cfg.mil.type == "linear_softmax":
            self.mil_pool = LinearSoftmaxMil(cfg.mil.linear_softmax.eps)
        else:
            raise ValueError(f"Unsupported MIL aggregator: {cfg.mil.type}")

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

    def _pool_encoder_output(self, hidden_states: Tensor) -> Tensor:
        if self.cfg.encoder.pooling == "cls":
            return hidden_states[:, 0, :]
        return hidden_states.mean(dim=1)

    def _scatter_embeddings(
        self, flat_embeddings: Tensor, instance_mask: Tensor
    ) -> Tensor:
        batch_size, max_instances = instance_mask.shape
        scattered = flat_embeddings.new_zeros(
            (batch_size, max_instances, flat_embeddings.shape[-1])
        )
        scattered[instance_mask] = flat_embeddings
        return scattered

    def _logits_to_probabilities(self, logits: Tensor) -> Tensor:
        if self.cfg.num_classes == 2:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _probabilities_to_logits(self, probabilities: Tensor) -> Tensor:
        eps = self.cfg.mil.linear_softmax.eps
        if self.cfg.num_classes == 2:
            return torch.logit(probabilities.clamp(eps, 1.0 - eps))
        normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            eps
        )
        return torch.log(normalized.clamp_min(eps))

    @staticmethod
    def _masked_mean(instance_embeddings: Tensor, instance_mask: Tensor) -> Tensor:
        weights = instance_mask.to(dtype=instance_embeddings.dtype).unsqueeze(-1)
        summed = torch.sum(instance_embeddings * weights, dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(self, input_values: Tensor, instance_mask: Tensor) -> AstMilOutput:
        if input_values.ndim != 4:
            raise ValueError(
                "input_values must have shape (B, M, max_length, num_mel_bins), "
                f"got {tuple(input_values.shape)}"
            )
        if instance_mask.ndim != 2:
            raise ValueError(
                "instance_mask must have shape (B, M), "
                f"got {tuple(instance_mask.shape)}"
            )
        if input_values.shape[:2] != instance_mask.shape:
            raise ValueError(
                "input_values and instance_mask must agree on bag shape; "
                f"got {tuple(input_values.shape[:2])} and {tuple(instance_mask.shape)}"
            )

        valid_mask = instance_mask.to(dtype=torch.bool)
        if not bool(valid_mask.any()):
            raise ValueError("Each batch must contain at least one valid instance")

        flat_inputs = input_values[valid_mask]
        encoder_output = self.encoder(input_values=flat_inputs, return_dict=True)
        flat_embeddings = self._pool_encoder_output(encoder_output.last_hidden_state)
        flat_embeddings = self.instance_head(flat_embeddings)
        instance_embeddings = self._scatter_embeddings(flat_embeddings, valid_mask)
        instance_logits = self.classifier(instance_embeddings)
        if instance_logits.ndim == 3 and instance_logits.shape[-1] == 1:
            instance_logits = instance_logits.squeeze(-1)
        instance_probabilities = self._logits_to_probabilities(instance_logits)

        attention_weights: Tensor | None = None
        if self.cfg.mil.type == "gated_attention":
            bag_embedding, attention_weights = self.mil_pool(
                instance_embeddings, valid_mask
            )
            bag_logits = self.classifier(bag_embedding)
            if bag_logits.ndim == 2 and bag_logits.shape[-1] == 1:
                bag_logits = bag_logits.squeeze(-1)
            bag_probabilities = self._logits_to_probabilities(bag_logits)
        else:
            bag_probabilities = self.mil_pool(instance_probabilities, valid_mask)
            bag_logits = self._probabilities_to_logits(bag_probabilities)
            bag_embedding = self._masked_mean(instance_embeddings, valid_mask)

        return AstMilOutput(
            bag_logits=bag_logits,
            bag_probabilities=bag_probabilities,
            instance_logits=instance_logits,
            instance_probabilities=instance_probabilities,
            instance_embeddings=instance_embeddings,
            bag_embedding=bag_embedding,
            instance_mask=valid_mask,
            attention_weights=attention_weights,
        )
