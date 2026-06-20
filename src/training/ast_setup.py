from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from torch.optim import AdamW
from transformers import ASTConfig

from src.models.model import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFeatureDims,
    AstModelConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    RespiratoryAstModel,
)
from src.utils.config import AstEncoderConfig as RunAstEncoderConfig
from src.utils.config import ModelConfig as RunModelConfig


@dataclass(frozen=True)
class EncoderAdaptationSummary:
    mode: str
    num_layers: int
    trainable_parameters: int
    frozen_parameters: int


@dataclass(frozen=True)
class OptimizerGroupSummary:
    encoder_lr: float
    head_lr: float
    encoder_trainable_parameters: int
    head_trainable_parameters: int
    param_group_count: int


@dataclass(frozen=True)
class AstPretrainedInfo:
    source: str
    name_or_path: str
    num_mel_bins: int
    max_length: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int


def _require_ast_encoder(cfg: RunModelConfig) -> RunAstEncoderConfig:
    if not isinstance(cfg.encoder, RunAstEncoderConfig):
        raise ValueError("AST setup requires model.encoder.type='ast'")
    return cfg.encoder


def build_ast_model(
    cfg: RunModelConfig,
    *,
    num_mel_bins: int,
    max_length: int,
    num_classes: int,
) -> RespiratoryAstModel:
    encoder = _require_ast_encoder(cfg)
    if cfg.classifier.type not in {"linear", "mlp"}:
        raise ValueError("AST setup supports only linear or mlp classifier heads")
    classifier_type = cast(Literal["linear", "mlp"], cfg.classifier.type)
    model_cfg = AstModelConfig(
        encoder=AstEncoderConfig(
            type=encoder.type,
            pretrained_name_or_path=encoder.pretrained_name_or_path,
            cache_dir=encoder.cache_dir,
            feature_dims=AstFeatureDims(
                num_mel_bins=num_mel_bins,
                max_length=max_length,
            ),
            adaptation=EncoderAdaptationConfig(
                mode=encoder.adaptation.mode,
                num_layers=encoder.adaptation.num_layers,
            ),
            architecture=AstArchitectureConfig(
                hidden_size=encoder.architecture.hidden_size,
                num_hidden_layers=encoder.architecture.num_hidden_layers,
                num_attention_heads=encoder.architecture.num_attention_heads,
                intermediate_size=encoder.architecture.intermediate_size,
                hidden_dropout_prob=encoder.architecture.hidden_dropout_prob,
                attention_probs_dropout_prob=(
                    encoder.architecture.attention_probs_dropout_prob
                ),
                frequency_stride=encoder.architecture.frequency_stride,
                time_stride=encoder.architecture.time_stride,
                patch_size=encoder.architecture.patch_size,
                qkv_bias=encoder.architecture.qkv_bias,
                layer_norm_eps=encoder.architecture.layer_norm_eps,
                initializer_range=encoder.architecture.initializer_range,
            ),
        ),
        classifier=ClassifierConfig(
            type=classifier_type,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
            pooling=cfg.classifier.pooling,
        ),
        num_classes=num_classes,
    )
    return RespiratoryAstModel(model_cfg)


def inspect_pretrained_encoder(cfg: RunModelConfig) -> AstPretrainedInfo | None:
    encoder = _require_ast_encoder(cfg)
    pretrained_name_or_path = encoder.pretrained_name_or_path
    if pretrained_name_or_path is None:
        return None
    pretrained_cfg = ASTConfig.from_pretrained(
        pretrained_name_or_path,
        cache_dir=encoder.cache_dir,
    )
    return AstPretrainedInfo(
        source="huggingface_pretrained",
        name_or_path=pretrained_name_or_path,
        num_mel_bins=int(pretrained_cfg.num_mel_bins),
        max_length=int(pretrained_cfg.max_length),
        hidden_size=int(pretrained_cfg.hidden_size),
        num_hidden_layers=int(pretrained_cfg.num_hidden_layers),
        num_attention_heads=int(pretrained_cfg.num_attention_heads),
    )


def apply_encoder_adaptation(
    model: RespiratoryAstModel,
    cfg: EncoderAdaptationConfig,
) -> EncoderAdaptationSummary:
    encoder = model.encoder

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    if cfg.mode == "full":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    elif cfg.mode == "partial":
        for layer in encoder.encoder.layer[-cfg.num_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in encoder.layernorm.parameters():
            parameter.requires_grad = True
    elif cfg.mode != "frozen":
        raise ValueError(f"Unsupported adaptation mode: {cfg.mode}")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    return EncoderAdaptationSummary(
        mode=cfg.mode,
        num_layers=cfg.num_layers,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
    )


def build_grouped_optimizer(
    model: RespiratoryAstModel,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[AdamW, OptimizerGroupSummary]:
    encoder_params = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    encoder_param_ids = {id(parameter) for parameter in model.encoder.parameters()}
    head_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_param_ids
    ]
    param_groups = []
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters available for optimizer creation")
    optimizer = AdamW(param_groups)
    summary = OptimizerGroupSummary(
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        encoder_trainable_parameters=sum(p.numel() for p in encoder_params),
        head_trainable_parameters=sum(p.numel() for p in head_params),
        param_group_count=len(param_groups),
    )
    return optimizer, summary
