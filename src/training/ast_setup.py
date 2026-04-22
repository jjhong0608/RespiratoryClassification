from __future__ import annotations

from dataclasses import dataclass

from torch.optim import AdamW

from src.models.model import (
    AstFeatureDims,
    ClassifierConfig,
    EncoderAdaptationConfig,
    MultiScaleRdtArchitectureConfig,
    MultiScaleRdtAstModel,
    MultiScaleRdtAstModelConfig,
    MultiScaleRdtEncoderConfig,
)
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
class ModelArchitectureSummary:
    encoder_type: str
    hidden_size: int
    num_attention_heads: int
    branch_token_counts: tuple[int, ...]
    total_token_count: int
    latent_query_count: int
    rdt_steps: int


def build_ast_model(
    cfg: RunModelConfig,
    *,
    num_mel_bins: int,
    max_length: int,
    num_classes: int,
) -> MultiScaleRdtAstModel:
    model_cfg = MultiScaleRdtAstModelConfig(
        encoder=MultiScaleRdtEncoderConfig(
            type=cfg.encoder.type,
            feature_dims=AstFeatureDims(
                num_mel_bins=num_mel_bins,
                max_length=max_length,
            ),
            adaptation=EncoderAdaptationConfig(
                mode=cfg.encoder.adaptation.mode,
                num_layers=cfg.encoder.adaptation.num_layers,
            ),
            architecture=MultiScaleRdtArchitectureConfig(
                hidden_size=cfg.encoder.architecture.hidden_size,
                num_attention_heads=cfg.encoder.architecture.num_attention_heads,
                mlp_ratio=cfg.encoder.architecture.mlp_ratio,
                hidden_dropout_prob=cfg.encoder.architecture.hidden_dropout_prob,
                attention_probs_dropout_prob=(
                    cfg.encoder.architecture.attention_probs_dropout_prob
                ),
                layer_norm_eps=cfg.encoder.architecture.layer_norm_eps,
                shared_stem_depth=cfg.encoder.architecture.shared_stem_depth,
                adapter_depth=cfg.encoder.architecture.adapter_depth,
                latent_query_count=cfg.encoder.architecture.latent_query_count,
                rdt_steps=cfg.encoder.architecture.rdt_steps,
                patch_branches=cfg.encoder.architecture.patch_branches,
            ),
        ),
        classifier=ClassifierConfig(
            type=cfg.classifier.type,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
            pooling=cfg.classifier.pooling,
        ),
        num_classes=num_classes,
    )
    return MultiScaleRdtAstModel(model_cfg)


def inspect_pretrained_encoder(cfg: RunModelConfig) -> object | None:
    del cfg
    return None


def summarize_model_architecture(
    model: MultiScaleRdtAstModel,
) -> ModelArchitectureSummary:
    architecture = model.cfg.encoder.architecture
    return ModelArchitectureSummary(
        encoder_type=model.cfg.encoder.type,
        hidden_size=architecture.hidden_size,
        num_attention_heads=architecture.num_attention_heads,
        branch_token_counts=model.encoder.branch_token_counts,
        total_token_count=model.encoder.total_token_count,
        latent_query_count=architecture.latent_query_count,
        rdt_steps=architecture.rdt_steps,
    )


def apply_encoder_adaptation(
    model: MultiScaleRdtAstModel,
    cfg: EncoderAdaptationConfig,
) -> EncoderAdaptationSummary:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if cfg.mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif cfg.mode == "frozen":
        for module in (model.latent_pooler, model.rdt_block, model.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError(
            "Unsupported adaptation mode for multiscale_rdt_ast: "
            f"{cfg.mode}. Use 'full' or 'frozen'."
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    return EncoderAdaptationSummary(
        mode=cfg.mode,
        num_layers=cfg.num_layers,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
    )


def build_grouped_optimizer(
    model: MultiScaleRdtAstModel,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[AdamW, OptimizerGroupSummary]:
    encoder_params = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head_modules = (model.latent_pooler, model.rdt_block, model.classifier)
    head_params = [
        parameter
        for module in head_modules
        for parameter in module.parameters()
        if parameter.requires_grad
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
