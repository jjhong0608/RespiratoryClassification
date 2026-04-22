from __future__ import annotations

from dataclasses import asdict

from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
    inspect_pretrained_encoder,
    summarize_model_architecture,
)
from src.utils.checkpoint import parse_model_cfg
from src.utils.config import (
    ClassifierConfig,
    EncoderAdaptationConfig,
    ModelConfig,
    ModelEncoderConfig,
    MultiScaleRdtArchitectureConfig,
)

from conftest import small_patch_branches


def _run_model_config() -> ModelConfig:
    return ModelConfig(
        encoder=ModelEncoderConfig(
            adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
            architecture=MultiScaleRdtArchitectureConfig(
                hidden_size=32,
                num_attention_heads=4,
                mlp_ratio=2.0,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                layer_norm_eps=1e-6,
                shared_stem_depth=1,
                adapter_depth=1,
                latent_query_count=4,
                rdt_steps=2,
                patch_branches=small_patch_branches(),
            ),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
            pooling="latent_mean",
        ),
    )


def test_full_adaptation_enables_all_parameters() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    apply_encoder_adaptation(
        model,
        EncoderAdaptationConfig(mode="full", num_layers=0),
    )

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_frozen_adaptation_freezes_only_encoder_parameters() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    summary = apply_encoder_adaptation(
        model,
        EncoderAdaptationConfig(mode="frozen", num_layers=0),
    )

    assert summary.mode == "frozen"
    assert summary.trainable_parameters == 0
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        parameter.requires_grad for parameter in model.latent_pooler.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.rdt_block.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_grouped_optimizer_uses_encoder_and_head_learning_rates() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )
    apply_encoder_adaptation(model, model.cfg.encoder.adaptation)

    optimizer, summary = build_grouped_optimizer(
        model,
        encoder_lr=1e-5,
        head_lr=1e-4,
        weight_decay=0.01,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["name"] == "encoder"
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["name"] == "head"
    assert optimizer.param_groups[1]["lr"] == 1e-4
    assert summary.encoder_trainable_parameters > 0
    assert summary.head_trainable_parameters > 0


def test_inspect_pretrained_encoder_returns_none_for_clean_break_model() -> None:
    assert inspect_pretrained_encoder(_run_model_config()) is None


def test_architecture_summary_reports_geometry() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    summary = summarize_model_architecture(model)

    assert summary.encoder_type == "multiscale_rdt_ast"
    assert summary.branch_token_counts == (28, 30, 31, 24)
    assert summary.total_token_count == 113
    assert summary.latent_query_count == 4
    assert summary.rdt_steps == 2


def test_parse_model_cfg_reconstructs_checkpoint_config() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=3,
    )

    parsed = parse_model_cfg(asdict(model.cfg))

    assert parsed.encoder.type == "multiscale_rdt_ast"
    assert parsed.encoder.feature_dims.max_length == 32
    assert parsed.encoder.architecture.patch_branches == small_patch_branches()
    assert parsed.num_classes == 3
