from __future__ import annotations

from dataclasses import asdict
from typing import Literal, cast

from src.models.model import BranchAwareGatedEvidencePooling
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
    EvidencePoolingConfig,
    MilConfig,
    ModelConfig,
    ModelEncoderConfig,
    MultiScaleRdtArchitectureConfig,
    RdtConfig,
)
from torch import nn

from conftest import small_patch_branches


def _run_model_config(
    *,
    rdt_enabled: bool = True,
    evidence_score_source: Literal[
        "attention_weight", "attention_logit", "instance_logit"
    ] = "attention_weight",
    exclude_branches_from_evidence: tuple[int, ...] = (),
    attention_temperature: float = 1.0,
    evidence_pooling: EvidencePoolingConfig | None = None,
) -> ModelConfig:
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
                patch_branches=small_patch_branches(),
                rdt=RdtConfig(
                    enabled=rdt_enabled,
                    steps=2,
                    top_tokens_per_branch=2,
                    gated_residual=True,
                    layerscale_init=0.01,
                    evidence_score_source=evidence_score_source,
                    exclude_branches_from_evidence=exclude_branches_from_evidence,
                ),
                mil=MilConfig(attention_temperature=attention_temperature),
                evidence_pooling=evidence_pooling or EvidencePoolingConfig(),
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


def test_frozen_adaptation_freezes_encoder_side_and_keeps_head_trainable() -> None:
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
    assert all(
        not parameter.requires_grad
        for module in model.encoder_side_modules()
        for parameter in module.parameters()
    )
    assert all(
        parameter.requires_grad
        for module in model.head_side_modules()
        for parameter in module.parameters()
    )


def test_frozen_adaptation_keeps_branch_gated_pooler_trainable() -> None:
    model = build_ast_model(
        _run_model_config(
            evidence_pooling=EvidencePoolingConfig(
                type="branch_gated",
                dropout=0.0,
            )
        ),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    apply_encoder_adaptation(
        model,
        EncoderAdaptationConfig(mode="frozen", num_layers=0),
    )

    pooler = cast(BranchAwareGatedEvidencePooling, model.evidence_pooler)

    assert any(parameter.requires_grad for parameter in pooler.parameters())


def test_grouped_optimizer_uses_event_mil_encoder_and_head_groups() -> None:
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
    assert optimizer.param_groups[1]["name"] == "head"
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-4
    assert summary.encoder_trainable_parameters > 0
    assert summary.head_trainable_parameters > 0

    encoder_param_ids = {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    }
    head_param_ids = {
        id(parameter) for parameter in optimizer.param_groups[1]["params"]
    }
    frequency_score = cast(nn.Linear, model.encoder.frequency_poolers[0].score)
    branch_logit_proj = cast(nn.Linear, model.branch_mil_heads[0].logit_proj)
    fusion_layer = model.fusion_projector[0]
    assert isinstance(fusion_layer, nn.Linear)
    assert id(frequency_score.weight) in encoder_param_ids
    assert id(branch_logit_proj.weight) in encoder_param_ids
    assert id(fusion_layer.weight) in head_param_ids
    assert id(fusion_layer.weight) not in encoder_param_ids


def test_grouped_optimizer_includes_branch_gated_pooler_in_head_group() -> None:
    model = build_ast_model(
        _run_model_config(
            evidence_pooling=EvidencePoolingConfig(
                type="branch_gated",
                dropout=0.0,
            )
        ),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )
    apply_encoder_adaptation(model, model.cfg.encoder.adaptation)

    optimizer, _ = build_grouped_optimizer(
        model,
        encoder_lr=1e-5,
        head_lr=1e-4,
        weight_decay=0.01,
    )

    head_param_ids = {
        id(parameter) for parameter in optimizer.param_groups[1]["params"]
    }
    pooler = cast(BranchAwareGatedEvidencePooling, model.evidence_pooler)
    gate_linear = cast(nn.Linear, pooler.gate[1])

    assert id(gate_linear.weight) in head_param_ids


def test_grouped_optimizer_still_has_head_group_when_rdt_disabled() -> None:
    model = build_ast_model(
        _run_model_config(rdt_enabled=False),
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
    assert optimizer.param_groups[1]["name"] == "head"
    assert summary.head_trainable_parameters > 0
    assert model.rdt_block is None


def test_inspect_pretrained_encoder_returns_none_for_clean_break_model() -> None:
    assert inspect_pretrained_encoder(_run_model_config()) is None


def test_build_ast_model_preserves_d_e_f_architecture_knobs() -> None:
    model = build_ast_model(
        _run_model_config(
            evidence_score_source="attention_logit",
            exclude_branches_from_evidence=(3,),
            attention_temperature=0.5,
        ),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    architecture = model.cfg.encoder.architecture

    assert architecture.rdt.evidence_score_source == "attention_logit"
    assert architecture.rdt.exclude_branches_from_evidence == (3,)
    assert architecture.mil.attention_temperature == 0.5
    assert architecture.evidence_pooling.type == "mean"
    assert model.branch_mil_heads[0].attention_temperature == 0.5


def test_architecture_summary_reports_event_geometry() -> None:
    model = build_ast_model(
        _run_model_config(),
        num_mel_bins=32,
        max_length=32,
        num_classes=2,
    )

    summary = summarize_model_architecture(model)

    assert summary.encoder_type == "multiscale_rdt_ast"
    assert summary.branch_token_counts == (28, 30, 31, 24)
    assert summary.branch_time_lengths == (7, 15, 31, 3)
    assert summary.total_patch_token_count == 113
    assert summary.total_temporal_length == 56
    assert summary.rdt_enabled is True
    assert summary.rdt_steps == 2
    assert summary.rdt_top_tokens_per_branch == 2
    assert summary.rdt_evidence_score_source == "attention_weight"
    assert summary.rdt_exclude_branches_from_evidence == ()
    assert summary.mil_attention_temperature == 1.0
    assert summary.evidence_pooling_type == "mean"


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
    assert parsed.encoder.architecture.rdt.enabled is True
    assert parsed.encoder.architecture.rdt.steps == 2
    assert parsed.encoder.architecture.rdt.top_tokens_per_branch == 2
    assert parsed.encoder.architecture.rdt.evidence_score_source == "attention_weight"
    assert parsed.encoder.architecture.mil.attention_temperature == 1.0
    assert parsed.encoder.architecture.evidence_pooling.type == "mean"
    assert parsed.num_classes == 3


def test_parse_model_cfg_rejects_legacy_architecture_keys() -> None:
    payload = {
        "encoder": {
            "type": "multiscale_rdt_ast",
            "feature_dims": {"num_mel_bins": 32, "max_length": 32},
            "adaptation": {"mode": "full", "num_layers": 0},
            "architecture": {
                "hidden_size": 32,
                "num_attention_heads": 4,
                "latent_query_count": 4,
            },
        },
        "classifier": {
            "type": "linear",
            "hidden_dim": 24,
            "dropout": 0.1,
            "pooling": "latent_mean",
        },
        "num_classes": 2,
    }

    try:
        parse_model_cfg(payload)
    except ValueError as exc:
        assert "event-MIL architecture" in str(exc)
    else:
        raise AssertionError("Expected parse_model_cfg to reject legacy keys")
