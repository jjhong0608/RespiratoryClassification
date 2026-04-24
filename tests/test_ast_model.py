from __future__ import annotations

from unittest import mock

import pytest
import torch
from src.models.model import (
    AstFeatureDims,
    ClassifierConfig,
    EncoderAdaptationConfig,
    MultiScaleRdtArchitectureConfig,
    MultiScaleRdtAstModel,
    MultiScaleRdtAstModelConfig,
    MultiScaleRdtEncoderConfig,
    PatchBranchConfig,
    RdtConfig,
)

from conftest import small_patch_branches


def _small_architecture(
    *,
    rdt_enabled: bool = True,
    rdt_steps: int = 3,
    top_tokens_per_branch: int = 2,
    patch_branches: tuple[PatchBranchConfig, ...] | None = None,
) -> MultiScaleRdtArchitectureConfig:
    return MultiScaleRdtArchitectureConfig(
        hidden_size=32,
        num_attention_heads=4,
        mlp_ratio=2.0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        shared_stem_depth=1,
        adapter_depth=1,
        patch_branches=patch_branches or small_patch_branches(),
        rdt=RdtConfig(
            enabled=rdt_enabled,
            steps=rdt_steps,
            top_tokens_per_branch=top_tokens_per_branch,
            gated_residual=True,
            layerscale_init=0.01,
        ),
    )


def _small_model_config(
    *,
    num_classes: int,
    rdt_enabled: bool = True,
    rdt_steps: int = 3,
    top_tokens_per_branch: int = 2,
    patch_branches: tuple[PatchBranchConfig, ...] | None = None,
) -> MultiScaleRdtAstModelConfig:
    return MultiScaleRdtAstModelConfig(
        encoder=MultiScaleRdtEncoderConfig(
            feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
            adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
            architecture=_small_architecture(
                rdt_enabled=rdt_enabled,
                rdt_steps=rdt_steps,
                top_tokens_per_branch=top_tokens_per_branch,
                patch_branches=patch_branches,
            ),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
            pooling="latent_mean",
        ),
        num_classes=num_classes,
    )


def test_binary_model_returns_expected_event_mil_outputs() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2,)
    assert output.pooled_embedding.shape == (2, 32)
    assert output.branch_logits is not None
    assert output.branch_logits.shape == (2, 4)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (2, 8, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (2, 8)
    assert output.selected_evidence_scores is not None
    assert output.selected_evidence_scores.shape == (2, 8)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids.shape == (2, 8)
    assert output.selected_evidence_branch_ids[0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert output.branch_attention_weights is not None
    assert tuple(attn.shape[1] for attn in output.branch_attention_weights) == (
        7,
        15,
        31,
        3,
    )


def test_multiclass_model_returns_expected_event_mil_outputs() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=3))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2, 3)
    assert output.pooled_embedding.shape == (2, 32)
    assert output.branch_logits is not None
    assert output.branch_logits.shape == (2, 4, 3)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (2, 8, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (2, 8)


def test_invalid_input_dims_raise_clear_value_error() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))

    with pytest.raises(ValueError, match="input_values must have shape"):
        model(torch.randn(2, 32))

    with pytest.raises(
        ValueError, match="feature dims do not match model feature dims"
    ):
        model(torch.randn(2, 31, 32))


def test_default_branch_geometry_matches_expected_token_and_time_counts() -> None:
    model = MultiScaleRdtAstModel(
        MultiScaleRdtAstModelConfig(
            encoder=MultiScaleRdtEncoderConfig(
                feature_dims=AstFeatureDims(num_mel_bins=128, max_length=1024),
                adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
                architecture=MultiScaleRdtArchitectureConfig(
                    hidden_size=48,
                    num_attention_heads=4,
                ),
            ),
            classifier=ClassifierConfig(pooling="latent_mean"),
            num_classes=2,
        )
    )

    assert model.encoder.branch_token_counts == (1016, 1020, 1022, 1023)
    assert model.encoder.branch_time_lengths == (127, 255, 511, 1023)
    assert model.encoder.total_token_count == 4081
    assert model.encoder.total_temporal_length == 1916


def test_encoder_returns_temporal_event_tokens_and_context() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    encoder_output = model.encoder(input_values.unsqueeze(1))

    assert tuple(tokens.shape for tokens in encoder_output.branch_event_tokens) == (
        (2, 7, 32),
        (2, 15, 32),
        (2, 31, 32),
        (2, 3, 32),
    )
    assert encoder_output.context_tokens.shape == (2, 56, 32)


def test_model_runs_without_rdt_when_disabled() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2, rdt_enabled=False))
    output = model(torch.randn(1, 32, 32))

    assert model.rdt_block is None
    assert output.logits.shape == (1,)
    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 8, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 8)


def test_three_scale_top2_forward_uses_dynamic_selected_evidence_length() -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            patch_branches=small_patch_branches()[:3],
        )
    )

    assert model.encoder.total_temporal_length == 53
    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 6, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 6)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids[0].tolist() == [0, 0, 1, 1, 2, 2]


def test_four_scale_top4_forward_uses_dynamic_selected_evidence_length() -> None:
    top4_patch_branches = (
        PatchBranchConfig(patch_size=(8, 8), stride=(4, 8)),
        PatchBranchConfig(patch_size=(4, 16), stride=(2, 16)),
        PatchBranchConfig(patch_size=(2, 32), stride=(1, 32)),
        PatchBranchConfig(patch_size=(8, 4), stride=(4, 4)),
    )
    model = MultiScaleRdtAstModel(
        _small_model_config(
            num_classes=2,
            top_tokens_per_branch=4,
            patch_branches=top4_patch_branches,
        )
    )

    output = model(torch.randn(1, 32, 32))

    assert output.selected_evidence_tokens is not None
    assert output.selected_evidence_tokens.shape == (1, 16, 32)
    assert output.selected_evidence_indices is not None
    assert output.selected_evidence_indices.shape == (1, 16)
    assert output.selected_evidence_branch_ids is not None
    assert output.selected_evidence_branch_ids[0].tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]


def test_rdt_block_is_reused_for_all_configured_steps() -> None:
    model = MultiScaleRdtAstModel(
        _small_model_config(num_classes=2, rdt_enabled=True, rdt_steps=2)
    )
    input_values = torch.randn(1, 32, 32)
    assert model.rdt_block is not None

    with mock.patch.object(
        model.rdt_block,
        "forward",
        wraps=model.rdt_block.forward,
    ) as wrapped:
        output = model(input_values)

    assert wrapped.call_count == 2
    assert output.pooled_embedding.shape == (1, 32)
