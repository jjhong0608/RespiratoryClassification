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
)

from conftest import small_patch_branches


def _small_architecture(
    *,
    latent_query_count: int = 4,
    rdt_steps: int = 3,
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
        latent_query_count=latent_query_count,
        rdt_steps=rdt_steps,
        patch_branches=small_patch_branches(),
    )


def _small_model_config(*, num_classes: int) -> MultiScaleRdtAstModelConfig:
    return MultiScaleRdtAstModelConfig(
        encoder=MultiScaleRdtEncoderConfig(
            feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
            adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
            architecture=_small_architecture(),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
            pooling="latent_mean",
        ),
        num_classes=num_classes,
    )


def test_binary_model_returns_single_logit_per_clip() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2,)
    assert output.pooled_embedding.shape == (2, 32)


def test_multiclass_model_returns_class_logits_per_clip() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=3))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2, 3)
    assert output.pooled_embedding.shape == (2, 32)


def test_invalid_input_dims_raise_clear_value_error() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))

    with pytest.raises(ValueError, match="input_values must have shape"):
        model(torch.randn(2, 32))

    with pytest.raises(
        ValueError, match="feature dims do not match model feature dims"
    ):
        model(torch.randn(2, 31, 32))


def test_default_branch_geometry_matches_expected_token_counts() -> None:
    model = MultiScaleRdtAstModel(
        MultiScaleRdtAstModelConfig(
            encoder=MultiScaleRdtEncoderConfig(
                feature_dims=AstFeatureDims(num_mel_bins=128, max_length=1024),
                adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
                architecture=MultiScaleRdtArchitectureConfig(
                    hidden_size=48,
                    num_attention_heads=6,
                ),
            ),
            classifier=ClassifierConfig(pooling="latent_mean"),
            num_classes=2,
        )
    )

    assert model.encoder.branch_token_counts == (1016, 1020, 1022, 1023)
    assert model.encoder.total_token_count == 4081


def test_latent_query_pooler_returns_expected_shape() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    evidence_memory = model.encoder(input_values.unsqueeze(1))
    latent = model.latent_pooler(evidence_memory)

    assert latent.shape == (2, 4, 32)


def test_rdt_block_is_reused_for_all_recurrent_steps() -> None:
    model = MultiScaleRdtAstModel(_small_model_config(num_classes=2))
    input_values = torch.randn(1, 32, 32)

    with mock.patch.object(
        model.rdt_block,
        "forward",
        wraps=model.rdt_block.forward,
    ) as wrapped:
        output = model(input_values)

    assert wrapped.call_count == 3
    assert output.pooled_embedding.shape == (1, 32)
