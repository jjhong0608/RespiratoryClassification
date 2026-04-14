from __future__ import annotations

import torch
from src.models.model import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFeatureDims,
    AstModelConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    RespiratoryAstModel,
)


def _model_config(*, num_classes: int) -> AstModelConfig:
    return AstModelConfig(
        encoder=AstEncoderConfig(
            feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
            pretrained_name_or_path=None,
            adaptation=EncoderAdaptationConfig(mode="partial", num_layers=1),
            architecture=AstArchitectureConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
            ),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
            pooling="cls",
        ),
        num_classes=num_classes,
    )


def test_binary_ast_model_returns_single_logit_per_clip() -> None:
    model = RespiratoryAstModel(_model_config(num_classes=2))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2,)
    assert output.pooled_embedding.shape == (2, 32)


def test_multiclass_ast_model_returns_class_logits_per_clip() -> None:
    model = RespiratoryAstModel(_model_config(num_classes=3))
    input_values = torch.randn(2, 32, 32)

    output = model(input_values)

    assert output.logits.shape == (2, 3)
    assert output.pooled_embedding.shape == (2, 32)
