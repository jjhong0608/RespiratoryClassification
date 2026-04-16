from __future__ import annotations

import torch
from src.models.mil import LinearSoftmaxMil, masked_softmax
from src.models.model import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFeatureDims,
    AstMilModelConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    GatedAttentionMilConfig,
    InstanceHeadConfig,
    LinearSoftmaxMilConfig,
    MilConfig,
    RespiratoryAstMilModel,
)


def _model_config(*, num_classes: int, mil_type: str) -> AstMilModelConfig:
    return AstMilModelConfig(
        encoder=AstEncoderConfig(
            feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
            pretrained_name_or_path=None,
            pooling="cls",
            adaptation=EncoderAdaptationConfig(mode="partial", num_layers=1),
            architecture=AstArchitectureConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
            ),
        ),
        instance_head=InstanceHeadConfig(
            projection_dim=24,
            dropout=0.1,
            normalize=True,
        ),
        mil=MilConfig(
            type=mil_type,  # type: ignore[arg-type]
            gated_attention=GatedAttentionMilConfig(attention_dim=16, dropout=0.1),
            linear_softmax=LinearSoftmaxMilConfig(eps=1e-6),
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=24,
            dropout=0.1,
        ),
        num_classes=num_classes,
    )


def test_gated_attention_binary_model_returns_bag_outputs() -> None:
    model = RespiratoryAstMilModel(
        _model_config(num_classes=2, mil_type="gated_attention")
    )
    input_values = torch.randn(2, 3, 32, 32)
    instance_mask = torch.tensor([[True, True, False], [True, True, True]])

    output = model(input_values, instance_mask)

    assert output.bag_logits.shape == (2,)
    assert output.bag_probabilities.shape == (2,)
    assert output.instance_logits.shape == (2, 3)
    assert output.instance_embeddings.shape == (2, 3, 24)
    assert output.attention_weights is not None
    assert torch.allclose(output.attention_weights[0, 2], torch.tensor(0.0))
    assert torch.allclose(output.attention_weights[0, :2].sum(), torch.tensor(1.0))


def test_linear_softmax_multiclass_model_returns_class_logits_per_bag() -> None:
    model = RespiratoryAstMilModel(
        _model_config(num_classes=3, mil_type="linear_softmax")
    )
    input_values = torch.randn(2, 4, 32, 32)
    instance_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    output = model(input_values, instance_mask)

    assert output.bag_logits.shape == (2, 3)
    assert output.bag_probabilities.shape == (2, 3)
    assert torch.allclose(
        output.bag_probabilities.sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert output.instance_logits.shape == (2, 4, 3)
    assert output.attention_weights is None


def test_masked_softmax_ignores_invalid_instances() -> None:
    scores = torch.tensor([[1.0, -5.0, 0.5]], dtype=torch.float32)
    mask = torch.tensor([[True, False, True]])

    weights = masked_softmax(scores, mask, dim=1)

    assert torch.allclose(weights[0, 1], torch.tensor(0.0))
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))


def test_linear_softmax_pooling_is_finite_for_small_probabilities() -> None:
    mil = LinearSoftmaxMil(eps=1e-6)
    instance_probabilities = torch.full((1, 3, 2), 1e-8, dtype=torch.float32)
    instance_mask = torch.tensor([[True, True, True]])

    pooled = mil(instance_probabilities, instance_mask)

    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled.sum(dim=-1), torch.ones(1), atol=1e-5)
