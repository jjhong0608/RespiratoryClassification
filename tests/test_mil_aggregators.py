from __future__ import annotations

import torch
from src.models.mil_aggregators import (
    AttentionMILAggregator,
    AttentionMILConfig,
    MILAggregatorOutput,
    NoisyOrMILAggregator,
    NoisyOrMILConfig,
    TopKMILAggregator,
    TopKMILConfig,
)


def test_topk_aggregator_returns_mean_of_selected_scores() -> None:
    aggregator = TopKMILAggregator(TopKMILConfig(k=2))
    logits = torch.tensor([[0.1, 0.9, 0.4, -1.0]], dtype=torch.float32)
    embeddings = torch.zeros(1, 4, 3)
    mask = torch.tensor([[True, True, True, False]])

    output = aggregator(
        instance_logits=logits, instance_embeddings=embeddings, instance_mask=mask
    )

    assert torch.allclose(output.bag_logits, torch.tensor([0.65]))
    assert output.topk_indices is not None
    assert output.topk_indices.tolist() == [[1, 2]]


def test_attention_aggregator_masks_invalid_instances() -> None:
    aggregator = AttentionMILAggregator(
        AttentionMILConfig(input_dim=4, hidden_dim=8, dropout=0.0, gated=True)
    )
    logits = torch.tensor([[0.3, 0.8, -0.2]], dtype=torch.float32)
    embeddings = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False]])

    output = aggregator(
        instance_logits=logits, instance_embeddings=embeddings, instance_mask=mask
    )

    assert output.attention_weights is not None
    weights = output.attention_weights
    assert torch.isclose(weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(weights[0, 2], torch.tensor(0.0), atol=1e-6)


def test_noisy_or_aggregator_returns_finite_bag_logit() -> None:
    aggregator = NoisyOrMILAggregator(NoisyOrMILConfig(clamp_eps=1e-6))
    logits = torch.tensor([[8.0, -4.0, 2.0]], dtype=torch.float32)
    embeddings = torch.zeros(1, 3, 2)
    mask = torch.tensor([[True, True, True]])

    output = aggregator(
        instance_logits=logits, instance_embeddings=embeddings, instance_mask=mask
    )

    assert isinstance(output, MILAggregatorOutput)
    assert torch.isfinite(output.bag_logits).all()
