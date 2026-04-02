from __future__ import annotations

import torch
from src.models.model import MILModelConfig, RespiratoryMILModel
from src.models.whisper_encoder import WhisperEncoderDims


def test_mil_model_returns_bag_and_instance_outputs() -> None:
    model = RespiratoryMILModel(
        MILModelConfig(
            encoder=WhisperEncoderDims(
                n_mels=8,
                n_audio_ctx=5,
                n_audio_state=16,
                n_audio_head=4,
                n_audio_layer=2,
            ),
            instance_head_type="mlp",
            instance_hidden_dim=12,
            instance_dropout=0.1,
            aggregator="topk",
            topk_k=2,
            attention_hidden_dim=8,
            attention_dropout=0.0,
            attention_gated=True,
            logsumexp_temperature=1.0,
            softmax_weighted_temperature=1.0,
            noisy_or_clamp_eps=1e-6,
        )
    )
    segments = torch.randn(2, 3, 8, 10)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    output = model(segments, mask)

    assert output.bag_logits.shape == (2,)
    assert output.instance_logits.shape == (2, 3)
    assert output.topk_indices is not None
    assert output.topk_indices.shape == (2, 2)


def test_attention_mil_model_emits_attention_weights() -> None:
    model = RespiratoryMILModel(
        MILModelConfig(
            encoder=WhisperEncoderDims(
                n_mels=8,
                n_audio_ctx=5,
                n_audio_state=16,
                n_audio_head=4,
                n_audio_layer=2,
            ),
            instance_head_type="linear",
            instance_hidden_dim=16,
            instance_dropout=0.0,
            aggregator="attention",
            topk_k=1,
            attention_hidden_dim=8,
            attention_dropout=0.0,
            attention_gated=False,
            logsumexp_temperature=1.0,
            softmax_weighted_temperature=1.0,
            noisy_or_clamp_eps=1e-6,
        )
    )
    segments = torch.randn(1, 4, 8, 10)
    mask = torch.tensor([[True, True, False, False]])

    output = model(segments, mask)

    assert output.attention_weights is not None
    assert torch.isclose(
        output.attention_weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5
    )
    assert torch.allclose(output.attention_weights[0, 2:], torch.zeros(2), atol=1e-6)
