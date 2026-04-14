from __future__ import annotations

import torch
from src.models.model import (
    InstanceHeadConfig,
    InterAttentionConfig,
    MILConfig,
    MILModelConfig,
    RespiratoryMILModel,
    SegmentEncoderAdaptationConfig,
    SegmentEncoderConfig,
    TopKConfig,
)
from src.models.segment_encoder import SegmentEncoderPoolingConfig
from src.models.whisper_encoder import WhisperEncoderDims


def _segment_encoder_config() -> SegmentEncoderConfig:
    return SegmentEncoderConfig(
        dims=WhisperEncoderDims(
            n_mels=8,
            n_audio_ctx=5,
            n_audio_state=16,
            n_audio_head=4,
            n_audio_layer=2,
        ),
        pooling=SegmentEncoderPoolingConfig(
            type="attention",
            hidden_dim=8,
            dropout=0.0,
            gated=True,
        ),
        adaptation=SegmentEncoderAdaptationConfig(mode="partial", num_layers=1),
    )


def test_mil_model_returns_hierarchical_outputs_for_topk() -> None:
    model = RespiratoryMILModel(
        MILModelConfig(
            segment_encoder=_segment_encoder_config(),
            instance_head=InstanceHeadConfig(
                type="mlp",
                hidden_dim=12,
                dropout=0.1,
            ),
            mil=MILConfig(
                aggregator="topk",
                topk=TopKConfig(k=2),
                attention=InterAttentionConfig(
                    hidden_dim=8,
                    dropout=0.0,
                    gated=True,
                ),
            ),
        )
    )
    segments = torch.randn(2, 3, 8, 10)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    output = model(segments, mask)

    assert output.bag_logits.shape == (2,)
    assert output.instance_logits.shape == (2, 3)
    assert output.instance_embeddings.shape == (2, 3, 16)
    assert output.intra_attention_weights.shape == (2, 3, 5)
    assert output.inter_attention_weights is None
    assert output.topk_indices is not None
    assert output.topk_indices.shape == (2, 2)


def test_attention_mil_model_emits_intra_and_inter_attention_weights() -> None:
    model = RespiratoryMILModel(
        MILModelConfig(
            segment_encoder=_segment_encoder_config(),
            instance_head=InstanceHeadConfig(
                type="linear",
                hidden_dim=16,
                dropout=0.0,
            ),
            mil=MILConfig(
                aggregator="attention",
                attention=InterAttentionConfig(
                    hidden_dim=8,
                    dropout=0.0,
                    gated=False,
                ),
                topk=TopKConfig(k=1),
            ),
        )
    )
    segments = torch.randn(1, 4, 8, 10)
    mask = torch.tensor([[True, True, False, False]])

    output = model(segments, mask)

    assert output.intra_attention_weights.shape == (1, 4, 5)
    assert torch.allclose(
        output.intra_attention_weights.sum(dim=-1),
        torch.ones(1, 4),
        atol=1e-5,
    )
    assert output.inter_attention_weights is not None
    assert torch.isclose(
        output.inter_attention_weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5
    )
    assert torch.allclose(
        output.inter_attention_weights[0, 2:], torch.zeros(2), atol=1e-6
    )
    assert output.topk_indices is None
