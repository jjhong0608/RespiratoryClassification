from __future__ import annotations

import pytest
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
from src.training.mil_setup import (
    apply_encoder_adaptation,
    build_grouped_optimizer,
    maybe_initialize_encoder,
)
from src.utils.config import (
    EncoderAdaptationConfig as RunEncoderAdaptationConfig,
)
from src.utils.config import (
    EncoderDimsConfig as RunEncoderDimsConfig,
)
from src.utils.config import (
    InstanceHeadConfig as RunInstanceHeadConfig,
)
from src.utils.config import (
    InterAttentionConfig as RunInterAttentionConfig,
)
from src.utils.config import (
    MILConfig as RunMILConfig,
)
from src.utils.config import (
    ModelConfig as RunModelConfig,
)
from src.utils.config import (
    SegmentEncoderConfig as RunSegmentEncoderConfig,
)
from src.utils.config import (
    TopKConfig as RunTopKConfig,
)


def _build_model() -> RespiratoryMILModel:
    return RespiratoryMILModel(
        MILModelConfig(
            segment_encoder=SegmentEncoderConfig(
                dims=WhisperEncoderDims(
                    n_mels=8,
                    n_audio_ctx=5,
                    n_audio_state=16,
                    n_audio_head=4,
                    n_audio_layer=3,
                ),
                pooling=SegmentEncoderPoolingConfig(
                    type="attention",
                    hidden_dim=8,
                    dropout=0.0,
                    gated=True,
                ),
                adaptation=SegmentEncoderAdaptationConfig(
                    mode="partial",
                    num_layers=1,
                ),
            ),
            instance_head=InstanceHeadConfig(
                type="mlp",
                hidden_dim=12,
                dropout=0.1,
            ),
            mil=MILConfig(
                aggregator="attention",
                attention=InterAttentionConfig(
                    hidden_dim=8,
                    dropout=0.0,
                    gated=True,
                ),
                topk=TopKConfig(k=2),
            ),
        )
    )


def test_partial_unfreezing_only_enables_last_block_and_ln_post() -> None:
    model = _build_model()

    summary = apply_encoder_adaptation(
        model,
        SegmentEncoderAdaptationConfig(mode="partial", num_layers=1),
    )
    trainable = {
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad
    }

    assert summary.mode == "partial"
    assert summary.trainable_parameters > 0
    assert any(name.startswith("blocks.2.") for name in trainable)
    assert any(name.startswith("ln_post.") for name in trainable)
    assert all(not name.startswith("blocks.0.") for name in trainable)
    assert all(not name.startswith("blocks.1.") for name in trainable)
    assert all(not name.startswith("conv1.") for name in trainable)
    assert all(not name.startswith("conv2.") for name in trainable)


def test_full_unfreezing_enables_all_encoder_parameters() -> None:
    model = _build_model()

    apply_encoder_adaptation(
        model,
        SegmentEncoderAdaptationConfig(mode="full", num_layers=1),
    )

    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_grouped_optimizer_uses_encoder_and_head_learning_rates() -> None:
    model = _build_model()
    apply_encoder_adaptation(
        model,
        SegmentEncoderAdaptationConfig(mode="partial", num_layers=1),
    )

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


def test_openai_whisper_registry_pretrained_is_rejected_for_ast_frontend() -> None:
    model = _build_model()
    cfg = RunModelConfig(
        segment_encoder=RunSegmentEncoderConfig(
            pretrained_name_or_path="tiny",
            dims=RunEncoderDimsConfig(
                n_audio_state=16,
                n_audio_head=4,
                n_audio_layer=3,
            ),
            adaptation=RunEncoderAdaptationConfig(mode="partial", num_layers=1),
        ),
        instance_head=RunInstanceHeadConfig(
            type="mlp",
            hidden_dim=12,
            dropout=0.1,
        ),
        mil=RunMILConfig(
            aggregator="attention",
            attention=RunInterAttentionConfig(
                hidden_dim=8,
                dropout=0.0,
                gated=True,
            ),
            topk=RunTopKConfig(k=2),
        ),
    )

    with pytest.raises(
        ValueError,
        match="OpenAI Whisper encoder checkpoints are only supported with",
    ):
        maybe_initialize_encoder(model, cfg, feature_type="ast_fbank")
