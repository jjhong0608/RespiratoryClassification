from __future__ import annotations

import torch
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import AudioEncoder, WhisperEncoderDims
from src.utils.checkpoint import parse_model_cfg


def test_audio_encoder_returns_hidden_states_for_weighted_layer_sum() -> None:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=4,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )
    encoder = AudioEncoder(dims)
    x = torch.randn(2, 80, 8)

    out = encoder(x, output_hidden_states=True)

    assert out.last_hidden_state.shape == (2, 4, 32)
    assert out.hidden_states is not None
    assert len(out.hidden_states) == dims.n_audio_layer + 1
    assert out.hidden_states[0].shape == (2, 4, 32)


def test_audio_encoder_prefix_tokens_extend_sequence_length() -> None:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=4,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )
    encoder = AudioEncoder(dims)
    x = torch.randn(2, 80, 8)
    prefix = torch.randn(2, 1, 32)

    out = encoder(x, output_hidden_states=True, prefix_tokens=prefix)

    assert out.last_hidden_state.shape == (2, 5, 32)
    assert out.hidden_states is not None
    assert out.hidden_states[0].shape == (2, 5, 32)


def test_hf_classifier_head_forward_shape_with_weighted_layer_sum() -> None:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=4,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=dims,
            num_classes=3,
            head_type="hf",
            use_weighted_layer_sum=True,
            classifier_proj_size=16,
        )
    )
    x = torch.randn(2, 80, 8)

    logits = model(x)

    assert logits.shape == (2, 3)
    layer_weights = getattr(model.classifier, "layer_weights", None)
    assert layer_weights is not None
    assert tuple(layer_weights.shape) == (dims.n_audio_layer + 1,)


def test_hf_classifier_head_forward_shape_with_cls_pooling() -> None:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=4,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=dims,
            num_classes=3,
            head_type="hf",
            pooling="cls",
            use_weighted_layer_sum=True,
            classifier_proj_size=16,
        )
    )
    x = torch.randn(2, 80, 8)

    logits = model(x)

    assert logits.shape == (2, 3)
    cls_token = getattr(model.classifier, "cls_token", None)
    cls_positional_embedding = getattr(
        model.classifier, "cls_positional_embedding", None
    )
    assert cls_token is not None
    assert cls_positional_embedding is not None
    assert tuple(cls_token.shape) == (1, 1, 32)
    assert tuple(cls_positional_embedding.shape) == (1, 1, 32)


def test_parse_model_cfg_supports_hf_head() -> None:
    cfg = parse_model_cfg(
        {
            "encoder": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
            },
            "num_classes": 2,
            "head_type": "hf",
            "pooling": "mean",
            "use_weighted_layer_sum": True,
            "classifier_proj_size": 128,
        }
    )

    assert cfg.head_type == "hf"
    assert cfg.use_weighted_layer_sum is True
    assert cfg.classifier_proj_size == 128


def test_parse_model_cfg_supports_cls_pooling_for_hf_head() -> None:
    cfg = parse_model_cfg(
        {
            "encoder": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
            },
            "num_classes": 2,
            "head_type": "hf",
            "pooling": "cls",
            "use_weighted_layer_sum": False,
            "classifier_proj_size": 128,
        }
    )

    assert cfg.head_type == "hf"
    assert cfg.pooling == "cls"
