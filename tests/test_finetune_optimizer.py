from __future__ import annotations

from src.cli.finetune import FineTuneRunner
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import WhisperEncoderDims


def test_finetune_optimizer_param_groups() -> None:
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=WhisperEncoderDims(
                n_mels=80,
                n_audio_ctx=2,
                n_audio_state=32,
                n_audio_head=4,
                n_audio_layer=1,
            ),
            num_classes=2,
            head_type="hf",
            pooling="mean",
        )
    )

    optimizer = FineTuneRunner.build_optimizer(
        model,
        encoder_lr=1e-5,
        classifier_lr=1e-4,
        weight_decay=0.01,
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.01


def test_finetune_optimizer_classifier_group_includes_cls_parameters() -> None:
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=WhisperEncoderDims(
                n_mels=80,
                n_audio_ctx=2,
                n_audio_state=32,
                n_audio_head=4,
                n_audio_layer=1,
            ),
            num_classes=2,
            head_type="hf",
            pooling="cls",
        )
    )

    optimizer = FineTuneRunner.build_optimizer(
        model,
        encoder_lr=1e-5,
        classifier_lr=1e-4,
        weight_decay=0.01,
    )
    classifier_param_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    cls_token = getattr(model.classifier, "cls_token", None)
    cls_positional_embedding = getattr(
        model.classifier, "cls_positional_embedding", None
    )

    assert cls_token is not None
    assert cls_positional_embedding is not None
    assert id(cls_token) in classifier_param_ids
    assert id(cls_positional_embedding) in classifier_param_ids
