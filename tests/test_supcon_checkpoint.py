from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.supcon import (
    SupervisedContrastiveEncoder,
    SupervisedContrastiveEncoderConfig,
)
from src.models.whisper_encoder import WhisperEncoderDims
from src.pretrained.whisper import OpenAIWhisperCheckpointLoader
from src.utils.config import PretrainedConfig


def _make_encoder_dims() -> WhisperEncoderDims:
    return WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=10,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )


def test_supcon_checkpoint_saves_encoder_only(tmp_path: Path) -> None:
    model = SupervisedContrastiveEncoder(
        SupervisedContrastiveEncoderConfig(
            encoder=_make_encoder_dims(),
            projection_hidden_dim=64,
            projection_output_dim=16,
        )
    )
    checkpoint = model.export_pretrained_checkpoint(
        epoch=3, train_losses=[1.0], val_losses=[0.8]
    )
    path = tmp_path / "supcon_last.pt"
    torch.save(checkpoint, path)

    state_dict = checkpoint["model_state_dict"]
    assert all(key.startswith("encoder.") for key in state_dict)
    assert not any("projector" in key for key in state_dict)
    assert checkpoint["dims"] == asdict(_make_encoder_dims())


def test_supcon_checkpoint_loads_into_classifier(tmp_path: Path) -> None:
    encoder_dims = _make_encoder_dims()
    pretrain_model = SupervisedContrastiveEncoder(
        SupervisedContrastiveEncoderConfig(
            encoder=encoder_dims,
            projection_hidden_dim=64,
            projection_output_dim=16,
        )
    )
    checkpoint = pretrain_model.export_pretrained_checkpoint(
        epoch=1,
        train_losses=[1.0],
        val_losses=[0.9],
    )
    path = tmp_path / "supcon_encoder.pt"
    torch.save(checkpoint, path)

    classifier_model = WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=encoder_dims,
            num_classes=2,
            head_type="hf",
            pooling="mean",
        )
    )
    loader = OpenAIWhisperCheckpointLoader()
    loader.load_encoder_into(
        classifier_model,
        PretrainedConfig(name_or_path=str(path), strict=True, freeze_encoder=False),
    )

    for key, value in pretrain_model.encoder.state_dict().items():
        assert torch.equal(value, classifier_model.encoder.state_dict()[key])
