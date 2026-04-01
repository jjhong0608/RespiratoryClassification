from __future__ import annotations

from pathlib import Path

import torch
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import WhisperEncoderDims
from src.pretrained.whisper import OpenAIWhisperCheckpointLoader
from src.utils.config import PretrainedConfig


def _make_model() -> WhisperEncoderClassifier:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=10,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )
    return WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=dims,
            num_classes=3,
            head_type="hf",
            pooling="mean",
        )
    )


def _make_cls_model() -> WhisperEncoderClassifier:
    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=10,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=2,
    )
    return WhisperEncoderClassifier(
        WhisperClassifierConfig(
            encoder=dims,
            num_classes=3,
            head_type="hf",
            pooling="cls",
        )
    )


def test_load_encoder_from_local_checkpoint(tmp_path: Path) -> None:
    src_model = _make_model()
    encoder_sd = src_model.encoder.state_dict()
    model_state_dict = {f"encoder.{k}": v.clone() for k, v in encoder_sd.items()}

    dims = {
        "n_mels": src_model.cfg.encoder.n_mels,
        "n_audio_ctx": src_model.cfg.encoder.n_audio_ctx,
        "n_audio_state": src_model.cfg.encoder.n_audio_state,
        "n_audio_head": src_model.cfg.encoder.n_audio_head,
        "n_audio_layer": src_model.cfg.encoder.n_audio_layer,
    }
    ckpt = {"dims": dims, "model_state_dict": model_state_dict}

    path = tmp_path / "whisper_like.pt"
    torch.save(ckpt, path)

    dst_model = _make_model()
    loader = OpenAIWhisperCheckpointLoader()
    info = loader.load_encoder_into(
        dst_model,
        PretrainedConfig(name_or_path=str(path), strict=True, freeze_encoder=False),
    )
    assert info.loaded_keys == len(encoder_sd)

    dst_sd = dst_model.encoder.state_dict()
    for k, v in encoder_sd.items():
        assert torch.equal(v, dst_sd[k])


def test_load_encoder_dim_mismatch_raises(tmp_path: Path) -> None:
    model = _make_model()
    encoder_sd = model.encoder.state_dict()
    model_state_dict = {f"encoder.{k}": v.clone() for k, v in encoder_sd.items()}

    dims = {
        "n_mels": model.cfg.encoder.n_mels,
        "n_audio_ctx": model.cfg.encoder.n_audio_ctx,
        "n_audio_state": model.cfg.encoder.n_audio_state,
        "n_audio_head": model.cfg.encoder.n_audio_head,
        "n_audio_layer": model.cfg.encoder.n_audio_layer + 1,
    }
    ckpt = {"dims": dims, "model_state_dict": model_state_dict}

    path = tmp_path / "bad_dims.pt"
    torch.save(ckpt, path)

    loader = OpenAIWhisperCheckpointLoader()
    try:
        loader.load_encoder_into(
            model, PretrainedConfig(name_or_path=str(path), strict=True)
        )
    except ValueError as e:
        assert "Encoder dims mismatch" in str(e)
        assert "python -m src.cli.pretrained_info" in str(e)
    else:
        raise AssertionError("Expected ValueError for encoder dims mismatch")


def test_freeze_encoder_sets_requires_grad_false(tmp_path: Path) -> None:
    src_model = _make_model()
    encoder_sd = src_model.encoder.state_dict()
    model_state_dict = {f"encoder.{k}": v.clone() for k, v in encoder_sd.items()}
    dims = {
        "n_mels": src_model.cfg.encoder.n_mels,
        "n_audio_ctx": src_model.cfg.encoder.n_audio_ctx,
        "n_audio_state": src_model.cfg.encoder.n_audio_state,
        "n_audio_head": src_model.cfg.encoder.n_audio_head,
        "n_audio_layer": src_model.cfg.encoder.n_audio_layer,
    }
    ckpt = {"dims": dims, "model_state_dict": model_state_dict}

    path = tmp_path / "freeze.pt"
    torch.save(ckpt, path)

    model = _make_model()
    loader = OpenAIWhisperCheckpointLoader()
    loader.load_encoder_into(
        model,
        PretrainedConfig(name_or_path=str(path), strict=True, freeze_encoder=True),
    )
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_inspect_encoder_dims_from_local_checkpoint(tmp_path: Path) -> None:
    model = _make_model()
    encoder_sd = model.encoder.state_dict()
    model_state_dict = {f"encoder.{k}": v.clone() for k, v in encoder_sd.items()}

    dims = {
        "n_mels": model.cfg.encoder.n_mels,
        "n_audio_ctx": model.cfg.encoder.n_audio_ctx,
        "n_audio_state": model.cfg.encoder.n_audio_state,
        "n_audio_head": model.cfg.encoder.n_audio_head,
        "n_audio_layer": model.cfg.encoder.n_audio_layer,
    }
    ckpt = {"dims": dims, "model_state_dict": model_state_dict}

    path = tmp_path / "inspect.pt"
    torch.save(ckpt, path)

    loader = OpenAIWhisperCheckpointLoader()
    got = loader.inspect_encoder_dims(str(path))
    assert got == model.cfg.encoder


def test_load_encoder_from_local_checkpoint_with_cls_pooling(tmp_path: Path) -> None:
    src_model = _make_model()
    encoder_sd = src_model.encoder.state_dict()
    model_state_dict = {f"encoder.{k}": v.clone() for k, v in encoder_sd.items()}

    dims = {
        "n_mels": src_model.cfg.encoder.n_mels,
        "n_audio_ctx": src_model.cfg.encoder.n_audio_ctx,
        "n_audio_state": src_model.cfg.encoder.n_audio_state,
        "n_audio_head": src_model.cfg.encoder.n_audio_head,
        "n_audio_layer": src_model.cfg.encoder.n_audio_layer,
    }
    ckpt = {"dims": dims, "model_state_dict": model_state_dict}

    path = tmp_path / "whisper_like_cls.pt"
    torch.save(ckpt, path)

    dst_model = _make_cls_model()
    loader = OpenAIWhisperCheckpointLoader()
    info = loader.load_encoder_into(
        dst_model,
        PretrainedConfig(name_or_path=str(path), strict=True, freeze_encoder=False),
    )
    assert info.loaded_keys == len(encoder_sd)

    dst_sd = dst_model.encoder.state_dict()
    for k, v in encoder_sd.items():
        assert torch.equal(v, dst_sd[k])
