from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.utils.config import JsonConfigLoader


def test_supcon_config_parses_nested_sections(tmp_path: Path) -> None:
    cfg = {
        "run_name": "supcon",
        "seed": 7,
        "device": "cpu",
        "output_dir": "checkpoints",
        "top_k": 2,
        "data": {
            "train_dirs": ["datasets/train"],
            "val_dirs": ["datasets/val"],
            "label_to_index": {"normal": 0, "abnormal": 1},
            "sample_rate": 16000,
            "clip_seconds": 1.0,
            "source_type": "original",
            "batch_size": 4,
            "num_workers": 0,
        },
        "encoder": {
            "n_mels": 80,
            "n_audio_ctx": 50,
            "n_audio_state": 32,
            "n_audio_head": 4,
            "n_audio_layer": 2,
        },
        "training": {
            "epochs": 3,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
        "supervised_contrastive": {
            "temperature": 0.07,
            "normalize": True,
            "projection_head": {
                "hidden_dim": 64,
                "output_dim": 16,
            },
            "augmentation": {
                "time_mask_param": 6,
                "time_mask_count": 2,
                "freq_mask_param": 4,
                "freq_mask_count": 1,
                "gaussian_noise_std": 0.05,
            },
        },
    }
    path = tmp_path / "supcon.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    parsed = JsonConfigLoader.load_supcon_pretrain(path)

    assert parsed.supervised_contrastive.temperature == pytest.approx(0.07)
    assert parsed.supervised_contrastive.projection_head.hidden_dim == 64
    assert parsed.supervised_contrastive.projection_head.output_dim == 16
    assert parsed.supervised_contrastive.augmentation.time_mask_count == 2
    assert (
        parsed.supervised_contrastive.augmentation.gaussian_noise_std
        == pytest.approx(0.05)
    )


def test_supcon_config_rejects_frozen_encoder(tmp_path: Path) -> None:
    cfg = {
        "run_name": "supcon",
        "seed": 7,
        "device": "cpu",
        "output_dir": "checkpoints",
        "top_k": 1,
        "pretrained": {
            "name_or_path": "base",
            "freeze_encoder": True,
        },
        "data": {
            "train_dirs": ["datasets/train"],
            "val_dirs": ["datasets/val"],
            "label_to_index": {"normal": 0, "abnormal": 1},
            "sample_rate": 16000,
            "clip_seconds": 1.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "encoder": {
            "n_mels": 80,
            "n_audio_ctx": 50,
            "n_audio_state": 32,
            "n_audio_head": 4,
            "n_audio_layer": 2,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
        },
        "supervised_contrastive": {
            "projection_head": {
                "hidden_dim": 32,
                "output_dim": 16,
            }
        },
    }
    path = tmp_path / "supcon_bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze_encoder"):
        JsonConfigLoader.load_supcon_pretrain(path)
