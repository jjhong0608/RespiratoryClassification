from __future__ import annotations

from pathlib import Path

import torch
from src.utils.checkpoint import parse_model_cfg


def test_checkpoint_loads_with_weights_only_true(tmp_path: Path) -> None:
    ckpt = {
        "dims": {},
        "model_state_dict": {},
        "model_cfg": {
            "encoder": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
            },
            "num_classes": 2,
            "pooling": "mean",
            "classifier_type": "linear",
            "hidden_dim": 256,
            "dropout": 0.0,
        },
        "preprocess_cfg": {"sample_rate": 16000, "n_mels": 80, "clip_seconds": 30.0},
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    if "weights_only" in torch.load.__code__.co_varnames:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    else:
        loaded = torch.load(path, map_location="cpu")

    cfg = parse_model_cfg(loaded["model_cfg"])
    assert cfg.encoder.n_audio_ctx == 1500
