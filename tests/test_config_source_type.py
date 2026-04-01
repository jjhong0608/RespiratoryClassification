from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import JsonConfigLoader


def test_training_config_parses_source_type(tmp_path: Path) -> None:
    cfg = {
        "run_name": "x",
        "seed": 1,
        "device": "cpu",
        "output_dir": "checkpoints",
        "top_k": 1,
        "data": {
            "train_dirs": ["datasets/train"],
            "val_dirs": ["datasets/val"],
            "label_to_index": {"a": 0, "b": 1},
            "sample_rate": 16000,
            "clip_seconds": 1.0,
            "batch_size": 2,
            "num_workers": 0,
            "source_type": "harmonic",
        },
        "model": {
            "n_mels": 80,
            "n_audio_ctx": 50,
            "n_audio_state": 32,
            "n_audio_head": 4,
            "n_audio_layer": 1,
            "pooling": "mean",
            "classifier": {"type": "linear"},
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
        },
    }
    p = tmp_path / "training.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    parsed = JsonConfigLoader.load_training(p)
    assert parsed.data.source_type == "harmonic"
