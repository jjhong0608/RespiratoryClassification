from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import JsonConfigLoader


def test_load_finetune_config(tmp_path: Path) -> None:
    cfg = {
        "run_name": "ft_run",
        "seed": 1,
        "device": "cpu",
        "output_dir": "checkpoints",
        "top_k": 2,
        "finetune": {
            "checkpoint_path": "checkpoints/run/last.pt",
            "encoder_lr": 1e-5,
            "classifier_lr": 1e-4,
        },
        "data": {
            "train_dirs": ["datasets/train"],
            "val_dirs": ["datasets/val"],
            "label_to_index": {"a": 0, "b": 1},
            "sample_rate": 16000,
            "clip_seconds": 30.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "max_grad_norm": 1.0,
        },
    }
    path = tmp_path / "finetune.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    parsed = JsonConfigLoader.load_finetune(path)
    assert parsed.run_name == "ft_run"
    assert parsed.finetune.checkpoint_path.endswith("last.pt")
    assert parsed.finetune.encoder_lr == 1e-5
    assert parsed.finetune.classifier_lr == 1e-4
