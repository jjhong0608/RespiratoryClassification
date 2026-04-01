from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.utils.config import JsonConfigLoader


def test_eval_config_parses_manual_threshold(tmp_path: Path) -> None:
    cfg = {
        "device": "cpu",
        "checkpoint_path": "checkpoints/run/last.pt",
        "data": {
            "eval_dirs": ["datasets/test"],
            "label_to_index": {"negative": 0, "positive": 1},
            "sample_rate": 16000,
            "clip_seconds": 1.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "threshold": {
            "manual": 0.7,
        },
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    parsed = JsonConfigLoader.load_eval(p)

    assert parsed.threshold.manual == pytest.approx(0.7)


def test_eval_config_rejects_invalid_manual_threshold(tmp_path: Path) -> None:
    cfg = {
        "device": "cpu",
        "checkpoint_path": "checkpoints/run/last.pt",
        "data": {
            "eval_dirs": ["datasets/test"],
            "label_to_index": {"negative": 0, "positive": 1},
            "sample_rate": 16000,
            "clip_seconds": 1.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "threshold": {
            "manual": 1.2,
        },
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold.manual"):
        JsonConfigLoader.load_eval(p)
