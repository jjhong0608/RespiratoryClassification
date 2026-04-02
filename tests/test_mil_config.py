from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.utils.config import JsonConfigLoader


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _training_payload() -> dict:
    return {
        "experiment": {
            "name": "wheeze_mil_topk",
            "task": "normal_vs_wheeze",
            "mode": "mil",
            "seed": 7,
            "device": "cpu",
            "output_dir": "checkpoints",
        },
        "data": {
            "train_dirs": ["datasets/train"],
            "val_dirs": ["datasets/val"],
            "label_to_index": {"normal": 0, "wheeze": 1},
            "batch_size": 2,
            "num_workers": 0,
            "audio": {
                "sample_rate": 16000,
                "clip_duration_sec": 10.0,
            },
            "preprocessing": {
                "feature_type": "log_mel",
                "source_type": "original",
                "n_mels": 80,
                "bandpass": {
                    "enabled": False,
                },
            },
            "segment": {
                "mode": "sliding_window",
                "length_sec": 5.0,
                "stride_sec": 2.5,
                "pad_last": True,
                "drop_last": False,
            },
        },
        "model": {
            "encoder": {
                "type": "whisper",
                "backbone": "tiny",
                "pretrained_name_or_path": "tiny",
                "freeze": False,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
            },
            "instance_head": {
                "type": "mlp",
                "hidden_dim": 64,
                "dropout": 0.1,
            },
            "mil": {
                "aggregator": "topk",
                "return_instance_scores": True,
                "topk": {
                    "k": 3,
                },
                "attention": {
                    "hidden_dim": 32,
                    "dropout": 0.1,
                    "gated": True,
                },
                "logsumexp": {
                    "temperature": 1.0,
                },
                "softmax_weighted": {
                    "temperature": 1.0,
                },
                "noisy_or": {
                    "clamp_eps": 1e-6,
                },
            },
        },
        "train": {
            "epochs": 3,
            "top_k": 2,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
            "loss": {
                "type": "bce",
                "pos_weight": None,
            },
            "sampler": {
                "weighted_random": True,
            },
        },
        "analysis": {
            "outputs": {
                "save_segment_scores": True,
                "save_attention_weights": True,
                "save_topk_indices": True,
                "save_bag_metadata": True,
            },
        },
    }


def test_load_training_config_uses_nested_mil_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _training_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.name == "wheeze_mil_topk"
    assert cfg.data.segment.mode == "sliding_window"
    assert cfg.data.segment.length_sec == pytest.approx(5.0)
    assert cfg.model.mil.aggregator == "topk"
    assert cfg.model.mil.topk.k == 3
    assert cfg.train.sampler.weighted_random is True
    assert cfg.analysis.outputs.save_attention_weights is True


def test_invalid_segment_config_rejects_conflicting_tail_policy(
    tmp_path: Path,
) -> None:
    payload = _training_payload()
    payload["data"]["segment"]["pad_last"] = True
    payload["data"]["segment"]["drop_last"] = True
    config_path = _write_json(tmp_path / "invalid_segment.json", payload)

    with pytest.raises(ValueError, match="pad_last and drop_last cannot both be true"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_topk_config_requires_positive_k(tmp_path: Path) -> None:
    payload = _training_payload()
    payload["model"]["mil"]["topk"]["k"] = 0
    config_path = _write_json(tmp_path / "invalid_topk.json", payload)

    with pytest.raises(ValueError, match="model.mil.topk.k must be greater than zero"):
        JsonConfigLoader.load_training(config_path)
