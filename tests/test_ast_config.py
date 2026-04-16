from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.utils.config import JsonConfigLoader
from transformers import ASTConfig


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _base_payload() -> dict:
    return {
        "experiment": {
            "name": "respiratory_ast_mil",
            "task": "normal_vs_wheeze",
            "mode": "recording_mil",
            "seed": 7,
            "device": "cpu",
            "output_dir": "checkpoints",
        },
        "data": {
            "metadata": {
                "label_to_index": {"normal": 0, "wheeze": 1},
                "splits": {
                    "train": {"roots": ["datasets/train"]},
                    "val": {"roots": ["datasets/val"]},
                    "eval": {"roots": ["datasets/eval"]},
                },
            },
            "audio": {
                "sample_rate": 16000,
            },
            "instance": {
                "window_sec": 2.0,
                "hop_sec": 1.0,
                "tail_policy": "cover_end",
            },
            "features": {
                "source_type": "original",
                "bandpass": {
                    "enabled": False,
                },
                "ast_fbank": {
                    "num_mel_bins": 32,
                    "max_length": 64,
                    "do_normalize": True,
                    "mean": -4.2677393,
                    "std": 4.5689974,
                },
            },
            "loader": {
                "num_workers": 0,
            },
        },
        "model": {
            "encoder": {
                "type": "ast",
                "pretrained_name_or_path": None,
                "cache_dir": None,
                "pooling": "cls",
                "adaptation": {
                    "mode": "partial",
                    "num_layers": 1,
                },
                "architecture": {
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 64,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1,
                    "frequency_stride": 10,
                    "time_stride": 10,
                    "patch_size": 16,
                    "qkv_bias": True,
                    "layer_norm_eps": 1e-12,
                    "initializer_range": 0.02,
                },
            },
            "instance_head": {
                "projection_dim": None,
                "dropout": 0.1,
                "normalize": False,
            },
            "mil": {
                "type": "gated_attention",
                "gated_attention": {
                    "attention_dim": 16,
                    "dropout": 0.1,
                },
                "linear_softmax": {
                    "eps": 1e-6,
                },
            },
            "classifier": {
                "type": "linear",
                "hidden_dim": 32,
                "dropout": 0.1,
            },
        },
        "train": {
            "batch_size": 2,
            "epochs": 3,
            "top_k": 2,
            "max_grad_norm": 1.0,
            "optimizer": {
                "encoder_lr": 1e-5,
                "head_lr": 1e-4,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "warmup_ratio": 0.1,
            },
            "loss": {
                "type": "bce",
                "auto_pos_weight": False,
                "pos_weight": None,
                "gamma": 2.0,
            },
            "sampler": {
                "weighted_random": True,
            },
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 5,
                "min_delta": 1e-4,
            },
        },
        "eval": {
            "batch_size": 2,
            "threshold_optimization": {
                "enabled": True,
                "metric": "f1",
            },
        },
        "logging": {
            "diagnostics": {
                "save_bag_logits": True,
                "save_bag_probabilities": True,
                "save_bag_embedding": False,
                "save_instance_logits": True,
                "save_instance_probabilities": True,
                "save_instance_embeddings": False,
                "save_attention_weights": True,
                "save_instance_metadata": True,
                "top_k_instances": 3,
            },
        },
    }


def _cv_payload() -> dict:
    payload = _base_payload()
    payload["cv"] = {
        "folds": [
            {
                "name": "fold_0",
                "train": {"roots": ["datasets/folds/fold_1"]},
                "val": {"roots": ["datasets/folds/fold_0"]},
            }
        ]
    }
    return payload


def _eval_payload() -> dict:
    payload = _base_payload()
    payload.pop("model")
    payload.pop("train")
    payload["eval"]["checkpoint_path"] = "checkpoints/respiratory_ast_mil/last.pt"
    return payload


def test_load_training_config_uses_recording_mil_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _base_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.mode == "recording_mil"
    assert cfg.data.instance.window_sec == 2.0
    assert cfg.model.encoder.pooling == "cls"
    assert cfg.model.mil.type == "gated_attention"
    assert cfg.eval.batch_size == 2


def test_load_multiclass_training_config_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["metadata"]["label_to_index"] = {
        "normal": 0,
        "wheeze": 1,
        "crackle": 2,
    }
    payload["train"]["loss"]["type"] = "cross_entropy"
    config_path = _write_json(tmp_path / "multiclass.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert len(cfg.data.label_to_index) == 3
    assert cfg.train.loss.type == "cross_entropy"


def test_legacy_clip_fields_are_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["audio"]["clip_duration_sec"] = 10.0
    config_path = _write_json(tmp_path / "legacy.json", payload)

    with pytest.raises(TypeError, match="clip_duration_sec"):
        JsonConfigLoader.load_training(config_path)


def test_multiclass_training_rejects_binary_loss(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["metadata"]["label_to_index"] = {
        "normal": 0,
        "wheeze": 1,
        "crackle": 2,
    }
    payload["train"]["loss"]["type"] = "bce"
    config_path = _write_json(tmp_path / "invalid_multiclass.json", payload)

    with pytest.raises(
        ValueError,
        match="Multi-class AST\\+MIL runs require train.loss.type='cross_entropy'",
    ):
        JsonConfigLoader.load_training(config_path)


def test_pretrained_input_dim_mismatch_is_rejected(tmp_path: Path) -> None:
    pretrained_dir = tmp_path / "pretrained_ast"
    ASTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_mel_bins=64,
        max_length=96,
    ).save_pretrained(pretrained_dir)

    payload = _base_payload()
    payload["model"]["encoder"]["pretrained_name_or_path"] = str(pretrained_dir)
    config_path = _write_json(tmp_path / "mismatch.json", payload)

    with pytest.raises(
        ValueError,
        match="Pretrained AST encoder input dims do not match",
    ):
        JsonConfigLoader.load_training(config_path)


def test_load_cv_and_eval_configs_use_recording_mil_schema(tmp_path: Path) -> None:
    cv_path = _write_json(tmp_path / "cv.json", _cv_payload())
    eval_path = _write_json(tmp_path / "eval.json", _eval_payload())

    cv_cfg = JsonConfigLoader.load_cv(cv_path)
    eval_cfg = JsonConfigLoader.load_eval(eval_path)

    assert cv_cfg.cv.folds[0].train.roots == ["datasets/folds/fold_1"]
    assert eval_cfg.eval.checkpoint_path == "checkpoints/respiratory_ast_mil/last.pt"
