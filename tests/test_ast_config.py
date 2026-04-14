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
            "name": "respiratory_ast_clip",
            "task": "normal_vs_wheeze",
            "mode": "clip",
            "seed": 7,
            "device": "cpu",
            "output_dir": "checkpoints",
        },
        "data": {
            "train_dirs": ["datasets/train"],
            "val_dirs": ["datasets/val"],
            "eval_dirs": ["datasets/eval"],
            "label_to_index": {"normal": 0, "wheeze": 1},
            "batch_size": 2,
            "num_workers": 0,
            "audio": {
                "sample_rate": 16000,
                "clip_duration_sec": 10.0,
            },
            "preprocessing": {
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
        },
        "model": {
            "encoder": {
                "type": "ast",
                "pretrained_name_or_path": None,
                "cache_dir": None,
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
            "classifier": {
                "type": "linear",
                "hidden_dim": 32,
                "dropout": 0.1,
                "pooling": "cls",
            },
        },
        "train": {
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
        "analysis": {
            "outputs": {
                "save_logits": True,
                "save_probabilities": True,
                "save_embeddings": False,
                "save_clip_metadata": True,
            },
        },
    }


def _cv_payload() -> dict:
    payload = _base_payload()
    payload["folds"] = [
        {
            "name": "fold_0",
            "train_dirs": ["datasets/folds/fold_1"],
            "val_dirs": ["datasets/folds/fold_0"],
        }
    ]
    return payload


def _eval_payload() -> dict:
    payload = _base_payload()
    payload.pop("train")
    payload["checkpoint_path"] = "checkpoints/respiratory_ast_clip/last.pt"
    payload["threshold_optimization"] = {
        "enabled": True,
        "metric": "f1",
    }
    return payload


def test_load_training_config_uses_ast_clip_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _base_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.mode == "clip"
    assert cfg.data.preprocessing.ast_fbank.max_length == 64
    assert cfg.model.encoder.type == "ast"
    assert cfg.model.encoder.adaptation.mode == "partial"
    assert cfg.model.classifier.pooling == "cls"
    assert cfg.train.loss.type == "bce"


def test_load_multiclass_training_config_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    config_path = _write_json(tmp_path / "multiclass.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert len(cfg.data.label_to_index) == 3
    assert cfg.train.loss.type == "cross_entropy"


def test_legacy_mil_fields_are_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["segment"] = {
        "mode": "sliding_window",
    }
    config_path = _write_json(tmp_path / "legacy.json", payload)

    with pytest.raises(TypeError, match="segment"):
        JsonConfigLoader.load_training(config_path)


def test_multiclass_training_rejects_binary_loss(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "bce"
    config_path = _write_json(tmp_path / "invalid_multiclass.json", payload)

    with pytest.raises(
        ValueError,
        match="Multi-class AST runs require train.loss.type='cross_entropy'",
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


def test_load_cv_and_eval_configs_use_clip_schema(tmp_path: Path) -> None:
    cv_path = _write_json(tmp_path / "cv.json", _cv_payload())
    eval_path = _write_json(tmp_path / "eval.json", _eval_payload())

    cv_cfg = JsonConfigLoader.load_cv(cv_path)
    eval_cfg = JsonConfigLoader.load_eval(eval_path)

    assert cv_cfg.folds[0].name == "fold_0"
    assert eval_cfg.threshold_optimization.metric == "f1"
