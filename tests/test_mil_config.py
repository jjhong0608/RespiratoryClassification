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
            "eval_dirs": ["datasets/eval"],
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
            "segment_encoder": {
                "type": "whisper",
                "backbone": "tiny",
                "pretrained_name_or_path": "tiny",
                "strict": True,
                "dims": {
                    "n_audio_state": 384,
                    "n_audio_head": 6,
                    "n_audio_layer": 4,
                },
                "pooling": {
                    "type": "attention",
                    "hidden_dim": 64,
                    "dropout": 0.1,
                    "gated": True,
                },
                "adaptation": {
                    "mode": "partial",
                    "num_layers": 1,
                },
            },
            "instance_head": {
                "type": "mlp",
                "hidden_dim": 64,
                "dropout": 0.1,
            },
            "mil": {
                "aggregator": "topk",
                "topk": {
                    "k": 3,
                },
                "attention": {
                    "hidden_dim": 32,
                    "dropout": 0.1,
                    "gated": True,
                },
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
                "save_segment_scores": True,
                "save_instance_logits": True,
                "save_intra_attention_weights": True,
                "save_inter_attention_weights": True,
                "save_topk_indices": True,
                "save_bag_metadata": True,
            },
        },
    }


def _cv_payload() -> dict:
    payload = _training_payload()
    payload["folds"] = [
        {
            "name": "fold_0",
            "train_dirs": ["datasets/folds/fold_1"],
            "val_dirs": ["datasets/folds/fold_0"],
        }
    ]
    return payload


def _eval_payload() -> dict:
    payload = _training_payload()
    payload.pop("train")
    payload["checkpoint_path"] = "checkpoints/wheeze_mil_topk/last.pt"
    payload["threshold_optimization"] = {
        "enabled": True,
        "metric": "f1",
    }
    return payload


def test_load_training_config_uses_nested_final_model_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _training_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.name == "wheeze_mil_topk"
    assert cfg.data.segment.mode == "sliding_window"
    assert cfg.model.segment_encoder.pooling.type == "attention"
    assert cfg.model.segment_encoder.adaptation.mode == "partial"
    assert cfg.model.mil.aggregator == "topk"
    assert cfg.model.mil.topk.k == 3
    assert cfg.train.optimizer.encoder_lr == pytest.approx(1e-5)
    assert cfg.train.optimizer.head_lr == pytest.approx(1e-4)
    assert cfg.train.scheduler.warmup_ratio == pytest.approx(0.1)
    assert cfg.train.early_stopping.enabled is True
    assert cfg.analysis.outputs.save_inter_attention_weights is True


def test_load_cv_config_uses_nested_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "cv.json", _cv_payload())

    cfg = JsonConfigLoader.load_cv(config_path)

    assert len(cfg.folds) == 1
    assert cfg.folds[0].name == "fold_0"
    assert cfg.model.segment_encoder.adaptation.num_layers == 1


def test_invalid_segment_config_rejects_conflicting_tail_policy(
    tmp_path: Path,
) -> None:
    payload = _training_payload()
    payload["data"]["segment"]["pad_last"] = True
    payload["data"]["segment"]["drop_last"] = True
    config_path = _write_json(tmp_path / "invalid_segment.json", payload)

    with pytest.raises(ValueError, match="pad_last and drop_last cannot both be true"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_pooling_type_is_rejected(tmp_path: Path) -> None:
    payload = _training_payload()
    payload["model"]["segment_encoder"]["pooling"]["type"] = "mean"
    config_path = _write_json(tmp_path / "invalid_pooling.json", payload)

    with pytest.raises(
        ValueError, match="model.segment_encoder.pooling.type must be 'attention'"
    ):
        JsonConfigLoader.load_training(config_path)


def test_invalid_topk_config_requires_positive_k(tmp_path: Path) -> None:
    payload = _training_payload()
    payload["model"]["mil"]["topk"]["k"] = 0
    config_path = _write_json(tmp_path / "invalid_topk.json", payload)

    with pytest.raises(ValueError, match="model.mil.topk.k must be greater than zero"):
        JsonConfigLoader.load_training(config_path)


def test_partial_unfreeze_rejects_num_layers_past_encoder_depth(
    tmp_path: Path,
) -> None:
    payload = _training_payload()
    payload["model"]["segment_encoder"]["adaptation"]["num_layers"] = 8
    config_path = _write_json(tmp_path / "invalid_adaptation.json", payload)

    with pytest.raises(
        ValueError,
        match="model.segment_encoder.adaptation.num_layers must not exceed",
    ):
        JsonConfigLoader.load_training(config_path)


def test_training_config_allows_auto_pos_weight_for_focal_loss(tmp_path: Path) -> None:
    payload = _training_payload()
    payload["train"]["loss"]["type"] = "focal"
    payload["train"]["loss"]["auto_pos_weight"] = True
    config_path = _write_json(tmp_path / "focal_auto_pos_weight.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.type == "focal"
    assert cfg.train.loss.auto_pos_weight is True
    assert cfg.train.loss.gamma == pytest.approx(2.0)


def test_training_config_rejects_conflicting_pos_weight_settings(
    tmp_path: Path,
) -> None:
    payload = _training_payload()
    payload["train"]["loss"]["auto_pos_weight"] = True
    payload["train"]["loss"]["pos_weight"] = 3.0
    config_path = _write_json(tmp_path / "conflicting_pos_weight.json", payload)

    with pytest.raises(
        ValueError,
        match="train.loss.auto_pos_weight and train.loss.pos_weight cannot both be set",
    ):
        JsonConfigLoader.load_training(config_path)


def test_training_config_rejects_invalid_early_stopping_monitor(
    tmp_path: Path,
) -> None:
    payload = _training_payload()
    payload["train"]["early_stopping"]["monitor"] = "val_f1"
    config_path = _write_json(tmp_path / "invalid_monitor.json", payload)

    with pytest.raises(
        ValueError,
        match="train.early_stopping.monitor must be one of",
    ):
        JsonConfigLoader.load_training(config_path)


def test_eval_config_parses_threshold_optimization(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "eval.json", _eval_payload())

    cfg = JsonConfigLoader.load_eval(config_path)

    assert cfg.threshold_optimization.enabled is True
    assert cfg.threshold_optimization.metric == "f1"


def test_eval_config_rejects_invalid_threshold_metric(tmp_path: Path) -> None:
    payload = _eval_payload()
    payload["threshold_optimization"]["metric"] = "invalid_metric"
    config_path = _write_json(tmp_path / "invalid_eval.json", payload)

    with pytest.raises(
        ValueError,
        match="threshold_optimization.metric must be one of",
    ):
        JsonConfigLoader.load_eval(config_path)
