from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.utils.config import JsonConfigLoader

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _small_patch_branches_payload() -> list[dict[str, list[int]]]:
    return [
        {"patch_size": [8, 8], "stride": [4, 8]},
        {"patch_size": [4, 16], "stride": [2, 16]},
        {"patch_size": [2, 32], "stride": [1, 32]},
        {"patch_size": [16, 4], "stride": [8, 4]},
    ]


def _base_payload() -> dict:
    return {
        "experiment": {
            "name": "respiratory_event_mil",
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
                    "max_length": 32,
                    "do_normalize": True,
                    "mean": -4.2677393,
                    "std": 4.5689974,
                },
            },
        },
        "model": {
            "encoder": {
                "type": "multiscale_rdt_ast",
                "adaptation": {
                    "mode": "full",
                    "num_layers": 0,
                },
                "architecture": {
                    "hidden_size": 32,
                    "num_attention_heads": 4,
                    "mlp_ratio": 2.0,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1,
                    "layer_norm_eps": 1e-6,
                    "shared_stem_depth": 1,
                    "adapter_depth": 1,
                    "patch_branches": _small_patch_branches_payload(),
                    "rdt": {
                        "enabled": True,
                        "steps": 2,
                        "top_tokens_per_branch": 2,
                        "gated_residual": True,
                        "layerscale_init": 0.01,
                    },
                },
            },
            "classifier": {
                "type": "linear",
                "hidden_dim": 32,
                "dropout": 0.1,
                "pooling": "latent_mean",
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
                "branch_auxiliary": {
                    "enabled": False,
                    "weight": 0.3,
                    "aggregation": "mean",
                },
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
    payload.pop("model")
    payload["checkpoint_path"] = "checkpoints/respiratory_event_mil/last.pt"
    payload["threshold_optimization"] = {
        "enabled": True,
        "metric": "f1",
    }
    return payload


def test_repo_example_configs_load() -> None:
    b0_cfg = JsonConfigLoader.load_training(ROOT / "configs/training_event_mil_b0.json")
    b1_cfg = JsonConfigLoader.load_training(ROOT / "configs/training_event_mil_b1.json")
    b2_cfg = JsonConfigLoader.load_training(ROOT / "configs/training_event_mil_b2.json")
    b3_cfg = JsonConfigLoader.load_training(ROOT / "configs/training_event_mil_b3.json")
    training_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_multiscale_rdt.json"
    )
    multiclass_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_multiclass.json"
    )
    cv_cfg = JsonConfigLoader.load_cv(ROOT / "configs/cv_multiscale_rdt.json")
    eval_cfg = JsonConfigLoader.load_eval(ROOT / "configs/eval_multiscale_rdt.json")

    assert b0_cfg.model.encoder.architecture.rdt.enabled is False
    assert b1_cfg.train.loss.branch_auxiliary.enabled is True
    assert b2_cfg.model.encoder.architecture.rdt.steps == 2
    assert b3_cfg.model.encoder.architecture.rdt.steps == 3
    assert training_cfg.model.encoder.type == "multiscale_rdt_ast"
    assert multiclass_cfg.train.loss.type == "cross_entropy"
    assert cv_cfg.folds[0].name == "fold_0"
    assert eval_cfg.threshold_optimization.metric == "f1"


def test_load_training_config_uses_event_mil_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _base_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.mode == "clip"
    assert cfg.data.preprocessing.ast_fbank.max_length == 32
    assert cfg.model.encoder.type == "multiscale_rdt_ast"
    assert cfg.model.classifier.pooling == "latent_mean"
    assert cfg.model.encoder.architecture.rdt.enabled is True
    assert cfg.train.loss.branch_auxiliary.enabled is False


def test_load_multiclass_training_config_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    config_path = _write_json(tmp_path / "multiclass.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert len(cfg.data.label_to_index) == 3
    assert cfg.train.loss.type == "cross_entropy"


def test_valid_disabled_rdt_ignores_steps_value(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"] = {
        "enabled": False,
        "steps": 0,
        "top_tokens_per_branch": 2,
        "gated_residual": True,
        "layerscale_init": 0.01,
    }
    config_path = _write_json(tmp_path / "b0.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.model.encoder.architecture.rdt.enabled is False
    assert cfg.model.encoder.architecture.rdt.steps == 0


def test_invalid_encoder_type_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["type"] = "ast"
    config_path = _write_json(tmp_path / "bad_type.json", payload)

    with pytest.raises(ValueError, match="multiscale_rdt_ast"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_pooling_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["classifier"]["pooling"] = "cls"
    config_path = _write_json(tmp_path / "bad_pooling.json", payload)

    with pytest.raises(ValueError, match="latent_mean"):
        JsonConfigLoader.load_training(config_path)


def test_partial_adaptation_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["adaptation"] = {"mode": "partial", "num_layers": 0}
    config_path = _write_json(tmp_path / "partial.json", payload)

    with pytest.raises(ValueError, match="not supported"):
        JsonConfigLoader.load_training(config_path)


def test_hidden_size_must_be_divisible_by_head_count(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["hidden_size"] = 30
    config_path = _write_json(tmp_path / "bad_hidden.json", payload)

    with pytest.raises(ValueError, match="divisible"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_rdt_steps_when_enabled_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"]["steps"] = 0
    config_path = _write_json(tmp_path / "bad_steps.json", payload)

    with pytest.raises(ValueError, match="rdt.steps"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_top_tokens_per_branch_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"]["top_tokens_per_branch"] = 0
    config_path = _write_json(tmp_path / "bad_top_tokens.json", payload)

    with pytest.raises(ValueError, match="top_tokens_per_branch"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_branch_auxiliary_weight_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["branch_auxiliary"] = {
        "enabled": True,
        "weight": 0.0,
        "aggregation": "mean",
    }
    config_path = _write_json(tmp_path / "bad_branch_aux_weight.json", payload)

    with pytest.raises(ValueError, match="branch_auxiliary.weight"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_branch_auxiliary_aggregation_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["branch_auxiliary"]["aggregation"] = "sum"
    config_path = _write_json(tmp_path / "bad_branch_aux_agg.json", payload)

    with pytest.raises(ValueError, match="aggregation"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    "field_name, field_value, match",
    [
        ("latent_query_count", 4, "latent_query_count"),
        ("summary_tokens_per_scale", 2, "summary_tokens_per_scale"),
        ("rdt_steps", 2, "rdt_steps"),
    ],
)
def test_legacy_architecture_fields_are_rejected(
    tmp_path: Path,
    field_name: str,
    field_value: int,
    match: str,
) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"][field_name] = field_value
    config_path = _write_json(tmp_path / f"legacy_{field_name}.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


def test_patch_geometry_must_fit_frontend_dims(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["patch_branches"][0]["patch_size"] = [
        40,
        8,
    ]
    config_path = _write_json(tmp_path / "bad_patch.json", payload)

    with pytest.raises(ValueError, match="does not fit"):
        JsonConfigLoader.load_training(config_path)


def test_load_cv_and_eval_configs_use_clip_schema(tmp_path: Path) -> None:
    cv_path = _write_json(tmp_path / "cv.json", _cv_payload())
    eval_path = _write_json(tmp_path / "eval.json", _eval_payload())

    cv_cfg = JsonConfigLoader.load_cv(cv_path)
    eval_cfg = JsonConfigLoader.load_eval(eval_path)

    assert cv_cfg.folds[0].name == "fold_0"
    assert eval_cfg.threshold_optimization.metric == "f1"
