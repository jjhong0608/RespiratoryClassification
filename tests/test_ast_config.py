from __future__ import annotations

import json
import logging as py_logging
from pathlib import Path

import pytest
from src.utils import logging as logging_utils
from src.utils.config import JsonConfigLoader
from src.utils.logging import LoggingMixin

ROOT = Path(__file__).resolve().parents[1]


class _LoggingProbe(LoggingMixin):
    pass


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


def _class_aware_evidence_pooling_payload() -> dict:
    return {
        "type": "class_aware_branch_gated",
        "gate_hidden_size": None,
        "dropout": 0.15,
        "temperature": 1.0,
        "class_gate": {
            "mode": "query",
            "scorer": "diagonal",
            "global_residual": {
                "enabled": True,
                "init_scale": 0.1,
                "learnable": True,
            },
            "evidence_auxiliary": {
                "enabled": False,
                "weight": 0.1,
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


def _augmentation_payload() -> dict:
    return {
        "enabled": True,
        "waveform": {
            "enabled": True,
            "probability": 1.0,
            "gain": {
                "enabled": True,
                "probability": 0.5,
                "min_db": -3.0,
                "max_db": 3.0,
            },
            "noise": {
                "enabled": True,
                "probability": 0.3,
                "snr_db_min": 15.0,
                "snr_db_max": 30.0,
            },
            "time_shift": {
                "enabled": True,
                "probability": 0.5,
                "max_shift_fraction": 0.05,
                "mode": "zero_pad",
            },
        },
        "fbank": {
            "enabled": True,
            "probability": 0.5,
            "time_mask": {
                "enabled": True,
                "num_masks": 1,
                "max_width": 32,
            },
            "freq_mask": {
                "enabled": True,
                "num_masks": 1,
                "max_width": 8,
            },
            "mask_value": 0.0,
        },
    }


def test_retained_repo_configs_load() -> None:
    current_training_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_CNUH_new_test_CNUH_3classes.json"
    )
    baseline_training_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_CNUH.json"
    )
    cv_cfg = JsonConfigLoader.load_cv(ROOT / "configs/cv_multiscale_rdt.json")
    eval_cfg = JsonConfigLoader.load_eval(ROOT / "configs/eval_multiscale_rdt.json")

    assert current_training_cfg.experiment.name == "new_test_CNUH_3classes_ver12"
    assert current_training_cfg.experiment.logging.terminal_width is None
    assert current_training_cfg.model.encoder.type == "multiscale_rdt_ast"
    assert current_training_cfg.train.loss.type == "cross_entropy"
    assert (
        current_training_cfg.model.encoder.architecture.evidence_pooling.type
        == "class_aware_branch_gated"
    )
    assert current_training_cfg.train.loss.top_branch_margin.enabled is True
    assert baseline_training_cfg.experiment.name == "test_CNUH_3classes"
    assert baseline_training_cfg.experiment.logging.terminal_width is None
    assert baseline_training_cfg.model.encoder.type == "multiscale_rdt_ast"
    assert baseline_training_cfg.train.loss.type == "cross_entropy"
    assert cv_cfg.folds[0].name == "fold_0"
    assert cv_cfg.experiment.logging.terminal_width is None
    assert cv_cfg.train.initialization.skip_mismatched_shapes is False
    assert eval_cfg.checkpoint_path
    assert eval_cfg.experiment.logging.terminal_width is None
    assert eval_cfg.threshold_optimization.metric == "f1"


def test_load_training_config_uses_event_mil_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _base_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.mode == "clip"
    assert cfg.experiment.logging.terminal_width is None
    assert cfg.data.preprocessing.ast_fbank.max_length == 32
    assert cfg.model.encoder.type == "multiscale_rdt_ast"
    assert cfg.model.classifier.pooling == "latent_mean"
    assert cfg.model.encoder.architecture.rdt.enabled is True
    assert cfg.model.encoder.architecture.evidence_pooling.type == "mean"
    assert cfg.data.augmentation.enabled is False
    assert cfg.data.augmentation.policy.type == "independent"
    assert cfg.data.augmentation.waveform.enabled is False
    assert cfg.data.augmentation.fbank.enabled is False
    assert cfg.train.loss.label_smoothing.enabled is False
    assert cfg.train.loss.label_smoothing.value == 0.0
    assert cfg.train.loss.gate_entropy_regularization.enabled is False
    assert cfg.train.loss.gate_entropy_regularization.weight == 0.0
    assert cfg.train.loss.gate_entropy_regularization.target == "evidence_gate"
    assert cfg.train.loss.gate_entropy_regularization.start_epoch == 1
    assert cfg.train.loss.gate_entropy_regularization.end_epoch is None
    assert cfg.train.loss.class_gate_diversity_regularization.enabled is False
    assert cfg.train.loss.class_gate_diversity_regularization.weight == 0.0
    assert (
        cfg.train.loss.class_gate_diversity_regularization.target
        == "class_evidence_gate"
    )
    assert cfg.train.loss.class_gate_diversity_regularization.metric == "js_divergence"
    assert cfg.train.loss.class_gate_diversity_regularization.start_epoch == 1
    assert cfg.train.loss.class_gate_diversity_regularization.end_epoch is None
    assert cfg.train.loss.class_evidence_margin.enabled is False
    assert cfg.train.loss.class_evidence_margin.weight == 0.0
    assert cfg.train.loss.class_evidence_margin.margin == 0.0
    assert cfg.train.loss.class_evidence_margin.target == "class_evidence_logits"
    assert cfg.train.loss.class_evidence_margin.mode == "minority_vs_major"
    assert cfg.train.loss.class_evidence_margin.major_class is None
    assert cfg.train.loss.class_evidence_margin.class_weighted is False
    assert cfg.train.loss.class_evidence_margin.reduction == "mean"
    assert cfg.train.loss.class_gated_branch_logit_margin.enabled is False
    assert cfg.train.loss.class_gated_branch_logit_margin.weight == 0.0
    assert cfg.train.loss.class_gated_branch_logit_margin.margin == 0.0
    assert (
        cfg.train.loss.class_gated_branch_logit_margin.target
        == "class_gated_branch_logits"
    )
    assert (
        cfg.train.loss.class_gated_branch_logit_margin.mode
        == "true_vs_hardest_negative"
    )
    assert cfg.train.loss.class_gated_branch_logit_margin.class_weighted is False
    assert cfg.train.loss.class_gated_branch_logit_margin.reduction == "mean"
    assert cfg.train.loss.gate_weighted_branch_margin.enabled is False
    assert cfg.train.loss.gate_weighted_branch_margin.weight == 0.0
    assert cfg.train.loss.gate_weighted_branch_margin.margin == 0.0
    assert cfg.train.loss.gate_weighted_branch_margin.target == "true_class_gate"
    assert cfg.train.loss.gate_weighted_branch_margin.source == "branch_logits"
    assert cfg.train.loss.gate_weighted_branch_margin.mode == "true_vs_hardest_negative"
    assert cfg.train.loss.gate_weighted_branch_margin.class_weighted is False
    assert cfg.train.loss.gate_weighted_branch_margin.warmup_epochs == 0
    assert cfg.train.loss.gate_weighted_branch_margin.reduction == "mean"
    assert (
        cfg.train.loss.gate_weighted_branch_margin.branch_selection == "gate_weighted"
    )
    assert cfg.train.loss.gate_branch_regret.enabled is False
    assert cfg.train.loss.gate_branch_regret.weight == 0.0
    assert cfg.train.loss.gate_branch_regret.target == "true_class_gate"
    assert cfg.train.loss.gate_branch_regret.source == "branch_logits"
    assert cfg.train.loss.gate_branch_regret.mode == "best_margin_regret"
    assert cfg.train.loss.top_branch_margin.enabled is False
    assert cfg.train.loss.top_branch_margin.weight == 0.0
    assert cfg.train.loss.top_branch_margin.margin == 0.0
    assert cfg.train.loss.top_branch_margin.target == "branch_logits"
    assert cfg.train.loss.top_branch_margin.mode == "true_vs_hardest_negative"
    assert cfg.train.loss.top_branch_margin.branch_reduction == "max"
    assert cfg.train.loss.top_branch_margin.class_weighted is False
    assert cfg.train.loss.top_branch_margin.reduction == "mean"
    assert cfg.train.loss.top_branch_margin.warmup_epochs == 0
    assert dict(cfg.train.loss.top_branch_margin.margin_by_label) == {}
    assert cfg.train.loss.top_branch_margin.auto_margin_by_train_stats.enabled is False
    assert cfg.train.loss.gate_branch_regret.margin_mode == ("true_vs_hardest_negative")
    assert cfg.train.loss.gate_branch_regret.positive_threshold == 0.0
    assert dict(cfg.train.loss.gate_branch_regret.positive_threshold_by_label) == {}
    assert cfg.train.loss.gate_branch_regret.tolerance == 0.0
    assert cfg.train.loss.gate_branch_regret.warmup_epochs == 0
    assert cfg.train.loss.gate_branch_regret.weight_schedule.enabled is False
    assert cfg.train.loss.gate_branch_regret.weight_schedule.start_epoch == 1
    assert cfg.train.loss.gate_branch_regret.weight_schedule.end_epoch == 1
    assert cfg.train.loss.gate_branch_regret.weight_schedule.start_multiplier == 1.0
    assert cfg.train.loss.gate_branch_regret.weight_schedule.end_multiplier == 1.0
    assert (
        cfg.train.loss.gate_branch_regret.auto_positive_threshold_by_train_stats.enabled
        is False
    )
    assert cfg.train.loss.branch_auxiliary.enabled is False
    assert cfg.train.sampler.weighted_random is True
    assert cfg.train.sampler.enabled is False
    assert cfg.train.sampler.type == "none"
    assert cfg.train.initialization.checkpoint_path is None


def test_experiment_logging_terminal_width_parses(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["experiment"]["logging"] = {"terminal_width": 120}
    config_path = _write_json(tmp_path / "train.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.logging.terminal_width == 120


@pytest.mark.parametrize("terminal_width", [0, -1, 120.5, "120", True])
def test_experiment_logging_terminal_width_rejects_invalid(
    tmp_path: Path,
    terminal_width: object,
) -> None:
    payload = _base_payload()
    payload["experiment"]["logging"] = {"terminal_width": terminal_width}
    config_path = _write_json(tmp_path / "train.json", payload)

    with pytest.raises((TypeError, ValueError), match="terminal_width"):
        JsonConfigLoader.load_training(config_path)


def test_configure_rich_logging_fixed_width_and_preserves_file_handler(
    tmp_path: Path,
) -> None:
    logging_utils.configure_rich_logging(None)
    logging_utils.enable_file_logging(tmp_path / "run.log", mode="w")
    file_handlers = [
        handler
        for handler in logging_utils.logger.handlers
        if isinstance(handler, py_logging.FileHandler)
    ]

    logging_utils.configure_rich_logging(120)

    assert logging_utils.handler.console.width == 120
    assert all(handler in logging_utils.logger.handlers for handler in file_handlers)
    assert logging_utils.handler in _LoggingProbe().logger.handlers
    logging_utils.configure_rich_logging(None)


def test_explicit_evidence_pooling_configs_load(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = {
        "type": "mean",
        "gate_hidden_size": None,
        "dropout": 0.1,
        "temperature": 1.0,
    }
    mean_path = _write_json(tmp_path / "mean_pooling.json", payload)

    mean_cfg = JsonConfigLoader.load_training(mean_path)

    assert mean_cfg.model.encoder.architecture.evidence_pooling.type == "mean"

    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = {
        "type": "branch_gated",
        "gate_hidden_size": 16,
        "dropout": 0.2,
        "temperature": 0.5,
    }
    gated_path = _write_json(tmp_path / "branch_gated_pooling.json", payload)

    gated_cfg = JsonConfigLoader.load_training(gated_path)

    assert gated_cfg.model.encoder.architecture.evidence_pooling.type == "branch_gated"
    assert gated_cfg.model.encoder.architecture.evidence_pooling.gate_hidden_size == 16
    assert gated_cfg.model.encoder.architecture.evidence_pooling.dropout == 0.2
    assert gated_cfg.model.encoder.architecture.evidence_pooling.temperature == 0.5

    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    class_aware_path = _write_json(
        tmp_path / "class_aware_branch_gated_pooling.json",
        payload,
    )

    class_aware_cfg = JsonConfigLoader.load_training(class_aware_path)

    class_gate = class_aware_cfg.model.encoder.architecture.evidence_pooling.class_gate
    assert (
        class_aware_cfg.model.encoder.architecture.evidence_pooling.type
        == "class_aware_branch_gated"
    )
    assert class_gate.mode == "query"
    assert class_gate.scorer == "diagonal"
    assert class_gate.global_residual.enabled is True
    assert class_gate.global_residual.init_scale == 0.1
    assert class_gate.global_residual.learnable is True
    assert class_gate.global_residual.warmup.enabled is False
    assert class_gate.global_residual.warmup.mode == "zero_to_learned"
    assert class_gate.global_residual.warmup.start_multiplier == 0.0
    assert class_gate.global_residual.warmup.end_multiplier == 1.0
    assert class_gate.evidence_auxiliary.enabled is False
    assert class_gate.evidence_auxiliary.weight == 0.1
    assert class_gate.branch_logit_feature.mode == "raw"
    assert class_gate.gate_mixing.enabled is False
    assert class_gate.gate_mixing.mode == "uniform_to_learned"
    assert class_gate.gate_mixing.start_alpha == 1.0
    assert class_gate.gate_mixing.end_alpha == 0.0


def test_class_aware_branch_logit_feature_mode_loads(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    evidence_pooling = _class_aware_evidence_pooling_payload()
    evidence_pooling["class_gate"]["branch_logit_feature"] = {
        "mode": "hardest_negative_margin"
    }
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = evidence_pooling
    config_path = _write_json(
        tmp_path / "class_aware_branch_logit_feature.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    class_gate = cfg.model.encoder.architecture.evidence_pooling.class_gate
    assert class_gate.branch_logit_feature.mode == "hardest_negative_margin"


def test_class_aware_gate_mixing_and_residual_warmup_load(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    evidence_pooling = _class_aware_evidence_pooling_payload()
    evidence_pooling["class_gate"]["gate_mixing"] = {
        "enabled": True,
        "mode": "uniform_to_learned",
        "start_alpha": 1.0,
        "end_alpha": 0.0,
        "hold_epochs": 10,
        "decay_epochs": 20,
    }
    evidence_pooling["class_gate"]["global_residual"]["warmup"] = {
        "enabled": True,
        "mode": "zero_to_learned",
        "start_multiplier": 0.0,
        "end_multiplier": 1.0,
        "hold_epochs": 10,
        "decay_epochs": 20,
    }
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = evidence_pooling
    config_path = _write_json(tmp_path / "class_aware_gate_schedules.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    class_gate = cfg.model.encoder.architecture.evidence_pooling.class_gate
    assert class_gate.gate_mixing.enabled is True
    assert class_gate.gate_mixing.mode == "uniform_to_learned"
    assert class_gate.gate_mixing.start_alpha == 1.0
    assert class_gate.gate_mixing.end_alpha == 0.0
    assert class_gate.gate_mixing.hold_epochs == 10
    assert class_gate.gate_mixing.decay_epochs == 20
    assert class_gate.global_residual.warmup.enabled is True
    assert class_gate.global_residual.warmup.mode == "zero_to_learned"
    assert class_gate.global_residual.warmup.start_multiplier == 0.0
    assert class_gate.global_residual.warmup.end_multiplier == 1.0
    assert class_gate.global_residual.warmup.hold_epochs == 10
    assert class_gate.global_residual.warmup.decay_epochs == 20


def test_invalid_class_aware_branch_logit_feature_mode_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    evidence_pooling = _class_aware_evidence_pooling_payload()
    evidence_pooling["class_gate"]["branch_logit_feature"] = {
        "mode": "true_vs_hardest_negative"
    }
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = evidence_pooling
    config_path = _write_json(
        tmp_path / "bad_class_aware_branch_logit_feature.json",
        payload,
    )

    with pytest.raises(ValueError, match="branch_logit_feature.mode"):
        JsonConfigLoader.load_training(config_path)


def test_two_label_class_aware_cross_entropy_config_loads(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    config_path = _write_json(tmp_path / "two_label_class_aware_ce.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert len(cfg.data.label_to_index) == 2
    assert cfg.train.loss.type == "cross_entropy"
    assert (
        cfg.model.encoder.architecture.evidence_pooling.type
        == "class_aware_branch_gated"
    )


@pytest.mark.parametrize("loss_type", ["bce", "focal"])
def test_two_label_class_aware_requires_cross_entropy(
    tmp_path: Path,
    loss_type: str,
) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["type"] = loss_type
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    config_path = _write_json(tmp_path / f"class_aware_{loss_type}.json", payload)

    with pytest.raises(ValueError, match="class_aware_branch_gated.*cross_entropy"):
        JsonConfigLoader.load_training(config_path)


def test_two_label_cross_entropy_requires_class_aware_pooling(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    config_path = _write_json(tmp_path / "two_label_ce_mean_pooling.json", payload)

    with pytest.raises(ValueError, match="Two-label cross_entropy.*class_aware"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("auto_pos_weight", True, "auto_pos_weight.*one-logit"),
        ("pos_weight", 2.0, "pos_weight.*one-logit"),
    ],
)
def test_two_label_class_aware_cross_entropy_rejects_binary_pos_weighting(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"][field] = value
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    config_path = _write_json(tmp_path / "two_label_ce_bad_pos_weight.json", payload)

    with pytest.raises(ValueError, match=error):
        JsonConfigLoader.load_training(config_path)


def test_two_label_class_aware_cross_entropy_allows_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    config_path = _write_json(tmp_path / "two_label_ce_class_weighting.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.class_weighting.enabled is True


def test_load_multiclass_training_config_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    config_path = _write_json(tmp_path / "multiclass.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert len(cfg.data.label_to_index) == 3
    assert cfg.train.loss.type == "cross_entropy"


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("class_gate", "mode"), "mlp", "class_gate.mode"),
        (("class_gate", "scorer"), "full", "class_gate.scorer"),
        (
            ("class_gate", "global_residual", "enabled"),
            "yes",
            "global_residual.enabled",
        ),
        (("class_gate", "global_residual", "init_scale"), -0.1, "init_scale"),
        (("class_gate", "global_residual", "learnable"), "yes", "learnable"),
        (
            ("class_gate", "global_residual", "warmup", "enabled"),
            "yes",
            "global_residual.warmup.enabled",
        ),
        (
            ("class_gate", "global_residual", "warmup", "mode"),
            "constant",
            "global_residual.warmup.mode",
        ),
        (
            ("class_gate", "global_residual", "warmup", "start_multiplier"),
            1.5,
            "global_residual.warmup.start_multiplier",
        ),
        (
            ("class_gate", "global_residual", "warmup", "hold_epochs"),
            -1,
            "global_residual.warmup.hold_epochs",
        ),
        (
            ("class_gate", "gate_mixing", "enabled"),
            "yes",
            "gate_mixing.enabled",
        ),
        (
            ("class_gate", "gate_mixing", "mode"),
            "learned_to_uniform",
            "gate_mixing.mode",
        ),
        (
            ("class_gate", "gate_mixing", "start_alpha"),
            -0.1,
            "gate_mixing.start_alpha",
        ),
        (
            ("class_gate", "gate_mixing", "decay_epochs"),
            -1,
            "gate_mixing.decay_epochs",
        ),
        (
            ("class_gate", "evidence_auxiliary", "enabled"),
            "yes",
            "evidence_auxiliary.enabled",
        ),
        (
            ("class_gate", "evidence_auxiliary", "weight"),
            0.0,
            "evidence_auxiliary.weight",
        ),
    ],
)
def test_invalid_class_aware_evidence_pooling_config_is_rejected(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    error: str,
) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "wheeze": 1, "crackle": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    evidence_pooling = _class_aware_evidence_pooling_payload()
    evidence_pooling["class_gate"]["evidence_auxiliary"]["enabled"] = True
    evidence_pooling["class_gate"]["global_residual"]["warmup"] = {
        "enabled": False,
        "mode": "zero_to_learned",
        "start_multiplier": 0.0,
        "end_multiplier": 1.0,
        "hold_epochs": 0,
        "decay_epochs": 0,
    }
    evidence_pooling["class_gate"]["gate_mixing"] = {
        "enabled": False,
        "mode": "uniform_to_learned",
        "start_alpha": 1.0,
        "end_alpha": 0.0,
        "hold_epochs": 0,
        "decay_epochs": 0,
    }
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = evidence_pooling
    target: dict = evidence_pooling
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path = _write_json(tmp_path / "bad_class_gate.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


def _fourclass_branch_binary_payload() -> dict:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {
        "normal": 0,
        "crackle": 1,
        "wheeze": 2,
        "rhonchi": 3,
    }
    payload["train"]["loss"] = {
        "type": "cross_entropy",
        "auto_pos_weight": False,
        "pos_weight": None,
        "gamma": 2.0,
        "class_weighting": {
            "enabled": True,
            "type": "sqrt_inverse_frequency",
            "normalize": "mean_one",
            "source": "train",
        },
        "branch_auxiliary": {
            "enabled": False,
            "weight": 0.3,
            "aggregation": "mean",
        },
        "branch_binary_auxiliary": {
            "enabled": True,
            "weight": 0.3,
            "label_to_index": {
                "normal": 0,
                "crackle": 1,
                "wheeze": 1,
                "rhonchi": 1,
            },
            "pos_weight": {
                "enabled": True,
                "type": "sqrt_normal_over_abnormal",
                "source": "train",
            },
            "aggregation": "mean",
        },
    }
    return payload


def test_valid_branch_binary_auxiliary_config_loads(tmp_path: Path) -> None:
    config_path = _write_json(
        tmp_path / "branch_binary.json",
        _fourclass_branch_binary_payload(),
    )

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.class_weighting.enabled is True
    assert cfg.train.loss.branch_binary_auxiliary.enabled is True
    assert cfg.train.loss.branch_binary_auxiliary.label_to_index == {
        "normal": 0,
        "crackle": 1,
        "wheeze": 1,
        "rhonchi": 1,
    }


def test_valid_label_smoothing_config_loads(tmp_path: Path) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["label_smoothing"] = {
        "enabled": True,
        "value": 0.05,
    }
    config_path = _write_json(tmp_path / "label_smoothing.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.label_smoothing.enabled is True
    assert cfg.train.loss.label_smoothing.value == 0.05


def test_valid_gate_entropy_regularization_config_loads(tmp_path: Path) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["gate_entropy_regularization"] = {
        "enabled": True,
        "weight": 0.001,
        "target": "evidence_gate",
    }
    config_path = _write_json(tmp_path / "gate_entropy_regularization.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.gate_entropy_regularization.enabled is True
    assert cfg.train.loss.gate_entropy_regularization.weight == 0.001
    assert cfg.train.loss.gate_entropy_regularization.target == "evidence_gate"


@pytest.mark.parametrize(
    "target",
    [
        "class_evidence_gate",
        "true_class_evidence_gate",
        "class_evidence_learned_gate",
    ],
)
def test_valid_class_gate_entropy_regularization_targets_load(
    tmp_path: Path,
    target: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["gate_entropy_regularization"] = {
        "enabled": True,
        "weight": 0.001,
        "target": target,
        "start_epoch": 11,
        "end_epoch": 30,
    }
    config_path = _write_json(
        tmp_path / f"gate_entropy_regularization_{target}.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.gate_entropy_regularization.target == target
    assert cfg.train.loss.gate_entropy_regularization.start_epoch == 11
    assert cfg.train.loss.gate_entropy_regularization.end_epoch == 30


def test_valid_class_gate_diversity_regularization_config_loads(
    tmp_path: Path,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["class_gate_diversity_regularization"] = {
        "enabled": True,
        "weight": 0.003,
        "target": "class_evidence_learned_gate",
        "metric": "js_divergence",
        "start_epoch": 11,
        "end_epoch": 30,
    }
    config_path = _write_json(
        tmp_path / "class_gate_diversity_regularization.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    diversity_cfg = cfg.train.loss.class_gate_diversity_regularization
    assert diversity_cfg.enabled is True
    assert diversity_cfg.weight == 0.003
    assert diversity_cfg.target == "class_evidence_learned_gate"
    assert diversity_cfg.metric == "js_divergence"
    assert diversity_cfg.start_epoch == 11
    assert diversity_cfg.end_epoch == 30


def _class_aware_cross_entropy_payload() -> dict:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    return payload


def test_valid_class_evidence_margin_minority_vs_major_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "minority_vs_major",
        "major_class": "normal",
    }
    config_path = _write_json(tmp_path / "class_evidence_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.class_evidence_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.weight == 0.05
    assert margin_cfg.margin == 0.5
    assert margin_cfg.target == "class_evidence_logits"
    assert margin_cfg.mode == "minority_vs_major"
    assert margin_cfg.major_class == "normal"


def test_valid_class_evidence_margin_true_vs_hardest_config_loads_without_major(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
    }
    config_path = _write_json(tmp_path / "class_evidence_margin_hardest.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.class_evidence_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.mode == "true_vs_hardest_negative"
    assert margin_cfg.major_class is None


def test_valid_class_weighted_class_evidence_margin_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
    }
    config_path = _write_json(
        tmp_path / "class_weighted_class_evidence_margin.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.class_evidence_margin.class_weighted is True
    assert (
        cfg.train.loss.class_evidence_margin.reduction
        == "class_balanced_violating_mean"
    )


def test_valid_class_gated_branch_logit_margin_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["class_gated_branch_logit_margin"] = {
        "enabled": True,
        "weight": 0.03,
        "margin": 0.3,
        "target": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
    }
    config_path = _write_json(tmp_path / "branch_logit_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.class_gated_branch_logit_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.weight == 0.03
    assert margin_cfg.margin == 0.3
    assert margin_cfg.target == "class_gated_branch_logits"
    assert margin_cfg.mode == "true_vs_hardest_negative"
    assert margin_cfg.class_weighted is True
    assert margin_cfg.reduction == "class_balanced_violating_mean"


def test_valid_gate_weighted_branch_margin_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["gate_weighted_branch_margin"] = {
        "enabled": True,
        "weight": 0.01,
        "margin": 0.3,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
        "warmup_epochs": 10,
        "reduction": "class_balanced_violating_mean",
        "branch_selection": "gate_weighted",
    }
    config_path = _write_json(tmp_path / "gate_weighted_branch_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.gate_weighted_branch_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.weight == 0.01
    assert margin_cfg.margin == 0.3
    assert margin_cfg.target == "true_class_gate"
    assert margin_cfg.source == "branch_logits"
    assert margin_cfg.mode == "true_vs_hardest_negative"
    assert margin_cfg.class_weighted is True
    assert margin_cfg.warmup_epochs == 10
    assert margin_cfg.reduction == "class_balanced_violating_mean"
    assert margin_cfg.branch_selection == "gate_weighted"


def test_valid_top_branch_margin_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["top_branch_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.3,
        "margin_by_label": {
            "normal": 0.3,
            "crackle": 0.3,
            "wheeze": 0.5,
        },
        "target": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "branch_reduction": "max",
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 0,
        "auto_margin_by_train_stats": {
            "enabled": True,
            "strategy": "ema_violation_controller",
            "start_epoch": 11,
            "update_interval_epochs": 1,
            "ema": 0.9,
            "step": 0.02,
            "target_violation_rate_by_label": {
                "normal": 0.2,
                "crackle": 0.45,
                "wheeze": 0.8,
            },
            "min_margin_by_label": {
                "normal": 0.3,
                "crackle": 0.3,
                "wheeze": 0.45,
            },
            "max_margin_by_label": {
                "normal": 0.3,
                "crackle": 0.3,
                "wheeze": 0.75,
            },
        },
    }
    config_path = _write_json(tmp_path / "top_branch_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.top_branch_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.weight == 0.05
    assert margin_cfg.margin == 0.3
    assert margin_cfg.target == "branch_logits"
    assert margin_cfg.mode == "true_vs_hardest_negative"
    assert margin_cfg.branch_reduction == "max"
    assert margin_cfg.class_weighted is True
    assert margin_cfg.reduction == "class_balanced_violating_mean"
    assert margin_cfg.warmup_epochs == 0
    assert dict(margin_cfg.margin_by_label) == {
        "normal": 0.3,
        "crackle": 0.3,
        "wheeze": 0.5,
    }
    assert margin_cfg.auto_margin_by_train_stats.enabled is True
    assert margin_cfg.auto_margin_by_train_stats.step == 0.02
    assert margin_cfg.auto_margin_by_train_stats.min_margin_by_label["wheeze"] == 0.45


def test_valid_gate_branch_regret_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_branch_regret"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "best_margin_regret",
        "margin_mode": "true_vs_hardest_negative",
        "positive_threshold": 0.3,
        "positive_threshold_by_label": {
            "normal": 0.3,
            "crackle": 0.3,
            "wheeze": 0.1,
        },
        "tolerance": 0.05,
        "warmup_epochs": 10,
        "weight_schedule": {
            "enabled": True,
            "start_epoch": 16,
            "end_epoch": 30,
            "start_multiplier": 0.2,
            "end_multiplier": 1.0,
        },
        "auto_positive_threshold_by_train_stats": {
            "enabled": True,
            "strategy": "ema_eligible_controller",
            "start_epoch": 31,
            "update_interval_epochs": 1,
            "ema": 0.9,
            "step": 0.02,
            "target_eligible_rate_by_label": {
                "normal": 0.7,
                "crackle": 0.7,
                "wheeze": 0.5,
            },
            "min_threshold_by_label": {
                "normal": 0.3,
                "crackle": 0.3,
                "wheeze": 0.0,
            },
            "max_threshold_by_label": {
                "normal": 0.3,
                "crackle": 0.3,
                "wheeze": 0.2,
            },
        },
    }
    config_path = _write_json(tmp_path / "gate_branch_regret.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    regret_cfg = cfg.train.loss.gate_branch_regret
    assert regret_cfg.enabled is True
    assert regret_cfg.weight == 0.01
    assert regret_cfg.target == "true_class_gate"
    assert regret_cfg.source == "branch_logits"
    assert regret_cfg.mode == "best_margin_regret"
    assert regret_cfg.margin_mode == "true_vs_hardest_negative"
    assert regret_cfg.positive_threshold == 0.3
    assert dict(regret_cfg.positive_threshold_by_label) == {
        "normal": 0.3,
        "crackle": 0.3,
        "wheeze": 0.1,
    }
    assert regret_cfg.tolerance == 0.05
    assert regret_cfg.warmup_epochs == 10
    assert regret_cfg.weight_schedule.enabled is True
    assert regret_cfg.weight_schedule.start_epoch == 16
    assert regret_cfg.weight_schedule.end_epoch == 30
    assert regret_cfg.weight_schedule.start_multiplier == 0.2
    assert regret_cfg.weight_schedule.end_multiplier == 1.0
    assert regret_cfg.auto_positive_threshold_by_train_stats.enabled is True
    assert regret_cfg.auto_positive_threshold_by_train_stats.step == 0.02


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "class_gate_diversity_regularization.enabled"),
        ("weight", 0.0, "class_gate_diversity_regularization.weight"),
        ("weight", -0.001, "class_gate_diversity_regularization.weight"),
        ("target", "evidence_gate", "class_gate_diversity_regularization.target"),
        ("metric", "kl_divergence", "class_gate_diversity_regularization.metric"),
        ("start_epoch", 0, "class_gate_diversity_regularization.start_epoch"),
        ("end_epoch", 10, "class_gate_diversity_regularization.end_epoch"),
    ],
)
def test_invalid_class_gate_diversity_regularization_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["class_gate_diversity_regularization"] = {
        "enabled": True,
        "weight": 0.0003,
        "target": "class_evidence_gate",
        "metric": "js_divergence",
        "start_epoch": 11,
        "end_epoch": 30,
    }
    payload["train"]["loss"]["class_gate_diversity_regularization"][field] = value
    config_path = _write_json(
        tmp_path / "bad_class_gate_diversity_regularization.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "class_evidence_margin.enabled"),
        ("weight", 0.0, "class_evidence_margin.weight"),
        ("weight", -0.001, "class_evidence_margin.weight"),
        ("margin", 0.0, "class_evidence_margin.margin"),
        ("margin", -0.1, "class_evidence_margin.margin"),
        ("target", "final_logits", "class_evidence_margin.target"),
        ("mode", "minority_vs_normal", "class_evidence_margin.mode"),
        ("major_class", "airway", "class_evidence_margin.major_class"),
        ("class_weighted", "yes", "class_evidence_margin.class_weighted"),
        ("reduction", "violating_mean", "class_evidence_margin.reduction"),
    ],
)
def test_invalid_class_evidence_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "minority_vs_major",
        "major_class": "normal",
    }
    payload["train"]["loss"]["class_evidence_margin"][field] = value
    config_path = _write_json(tmp_path / "bad_class_evidence_margin.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "class_gated_branch_logit_margin.enabled"),
        ("weight", 0.0, "class_gated_branch_logit_margin.weight"),
        ("weight", -0.001, "class_gated_branch_logit_margin.weight"),
        ("margin", 0.0, "class_gated_branch_logit_margin.margin"),
        ("margin", -0.1, "class_gated_branch_logit_margin.margin"),
        (
            "target",
            "class_gated_branch_logit_features",
            "class_gated_branch_logit_margin.target",
        ),
        ("mode", "minority_vs_major", "class_gated_branch_logit_margin.mode"),
        ("class_weighted", "yes", "class_gated_branch_logit_margin.class_weighted"),
        (
            "reduction",
            "violating_mean",
            "class_gated_branch_logit_margin.reduction",
        ),
    ],
)
def test_invalid_class_gated_branch_logit_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_gated_branch_logit_margin"] = {
        "enabled": True,
        "weight": 0.03,
        "margin": 0.3,
        "target": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": False,
    }
    payload["train"]["loss"]["class_gated_branch_logit_margin"][field] = value
    config_path = _write_json(tmp_path / "bad_branch_logit_margin.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "gate_weighted_branch_margin.enabled"),
        ("weight", 0.0, "gate_weighted_branch_margin.weight"),
        ("weight", -0.001, "gate_weighted_branch_margin.weight"),
        ("margin", 0.0, "gate_weighted_branch_margin.margin"),
        ("margin", -0.1, "gate_weighted_branch_margin.margin"),
        ("target", "class_evidence_gate", "gate_weighted_branch_margin.target"),
        ("source", "branch_logit_margin", "gate_weighted_branch_margin.source"),
        ("mode", "detached_soft_target_kl", "gate_weighted_branch_margin.mode"),
        ("class_weighted", "yes", "gate_weighted_branch_margin.class_weighted"),
        ("warmup_epochs", -1, "gate_weighted_branch_margin.warmup_epochs"),
        ("warmup_epochs", "ten", "gate_weighted_branch_margin.warmup_epochs"),
        ("reduction", "violating_mean", "gate_weighted_branch_margin.reduction"),
        (
            "branch_selection",
            "soft_oracle",
            "gate_weighted_branch_margin.branch_selection",
        ),
        (
            "branch_selection",
            "topk_gate",
            "gate_weighted_branch_margin.branch_selection",
        ),
    ],
)
def test_invalid_gate_weighted_branch_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_weighted_branch_margin"] = {
        "enabled": True,
        "weight": 0.01,
        "margin": 0.3,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": False,
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["gate_weighted_branch_margin"][field] = value
    config_path = _write_json(
        tmp_path / "bad_gate_weighted_branch_margin.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


def test_gate_weighted_branch_margin_top_k_is_rejected(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_weighted_branch_margin"] = {
        "enabled": True,
        "weight": 0.01,
        "margin": 0.3,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": False,
        "warmup_epochs": 10,
        "top_k": 2,
    }
    config_path = _write_json(
        tmp_path / "bad_gate_weighted_branch_margin_top_k.json",
        payload,
    )

    with pytest.raises(ValueError, match="gate_weighted_branch_margin.top_k"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "top_branch_margin.enabled"),
        ("weight", 0.0, "top_branch_margin.weight"),
        ("weight", -0.001, "top_branch_margin.weight"),
        ("margin", 0.0, "top_branch_margin.margin"),
        ("margin", -0.1, "top_branch_margin.margin"),
        ("target", "class_gated_branch_logits", "top_branch_margin.target"),
        ("mode", "minority_vs_major", "top_branch_margin.mode"),
        ("branch_reduction", "mean", "top_branch_margin.branch_reduction"),
        ("class_weighted", "yes", "top_branch_margin.class_weighted"),
        ("reduction", "violating_mean", "top_branch_margin.reduction"),
        ("warmup_epochs", -1, "top_branch_margin.warmup_epochs"),
        ("warmup_epochs", "ten", "top_branch_margin.warmup_epochs"),
    ],
)
def test_invalid_top_branch_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["top_branch_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.3,
        "target": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "branch_reduction": "max",
        "class_weighted": False,
        "warmup_epochs": 0,
    }
    payload["train"]["loss"]["top_branch_margin"][field] = value
    config_path = _write_json(tmp_path / "bad_top_branch_margin.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("nested", "error"),
    [
        ({"margin_by_label": {"unknown": 0.4}}, "margin_by_label"),
        ({"margin_by_label": {"wheeze": 0.0}}, "margin_by_label"),
        (
            {
                "auto_margin_by_train_stats": {
                    "enabled": True,
                    "strategy": "bad",
                    "start_epoch": 11,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.02,
                    "target_violation_rate_by_label": {
                        "normal": 0.2,
                        "crackle": 0.45,
                        "wheeze": 0.8,
                    },
                    "min_margin_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.45,
                    },
                    "max_margin_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.75,
                    },
                }
            },
            "auto_margin_by_train_stats.strategy",
        ),
        (
            {
                "auto_margin_by_train_stats": {
                    "enabled": True,
                    "strategy": "ema_violation_controller",
                    "start_epoch": 11,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.0,
                    "target_violation_rate_by_label": {
                        "normal": 0.2,
                        "crackle": 0.45,
                        "wheeze": 0.8,
                    },
                    "min_margin_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.45,
                    },
                    "max_margin_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.75,
                    },
                }
            },
            "auto_margin_by_train_stats.step",
        ),
    ],
)
def test_invalid_top_branch_margin_label_adaptive_config_is_rejected(
    tmp_path: Path,
    nested: dict[str, object],
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["top_branch_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.3,
        "target": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "branch_reduction": "max",
        "class_weighted": False,
        "warmup_epochs": 0,
        **nested,
    }
    config_path = _write_json(
        tmp_path / "bad_top_branch_margin_label_adaptive.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "gate_branch_regret.enabled"),
        ("weight", 0.0, "gate_branch_regret.weight"),
        ("weight", -0.001, "gate_branch_regret.weight"),
        ("target", "class_evidence_gate", "gate_branch_regret.target"),
        ("source", "branch_logit_margin", "gate_branch_regret.source"),
        ("mode", "detached_soft_target_kl", "gate_branch_regret.mode"),
        ("margin_mode", "minority_vs_major", "gate_branch_regret.margin_mode"),
        ("positive_threshold", "high", "gate_branch_regret.positive_threshold"),
        ("positive_threshold", -0.1, "gate_branch_regret.positive_threshold"),
        ("tolerance", "low", "gate_branch_regret.tolerance"),
        ("tolerance", -0.1, "gate_branch_regret.tolerance"),
        ("warmup_epochs", -1, "gate_branch_regret.warmup_epochs"),
        ("warmup_epochs", "ten", "gate_branch_regret.warmup_epochs"),
    ],
)
def test_invalid_gate_branch_regret_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_branch_regret"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "best_margin_regret",
        "margin_mode": "true_vs_hardest_negative",
        "positive_threshold": 0.3,
        "tolerance": 0.05,
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["gate_branch_regret"][field] = value
    config_path = _write_json(tmp_path / "bad_gate_branch_regret.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("nested", "error"),
    [
        (
            {"positive_threshold_by_label": {"unknown": 0.1}},
            "positive_threshold_by_label",
        ),
        (
            {"positive_threshold_by_label": {"wheeze": -0.1}},
            "positive_threshold_by_label",
        ),
        (
            {"weight_schedule": {"enabled": "yes"}},
            "weight_schedule.enabled",
        ),
        (
            {"weight_schedule": {"enabled": True, "start_epoch": 0}},
            "weight_schedule.start_epoch",
        ),
        (
            {
                "weight_schedule": {
                    "enabled": True,
                    "start_epoch": 16,
                    "end_epoch": 15,
                }
            },
            "weight_schedule.end_epoch",
        ),
        (
            {"weight_schedule": {"enabled": True, "start_multiplier": -0.1}},
            "weight_schedule.start_multiplier",
        ),
        (
            {"weight_schedule": {"enabled": True, "end_multiplier": -0.1}},
            "weight_schedule.end_multiplier",
        ),
        (
            {
                "auto_positive_threshold_by_train_stats": {
                    "enabled": True,
                    "strategy": "bad",
                    "start_epoch": 31,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.02,
                    "target_eligible_rate_by_label": {
                        "normal": 0.7,
                        "crackle": 0.7,
                        "wheeze": 0.5,
                    },
                    "min_threshold_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.0,
                    },
                    "max_threshold_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.2,
                    },
                }
            },
            "auto_positive_threshold_by_train_stats.strategy",
        ),
        (
            {
                "positive_threshold_by_label": {"wheeze": 0.4},
                "auto_positive_threshold_by_train_stats": {
                    "enabled": True,
                    "strategy": "ema_eligible_controller",
                    "start_epoch": 31,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.02,
                    "target_eligible_rate_by_label": {
                        "normal": 0.7,
                        "crackle": 0.7,
                        "wheeze": 0.5,
                    },
                    "min_threshold_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.0,
                    },
                    "max_threshold_by_label": {
                        "normal": 0.3,
                        "crackle": 0.3,
                        "wheeze": 0.2,
                    },
                },
            },
            "initial threshold",
        ),
    ],
)
def test_invalid_gate_branch_regret_label_adaptive_config_is_rejected(
    tmp_path: Path,
    nested: dict[str, object],
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_branch_regret"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "best_margin_regret",
        "margin_mode": "true_vs_hardest_negative",
        "positive_threshold": 0.3,
        "tolerance": 0.05,
        "warmup_epochs": 10,
        **nested,
    }
    config_path = _write_json(
        tmp_path / "bad_gate_branch_regret_label_adaptive.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


def test_gate_branch_alignment_config_is_no_longer_supported(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_branch_alignment"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "true_class_gate",
        "source": "branch_logit_margin",
        "mode": "detached_soft_target_kl",
        "margin_mode": "true_vs_hardest_negative",
        "temperature": 1.0,
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "gate_branch_alignment_removed.json", payload)

    with pytest.raises(
        ValueError, match="gate_branch_alignment is no longer supported"
    ):
        JsonConfigLoader.load_training(config_path)


def test_class_evidence_margin_minority_vs_major_requires_major_class(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "minority_vs_major",
    }
    config_path = _write_json(
        tmp_path / "class_evidence_margin_missing_major.json",
        payload,
    )

    with pytest.raises(ValueError, match="major_class is required"):
        JsonConfigLoader.load_training(config_path)


def test_class_evidence_margin_requires_class_aware_pooling(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "minority_vs_major",
        "major_class": "normal",
    }
    config_path = _write_json(
        tmp_path / "class_evidence_margin_mean_pooling.json",
        payload,
    )

    with pytest.raises(ValueError, match="class_evidence_margin requires"):
        JsonConfigLoader.load_training(config_path)


def test_class_evidence_margin_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "minority_vs_major",
        "major_class": "normal",
    }
    config_path = _write_json(
        tmp_path / "class_evidence_margin_bce.json",
        payload,
    )

    with pytest.raises(ValueError, match="class_evidence_margin.*cross_entropy"):
        JsonConfigLoader.load_training(config_path)


def test_class_weighted_margins_require_class_weighting(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
    }
    config_path = _write_json(
        tmp_path / "class_weighted_margin_without_weights.json",
        payload,
    )

    with pytest.raises(ValueError, match="class_evidence_margin.class_weighted"):
        JsonConfigLoader.load_training(config_path)


def test_class_weighted_branch_logit_margin_requires_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_gated_branch_logit_margin"] = {
        "enabled": True,
        "weight": 0.03,
        "margin": 0.3,
        "target": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
    }
    config_path = _write_json(
        tmp_path / "class_weighted_branch_margin_without_weights.json",
        payload,
    )

    with pytest.raises(
        ValueError,
        match="class_gated_branch_logit_margin.class_weighted",
    ):
        JsonConfigLoader.load_training(config_path)


def test_class_weighted_gate_weighted_branch_margin_requires_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_weighted_branch_margin"] = {
        "enabled": True,
        "weight": 0.01,
        "margin": 0.3,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
        "warmup_epochs": 10,
    }
    config_path = _write_json(
        tmp_path / "class_weighted_gate_branch_margin_without_weights.json",
        payload,
    )

    with pytest.raises(
        ValueError,
        match="gate_weighted_branch_margin.class_weighted",
    ):
        JsonConfigLoader.load_training(config_path)


def test_class_weighted_top_branch_margin_requires_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["top_branch_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.3,
        "target": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "branch_reduction": "max",
        "class_weighted": True,
    }
    config_path = _write_json(
        tmp_path / "class_weighted_top_branch_margin_without_weights.json",
        payload,
    )

    with pytest.raises(ValueError, match="top_branch_margin.class_weighted"):
        JsonConfigLoader.load_training(config_path)


def test_class_gated_branch_logit_margin_requires_class_aware_pooling(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["class_gated_branch_logit_margin"] = {
        "enabled": True,
        "weight": 0.03,
        "margin": 0.3,
        "target": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
    }
    config_path = _write_json(tmp_path / "branch_margin_mean_pooling.json", payload)

    with pytest.raises(ValueError, match="class_gated_branch_logit_margin requires"):
        JsonConfigLoader.load_training(config_path)


def test_class_gated_branch_logit_margin_requires_cross_entropy(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    payload["train"]["loss"]["class_gated_branch_logit_margin"] = {
        "enabled": True,
        "weight": 0.03,
        "margin": 0.3,
        "target": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
    }
    config_path = _write_json(tmp_path / "branch_margin_bce.json", payload)

    with pytest.raises(
        ValueError,
        match="class_gated_branch_logit_margin.*cross_entropy",
    ):
        JsonConfigLoader.load_training(config_path)


def test_gate_weighted_branch_margin_requires_class_aware_pooling(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["gate_weighted_branch_margin"] = {
        "enabled": True,
        "weight": 0.01,
        "margin": 0.3,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "warmup_epochs": 10,
    }
    config_path = _write_json(
        tmp_path / "gate_branch_margin_mean_pooling.json", payload
    )

    with pytest.raises(ValueError, match="gate_weighted_branch_margin requires"):
        JsonConfigLoader.load_training(config_path)


def test_gate_weighted_branch_margin_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    payload["train"]["loss"]["gate_weighted_branch_margin"] = {
        "enabled": True,
        "weight": 0.01,
        "margin": 0.3,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "gate_branch_margin_bce.json", payload)

    with pytest.raises(ValueError, match="gate_weighted_branch_margin.*cross_entropy"):
        JsonConfigLoader.load_training(config_path)


def test_top_branch_margin_requires_class_aware_pooling(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["top_branch_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.3,
        "target": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "branch_reduction": "max",
    }
    config_path = _write_json(tmp_path / "top_branch_margin_mean_pooling.json", payload)

    with pytest.raises(ValueError, match="top_branch_margin requires"):
        JsonConfigLoader.load_training(config_path)


def test_top_branch_margin_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    payload["train"]["loss"]["top_branch_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.3,
        "target": "branch_logits",
        "mode": "true_vs_hardest_negative",
        "branch_reduction": "max",
    }
    config_path = _write_json(tmp_path / "top_branch_margin_bce.json", payload)

    with pytest.raises(ValueError, match="top_branch_margin.*cross_entropy"):
        JsonConfigLoader.load_training(config_path)


def test_gate_branch_regret_requires_class_aware_pooling(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = {
        "type": "branch_gated",
        "gate_hidden_size": None,
        "dropout": 0.15,
        "temperature": 1.0,
    }
    payload["train"]["loss"]["gate_branch_regret"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "best_margin_regret",
        "margin_mode": "true_vs_hardest_negative",
        "positive_threshold": 0.3,
        "tolerance": 0.05,
    }
    config_path = _write_json(tmp_path / "bad_gate_branch_regret_pooling.json", payload)

    with pytest.raises(ValueError, match="gate_branch_regret requires"):
        JsonConfigLoader.load_training(config_path)


def test_gate_branch_regret_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["type"] = "bce"
    payload["train"]["loss"]["gate_branch_regret"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "best_margin_regret",
        "margin_mode": "true_vs_hardest_negative",
        "positive_threshold": 0.3,
        "tolerance": 0.05,
    }
    config_path = _write_json(tmp_path / "bad_gate_branch_regret_loss.json", payload)

    with pytest.raises(ValueError, match="gate_branch_regret.*cross_entropy"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "gate_entropy_regularization.enabled"),
        ("weight", 0.0, "gate_entropy_regularization.weight"),
        ("weight", -0.001, "gate_entropy_regularization.weight"),
        ("target", "branch_attention", "gate_entropy_regularization.target"),
        ("start_epoch", 0, "gate_entropy_regularization.start_epoch"),
        ("end_epoch", 10, "gate_entropy_regularization.end_epoch"),
    ],
)
def test_invalid_gate_entropy_regularization_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["gate_entropy_regularization"] = {
        "enabled": True,
        "weight": 0.001,
        "target": "evidence_gate",
        "start_epoch": 11,
        "end_epoch": 30,
    }
    payload["train"]["loss"]["gate_entropy_regularization"][field] = value
    config_path = _write_json(
        tmp_path / "bad_gate_entropy_regularization.json", payload
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "label_smoothing.enabled"),
        ("value", -0.1, "label_smoothing.value"),
        ("value", 1.0, "label_smoothing.value"),
    ],
)
def test_invalid_label_smoothing_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["label_smoothing"] = {
        "enabled": True,
        "value": 0.05,
    }
    payload["train"]["loss"]["label_smoothing"][field] = value
    config_path = _write_json(tmp_path / "bad_label_smoothing.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


def test_valid_branch_binary_cosine_schedule_config_loads(tmp_path: Path) -> None:
    payload = _fourclass_branch_binary_payload()
    branch_binary = payload["train"]["loss"]["branch_binary_auxiliary"]
    branch_binary["schedule"] = {
        "enabled": True,
        "type": "cosine_floor",
        "max_weight": 0.4,
        "min_weight": 0.1,
        "total_epochs": 120,
    }
    branch_binary["monitor"] = {"loss_weight": 0.3}
    payload["checkpointing"] = {
        "monitors": [
            {
                "name": "val_loss_total_monitor",
                "mode": "min",
                "keep_top_k": 3,
                "filename_prefix": "best_loss",
            },
            {
                "name": "last",
                "mode": "last",
                "keep_top_k": 3,
                "filename_prefix": "last",
            },
        ]
    }
    config_path = _write_json(tmp_path / "branch_binary_cosine.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.branch_binary_auxiliary.schedule.enabled is True
    assert cfg.train.loss.branch_binary_auxiliary.schedule.max_weight == 0.4
    assert cfg.train.loss.branch_binary_auxiliary.schedule.min_weight == 0.1
    assert cfg.train.loss.branch_binary_auxiliary.schedule.total_epochs == 120
    assert cfg.train.loss.branch_binary_auxiliary.monitor.loss_weight == 0.3
    assert cfg.checkpointing is not None
    assert cfg.checkpointing.monitors[0].name == "val_loss_total_monitor"
    assert cfg.checkpointing.monitors[0].top_k == 3
    assert cfg.checkpointing.monitors[0].filename_prefix == "best_loss"
    assert cfg.checkpointing.monitors[1].mode == "last"


def test_valid_sqrt_inverse_sampler_config_loads(tmp_path: Path) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["sampler"] = {
        "enabled": True,
        "type": "sqrt_inverse_class",
        "replacement": True,
        "num_samples": "dataset_size",
        "source": "train",
    }
    config_path = _write_json(tmp_path / "sqrt_sampler.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.sampler.enabled is True
    assert cfg.train.sampler.type == "sqrt_inverse_class"
    assert cfg.train.sampler.replacement is True
    assert cfg.train.sampler.num_samples == "dataset_size"
    assert cfg.train.sampler.source == "train"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("type", "inverse_class", "train.sampler.type"),
        ("source", "val", "train.sampler.source"),
        ("replacement", False, "train.sampler.replacement"),
        ("num_samples", 0, "train.sampler.num_samples"),
        ("num_samples", "all", "train.sampler.num_samples"),
    ],
)
def test_invalid_sqrt_inverse_sampler_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    sampler = {
        "enabled": True,
        "type": "sqrt_inverse_class",
        "replacement": True,
        "num_samples": "dataset_size",
        "source": "train",
    }
    sampler[field] = value
    payload["train"]["sampler"] = sampler
    config_path = _write_json(tmp_path / "bad_sqrt_sampler.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("type", "linear", "schedule.type"),
        ("max_weight", 0.0, "schedule.max_weight"),
        ("min_weight", -0.1, "schedule.min_weight"),
        ("total_epochs", 0, "schedule.total_epochs"),
    ],
)
def test_invalid_branch_binary_schedule_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    schedule = {
        "enabled": True,
        "type": "cosine_floor",
        "max_weight": 0.4,
        "min_weight": 0.1,
        "total_epochs": 120,
    }
    schedule[field] = value
    payload["train"]["loss"]["branch_binary_auxiliary"]["schedule"] = schedule
    config_path = _write_json(tmp_path / "bad_branch_binary_schedule.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


def test_invalid_branch_binary_schedule_weight_order_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["branch_binary_auxiliary"]["schedule"] = {
        "enabled": True,
        "type": "cosine_floor",
        "max_weight": 0.1,
        "min_weight": 0.4,
        "total_epochs": 120,
    }
    config_path = _write_json(
        tmp_path / "bad_branch_binary_schedule_order.json", payload
    )

    with pytest.raises(ValueError, match="schedule.max_weight"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_branch_binary_monitor_config_is_rejected(tmp_path: Path) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["branch_binary_auxiliary"]["monitor"] = {
        "loss_weight": -0.1
    }
    config_path = _write_json(tmp_path / "bad_branch_binary_monitor.json", payload)

    with pytest.raises(ValueError, match="monitor.loss_weight"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("label_map", "error"),
    [
        (
            {"normal": 2, "crackle": 1, "wheeze": 1, "rhonchi": 1},
            "values must be 0 or 1",
        ),
        (
            {"normal": 0, "crackle": 0, "wheeze": 0, "rhonchi": 0},
            "at least one abnormal label",
        ),
        (
            {"normal": 1, "crackle": 1, "wheeze": 1, "rhonchi": 1},
            "at least one normal label",
        ),
        (
            {"normal": 0, "crackle": 1, "wheeze": 1},
            "missing=\\['rhonchi'\\]",
        ),
        (
            {"normal": 0, "crackle": 1, "wheeze": 1, "rhonchi": 1, "extra": 1},
            "extra=\\['extra'\\]",
        ),
    ],
)
def test_invalid_branch_binary_label_maps_are_rejected(
    tmp_path: Path,
    label_map: dict[str, int],
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["branch_binary_auxiliary"]["label_to_index"] = label_map
    config_path = _write_json(tmp_path / "bad_branch_binary.json", payload)

    with pytest.raises(ValueError, match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("type", "inverse", "class_weighting.type"),
        ("normalize", "sum_one", "class_weighting.normalize"),
        ("source", "val", "class_weighting.source"),
    ],
)
def test_invalid_class_weighting_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["class_weighting"][field] = value
    config_path = _write_json(tmp_path / "bad_class_weighting.json", payload)

    with pytest.raises(ValueError, match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("type", "normal_over_abnormal", "pos_weight.type"),
        ("source", "val", "pos_weight.source"),
    ],
)
def test_invalid_branch_binary_pos_weight_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["branch_binary_auxiliary"]["pos_weight"][field] = value
    config_path = _write_json(tmp_path / "bad_binary_pos_weight.json", payload)

    with pytest.raises(ValueError, match=error):
        JsonConfigLoader.load_training(config_path)


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


def test_top_tokens_validation_ignores_excluded_branches(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"]["top_tokens_per_branch"] = 4
    payload["model"]["encoder"]["architecture"]["rdt"][
        "exclude_branches_from_evidence"
    ] = [3]
    config_path = _write_json(
        tmp_path / "top_tokens_exclude_short_branch.json", payload
    )

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.model.encoder.architecture.rdt.top_tokens_per_branch == 4
    assert cfg.model.encoder.architecture.rdt.exclude_branches_from_evidence == (3,)


def test_invalid_evidence_score_source_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"]["evidence_score_source"] = (
        "unsupported"
    )
    config_path = _write_json(tmp_path / "bad_source.json", payload)

    with pytest.raises(ValueError, match="evidence_score_source"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_attention_temperature_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["mil"] = {"attention_temperature": 0.0}
    config_path = _write_json(tmp_path / "bad_temperature.json", payload)

    with pytest.raises(ValueError, match="attention_temperature"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field_path", "field_value", "match"),
    [
        (("waveform", "probability"), -0.1, "waveform.probability"),
        (("waveform", "gain", "probability"), 1.1, "gain.probability"),
        (("waveform", "noise", "probability"), -0.1, "noise.probability"),
        (("waveform", "time_shift", "probability"), 1.1, "time_shift.probability"),
        (("fbank", "probability"), -0.1, "fbank.probability"),
    ],
)
def test_invalid_augmentation_probability_is_rejected(
    tmp_path: Path,
    field_path: tuple[str, ...],
    field_value: object,
    match: str,
) -> None:
    payload = _base_payload()
    augmentation = _augmentation_payload()
    target = augmentation
    for field_name in field_path[:-1]:
        target = target[field_name]
    target[field_path[-1]] = field_value
    payload["data"]["augmentation"] = augmentation
    config_path = _write_json(tmp_path / "bad_augmentation_probability.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("waveform", "gain", "min_db", 4.0), "gain.min_db"),
        (("waveform", "noise", "snr_db_min", 0.0), "noise.snr_db_min"),
        (("waveform", "noise", "snr_db_max", 10.0), "noise.snr_db_min"),
        (
            ("waveform", "time_shift", "max_shift_fraction", -0.1),
            "max_shift_fraction",
        ),
        (
            ("waveform", "time_shift", "max_shift_fraction", 1.0),
            "max_shift_fraction",
        ),
        (("waveform", "time_shift", "mode", "wrap"), "time_shift.mode"),
        (("fbank", "time_mask", "num_masks", -1), "time_mask.num_masks"),
        (("fbank", "freq_mask", "max_width", -1), "freq_mask.max_width"),
    ],
)
def test_invalid_augmentation_config_is_rejected(
    tmp_path: Path,
    mutation: tuple[str, str, str, object],
    match: str,
) -> None:
    payload = _base_payload()
    augmentation = _augmentation_payload()
    section, subsection, field_name, field_value = mutation
    augmentation[section][subsection][field_name] = field_value
    payload["data"]["augmentation"] = augmentation
    config_path = _write_json(tmp_path / "bad_augmentation.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


def test_zero_width_augmentation_mask_is_valid(tmp_path: Path) -> None:
    payload = _base_payload()
    augmentation = _augmentation_payload()
    augmentation["fbank"]["time_mask"]["max_width"] = 0
    augmentation["fbank"]["freq_mask"]["max_width"] = 0
    payload["data"]["augmentation"] = augmentation
    config_path = _write_json(tmp_path / "zero_width_mask.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.data.augmentation.fbank.time_mask.max_width == 0
    assert cfg.data.augmentation.fbank.freq_mask.max_width == 0


@pytest.mark.parametrize(
    ("choices", "match"),
    [
        (
            [
                {"name": "none", "probability": 0.5},
                {"name": "unsupported", "probability": 0.5},
            ],
            "choices.name",
        ),
        (
            [
                {"name": "none", "probability": -0.1},
                {"name": "waveform", "probability": 1.1},
            ],
            "probability",
        ),
        (
            [
                {"name": "none", "probability": 0.5},
                {"name": "waveform", "probability": 0.4},
            ],
            "sum to 1.0",
        ),
        (
            [
                {"name": "none", "probability": 0.0},
                {"name": "waveform", "probability": 0.0},
            ],
            "positive probability",
        ),
    ],
)
def test_invalid_oneof_augmentation_policy_is_rejected(
    tmp_path: Path,
    choices: list[dict[str, object]],
    match: str,
) -> None:
    payload = _base_payload()
    augmentation = _augmentation_payload()
    augmentation["policy"] = {
        "type": "one_of",
        "choices": choices,
    }
    payload["data"]["augmentation"] = augmentation
    config_path = _write_json(tmp_path / "bad_oneof_policy.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


def test_invalid_augmentation_policy_type_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    augmentation = _augmentation_payload()
    augmentation["policy"] = {"type": "sometimes"}
    payload["data"]["augmentation"] = augmentation
    config_path = _write_json(tmp_path / "bad_policy_type.json", payload)

    with pytest.raises(ValueError, match="policy.type"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("section", "field_name", "field_value", "match"),
    [
        (
            "branch_event_dropout",
            "probability",
            1.0,
            "branch_event_dropout.probability",
        ),
        (
            "selected_evidence_dropout",
            "probability",
            -0.1,
            "selected_evidence_dropout.probability",
        ),
        ("branch_event_dropout", "mode", "zero", "branch_event_dropout.mode"),
        (
            "selected_evidence_dropout",
            "mode",
            "zero_mask",
            "selected_evidence_dropout.mode",
        ),
        (
            "branch_event_dropout",
            "min_keep_tokens",
            0,
            "branch_event_dropout.min_keep_tokens",
        ),
        (
            "selected_evidence_dropout",
            "min_keep_per_branch",
            0,
            "selected_evidence_dropout.min_keep_per_branch",
        ),
    ],
)
def test_invalid_token_augmentation_config_is_rejected(
    tmp_path: Path,
    section: str,
    field_name: str,
    field_value: object,
    match: str,
) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["token_augmentation"] = {
        "branch_event_dropout": {
            "enabled": False,
            "probability": 0.05,
            "mode": "zero_mask",
            "min_keep_tokens": 1,
        },
        "selected_evidence_dropout": {
            "enabled": False,
            "probability": 0.05,
            "mode": "zero",
            "min_keep_per_branch": 1,
        },
    }
    payload["model"]["encoder"]["architecture"]["token_augmentation"][section][
        field_name
    ] = field_value
    config_path = _write_json(tmp_path / "bad_token_augmentation.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field_name", "field_value", "match"),
    [
        ("type", "invalid", "evidence_pooling.type"),
        ("temperature", 0.0, "evidence_pooling.temperature"),
        ("dropout", -0.1, "evidence_pooling.dropout"),
        ("dropout", 1.0, "evidence_pooling.dropout"),
        ("gate_hidden_size", 0, "evidence_pooling.gate_hidden_size"),
    ],
)
def test_invalid_evidence_pooling_config_is_rejected(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    match: str,
) -> None:
    payload = _base_payload()
    evidence_pooling: dict[str, object] = {
        "type": "branch_gated",
        "gate_hidden_size": None,
        "dropout": 0.1,
        "temperature": 1.0,
    }
    evidence_pooling[field_name] = field_value
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = evidence_pooling
    config_path = _write_json(tmp_path / f"bad_pooling_{field_name}.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


def test_invalid_excluded_evidence_branch_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"][
        "exclude_branches_from_evidence"
    ] = [4]
    config_path = _write_json(tmp_path / "bad_excluded_branch.json", payload)

    with pytest.raises(ValueError, match="exclude_branches_from_evidence"):
        JsonConfigLoader.load_training(config_path)


def test_duplicate_excluded_evidence_branch_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["rdt"][
        "exclude_branches_from_evidence"
    ] = [2, 2]
    config_path = _write_json(tmp_path / "duplicate_excluded_branch.json", payload)

    with pytest.raises(ValueError, match="duplicate"):
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


def test_invalid_branch_auxiliary_weights_length_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["branch_auxiliary"] = {
        "enabled": True,
        "weight": 0.1,
        "weights": [0.1, 0.1, 0.1],
        "aggregation": "mean",
    }
    config_path = _write_json(tmp_path / "bad_branch_weights_length.json", payload)

    with pytest.raises(ValueError, match="weights length"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_branch_auxiliary_weights_value_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["branch_auxiliary"] = {
        "enabled": True,
        "weight": 0.1,
        "weights": [0.1, 0.1, 0.0, 0.1],
        "aggregation": "mean",
    }
    config_path = _write_json(tmp_path / "bad_branch_weights_value.json", payload)

    with pytest.raises(ValueError, match="weights values"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_attention_entropy_weight_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["loss"]["attention_entropy"] = {
        "enabled": True,
        "weight": 0.0,
    }
    config_path = _write_json(tmp_path / "bad_entropy.json", payload)

    with pytest.raises(ValueError, match="attention_entropy.weight"):
        JsonConfigLoader.load_training(config_path)


def test_training_initialization_accepts_checkpoint_path(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["train"]["initialization"] = {
        "checkpoint_path": "checkpoints/stage1/best_loss_0.123456.pt",
        "load_model_state": True,
        "strict": False,
        "load_optimizer_state": False,
    }
    config_path = _write_json(tmp_path / "warmstart.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert (
        cfg.train.initialization.checkpoint_path
        == "checkpoints/stage1/best_loss_0.123456.pt"
    )
    assert cfg.train.initialization.load_model_state is True
    assert cfg.train.initialization.strict is False
    assert cfg.train.initialization.load_optimizer_state is False
    assert cfg.train.initialization.skip_mismatched_shapes is False


def test_training_initialization_accepts_skip_mismatched_shapes(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["train"]["initialization"] = {
        "checkpoint_path": "checkpoints/stage1/best_loss_0.123456.pt",
        "load_model_state": True,
        "strict": False,
        "load_optimizer_state": False,
        "skip_mismatched_shapes": True,
    }
    config_path = _write_json(tmp_path / "filtered_warmstart.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.initialization.skip_mismatched_shapes is True


@pytest.mark.parametrize(
    "field_name, field_value, match",
    [
        ("load_model_state", "yes", "load_model_state"),
        ("strict", "false", "strict"),
        ("load_optimizer_state", 1, "load_optimizer_state"),
        ("skip_mismatched_shapes", "true", "skip_mismatched_shapes"),
    ],
)
def test_training_initialization_flags_must_be_booleans(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    match: str,
) -> None:
    payload = _base_payload()
    payload["train"]["initialization"] = {
        "checkpoint_path": None,
        "load_model_state": True,
        "strict": False,
        "load_optimizer_state": False,
        "skip_mismatched_shapes": False,
    }
    payload["train"]["initialization"][field_name] = field_value
    config_path = _write_json(tmp_path / f"bad_init_{field_name}.json", payload)

    with pytest.raises(TypeError, match=match):
        JsonConfigLoader.load_training(config_path)


def test_training_initialization_checkpoint_path_must_be_string_or_null(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["train"]["initialization"] = {
        "checkpoint_path": 123,
        "load_model_state": True,
        "strict": False,
        "load_optimizer_state": False,
        "skip_mismatched_shapes": False,
    }
    config_path = _write_json(tmp_path / "bad_init_path.json", payload)

    with pytest.raises(TypeError, match="checkpoint_path"):
        JsonConfigLoader.load_training(config_path)


def test_training_initialization_rejects_optimizer_state_with_shape_filter(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["train"]["initialization"] = {
        "checkpoint_path": "checkpoints/stage1/best_loss_0.123456.pt",
        "load_model_state": True,
        "strict": False,
        "load_optimizer_state": True,
        "skip_mismatched_shapes": True,
    }
    config_path = _write_json(tmp_path / "bad_init_optimizer_filter.json", payload)

    with pytest.raises(ValueError, match="load_optimizer_state"):
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
