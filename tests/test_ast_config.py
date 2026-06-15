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
    disease_training_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_CNUH_disease_3classes.json"
    )
    baseline_training_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_CNUH.json"
    )
    cv_cfg = JsonConfigLoader.load_cv(ROOT / "configs/cv_multiscale_rdt.json")
    eval_cfg = JsonConfigLoader.load_eval(ROOT / "configs/eval_multiscale_rdt.json")

    assert current_training_cfg.experiment.name == "new_test_CNUH_3classes_ver36"
    assert current_training_cfg.experiment.logging.terminal_width == 310
    assert current_training_cfg.model.encoder.type == "multiscale_rdt_ast"
    assert current_training_cfg.model.encoder.architecture.hidden_size == 512
    assert current_training_cfg.model.encoder.architecture.adapter_depth == 8
    assert current_training_cfg.train.loss.type == "cross_entropy"
    assert (
        current_training_cfg.model.encoder.architecture.evidence_pooling.type
        == "class_aware_branch_gated"
    )
    current_class_gate = (
        current_training_cfg.model.encoder.architecture.evidence_pooling.class_gate
    )
    assert current_class_gate.evidence_scorer.type == "class_axis_attention"
    assert current_class_gate.evidence_scorer.embedding_hidden_size == 768
    assert current_class_gate.evidence_scorer.branch_hidden_size == 128
    assert current_class_gate.evidence_scorer.fusion_hidden_size == 1024
    assert current_class_gate.evidence_scorer.num_attention_heads == 4
    assert current_class_gate.evidence_scorer.num_attention_layers == 2
    assert current_class_gate.evidence_scorer.use_class_embedding is True
    assert current_class_gate.evidence_scorer.logit_centering is True
    assert current_class_gate.evidence_scorer.dropout == pytest.approx(0.05)
    assert current_class_gate.evidence_scorer.score_decomposition.enabled is True
    assert (
        current_class_gate.evidence_scorer.score_decomposition.branch_scale_mode
        == "bounded_sigmoid"
    )
    assert (
        current_class_gate.evidence_scorer.score_decomposition.branch_scale_min
        == pytest.approx(0.7)
    )
    assert (
        current_class_gate.evidence_scorer.score_decomposition.branch_scale_init
        == pytest.approx(1.0)
    )
    assert (
        current_class_gate.evidence_scorer.score_decomposition.branch_scale_max
        == pytest.approx(2.0)
    )
    assert (
        current_class_gate.evidence_scorer.score_decomposition.interaction_scale_mode
        == "bounded_sigmoid"
    )
    assert (
        current_class_gate.evidence_scorer.score_decomposition.interaction_scale_min
        == pytest.approx(0.3)
    )
    assert (
        current_class_gate.evidence_scorer.score_decomposition.interaction_scale_init
        == pytest.approx(0.7)
    )
    schedule = current_class_gate.evidence_scorer.score_decomposition.interaction_scale_schedule
    assert schedule.enabled is True
    assert schedule.start_epoch == 11
    assert schedule.end_epoch == 30
    assert schedule.start_multiplier == pytest.approx(0.2)
    assert schedule.end_multiplier == pytest.approx(1.0)
    assert current_class_gate.evidence_scorer.branch_direct_score.enabled is True
    assert (
        current_class_gate.evidence_scorer.branch_direct_score.top_support_mode
        == "raw_existential_plus_relative_correction"
    )
    correction = (
        current_class_gate.evidence_scorer.branch_direct_score.top_relative_correction
    )
    assert correction.enabled is True
    assert correction.positive_scale == pytest.approx(0.2)
    assert correction.negative_scale == pytest.approx(0.05)
    assert correction.negative_clip == pytest.approx(1.0)
    assert (
        current_class_gate.evidence_scorer.branch_direct_score.top_scale_init
        == pytest.approx(1.0)
    )
    assert (
        current_class_gate.evidence_scorer.branch_direct_score.gated_scale_init
        == pytest.approx(0.5)
    )
    reliability = (
        current_class_gate.evidence_scorer.branch_direct_score.gate_reliability_mixture
    )
    assert reliability.enabled is True
    assert reliability.source == "top_vs_gated_margin_regret"
    assert reliability.mode == "exp_neg_regret"
    assert reliability.temperature == pytest.approx(1.0)
    assert reliability.tolerance == pytest.approx(0.05)
    assert reliability.detach is True
    assert (
        current_class_gate.evidence_scorer.branch_direct_score.residual_scale_init
        == pytest.approx(0.2)
    )
    score_bounding = (
        current_class_gate.evidence_scorer.score_decomposition.score_bounding
    )
    assert score_bounding.enabled is True
    assert score_bounding.embedding.enabled is True
    assert score_bounding.embedding.bound == pytest.approx(8.0)
    assert score_bounding.embedding.temperature == pytest.approx(1.0)
    assert score_bounding.interaction.enabled is True
    assert score_bounding.interaction.bound == pytest.approx(6.0)
    assert score_bounding.interaction.temperature == pytest.approx(1.0)
    direct_path = (
        current_class_gate.evidence_scorer.branch_direct_score.top_support_direct_path
    )
    assert direct_path.enabled is True
    assert direct_path.mode == "monotonic_raw_relative"
    assert direct_path.raw_scale_min == pytest.approx(0.7)
    assert direct_path.raw_scale_init == pytest.approx(1.0)
    assert direct_path.raw_scale_max == pytest.approx(2.0)
    assert direct_path.relative_positive_scale_min == pytest.approx(0.2)
    assert direct_path.relative_positive_scale_init == pytest.approx(0.5)
    assert direct_path.relative_positive_scale_max == pytest.approx(1.5)
    assert direct_path.relative_negative_scale_init == pytest.approx(0.1)
    assert direct_path.relative_negative_scale_max == pytest.approx(0.5)
    assert direct_path.residual_hidden_size == 64
    assert direct_path.residual_scale_init == pytest.approx(0.05)
    assert direct_path.residual_scale_max == pytest.approx(0.3)
    current_negative_cap = direct_path.negative_relative_cap
    assert current_negative_cap.enabled is True
    assert current_negative_cap.mode == "raw_fraction_cap"
    assert current_negative_cap.max_negative_fraction == pytest.approx(0.75)
    assert current_negative_cap.negative_cap == pytest.approx(1.5)
    assert current_negative_cap.max_negative_fraction_by_label == {"wheeze": 0.5}
    assert current_negative_cap.negative_cap_by_label == {"wheeze": 1.0}
    assert current_negative_cap.max_negative_fraction_by_class == pytest.approx(
        (0.75, 0.75, 0.5)
    )
    assert current_negative_cap.negative_cap_by_class == pytest.approx((1.5, 1.5, 1.0))
    assert current_training_cfg.train.loss.top_support_score_margin.enabled is True
    assert (
        current_training_cfg.train.loss.top_support_score_margin.weight
        == pytest.approx(0.05)
    )
    assert (
        current_training_cfg.train.loss.top_support_score_margin.label_weight_by_label
        == {
            "normal": 1.0,
            "crackle": 1.0,
            "wheeze": 2.0,
        }
    )
    hardness = (
        current_training_cfg.train.loss.top_support_score_margin.hardness_weighting
    )
    assert hardness.enabled is True
    assert hardness.source == "top_support_gap"
    assert hardness.mode == "negative_gap"
    assert hardness.gain == pytest.approx(1.0)
    assert hardness.cap == pytest.approx(3.0)
    support_multiplier = current_training_cfg.train.loss.top_support_score_margin.support_conditioned_multiplier
    assert support_multiplier.enabled is True
    assert support_multiplier.source == "top_branch_margin"
    assert support_multiplier.mode == "linear"
    assert support_multiplier.gain == pytest.approx(1.0)
    assert support_multiplier.cap == pytest.approx(2.0)
    min_gap = current_training_cfg.train.loss.top_support_gap_min_constraint
    assert min_gap.enabled is True
    assert min_gap.weight == pytest.approx(0.05)
    assert min_gap.base_min_gap_by_label == {
        "normal": 0.0,
        "crackle": 0.0,
        "wheeze": 0.2,
    }
    assert min_gap.support_gain == pytest.approx(0.5)
    assert min_gap.support_cap == pytest.approx(2.0)
    teacher_rel = current_training_cfg.train.loss.class_top_branch_relative_margin
    assert teacher_rel.enabled is True
    assert teacher_rel.weight == pytest.approx(0.05)
    assert teacher_rel.target == "class_top_branch_margin_features"
    assert teacher_rel.mode == "true_vs_hardest_negative_hinge"
    assert teacher_rel.margin == pytest.approx(0.3)
    assert teacher_rel.support_weighting.enabled is True
    assert teacher_rel.support_weighting.source == "top_branch_margin"
    assert teacher_rel.support_weighting.mode == "linear"
    assert teacher_rel.support_weighting.gain == pytest.approx(0.75)
    assert teacher_rel.support_weighting.cap == pytest.approx(2.0)
    assert teacher_rel.hardness_weighting.enabled is True
    assert teacher_rel.hardness_weighting.source == "teacher_gap_deficit"
    assert teacher_rel.hardness_weighting.mode == "linear"
    assert teacher_rel.hardness_weighting.gain == pytest.approx(0.75)
    assert teacher_rel.hardness_weighting.cap == pytest.approx(2.0)
    assert teacher_rel.margin_by_label == {
        "normal": 0.3,
        "crackle": 0.3,
        "wheeze": 0.8,
    }
    assert set(teacher_rel.hardness_weighting_by_label) == {
        "normal",
        "crackle",
        "wheeze",
    }
    assert teacher_rel.hardness_weighting_by_label["wheeze"].gain == pytest.approx(1.5)
    assert teacher_rel.hardness_weighting_by_label["wheeze"].cap == pytest.approx(4.0)
    assert teacher_rel.weak_positive_support_weighting.enabled is True
    assert teacher_rel.weak_positive_support_weighting.min_support == pytest.approx(0.2)
    assert teacher_rel.weak_positive_support_weighting.max_support == pytest.approx(1.0)
    assert teacher_rel.weak_positive_support_weighting.multiplier == pytest.approx(1.25)
    assert teacher_rel.weak_positive_support_weighting.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    assert teacher_rel.weak_positive_margin_boost.boost_by_label == {
        "normal": 0.1,
        "crackle": 0.2,
        "wheeze": 0.8,
    }
    assert teacher_rel.weak_positive_margin_boost.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    teacher_min = current_training_cfg.train.loss.top_teacher_gap_min_constraint
    assert teacher_min.enabled is True
    assert teacher_min.weight == pytest.approx(0.12)
    assert teacher_min.target == "class_top_branch_margin_features"
    assert teacher_min.mode == "support_conditioned_min_gap"
    assert teacher_min.base_min_gap == pytest.approx(0.0)
    assert teacher_min.support_source == "top_branch_margin"
    assert teacher_min.support_gain == pytest.approx(0.75)
    assert teacher_min.support_cap == pytest.approx(2.0)
    assert teacher_min.base_min_gap_by_label == {
        "normal": 0.0,
        "crackle": 0.0,
        "wheeze": 0.4,
    }
    assert teacher_min.support_gain_by_label == {
        "normal": 0.5,
        "crackle": 0.5,
        "wheeze": 1.25,
    }
    assert teacher_min.weak_positive_support_weighting.enabled is True
    assert teacher_min.weak_positive_support_weighting.min_support == pytest.approx(0.2)
    assert teacher_min.weak_positive_support_weighting.max_support == pytest.approx(1.0)
    assert teacher_min.weak_positive_support_weighting.multiplier == pytest.approx(1.5)
    assert teacher_min.weak_positive_support_weighting.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    assert teacher_min.weak_positive_target_boost.boost_by_label == {
        "normal": 0.2,
        "crackle": 0.2,
        "wheeze": 1.0,
    }
    assert teacher_min.weak_positive_target_boost.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    phase_schedule = (
        current_training_cfg.train.loss.top_branch_margin.phase_weight_schedule
    )
    assert phase_schedule.enabled is True
    assert phase_schedule.start_epoch == 21
    assert phase_schedule.end_epoch == 31
    assert phase_schedule.start_multiplier_by_label == {
        "normal": 1.0,
        "crackle": 1.0,
        "wheeze": 1.0,
    }
    assert phase_schedule.label_multiplier_by_label == {
        "normal": 1.0,
        "crackle": 1.0,
        "wheeze": 2.0,
    }
    top_branch_hardness = (
        current_training_cfg.train.loss.top_branch_margin.hardness_weighting
    )
    assert top_branch_hardness.enabled is True
    assert top_branch_hardness.source == "margin_deficit"
    assert top_branch_hardness.mode == "linear"
    assert top_branch_hardness.gain == pytest.approx(1.0)
    assert top_branch_hardness.cap == pytest.approx(3.0)
    assert current_training_cfg.train.loss.branch_direct_score_margin.enabled is True
    assert (
        current_training_cfg.train.loss.branch_direct_score_margin.weight
        == pytest.approx(0.05)
    )
    assert current_training_cfg.train.loss.branch_support_score_margin.enabled is True
    assert (
        current_training_cfg.train.loss.branch_support_score_margin.weight
        == pytest.approx(0.05)
    )
    dominance = current_training_cfg.train.loss.branch_path_dominance_constraint
    assert dominance.enabled is True
    assert dominance.weight == pytest.approx(0.05)
    assert dominance.allowed_drop == pytest.approx(0.5)
    assert dominance.label_weight_by_label == {
        "normal": 0.75,
        "crackle": 0.75,
        "wheeze": 1.5,
    }
    disagreement_cap = (
        current_training_cfg.train.loss.branch_support_disagreement_cap_regularization
    )
    assert disagreement_cap.enabled is True
    assert disagreement_cap.weight == pytest.approx(0.02)
    assert disagreement_cap.embedding_gap_cap == pytest.approx(6.0)
    assert disagreement_cap.interaction_gap_cap == pytest.approx(4.0)
    assert dominance.allowed_drop_by_label == {
        "normal": 0.5,
        "crackle": 0.75,
        "wheeze": 0.3,
    }
    assert dominance.support_weighting.enabled is True
    assert dominance.support_weighting.gain == pytest.approx(0.5)
    assert dominance.support_weighting.cap == pytest.approx(3.0)
    assert current_class_gate.evidence_scorer.branch_feature_transform.mode == "tanh"
    assert (
        current_class_gate.evidence_scorer.branch_feature_transform.temperature
        == pytest.approx(1.0)
    )
    assert (
        current_training_cfg.train.loss.branch_to_evidence_ranking_consistency.enabled
        is True
    )
    assert (
        current_training_cfg.train.loss.branch_to_evidence_ranking_consistency.mode
        == "true_label_anchored_softplus"
    )
    assert (
        current_training_cfg.train.loss.branch_to_evidence_ranking_consistency.source
        == "class_top_branch_margin_relative_features"
    )
    assert (
        current_training_cfg.train.loss.branch_to_evidence_ranking_consistency.weight
        == pytest.approx(0.1)
    )
    assert current_training_cfg.train.loss.global_residual_anti_veto.enabled is True
    assert current_training_cfg.train.loss.top_branch_margin.enabled is True
    assert current_training_cfg.train.loss.gate_bad_branch_suppression.enabled is True
    assert (
        current_training_cfg.train.loss.class_weighting.type
        == "power_inverse_frequency"
    )
    assert current_training_cfg.train.loss.class_weighting.power == pytest.approx(0.75)
    assert (
        current_training_cfg.train.loss.class_evidence_margin.mode
        == "softplus_true_vs_hardest_negative"
    )
    assert current_training_cfg.train.loss.class_evidence_margin.margin == 0.0
    assert current_training_cfg.train.loss.class_evidence_margin.reduction == "mean"
    assert (
        current_training_cfg.train.loss.class_evidence_gap_cap_regularization.enabled
        is True
    )
    assert (
        current_training_cfg.train.loss.class_evidence_gap_cap_regularization.weight
        == pytest.approx(0.02)
    )
    assert (
        current_training_cfg.train.loss.class_evidence_gap_cap_regularization.negative_gap_cap
        == pytest.approx(3.0)
    )
    assert (
        current_training_cfg.train.loss.class_evidence_gap_cap_regularization.label_weight_by_label
        == {
            "normal": 1.0,
            "crackle": 1.0,
            "wheeze": 1.5,
        }
    )
    assert (
        current_training_cfg.train.loss.class_evidence_gap_cap_regularization.negative_gap_cap_by_label
        == {
            "normal": 3.0,
            "crackle": 3.0,
            "wheeze": 2.5,
        }
    )
    assert (
        current_training_cfg.train.loss.class_evidence_positive_gap_cap_regularization.enabled
        is True
    )
    assert (
        current_training_cfg.train.loss.class_evidence_positive_gap_cap_regularization.positive_gap_cap
        == pytest.approx(8.0)
    )
    assert (
        current_training_cfg.train.loss.class_evidence_positive_gap_cap_regularization.weight
        == pytest.approx(0.015)
    )
    assert current_training_cfg.train.loss.interaction_gap_cap_regularization.enabled
    assert (
        current_training_cfg.train.loss.interaction_gap_cap_regularization.gap_cap
        == pytest.approx(6.0)
    )
    assert (
        current_training_cfg.train.loss.interaction_gap_cap_regularization.weight
        == pytest.approx(0.015)
    )
    assert current_training_cfg.model.classifier.hidden_dim == 1024
    assert current_training_cfg.model.classifier.fusion_projector.type == "mlp"
    assert current_training_cfg.model.classifier.fusion_projector.hidden_dim == 640
    assert current_training_cfg.model.classifier.fusion_projector.layer_norm is True
    assert current_class_gate.global_residual.correction.confidence_aware_gate.enabled
    assert (
        current_class_gate.global_residual.correction.confidence_aware_gate.damping
        == pytest.approx(0.7)
    )
    assert disease_training_cfg.experiment.name == "CNUH_DISEASE_VER20"
    assert disease_training_cfg.model.encoder.architecture.hidden_size == 512
    assert disease_training_cfg.model.encoder.architecture.adapter_depth == 8
    assert disease_training_cfg.model.classifier.hidden_dim == 1024
    assert disease_training_cfg.model.classifier.fusion_projector.type == "mlp"
    assert disease_training_cfg.model.classifier.fusion_projector.hidden_dim == 640
    assert disease_training_cfg.model.classifier.fusion_projector.layer_norm is True
    assert (
        disease_training_cfg.train.loss.branch_to_evidence_ranking_consistency.mode
        == "true_label_anchored_softplus"
    )
    assert (
        disease_training_cfg.train.loss.branch_to_evidence_ranking_consistency.source
        == "class_top_branch_margin_relative_features"
    )
    assert (
        disease_training_cfg.train.loss.branch_to_evidence_ranking_consistency.weight
        == pytest.approx(0.1)
    )
    assert (
        disease_training_cfg.train.loss.class_weighting.type
        == "power_inverse_frequency"
    )
    assert disease_training_cfg.train.loss.class_weighting.power == pytest.approx(0.75)
    assert (
        disease_training_cfg.train.loss.class_evidence_margin.mode
        == "softplus_true_vs_hardest_negative"
    )
    assert disease_training_cfg.train.loss.branch_support_score_margin.enabled is True
    assert disease_training_cfg.train.loss.branch_support_score_margin.weight == (
        pytest.approx(0.05)
    )
    assert disease_training_cfg.train.loss.top_support_score_margin.enabled is True
    assert disease_training_cfg.train.loss.top_support_score_margin.weight == (
        pytest.approx(0.05)
    )
    disease_min_gap = disease_training_cfg.train.loss.top_support_gap_min_constraint
    assert disease_min_gap.enabled is True
    assert disease_min_gap.base_min_gap_by_label == {
        "Normal": 0.0,
        "Lung_Parenchymal": 0.0,
        "Airway": 0.2,
    }
    disease_teacher_rel = (
        disease_training_cfg.train.loss.class_top_branch_relative_margin
    )
    assert disease_teacher_rel.enabled is True
    assert disease_teacher_rel.weight == pytest.approx(0.05)
    assert disease_teacher_rel.margin == pytest.approx(0.3)
    assert disease_teacher_rel.support_weighting.enabled is True
    assert disease_teacher_rel.support_weighting.source == "top_branch_margin"
    assert disease_teacher_rel.support_weighting.gain == pytest.approx(0.75)
    assert disease_teacher_rel.support_weighting.cap == pytest.approx(2.0)
    assert disease_teacher_rel.hardness_weighting.enabled is True
    assert disease_teacher_rel.hardness_weighting.source == "teacher_gap_deficit"
    assert disease_teacher_rel.hardness_weighting.mode == "linear"
    assert disease_teacher_rel.hardness_weighting.gain == pytest.approx(0.75)
    assert disease_teacher_rel.hardness_weighting.cap == pytest.approx(2.0)
    assert disease_teacher_rel.margin_by_label == {
        "Normal": 0.3,
        "Lung_Parenchymal": 0.3,
        "Airway": 0.8,
    }
    assert disease_teacher_rel.hardness_weighting_by_label["Normal"].gain == (
        pytest.approx(0.5)
    )
    assert disease_teacher_rel.hardness_weighting_by_label[
        "Lung_Parenchymal"
    ].cap == pytest.approx(2.5)
    assert disease_teacher_rel.hardness_weighting_by_label["Airway"].gain == (
        pytest.approx(1.5)
    )
    assert disease_teacher_rel.hardness_weighting_by_label["Airway"].cap == (
        pytest.approx(4.0)
    )
    assert disease_teacher_rel.weak_positive_support_weighting.enabled is True
    assert (
        disease_teacher_rel.weak_positive_support_weighting.multiplier
        == pytest.approx(1.25)
    )
    assert disease_teacher_rel.weak_positive_support_weighting.support_band_by_label[
        "Airway"
    ].max_support == pytest.approx(1.2)
    assert disease_teacher_rel.weak_positive_margin_boost.enabled is True
    assert disease_teacher_rel.weak_positive_margin_boost.boost == pytest.approx(0.2)
    assert disease_teacher_rel.weak_positive_margin_boost.boost_by_label == {
        "Normal": 0.1,
        "Lung_Parenchymal": 0.2,
        "Airway": 0.8,
    }
    assert disease_teacher_rel.weak_positive_margin_boost.support_band_by_label[
        "Airway"
    ].max_support == pytest.approx(1.2)
    disease_teacher_min = disease_training_cfg.train.loss.top_teacher_gap_min_constraint
    assert disease_teacher_min.enabled is True
    assert disease_teacher_min.weight == pytest.approx(0.12)
    assert disease_teacher_min.base_min_gap == pytest.approx(0.0)
    assert disease_teacher_min.support_gain == pytest.approx(0.75)
    assert disease_teacher_min.support_cap == pytest.approx(2.0)
    assert disease_teacher_min.base_min_gap_by_label == {
        "Normal": 0.0,
        "Lung_Parenchymal": 0.0,
        "Airway": 0.4,
    }
    assert disease_teacher_min.support_gain_by_label == {
        "Normal": 0.5,
        "Lung_Parenchymal": 0.5,
        "Airway": 1.25,
    }
    assert disease_teacher_min.weak_positive_support_weighting.enabled is True
    assert (
        disease_teacher_min.weak_positive_support_weighting.multiplier
        == pytest.approx(1.5)
    )
    assert disease_teacher_min.weak_positive_support_weighting.support_band_by_label[
        "Airway"
    ].max_support == pytest.approx(1.2)
    assert disease_teacher_min.weak_positive_target_boost.enabled is True
    assert disease_teacher_min.weak_positive_target_boost.boost == pytest.approx(0.3)
    assert disease_teacher_min.weak_positive_target_boost.boost_by_label == {
        "Normal": 0.2,
        "Lung_Parenchymal": 0.2,
        "Airway": 1.0,
    }
    assert disease_teacher_min.weak_positive_target_boost.support_band_by_label[
        "Airway"
    ].max_support == pytest.approx(1.2)
    disease_gate_align = disease_training_cfg.train.loss.gate_best_branch_alignment
    assert disease_gate_align.enabled is True
    assert disease_gate_align.weight == pytest.approx(0.04)
    assert disease_gate_align.min_best_margin == pytest.approx(0.2)
    assert disease_gate_align.max_best_margin == pytest.approx(1.0)
    assert disease_gate_align.mismatch_margin_drop == pytest.approx(0.5)
    assert disease_gate_align.class_weighted is False
    assert disease_gate_align.label_weight_by_label == {
        "Normal": 1.5,
        "Lung_Parenchymal": 1.0,
        "Airway": 0.1,
    }
    assert disease_gate_align.max_best_margin_by_label == {"Airway": 1.2}
    negative_cap = disease_training_cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_direct_score.top_support_direct_path.negative_relative_cap
    assert negative_cap.enabled is True
    assert negative_cap.mode == "raw_fraction_cap"
    assert negative_cap.max_negative_fraction == pytest.approx(0.75)
    assert negative_cap.negative_cap == pytest.approx(1.5)
    assert negative_cap.max_negative_fraction_by_label == {"Airway": 0.5}
    assert negative_cap.negative_cap_by_label == {"Airway": 1.0}
    assert negative_cap.max_negative_fraction_by_class == pytest.approx(
        (0.75, 0.75, 0.5)
    )
    assert negative_cap.negative_cap_by_class == pytest.approx((1.5, 1.5, 1.0))
    assert (
        disease_training_cfg.train.loss.top_support_score_margin.label_weight_by_label
        == {
            "Normal": 1.0,
            "Lung_Parenchymal": 1.0,
            "Airway": 2.0,
        }
    )
    disease_hardness = (
        disease_training_cfg.train.loss.top_support_score_margin.hardness_weighting
    )
    assert disease_hardness.enabled is True
    assert disease_hardness.source == "top_support_gap"
    assert disease_hardness.mode == "negative_gap"
    assert disease_hardness.gain == pytest.approx(1.0)
    assert disease_hardness.cap == pytest.approx(3.0)
    assert (
        disease_training_cfg.train.loss.top_branch_margin.phase_weight_schedule.label_multiplier_by_label
        == {
            "Normal": 1.0,
            "Lung_Parenchymal": 1.0,
            "Airway": 2.0,
        }
    )
    assert (
        disease_training_cfg.train.loss.top_branch_margin.phase_weight_schedule.enabled
        is True
    )
    assert (
        disease_training_cfg.train.loss.top_branch_margin.phase_weight_schedule.start_epoch
        == 21
    )
    assert (
        disease_training_cfg.train.loss.top_branch_margin.phase_weight_schedule.end_epoch
        == 31
    )
    assert (
        disease_training_cfg.train.loss.top_branch_margin.phase_weight_schedule.start_multiplier_by_label
        == {
            "Normal": 1.0,
            "Lung_Parenchymal": 1.0,
            "Airway": 1.0,
        }
    )
    assert disease_training_cfg.train.loss.branch_direct_score_margin.enabled is True
    assert disease_training_cfg.train.loss.branch_direct_score_margin.weight == (
        pytest.approx(0.05)
    )
    assert (
        disease_training_cfg.train.loss.branch_path_dominance_constraint.enabled is True
    )
    assert (
        disease_training_cfg.train.loss.branch_path_dominance_constraint.label_weight_by_label
        == {
            "Normal": 0.75,
            "Lung_Parenchymal": 0.75,
            "Airway": 1.5,
        }
    )
    assert (
        disease_training_cfg.train.loss.branch_path_dominance_constraint.allowed_drop_by_label
        == {
            "Normal": 0.5,
            "Lung_Parenchymal": 0.75,
            "Airway": 0.3,
        }
    )
    disease_disagreement_cap = (
        disease_training_cfg.train.loss.branch_support_disagreement_cap_regularization
    )
    assert disease_disagreement_cap.enabled is True
    assert disease_disagreement_cap.embedding_gap_cap == pytest.approx(6.0)
    assert disease_disagreement_cap.interaction_gap_cap == pytest.approx(4.0)
    assert (
        disease_training_cfg.train.loss.class_evidence_gap_cap_regularization.label_weight_by_label
        == {
            "Normal": 1.0,
            "Lung_Parenchymal": 1.0,
            "Airway": 1.5,
        }
    )
    assert (
        disease_training_cfg.train.loss.class_evidence_gap_cap_regularization.negative_gap_cap_by_label
        == {
            "Normal": 3.0,
            "Lung_Parenchymal": 3.0,
            "Airway": 2.5,
        }
    )
    assert (
        disease_training_cfg.train.loss.class_evidence_positive_gap_cap_regularization.enabled
        is True
    )
    assert (
        disease_training_cfg.train.loss.class_evidence_positive_gap_cap_regularization.positive_gap_cap
        == pytest.approx(8.0)
    )
    assert (
        disease_training_cfg.train.loss.class_evidence_positive_gap_cap_regularization.weight
        == pytest.approx(0.015)
    )
    assert disease_training_cfg.train.loss.interaction_gap_cap_regularization.enabled
    assert (
        disease_training_cfg.train.loss.interaction_gap_cap_regularization.gap_cap
        == pytest.approx(6.0)
    )
    assert (
        disease_training_cfg.train.loss.interaction_gap_cap_regularization.weight
        == pytest.approx(0.015)
    )
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
    assert cfg.model.classifier.fusion_projector.type == "linear"
    assert cfg.model.classifier.fusion_projector.hidden_dim == 640
    assert cfg.model.classifier.fusion_projector.layer_norm is False
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
    assert cfg.train.loss.class_evidence_margin.temperature == 1.0
    assert cfg.train.loss.class_weighting.power == 0.5
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
    assert cfg.train.loss.gate_bad_branch_suppression.enabled is False
    assert cfg.train.loss.gate_bad_branch_suppression.weight == 0.0
    assert cfg.train.loss.gate_bad_branch_suppression.target == "true_class_gate"
    assert cfg.train.loss.gate_bad_branch_suppression.source == "branch_logits"
    assert cfg.train.loss.gate_bad_branch_suppression.mode == "margin_below_threshold"
    assert (
        cfg.train.loss.gate_bad_branch_suppression.margin_mode
        == "true_vs_hardest_negative"
    )
    assert cfg.train.loss.gate_bad_branch_suppression.bad_margin_threshold == 0.0
    assert (
        dict(cfg.train.loss.gate_bad_branch_suppression.bad_margin_threshold_by_label)
        == {}
    )
    assert cfg.train.loss.gate_bad_branch_suppression.warmup_epochs == 0
    assert cfg.train.loss.gate_bad_branch_suppression.weight_schedule.enabled is False
    assert (
        cfg.train.loss.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats.enabled
        is False
    )
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


def test_classifier_fusion_projector_mlp_config_loads(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["classifier"]["fusion_projector"] = {
        "type": "mlp",
        "hidden_dim": 640,
        "layer_norm": True,
    }
    config_path = _write_json(tmp_path / "fusion_projector_mlp.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    fusion_projector = cfg.model.classifier.fusion_projector
    assert fusion_projector.type == "mlp"
    assert fusion_projector.hidden_dim == 640
    assert fusion_projector.layer_norm is True


@pytest.mark.parametrize(
    ("field_name", "field_value", "error"),
    [
        ("type", "residual_mlp", "fusion_projector.type"),
        ("hidden_dim", 0, "fusion_projector.hidden_dim"),
        ("hidden_dim", "640", "fusion_projector.hidden_dim"),
        ("layer_norm", "true", "fusion_projector.layer_norm"),
    ],
)
def test_invalid_classifier_fusion_projector_config_is_rejected(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    error: str,
) -> None:
    payload = _base_payload()
    payload["model"]["classifier"]["fusion_projector"] = {
        "type": "mlp",
        "hidden_dim": 640,
        "layer_norm": True,
    }
    payload["model"]["classifier"]["fusion_projector"][field_name] = field_value
    config_path = _write_json(
        tmp_path / f"bad_fusion_projector_{field_name}.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


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
    assert class_gate.scorer_hidden_size is None
    assert class_gate.scorer_dropout == 0.0
    assert class_gate.global_residual.enabled is True
    assert class_gate.global_residual.init_scale == 0.1
    assert class_gate.global_residual.learnable is True
    assert class_gate.global_residual.bounding.enabled is False
    assert class_gate.global_residual.bounding.bound == 1.0
    assert class_gate.global_residual.bounding.temperature == 1.0
    assert class_gate.global_residual.warmup.enabled is False
    assert class_gate.global_residual.warmup.mode == "zero_to_learned"
    assert class_gate.global_residual.warmup.start_multiplier == 0.0
    assert class_gate.global_residual.warmup.end_multiplier == 1.0
    assert class_gate.evidence_auxiliary.enabled is False
    assert class_gate.evidence_auxiliary.weight == 0.1
    assert class_gate.evidence_scorer.type == "embedding_mlp"
    assert class_gate.evidence_scorer.embedding_hidden_size == 512
    assert class_gate.evidence_scorer.branch_hidden_size == 64
    assert class_gate.evidence_scorer.fusion_hidden_size == 512
    assert class_gate.evidence_scorer.dropout == pytest.approx(0.05)
    assert class_gate.evidence_scorer.branch_feature_transform.mode == "tanh"
    assert (
        class_gate.evidence_scorer.branch_feature_transform.temperature
        == pytest.approx(1.0)
    )
    assert class_gate.branch_logit_feature.mode == "raw"
    assert class_gate.gate_mixing.enabled is False


def test_class_aware_two_tower_evidence_scorer_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "two_tower_mlp",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "dropout": 0.2,
        "branch_feature_transform": {
            "mode": "tanh",
            "temperature": 0.75,
        },
    }
    config_path = _write_json(tmp_path / "two_tower_scorer.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    loaded = cfg.model.encoder.architecture.evidence_pooling.class_gate
    assert loaded.evidence_scorer.type == "two_tower_mlp"
    assert loaded.evidence_scorer.embedding_hidden_size == 32
    assert loaded.evidence_scorer.branch_hidden_size == 8
    assert loaded.evidence_scorer.fusion_hidden_size == 24
    assert loaded.evidence_scorer.dropout == pytest.approx(0.2)
    assert loaded.evidence_scorer.branch_feature_transform.mode == "tanh"
    assert loaded.evidence_scorer.branch_feature_transform.temperature == pytest.approx(
        0.75
    )
    assert loaded.gate_mixing.mode == "uniform_to_learned"
    assert loaded.gate_mixing.start_alpha == 1.0
    assert loaded.gate_mixing.end_alpha == 0.0


def test_class_aware_class_axis_attention_evidence_scorer_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "dropout": 0.2,
        "use_class_embedding": True,
        "logit_centering": True,
        "score_decomposition": {
            "enabled": True,
            "branch_scale_mode": "bounded_sigmoid",
            "branch_scale_min": 0.2,
            "branch_scale_init": 0.5,
            "branch_scale_max": 1.0,
            "interaction_scale_mode": "bounded_sigmoid",
            "interaction_scale_min": 0.1,
            "interaction_scale_init": 0.8,
            "interaction_scale_max": 1.5,
            "interaction_scale_schedule": {
                "enabled": True,
                "start_epoch": 11,
                "end_epoch": 30,
                "start_multiplier": 0.2,
                "end_multiplier": 1.0,
            },
        },
        "branch_feature_transform": {
            "mode": "tanh",
            "temperature": 0.75,
        },
    }
    class_gate["global_residual"]["correction"] = {
        "mode": "gated_zero_mean",
        "gate_hidden_size": 16,
        "dropout": 0.1,
        "zero_mean": True,
        "rebound": True,
        "confidence_aware_gate": {
            "enabled": True,
            "source": "evidence_gap",
            "mode": "damped_sigmoid",
            "temperature": 1.25,
            "damping": 0.6,
        },
    }
    config_path = _write_json(tmp_path / "class_axis_attention_scorer.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    loaded = cfg.model.encoder.architecture.evidence_pooling.class_gate
    assert loaded.evidence_scorer.type == "class_axis_attention"
    assert loaded.evidence_scorer.embedding_hidden_size == 32
    assert loaded.evidence_scorer.branch_hidden_size == 8
    assert loaded.evidence_scorer.fusion_hidden_size == 24
    assert loaded.evidence_scorer.num_attention_heads == 4
    assert loaded.evidence_scorer.num_attention_layers == 1
    assert loaded.evidence_scorer.use_class_embedding is True
    assert loaded.evidence_scorer.logit_centering is True
    assert loaded.evidence_scorer.score_decomposition.enabled is True
    assert loaded.evidence_scorer.score_decomposition.branch_scale_mode == (
        "bounded_sigmoid"
    )
    assert loaded.evidence_scorer.score_decomposition.branch_scale_min == (
        pytest.approx(0.2)
    )
    assert loaded.evidence_scorer.score_decomposition.branch_scale_init == (
        pytest.approx(0.5)
    )
    assert loaded.evidence_scorer.score_decomposition.branch_scale_max == (
        pytest.approx(1.0)
    )
    assert loaded.evidence_scorer.score_decomposition.interaction_scale_mode == (
        "bounded_sigmoid"
    )
    assert loaded.evidence_scorer.score_decomposition.interaction_scale_min == (
        pytest.approx(0.1)
    )
    assert loaded.evidence_scorer.score_decomposition.interaction_scale_init == (
        pytest.approx(0.8)
    )
    assert loaded.evidence_scorer.score_decomposition.interaction_scale_max == (
        pytest.approx(1.5)
    )
    schedule = loaded.evidence_scorer.score_decomposition.interaction_scale_schedule
    assert schedule.enabled is True
    assert schedule.start_epoch == 11
    assert schedule.end_epoch == 30
    assert schedule.start_multiplier == pytest.approx(0.2)
    assert schedule.end_multiplier == pytest.approx(1.0)
    assert loaded.global_residual.correction.mode == "gated_zero_mean"
    assert loaded.global_residual.correction.gate_hidden_size == 16
    assert loaded.global_residual.correction.dropout == pytest.approx(0.1)
    assert loaded.global_residual.correction.zero_mean is True
    assert loaded.global_residual.correction.rebound is True
    confidence_gate = loaded.global_residual.correction.confidence_aware_gate
    assert confidence_gate.enabled is True
    assert confidence_gate.source == "evidence_gap"
    assert confidence_gate.mode == "damped_sigmoid"
    assert confidence_gate.temperature == pytest.approx(1.25)
    assert confidence_gate.damping == pytest.approx(0.6)


def test_class_aware_two_tower_evidence_scorer_and_bounding_load(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "two_tower_mlp",
        "embedding_hidden_size": 512,
        "branch_hidden_size": 64,
        "fusion_hidden_size": 512,
        "dropout": 0.05,
        "branch_feature_transform": {"mode": "tanh", "temperature": 1.0},
    }
    class_gate["global_residual"]["bounding"] = {
        "enabled": True,
        "bound": 1.0,
        "temperature": 0.75,
    }
    config_path = _write_json(
        tmp_path / "evidence_scorer_branch_features.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    loaded = cfg.model.encoder.architecture.evidence_pooling.class_gate
    assert loaded.evidence_scorer.type == "two_tower_mlp"
    assert loaded.evidence_scorer.embedding_hidden_size == 512
    assert loaded.evidence_scorer.branch_hidden_size == 64
    assert loaded.evidence_scorer.fusion_hidden_size == 512
    assert loaded.global_residual.bounding.enabled is True
    assert loaded.global_residual.bounding.bound == pytest.approx(1.0)
    assert loaded.global_residual.bounding.temperature == pytest.approx(0.75)


def test_class_aware_two_tower_evidence_scorer_rejects_invalid_transform(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "two_tower_mlp",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "dropout": 0.0,
        "branch_feature_transform": {"mode": "identity", "temperature": 1.0},
    }
    config_path = _write_json(
        tmp_path / "bad_two_tower_transform.json",
        payload,
    )

    with pytest.raises(ValueError, match="branch_feature_transform.mode"):
        JsonConfigLoader.load_training(config_path)


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
        (("class_gate", "scorer_hidden_size"), 0, "scorer_hidden_size"),
        (("class_gate", "scorer_hidden_size"), "32", "scorer_hidden_size"),
        (("class_gate", "scorer_dropout"), 1.0, "scorer_dropout"),
        (("class_gate", "scorer_dropout"), "0.1", "scorer_dropout"),
        (("class_gate", "evidence_scorer", "type"), "concat", "evidence_scorer.type"),
        (
            ("class_gate", "evidence_scorer", "embedding_hidden_size"),
            0,
            "embedding_hidden_size",
        ),
        (
            ("class_gate", "evidence_scorer", "branch_hidden_size"),
            "64",
            "branch_hidden_size",
        ),
        (
            ("class_gate", "evidence_scorer", "fusion_hidden_size"),
            0,
            "fusion_hidden_size",
        ),
        (("class_gate", "evidence_scorer", "dropout"), 1.0, "evidence_scorer.dropout"),
        (
            (
                "class_gate",
                "evidence_scorer",
                "branch_feature_transform",
                "mode",
            ),
            "identity",
            "branch_feature_transform.mode",
        ),
        (
            (
                "class_gate",
                "evidence_scorer",
                "branch_feature_transform",
                "temperature",
            ),
            0.0,
            "branch_feature_transform.temperature",
        ),
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
    if "evidence_scorer" in path:
        evidence_pooling["class_gate"]["evidence_scorer"] = {
            "type": "two_tower_mlp",
            "embedding_hidden_size": 512,
            "branch_hidden_size": 64,
            "fusion_hidden_size": 512,
            "dropout": 0.05,
            "branch_feature_transform": {
                "mode": "tanh",
                "temperature": 1.0,
            },
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


def test_valid_power_inverse_class_weighting_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    config_path = _write_json(tmp_path / "power_class_weighting.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.class_weighting.type == "power_inverse_frequency"
    assert cfg.train.loss.class_weighting.power == 0.75


def test_valid_class_evidence_margin_softplus_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.0,
        "target": "class_evidence_logits",
        "mode": "softplus_true_vs_hardest_negative",
        "major_class": None,
        "class_weighted": True,
        "reduction": "mean",
        "temperature": 1.0,
    }
    config_path = _write_json(tmp_path / "class_evidence_margin_softplus.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.class_evidence_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.mode == "softplus_true_vs_hardest_negative"
    assert margin_cfg.margin == 0.0
    assert margin_cfg.temperature == 1.0
    assert margin_cfg.reduction == "mean"


def test_valid_branch_support_score_margin_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "dropout": 0.1,
        "use_class_embedding": True,
        "logit_centering": True,
        "score_decomposition": {
            "enabled": True,
            "branch_scale_mode": "bounded_sigmoid",
            "branch_scale_min": 0.5,
            "branch_scale_init": 0.7,
            "branch_scale_max": 1.5,
            "interaction_scale_mode": "bounded_sigmoid",
            "interaction_scale_min": 0.3,
            "interaction_scale_init": 0.7,
            "interaction_scale_max": 1.5,
        },
    }
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["branch_support_score_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_branch_support_scores",
        "mode": "softplus_true_vs_hardest_negative",
        "temperature": 1.0,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "branch_support_score_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.branch_support_score_margin
    assert margin_cfg.enabled is True
    assert margin_cfg.weight == pytest.approx(0.05)
    assert margin_cfg.class_weighted is True
    assert margin_cfg.warmup_epochs == 10


def test_valid_top_support_score_margin_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "dropout": 0.1,
        "use_class_embedding": True,
        "logit_centering": True,
        "score_decomposition": {"enabled": True},
        "branch_direct_score": {
            "enabled": True,
            "top_support_mode": "raw_existential_plus_relative_correction",
            "top_scale_mode": "bounded_sigmoid",
            "top_scale_min": 0.7,
            "top_scale_init": 1.0,
            "top_scale_max": 2.0,
            "gated_scale_mode": "bounded_sigmoid",
            "gated_scale_min": 0.0,
            "gated_scale_init": 0.5,
            "gated_scale_max": 1.5,
            "gate_reliability_mixture": {
                "enabled": True,
                "source": "top_vs_gated_margin_regret",
                "mode": "exp_neg_regret",
                "temperature": 1.0,
                "tolerance": 0.05,
                "detach": True,
            },
            "top_relative_correction": {
                "enabled": True,
                "positive_scale": 0.2,
                "negative_scale": 0.05,
                "negative_clip": 1.0,
            },
        },
    }
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["top_support_score_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_top_support_scores",
        "mode": "softplus_true_vs_hardest_negative",
        "temperature": 1.0,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
        "label_weight_by_label": {
            "normal": 1.0,
            "crackle": 1.5,
            "wheeze": 2.0,
        },
        "support_conditioned_multiplier": {
            "enabled": True,
            "source": "top_branch_margin",
            "mode": "linear",
            "gain": 1.0,
            "cap": 2.0,
        },
        "hardness_weighting": {
            "enabled": True,
            "source": "top_support_gap",
            "mode": "negative_gap",
            "gain": 1.0,
            "cap": 3.0,
        },
    }
    config_path = _write_json(tmp_path / "top_support_score_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    direct_cfg = cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_direct_score
    margin_cfg = cfg.train.loss.top_support_score_margin
    assert direct_cfg.gate_reliability_mixture.enabled is True
    assert direct_cfg.top_support_mode == "raw_existential_plus_relative_correction"
    assert direct_cfg.top_relative_correction.enabled is True
    assert margin_cfg.enabled is True
    assert margin_cfg.target == "class_evidence_top_support_scores"
    assert margin_cfg.weight == pytest.approx(0.05)
    assert margin_cfg.class_weighted is True
    assert margin_cfg.warmup_epochs == 10
    assert margin_cfg.label_weight_by_label["wheeze"] == pytest.approx(2.0)
    assert margin_cfg.support_conditioned_multiplier.enabled is True
    assert margin_cfg.hardness_weighting.enabled is True
    assert margin_cfg.hardness_weighting.source == "top_support_gap"


def test_valid_class_evidence_gap_cap_regularization_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["class_evidence_gap_cap_regularization"] = {
        "enabled": True,
        "weight": 0.02,
        "target": "class_evidence_logits",
        "mode": "negative_gap_hinge",
        "negative_gap_cap": 3.0,
        "negative_gap_cap_by_label": {
            "normal": 3.0,
            "crackle": 3.0,
            "wheeze": 2.5,
        },
        "label_weight_by_label": {
            "normal": 1.0,
            "crackle": 1.0,
            "wheeze": 1.5,
        },
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "evidence_gap_cap.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    gap_cap_cfg = cfg.train.loss.class_evidence_gap_cap_regularization
    assert gap_cap_cfg.enabled is True
    assert gap_cap_cfg.weight == pytest.approx(0.02)
    assert gap_cap_cfg.target == "class_evidence_logits"
    assert gap_cap_cfg.mode == "negative_gap_hinge"
    assert gap_cap_cfg.negative_gap_cap == pytest.approx(3.0)
    assert gap_cap_cfg.negative_gap_cap_by_label == {
        "normal": 3.0,
        "crackle": 3.0,
        "wheeze": 2.5,
    }
    assert gap_cap_cfg.label_weight_by_label == {
        "normal": 1.0,
        "crackle": 1.0,
        "wheeze": 1.5,
    }
    assert gap_cap_cfg.class_weighted is True
    assert gap_cap_cfg.reduction == "mean"
    assert gap_cap_cfg.warmup_epochs == 10


def test_valid_positive_and_interaction_gap_cap_configs_load(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "dropout": 0.1,
        "use_class_embedding": True,
        "logit_centering": True,
        "score_decomposition": {"enabled": True},
    }
    payload["train"]["loss"]["class_evidence_positive_gap_cap_regularization"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "class_evidence_logits",
        "mode": "positive_gap_hinge",
        "positive_gap_cap": 8.0,
        "class_weighted": False,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["interaction_gap_cap_regularization"] = {
        "enabled": True,
        "weight": 0.01,
        "target": "class_evidence_interaction_scores",
        "mode": "absolute_gap_hinge",
        "gap_cap": 6.0,
        "class_weighted": False,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "positive_interaction_gap_cap.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    positive_cfg = cfg.train.loss.class_evidence_positive_gap_cap_regularization
    assert positive_cfg.enabled is True
    assert positive_cfg.target == "class_evidence_logits"
    assert positive_cfg.mode == "positive_gap_hinge"
    assert positive_cfg.positive_gap_cap == pytest.approx(8.0)
    assert positive_cfg.reduction == "mean"
    interaction_cfg = cfg.train.loss.interaction_gap_cap_regularization
    assert interaction_cfg.enabled is True
    assert interaction_cfg.target == "class_evidence_interaction_scores"
    assert interaction_cfg.mode == "absolute_gap_hinge"
    assert interaction_cfg.gap_cap == pytest.approx(6.0)
    assert interaction_cfg.reduction == "mean"


def test_valid_branch_direct_score_margin_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "dropout": 0.1,
        "use_class_embedding": True,
        "logit_centering": True,
        "score_decomposition": {"enabled": True},
        "branch_direct_score": {
            "enabled": True,
            "positive_weight_mode": "softplus",
            "top_scale_mode": "bounded_sigmoid",
            "top_scale_min": 0.7,
            "top_scale_init": 1.0,
            "top_scale_max": 2.0,
            "gated_scale_mode": "bounded_sigmoid",
            "gated_scale_min": 0.0,
            "gated_scale_init": 0.5,
            "gated_scale_max": 1.5,
            "residual_hidden_size": 128,
            "residual_scale_mode": "sigmoid_max",
            "residual_scale_init": 0.2,
            "residual_scale_max": 0.5,
            "residual_bound": 1.0,
            "residual_temperature": 1.0,
            "gate_reliability_mixture": {
                "enabled": True,
                "source": "top_vs_gated_margin_regret",
                "mode": "exp_neg_regret",
                "temperature": 1.0,
                "tolerance": 0.05,
                "detach": True,
            },
        },
    }
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["branch_direct_score_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_branch_direct_scores",
        "mode": "softplus_true_vs_hardest_negative",
        "temperature": 1.0,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "branch_direct_score_margin.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    direct_cfg = cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_direct_score
    margin_cfg = cfg.train.loss.branch_direct_score_margin
    assert direct_cfg.enabled is True
    assert direct_cfg.top_scale_init == pytest.approx(1.0)
    assert direct_cfg.gated_scale_init == pytest.approx(0.5)
    assert direct_cfg.gate_reliability_mixture.enabled is True
    assert margin_cfg.enabled is True
    assert margin_cfg.target == "class_evidence_branch_direct_scores"
    assert margin_cfg.weight == pytest.approx(0.05)
    assert margin_cfg.class_weighted is True
    assert margin_cfg.warmup_epochs == 10


def test_valid_teacher_relative_top_branch_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["class_top_branch_relative_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_top_branch_margin_features",
        "mode": "true_vs_hardest_negative_hinge",
        "margin": 0.3,
        "margin_by_label": {"wheeze": 0.5},
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
        "support_weighting": {
            "enabled": True,
            "source": "top_branch_margin",
            "mode": "linear",
            "gain": 0.5,
            "cap": 3.0,
        },
        "hardness_weighting": {
            "enabled": True,
            "source": "teacher_gap_deficit",
            "mode": "linear",
            "gain": 0.5,
            "cap": 2.0,
        },
        "hardness_weighting_by_label": {
            "wheeze": {
                "gain": 1.0,
                "cap": 3.0,
            }
        },
        "weak_positive_support_weighting": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "multiplier": 1.25,
            "support_band_by_label": {
                "wheeze": {
                    "min_support": 0.2,
                    "max_support": 1.2,
                }
            },
        },
        "weak_positive_margin_boost": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "boost": 0.2,
            "boost_by_label": {"wheeze": 0.4},
            "support_band_by_label": {
                "wheeze": {
                    "min_support": 0.2,
                    "max_support": 1.2,
                }
            },
        },
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"] = {
        "enabled": True,
        "weight": 0.075,
        "target": "class_top_branch_margin_features",
        "mode": "support_conditioned_min_gap",
        "base_min_gap": 0.0,
        "base_min_gap_by_label": {"wheeze": 0.2},
        "support_source": "top_branch_margin",
        "support_gain": 0.75,
        "support_gain_by_label": {"wheeze": 1.0},
        "support_cap": 2.0,
        "weak_positive_support_weighting": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "multiplier": 1.5,
            "support_band_by_label": {
                "wheeze": {
                    "min_support": 0.2,
                    "max_support": 1.2,
                }
            },
        },
        "weak_positive_target_boost": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "boost": 0.3,
            "boost_by_label": {"wheeze": 0.6},
            "support_band_by_label": {
                "wheeze": {
                    "min_support": 0.2,
                    "max_support": 1.2,
                }
            },
        },
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "teacher_relative_top_branch.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    relative_cfg = cfg.train.loss.class_top_branch_relative_margin
    assert relative_cfg.enabled is True
    assert relative_cfg.margin == pytest.approx(0.3)
    assert relative_cfg.margin_by_label == {"wheeze": 0.5}
    assert relative_cfg.support_weighting.source == "top_branch_margin"
    assert relative_cfg.support_weighting.gain == pytest.approx(0.5)
    assert relative_cfg.hardness_weighting.source == "teacher_gap_deficit"
    assert relative_cfg.hardness_weighting.mode == "linear"
    assert relative_cfg.hardness_weighting.gain == pytest.approx(0.5)
    assert relative_cfg.hardness_weighting.cap == pytest.approx(2.0)
    assert relative_cfg.hardness_weighting_by_label["wheeze"].gain == pytest.approx(1.0)
    assert relative_cfg.hardness_weighting_by_label["wheeze"].cap == pytest.approx(3.0)
    assert relative_cfg.weak_positive_support_weighting.enabled is True
    assert relative_cfg.weak_positive_support_weighting.min_support == pytest.approx(
        0.2
    )
    assert relative_cfg.weak_positive_support_weighting.max_support == pytest.approx(
        1.0
    )
    assert relative_cfg.weak_positive_support_weighting.multiplier == pytest.approx(
        1.25
    )
    assert relative_cfg.weak_positive_support_weighting.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    assert relative_cfg.weak_positive_margin_boost.enabled is True
    assert relative_cfg.weak_positive_margin_boost.boost == pytest.approx(0.2)
    assert relative_cfg.weak_positive_margin_boost.boost_by_label == {"wheeze": 0.4}
    assert relative_cfg.weak_positive_margin_boost.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    teacher_min_cfg = cfg.train.loss.top_teacher_gap_min_constraint
    assert teacher_min_cfg.enabled is True
    assert teacher_min_cfg.weight == pytest.approx(0.075)
    assert teacher_min_cfg.base_min_gap_by_label == {"wheeze": 0.2}
    assert teacher_min_cfg.support_source == "top_branch_margin"
    assert teacher_min_cfg.support_gain == pytest.approx(0.75)
    assert teacher_min_cfg.support_gain_by_label == {"wheeze": 1.0}
    assert teacher_min_cfg.support_cap == pytest.approx(2.0)
    assert teacher_min_cfg.weak_positive_support_weighting.enabled is True
    assert teacher_min_cfg.weak_positive_support_weighting.multiplier == pytest.approx(
        1.5
    )
    assert teacher_min_cfg.weak_positive_support_weighting.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)
    assert teacher_min_cfg.weak_positive_target_boost.enabled is True
    assert teacher_min_cfg.weak_positive_target_boost.boost == pytest.approx(0.3)
    assert teacher_min_cfg.weak_positive_target_boost.boost_by_label == {"wheeze": 0.6}
    assert teacher_min_cfg.weak_positive_target_boost.support_band_by_label[
        "wheeze"
    ].max_support == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("section", "updates", "error"),
    [
        (
            "class_top_branch_relative_margin",
            {"margin_by_label": {"ghost": 0.3}},
            "class_top_branch_relative_margin.margin_by_label",
        ),
        (
            "class_top_branch_relative_margin",
            {
                "hardness_weighting_by_label": {
                    "ghost": {
                        "gain": 1.0,
                        "cap": 3.0,
                    }
                }
            },
            "class_top_branch_relative_margin.hardness_weighting_by_label",
        ),
        (
            "class_top_branch_relative_margin",
            {
                "weak_positive_margin_boost": {
                    "enabled": True,
                    "min_support": 0.2,
                    "max_support": 1.0,
                    "boost": 0.2,
                    "support_band_by_label": {
                        "ghost": {
                            "min_support": 0.2,
                            "max_support": 1.2,
                        }
                    },
                }
            },
            "weak_positive_margin_boost.support_band_by_label",
        ),
        (
            "top_teacher_gap_min_constraint",
            {"support_gain_by_label": {"ghost": 1.0}},
            "top_teacher_gap_min_constraint.support_gain_by_label",
        ),
    ],
)
def test_invalid_teacher_relative_label_specific_keys_are_rejected(
    tmp_path: Path,
    section: str,
    updates: dict[str, object],
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["class_top_branch_relative_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_top_branch_margin_features",
        "mode": "true_vs_hardest_negative_hinge",
        "margin": 0.3,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"] = {
        "enabled": True,
        "weight": 0.075,
        "target": "class_top_branch_margin_features",
        "mode": "support_conditioned_min_gap",
        "base_min_gap": 0.0,
        "support_source": "top_branch_margin",
        "support_gain": 0.75,
        "support_cap": 2.0,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"][section].update(updates)
    config_path = _write_json(
        tmp_path / "bad_teacher_relative_label_keys.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("target", "class_evidence_logits", "class_top_branch_relative_margin.target"),
        ("mode", "softplus", "class_top_branch_relative_margin.mode"),
        ("margin", 0.0, "class_top_branch_relative_margin.margin"),
        ("weight", 0.0, "class_top_branch_relative_margin.weight"),
        ("warmup_epochs", -1, "class_top_branch_relative_margin.warmup_epochs"),
    ],
)
def test_invalid_class_top_branch_relative_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["class_top_branch_relative_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_top_branch_margin_features",
        "mode": "true_vs_hardest_negative_hinge",
        "margin": 0.3,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["class_top_branch_relative_margin"][field] = value
    config_path = _write_json(
        tmp_path / "bad_class_top_branch_relative_margin.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        (
            "support_weighting",
            "source",
            "same_as_source",
            "class_top_branch_relative_margin.support_weighting",
        ),
        (
            "support_weighting",
            "mode",
            "positive_linear",
            "class_top_branch_relative_margin.support_weighting",
        ),
        (
            "hardness_weighting",
            "source",
            "evidence_gap",
            "class_top_branch_relative_margin.hardness_weighting",
        ),
        (
            "hardness_weighting",
            "mode",
            "negative_gap",
            "class_top_branch_relative_margin.hardness_weighting",
        ),
        (
            "weak_positive_support_weighting",
            "max_support",
            0.1,
            "class_top_branch_relative_margin.weak_positive_support_weighting",
        ),
        (
            "weak_positive_support_weighting",
            "multiplier",
            0.9,
            "class_top_branch_relative_margin.weak_positive_support_weighting",
        ),
        (
            "weak_positive_margin_boost",
            "max_support",
            0.1,
            "class_top_branch_relative_margin.weak_positive_margin_boost",
        ),
        (
            "weak_positive_margin_boost",
            "boost",
            -0.1,
            "class_top_branch_relative_margin.weak_positive_margin_boost",
        ),
        (
            "weak_positive_margin_boost",
            "enabled",
            "true",
            "class_top_branch_relative_margin.weak_positive_margin_boost",
        ),
    ],
)
def test_invalid_class_top_branch_relative_margin_weighting_is_rejected(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["class_top_branch_relative_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_top_branch_margin_features",
        "mode": "true_vs_hardest_negative_hinge",
        "margin": 0.3,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
        "support_weighting": {
            "enabled": True,
            "source": "top_branch_margin",
            "mode": "linear",
            "gain": 0.5,
            "cap": 3.0,
        },
        "hardness_weighting": {
            "enabled": True,
            "source": "teacher_gap_deficit",
            "mode": "linear",
            "gain": 1.0,
            "cap": 3.0,
        },
        "weak_positive_support_weighting": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "multiplier": 1.25,
        },
        "weak_positive_margin_boost": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "boost": 0.2,
        },
    }
    payload["train"]["loss"]["class_top_branch_relative_margin"][section][field] = value
    config_path = _write_json(
        tmp_path / "bad_class_top_branch_relative_weighting.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("target", "class_evidence_logits", "top_teacher_gap_min_constraint.target"),
        ("mode", "true_vs_hardest_negative", "top_teacher_gap_min_constraint.mode"),
        (
            "support_source",
            "same_as_source",
            "top_teacher_gap_min_constraint.support_source",
        ),
        ("support_cap", 0.0, "top_teacher_gap_min_constraint.support_cap"),
        ("warmup_epochs", -1, "top_teacher_gap_min_constraint.warmup_epochs"),
    ],
)
def test_invalid_top_teacher_gap_min_constraint_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_top_branch_margin_features",
        "mode": "support_conditioned_min_gap",
        "base_min_gap": 0.0,
        "support_source": "top_branch_margin",
        "support_gain": 0.5,
        "support_cap": 2.0,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"][field] = value
    config_path = _write_json(
        tmp_path / "bad_top_teacher_gap_min_constraint.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "min_support",
            -0.1,
            "top_teacher_gap_min_constraint.weak_positive_support_weighting",
        ),
        (
            "max_support",
            0.2,
            "top_teacher_gap_min_constraint.weak_positive_support_weighting",
        ),
        (
            "multiplier",
            0.9,
            "top_teacher_gap_min_constraint.weak_positive_support_weighting",
        ),
        (
            "enabled",
            "true",
            "top_teacher_gap_min_constraint.weak_positive_support_weighting",
        ),
    ],
)
def test_invalid_top_teacher_gap_min_weak_positive_weighting_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_top_branch_margin_features",
        "mode": "support_conditioned_min_gap",
        "base_min_gap": 0.0,
        "support_source": "top_branch_margin",
        "support_gain": 0.75,
        "support_cap": 2.0,
        "weak_positive_support_weighting": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "multiplier": 1.5,
        },
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"][
        "weak_positive_support_weighting"
    ][field] = value
    config_path = _write_json(
        tmp_path / "bad_top_teacher_gap_min_weak_positive.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "min_support",
            -0.1,
            "top_teacher_gap_min_constraint.weak_positive_target_boost",
        ),
        (
            "max_support",
            0.2,
            "top_teacher_gap_min_constraint.weak_positive_target_boost",
        ),
        (
            "boost",
            -0.1,
            "top_teacher_gap_min_constraint.weak_positive_target_boost",
        ),
        (
            "enabled",
            "true",
            "top_teacher_gap_min_constraint.weak_positive_target_boost",
        ),
    ],
)
def test_invalid_top_teacher_gap_min_weak_positive_target_boost_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_top_branch_margin_features",
        "mode": "support_conditioned_min_gap",
        "base_min_gap": 0.0,
        "support_source": "top_branch_margin",
        "support_gain": 0.75,
        "support_cap": 2.0,
        "weak_positive_target_boost": {
            "enabled": True,
            "min_support": 0.2,
            "max_support": 1.0,
            "boost": 0.3,
        },
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["top_teacher_gap_min_constraint"][
        "weak_positive_target_boost"
    ][field] = value
    config_path = _write_json(
        tmp_path / "bad_top_teacher_gap_min_weak_positive_target_boost.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


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


def test_valid_branch_to_evidence_consistency_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_logits",
        "source": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
        "teacher_detach": True,
        "tolerance": 0.1,
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "branch_to_evidence.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    consistency_cfg = cfg.train.loss.branch_to_evidence_ranking_consistency
    assert consistency_cfg.enabled is True
    assert consistency_cfg.weight == pytest.approx(0.05)
    assert consistency_cfg.teacher_detach is True
    assert consistency_cfg.tolerance == pytest.approx(0.1)
    assert consistency_cfg.class_weighted is True
    assert consistency_cfg.reduction == "class_balanced_violating_mean"
    assert consistency_cfg.warmup_epochs == 10


def test_valid_top_branch_to_evidence_consistency_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "top_branch_margin",
        "mode": "true_vs_hardest_negative",
        "teacher_detach": True,
        "teacher_gap_cap": 1.0,
        "teacher_floor_by_label": {
            "normal": 0.0,
            "crackle": 0.2,
            "wheeze": 0.5,
        },
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "top_branch_to_evidence.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    consistency_cfg = cfg.train.loss.branch_to_evidence_ranking_consistency
    assert consistency_cfg.source == "top_branch_margin"
    assert consistency_cfg.teacher_gap_cap == pytest.approx(1.0)
    assert dict(consistency_cfg.teacher_floor_by_label) == {
        "normal": 0.0,
        "crackle": 0.2,
        "wheeze": 0.5,
    }


def test_valid_weighted_top_branch_to_evidence_consistency_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "top_branch_margin",
        "mode": "true_vs_hardest_negative",
        "teacher_detach": True,
        "teacher_gap_cap": 1.0,
        "teacher_floor_by_label": {
            "normal": 0.0,
            "crackle": 0.2,
            "wheeze": 0.5,
        },
        "support_weighting": {
            "enabled": True,
            "source": "top_branch_margin",
            "mode": "linear",
            "gain": 1.0,
            "cap": 2.0,
        },
        "hardness_weighting": {
            "enabled": True,
            "source": "evidence_gap",
            "mode": "negative_gap",
            "gain": 1.0,
            "cap": 3.0,
        },
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(
        tmp_path / "weighted_top_branch_to_evidence.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    consistency_cfg = cfg.train.loss.branch_to_evidence_ranking_consistency
    assert consistency_cfg.support_weighting.enabled is True
    assert consistency_cfg.support_weighting.gain == pytest.approx(1.0)
    assert consistency_cfg.support_weighting.cap == pytest.approx(2.0)
    assert consistency_cfg.hardness_weighting.enabled is True
    assert consistency_cfg.hardness_weighting.gain == pytest.approx(1.0)
    assert consistency_cfg.hardness_weighting.cap == pytest.approx(3.0)


def test_valid_teacher_distribution_kl_branch_to_evidence_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "class_top_branch_margin_features",
        "mode": "teacher_distribution_kl",
        "teacher_detach": True,
        "teacher_temperature": 0.7,
        "student_temperature": 1.0,
        "class_weighted": True,
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "teacher_distribution_kl.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    consistency_cfg = cfg.train.loss.branch_to_evidence_ranking_consistency
    assert consistency_cfg.source == "class_top_branch_margin_features"
    assert consistency_cfg.mode == "teacher_distribution_kl"
    assert consistency_cfg.teacher_temperature == pytest.approx(0.7)
    assert consistency_cfg.student_temperature == pytest.approx(1.0)
    assert consistency_cfg.teacher_detach is True
    assert consistency_cfg.class_weighted is True
    assert consistency_cfg.reduction == "mean"


def test_valid_true_label_anchored_softplus_branch_to_evidence_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "class_top_branch_margin_relative_features",
        "mode": "true_label_anchored_softplus",
        "teacher_detach": True,
        "temperature": 1.0,
        "support_weighting": {
            "enabled": True,
            "source": "same_as_source",
            "mode": "positive_linear",
            "gain": 1.0,
            "cap": 2.0,
        },
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "true_label_anchored_b2e.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    consistency_cfg = cfg.train.loss.branch_to_evidence_ranking_consistency
    assert consistency_cfg.source == "class_top_branch_margin_relative_features"
    assert consistency_cfg.mode == "true_label_anchored_softplus"
    assert consistency_cfg.temperature == pytest.approx(1.0)
    assert consistency_cfg.support_weighting.enabled is True
    assert consistency_cfg.support_weighting.source == "same_as_source"
    assert consistency_cfg.support_weighting.mode == "positive_linear"
    assert consistency_cfg.class_weighted is True
    assert consistency_cfg.reduction == "mean"


def test_class_evidence_margin_support_weighting_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.2,
        "margin": 0.5,
        "target": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": False,
        "support_weighting": {
            "enabled": True,
            "source": "top_branch_margin",
            "mode": "linear",
            "gain": 1.0,
            "cap": 2.0,
        },
    }
    config_path = _write_json(
        tmp_path / "class_evidence_margin_support_weighting.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    margin_cfg = cfg.train.loss.class_evidence_margin
    assert margin_cfg.support_weighting.enabled is True
    assert margin_cfg.support_weighting.source == "top_branch_margin"
    assert margin_cfg.support_weighting.mode == "linear"


def test_valid_global_residual_anti_veto_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["global_residual_anti_veto"] = {
        "enabled": True,
        "weight": 0.02,
        "target": "global_residual_logits",
        "reference": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
        "evidence_confidence_threshold": 0.0,
        "min_residual_gap": -0.5,
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "global_residual_anti_veto.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    anti_veto_cfg = cfg.train.loss.global_residual_anti_veto
    assert anti_veto_cfg.enabled is True
    assert anti_veto_cfg.weight == pytest.approx(0.02)
    assert anti_veto_cfg.min_residual_gap == pytest.approx(-0.5)
    assert anti_veto_cfg.class_weighted is True
    assert anti_veto_cfg.reduction == "class_balanced_violating_mean"
    assert anti_veto_cfg.warmup_epochs == 10


def test_valid_final_gap_anti_veto_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["global_residual_anti_veto"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "final_logits",
        "reference": "class_evidence_logits",
        "support_source": "top_branch_margin",
        "mode": "final_gap_preservation",
        "margin_mode": "true_vs_hardest_negative",
        "support_threshold": 0.3,
        "allowed_gap_drop": 0.3,
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "final_gap_anti_veto.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    anti_veto_cfg = cfg.train.loss.global_residual_anti_veto
    assert anti_veto_cfg.target == "final_logits"
    assert anti_veto_cfg.support_source == "top_branch_margin"
    assert anti_veto_cfg.mode == "final_gap_preservation"
    assert anti_veto_cfg.support_threshold == pytest.approx(0.3)
    assert anti_veto_cfg.allowed_gap_drop == pytest.approx(0.3)


def test_valid_residual_contradiction_regularization_config_loads(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "sqrt_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
    }
    payload["train"]["loss"]["residual_contradiction_regularization"] = {
        "enabled": True,
        "weight": 0.02,
        "target": "global_residual_logits",
        "reference": "class_evidence_logits",
        "mode": "opposite_gap_penalty",
        "margin_mode": "true_vs_hardest_negative",
        "evidence_gap_threshold": 0.0,
        "min_residual_gap": -0.3,
        "class_weighted": True,
        "reduction": "class_balanced_violating_mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(
        tmp_path / "residual_contradiction_regularization.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    residual_cfg = cfg.train.loss.residual_contradiction_regularization
    assert residual_cfg.enabled is True
    assert residual_cfg.weight == pytest.approx(0.02)
    assert residual_cfg.min_residual_gap == pytest.approx(-0.3)
    assert residual_cfg.class_weighted is True
    assert residual_cfg.reduction == "class_balanced_violating_mean"


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


def test_valid_gate_bad_branch_suppression_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_bad_branch_suppression"] = {
        "enabled": True,
        "weight": 0.025,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "margin_below_threshold",
        "margin_mode": "true_vs_hardest_negative",
        "bad_margin_threshold": 0.0,
        "bad_margin_threshold_by_label": {
            "normal": -0.2,
            "crackle": 0.1,
            "wheeze": 0.1,
        },
        "warmup_epochs": 15,
        "weight_schedule": {
            "enabled": True,
            "start_epoch": 16,
            "end_epoch": 25,
            "start_multiplier": 0.3,
            "end_multiplier": 1.0,
        },
        "auto_bad_margin_threshold_by_train_stats": {
            "enabled": True,
            "strategy": "ema_bad_gate_mass_controller",
            "start_epoch": 31,
            "update_interval_epochs": 1,
            "ema": 0.9,
            "step": 0.01,
            "target_bad_gate_mass_by_label": {
                "normal": 0.05,
                "crackle": 0.15,
                "wheeze": 0.1,
            },
            "min_threshold_by_label": {
                "normal": -0.2,
                "crackle": 0.0,
                "wheeze": 0.0,
            },
            "max_threshold_by_label": {
                "normal": -0.2,
                "crackle": 0.35,
                "wheeze": 0.3,
            },
        },
    }
    config_path = _write_json(tmp_path / "gate_bad_branch_suppression.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    bad_cfg = cfg.train.loss.gate_bad_branch_suppression
    assert bad_cfg.enabled is True
    assert bad_cfg.weight == 0.025
    assert bad_cfg.target == "true_class_gate"
    assert bad_cfg.source == "branch_logits"
    assert bad_cfg.mode == "margin_below_threshold"
    assert bad_cfg.margin_mode == "true_vs_hardest_negative"
    assert bad_cfg.bad_margin_threshold == 0.0
    assert dict(bad_cfg.bad_margin_threshold_by_label) == {
        "normal": -0.2,
        "crackle": 0.1,
        "wheeze": 0.1,
    }
    assert bad_cfg.warmup_epochs == 15
    assert bad_cfg.weight_schedule.enabled is True
    assert bad_cfg.weight_schedule.start_epoch == 16
    assert bad_cfg.weight_schedule.end_epoch == 25
    assert bad_cfg.weight_schedule.start_multiplier == 0.3
    assert bad_cfg.weight_schedule.end_multiplier == 1.0
    auto_cfg = bad_cfg.auto_bad_margin_threshold_by_train_stats
    assert auto_cfg.enabled is True
    assert auto_cfg.strategy == "ema_bad_gate_mass_controller"
    assert auto_cfg.step == 0.01
    assert dict(auto_cfg.min_threshold_by_label)["normal"] == -0.2


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
        ("margin", -0.1, "class_evidence_margin.margin"),
        ("temperature", 0.0, "class_evidence_margin.temperature"),
        ("temperature", -1.0, "class_evidence_margin.temperature"),
        ("temperature", "cold", "class_evidence_margin.temperature"),
        (
            "reduction",
            "class_balanced_violating_mean",
            "class_evidence_margin.reduction",
        ),
    ],
)
def test_invalid_softplus_class_evidence_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_evidence_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "margin": 0.0,
        "target": "class_evidence_logits",
        "mode": "softplus_true_vs_hardest_negative",
        "class_weighted": False,
        "reduction": "mean",
        "temperature": 1.0,
    }
    payload["train"]["loss"]["class_evidence_margin"][field] = value
    config_path = _write_json(
        tmp_path / "bad_softplus_class_evidence_margin.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "branch_support_score_margin.enabled"),
        ("weight", 0.0, "branch_support_score_margin.weight"),
        ("target", "class_evidence_logits", "branch_support_score_margin.target"),
        ("mode", "true_vs_hardest_negative", "branch_support_score_margin.mode"),
        ("temperature", 0.0, "branch_support_score_margin.temperature"),
        ("class_weighted", "yes", "branch_support_score_margin.class_weighted"),
        (
            "reduction",
            "class_balanced_violating_mean",
            "branch_support_score_margin.reduction",
        ),
        ("warmup_epochs", -1, "branch_support_score_margin.warmup_epochs"),
    ],
)
def test_invalid_branch_support_score_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "score_decomposition": {"enabled": True},
    }
    payload["train"]["loss"]["branch_support_score_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_branch_support_scores",
        "mode": "softplus_true_vs_hardest_negative",
        "temperature": 1.0,
        "class_weighted": False,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["branch_support_score_margin"][field] = value
    config_path = _write_json(
        tmp_path / "bad_branch_support_score_margin.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "branch_direct_score_margin.enabled"),
        ("weight", 0.0, "branch_direct_score_margin.weight"),
        ("target", "class_evidence_logits", "branch_direct_score_margin.target"),
        ("mode", "true_vs_hardest_negative", "branch_direct_score_margin.mode"),
        ("temperature", 0.0, "branch_direct_score_margin.temperature"),
        ("class_weighted", "yes", "branch_direct_score_margin.class_weighted"),
        (
            "reduction",
            "class_balanced_violating_mean",
            "branch_direct_score_margin.reduction",
        ),
        ("warmup_epochs", -1, "branch_direct_score_margin.warmup_epochs"),
    ],
)
def test_invalid_branch_direct_score_margin_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    class_gate = payload["model"]["encoder"]["architecture"]["evidence_pooling"][
        "class_gate"
    ]
    class_gate["evidence_scorer"] = {
        "type": "class_axis_attention",
        "embedding_hidden_size": 32,
        "branch_hidden_size": 8,
        "fusion_hidden_size": 24,
        "num_attention_heads": 4,
        "num_attention_layers": 1,
        "score_decomposition": {"enabled": True},
        "branch_direct_score": {"enabled": True},
    }
    payload["train"]["loss"]["branch_direct_score_margin"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_branch_direct_scores",
        "mode": "softplus_true_vs_hardest_negative",
        "temperature": 1.0,
        "class_weighted": False,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["branch_direct_score_margin"][field] = value
    config_path = _write_json(
        tmp_path / "bad_branch_direct_score_margin.json",
        payload,
    )

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


def test_valid_gate_best_branch_alignment_config_loads(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["gate_best_branch_alignment"] = {
        "enabled": True,
        "weight": 0.03,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "weak_positive_best_branch_alignment",
        "margin_mode": "true_vs_hardest_negative",
        "min_best_margin": 0.2,
        "min_best_margin_by_label": {"wheeze": 0.2},
        "max_best_margin": 1.0,
        "max_best_margin_by_label": {"wheeze": 1.2},
        "mismatch_margin_drop": 0.5,
        "mismatch_margin_drop_by_label": {"wheeze": 0.4},
        "label_weight_by_label": {"normal": 1.0, "crackle": 0.75, "wheeze": 0.25},
        "loss": "negative_log_best_gate",
        "detach_branch_margin": True,
        "class_weighted": False,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(tmp_path / "gate_best_branch_alignment.json", payload)

    cfg = JsonConfigLoader.load_training(config_path)

    align_cfg = cfg.train.loss.gate_best_branch_alignment
    assert align_cfg.enabled is True
    assert align_cfg.weight == pytest.approx(0.03)
    assert align_cfg.target == "true_class_gate"
    assert align_cfg.source == "branch_logits"
    assert align_cfg.mode == "weak_positive_best_branch_alignment"
    assert align_cfg.min_best_margin == pytest.approx(0.2)
    assert align_cfg.min_best_margin_by_label == {"wheeze": 0.2}
    assert align_cfg.max_best_margin == pytest.approx(1.0)
    assert align_cfg.max_best_margin_by_label == {"wheeze": 1.2}
    assert align_cfg.mismatch_margin_drop == pytest.approx(0.5)
    assert align_cfg.mismatch_margin_drop_by_label == {"wheeze": 0.4}
    assert align_cfg.label_weight_by_label == {
        "normal": 1.0,
        "crackle": 0.75,
        "wheeze": 0.25,
    }
    assert align_cfg.detach_branch_margin is True
    assert align_cfg.class_weighted is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "gate_best_branch_alignment.enabled"),
        ("weight", 0.0, "gate_best_branch_alignment.weight"),
        ("target", "class_gate", "gate_best_branch_alignment.target"),
        ("source", "margin", "gate_best_branch_alignment.source"),
        ("mode", "best_branch_ce", "gate_best_branch_alignment.mode"),
        ("margin_mode", "minority_vs_major", "gate_best_branch_alignment.margin_mode"),
        ("min_best_margin", -0.1, "gate_best_branch_alignment.min_best_margin"),
        ("max_best_margin", 0.0, "gate_best_branch_alignment.max_best_margin"),
        (
            "label_weight_by_label",
            {"ghost": 1.0},
            "gate_best_branch_alignment.label_weight_by_label",
        ),
        (
            "max_best_margin_by_label",
            {"ghost": 1.2},
            "gate_best_branch_alignment.max_best_margin_by_label",
        ),
        (
            "max_best_margin_by_label",
            {"normal": 0.1},
            "gate_best_branch_alignment max_best_margin",
        ),
        (
            "mismatch_margin_drop_by_label",
            {"ghost": 0.4},
            "gate_best_branch_alignment.mismatch_margin_drop_by_label",
        ),
        (
            "mismatch_margin_drop",
            -0.1,
            "gate_best_branch_alignment.mismatch_margin_drop",
        ),
        ("loss", "ce", "gate_best_branch_alignment.loss"),
        (
            "detach_branch_margin",
            "true",
            "gate_best_branch_alignment.detach_branch_margin",
        ),
        ("class_weighted", "true", "gate_best_branch_alignment.class_weighted"),
        (
            "reduction",
            "class_balanced_violating_mean",
            "gate_best_branch_alignment.reduction",
        ),
        ("warmup_epochs", -1, "gate_best_branch_alignment.warmup_epochs"),
    ],
)
def test_invalid_gate_best_branch_alignment_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["class_weighting"] = {
        "enabled": True,
        "type": "power_inverse_frequency",
        "normalize": "mean_one",
        "source": "train",
        "power": 0.75,
    }
    payload["train"]["loss"]["gate_best_branch_alignment"] = {
        "enabled": True,
        "weight": 0.03,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "weak_positive_best_branch_alignment",
        "margin_mode": "true_vs_hardest_negative",
        "min_best_margin": 0.2,
        "max_best_margin": 1.0,
        "mismatch_margin_drop": 0.5,
        "loss": "negative_log_best_gate",
        "detach_branch_margin": True,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    payload["train"]["loss"]["gate_best_branch_alignment"][field] = value
    config_path = _write_json(tmp_path / "bad_gate_best_branch_alignment.json", payload)

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


def test_gate_best_branch_alignment_class_weighted_requires_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_best_branch_alignment"] = {
        "enabled": True,
        "weight": 0.03,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "weak_positive_best_branch_alignment",
        "margin_mode": "true_vs_hardest_negative",
        "min_best_margin": 0.2,
        "max_best_margin": 1.0,
        "mismatch_margin_drop": 0.5,
        "loss": "negative_log_best_gate",
        "detach_branch_margin": True,
        "class_weighted": True,
        "reduction": "mean",
        "warmup_epochs": 10,
    }
    config_path = _write_json(
        tmp_path / "bad_gate_best_branch_alignment_class_weighted.json",
        payload,
    )

    with pytest.raises(ValueError, match="gate_best_branch_alignment.class_weighted"):
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "gate_bad_branch_suppression.enabled"),
        ("weight", 0.0, "gate_bad_branch_suppression.weight"),
        ("weight", -0.001, "gate_bad_branch_suppression.weight"),
        ("target", "class_evidence_gate", "gate_bad_branch_suppression.target"),
        ("source", "branch_logit_margin", "gate_bad_branch_suppression.source"),
        ("mode", "best_margin_regret", "gate_bad_branch_suppression.mode"),
        (
            "margin_mode",
            "minority_vs_major",
            "gate_bad_branch_suppression.margin_mode",
        ),
        (
            "bad_margin_threshold",
            "low",
            "gate_bad_branch_suppression.bad_margin_threshold",
        ),
        ("warmup_epochs", -1, "gate_bad_branch_suppression.warmup_epochs"),
        ("warmup_epochs", "ten", "gate_bad_branch_suppression.warmup_epochs"),
    ],
)
def test_invalid_gate_bad_branch_suppression_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_bad_branch_suppression"] = {
        "enabled": True,
        "weight": 0.025,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "margin_below_threshold",
        "margin_mode": "true_vs_hardest_negative",
        "bad_margin_threshold": 0.0,
        "warmup_epochs": 15,
    }
    payload["train"]["loss"]["gate_bad_branch_suppression"][field] = value
    config_path = _write_json(
        tmp_path / "bad_gate_bad_branch_suppression.json",
        payload,
    )

    with pytest.raises((TypeError, ValueError), match=error):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("nested", "error"),
    [
        (
            {"bad_margin_threshold_by_label": {"unknown": 0.1}},
            "bad_margin_threshold_by_label",
        ),
        (
            {"bad_margin_threshold_by_label": {"wheeze": "low"}},
            "bad_margin_threshold_by_label",
        ),
        (
            {"weight_schedule": {"enabled": "yes"}},
            "gate_bad_branch_suppression.weight_schedule.enabled",
        ),
        (
            {"weight_schedule": {"enabled": True, "start_epoch": 0}},
            "gate_bad_branch_suppression.weight_schedule.start_epoch",
        ),
        (
            {
                "weight_schedule": {
                    "enabled": True,
                    "start_epoch": 16,
                    "end_epoch": 15,
                }
            },
            "gate_bad_branch_suppression.weight_schedule.end_epoch",
        ),
        (
            {
                "auto_bad_margin_threshold_by_train_stats": {
                    "enabled": True,
                    "strategy": "bad",
                    "start_epoch": 31,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.01,
                    "target_bad_gate_mass_by_label": {
                        "normal": 0.05,
                        "crackle": 0.15,
                        "wheeze": 0.1,
                    },
                    "min_threshold_by_label": {
                        "normal": -0.2,
                        "crackle": 0.0,
                        "wheeze": 0.0,
                    },
                    "max_threshold_by_label": {
                        "normal": -0.2,
                        "crackle": 0.35,
                        "wheeze": 0.3,
                    },
                }
            },
            "auto_bad_margin_threshold_by_train_stats.strategy",
        ),
        (
            {
                "auto_bad_margin_threshold_by_train_stats": {
                    "enabled": True,
                    "strategy": "ema_bad_gate_mass_controller",
                    "start_epoch": 31,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.0,
                    "target_bad_gate_mass_by_label": {
                        "normal": 0.05,
                        "crackle": 0.15,
                        "wheeze": 0.1,
                    },
                    "min_threshold_by_label": {
                        "normal": -0.2,
                        "crackle": 0.0,
                        "wheeze": 0.0,
                    },
                    "max_threshold_by_label": {
                        "normal": -0.2,
                        "crackle": 0.35,
                        "wheeze": 0.3,
                    },
                }
            },
            "auto_bad_margin_threshold_by_train_stats.step",
        ),
        (
            {
                "bad_margin_threshold_by_label": {"wheeze": 0.4},
                "auto_bad_margin_threshold_by_train_stats": {
                    "enabled": True,
                    "strategy": "ema_bad_gate_mass_controller",
                    "start_epoch": 31,
                    "update_interval_epochs": 1,
                    "ema": 0.9,
                    "step": 0.01,
                    "target_bad_gate_mass_by_label": {
                        "normal": 0.05,
                        "crackle": 0.15,
                        "wheeze": 0.1,
                    },
                    "min_threshold_by_label": {
                        "normal": -0.2,
                        "crackle": 0.0,
                        "wheeze": 0.0,
                    },
                    "max_threshold_by_label": {
                        "normal": -0.2,
                        "crackle": 0.35,
                        "wheeze": 0.3,
                    },
                },
            },
            "initial threshold",
        ),
    ],
)
def test_invalid_gate_bad_branch_suppression_label_adaptive_config_is_rejected(
    tmp_path: Path,
    nested: dict[str, object],
    error: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["gate_bad_branch_suppression"] = {
        "enabled": True,
        "weight": 0.025,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "margin_below_threshold",
        "margin_mode": "true_vs_hardest_negative",
        "bad_margin_threshold": 0.0,
        "warmup_epochs": 15,
        **nested,
    }
    config_path = _write_json(
        tmp_path / "bad_gate_bad_branch_suppression_label_adaptive.json",
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


def test_invalid_branch_to_evidence_teacher_floor_config_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "top_branch_margin",
        "mode": "true_vs_hardest_negative",
        "teacher_gap_cap": 0.4,
        "teacher_floor_by_label": {"wheeze": 0.5},
    }
    config_path = _write_json(tmp_path / "bad_teacher_floor.json", payload)

    with pytest.raises(ValueError, match="teacher_floor_by_label.wheeze"):
        JsonConfigLoader.load_training(config_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("teacher_temperature", 0.0, "teacher_temperature"),
        ("student_temperature", -1.0, "student_temperature"),
    ],
)
def test_invalid_teacher_distribution_kl_temperature_is_rejected(
    tmp_path: Path,
    field: str,
    value: float,
    match: str,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "class_top_branch_margin_features",
        "mode": "teacher_distribution_kl",
        "teacher_temperature": 0.7,
        "student_temperature": 1.0,
    }
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"][field] = value
    config_path = _write_json(tmp_path / "bad_kl_temperature.json", payload)

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_training(config_path)


def test_teacher_distribution_kl_rejects_hard_margin_options(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.1,
        "target": "class_evidence_logits",
        "source": "class_top_branch_margin_features",
        "mode": "teacher_distribution_kl",
        "teacher_temperature": 0.7,
        "student_temperature": 1.0,
        "teacher_floor_by_label": {"wheeze": 0.5},
    }
    config_path = _write_json(tmp_path / "kl_with_hard_options.json", payload)

    with pytest.raises(ValueError, match="teacher_floor_by_label"):
        JsonConfigLoader.load_training(config_path)


def test_invalid_final_gap_anti_veto_config_is_rejected(tmp_path: Path) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["global_residual_anti_veto"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "global_residual_logits",
        "reference": "class_evidence_logits",
        "support_source": "top_branch_margin",
        "mode": "final_gap_preservation",
        "margin_mode": "true_vs_hardest_negative",
        "support_threshold": 0.3,
        "allowed_gap_drop": 0.3,
    }
    config_path = _write_json(tmp_path / "bad_final_gap_anti_veto.json", payload)

    with pytest.raises(ValueError, match="target.*final_logits"):
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


def test_class_weighted_branch_to_evidence_requires_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_logits",
        "source": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
    }
    config_path = _write_json(
        tmp_path / "class_weighted_b2e_without_weights.json",
        payload,
    )

    with pytest.raises(
        ValueError,
        match="branch_to_evidence_ranking_consistency.class_weighted",
    ):
        JsonConfigLoader.load_training(config_path)


def test_class_weighted_anti_veto_requires_class_weighting(
    tmp_path: Path,
) -> None:
    payload = _class_aware_cross_entropy_payload()
    payload["train"]["loss"]["global_residual_anti_veto"] = {
        "enabled": True,
        "weight": 0.02,
        "target": "global_residual_logits",
        "reference": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
        "class_weighted": True,
    }
    config_path = _write_json(
        tmp_path / "class_weighted_anti_veto_without_weights.json",
        payload,
    )

    with pytest.raises(
        ValueError,
        match="global_residual_anti_veto.class_weighted",
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


def test_branch_to_evidence_consistency_requires_class_aware_pooling(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["branch_to_evidence_ranking_consistency"] = {
        "enabled": True,
        "weight": 0.05,
        "target": "class_evidence_logits",
        "source": "class_gated_branch_logits",
        "mode": "true_vs_hardest_negative",
    }
    config_path = _write_json(tmp_path / "b2e_mean_pooling.json", payload)

    with pytest.raises(
        ValueError,
        match="branch_to_evidence_ranking_consistency requires",
    ):
        JsonConfigLoader.load_training(config_path)


def test_global_residual_anti_veto_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    payload["train"]["loss"]["global_residual_anti_veto"] = {
        "enabled": True,
        "weight": 0.02,
        "target": "global_residual_logits",
        "reference": "class_evidence_logits",
        "mode": "true_vs_hardest_negative",
    }
    config_path = _write_json(tmp_path / "anti_veto_bce.json", payload)

    with pytest.raises(ValueError, match="global_residual_anti_veto.*cross_entropy"):
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


def test_gate_bad_branch_suppression_requires_class_aware_pooling(
    tmp_path: Path,
) -> None:
    payload = _base_payload()
    payload["data"]["label_to_index"] = {"normal": 0, "crackle": 1, "wheeze": 2}
    payload["train"]["loss"]["type"] = "cross_entropy"
    payload["train"]["loss"]["auto_pos_weight"] = False
    payload["train"]["loss"]["pos_weight"] = None
    payload["train"]["loss"]["gate_bad_branch_suppression"] = {
        "enabled": True,
        "weight": 0.025,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "margin_below_threshold",
        "margin_mode": "true_vs_hardest_negative",
        "bad_margin_threshold": 0.0,
    }
    config_path = _write_json(
        tmp_path / "gate_bad_branch_suppression_mean_pooling.json",
        payload,
    )

    with pytest.raises(ValueError, match="gate_bad_branch_suppression requires"):
        JsonConfigLoader.load_training(config_path)


def test_gate_bad_branch_suppression_requires_cross_entropy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["model"]["encoder"]["architecture"]["evidence_pooling"] = (
        _class_aware_evidence_pooling_payload()
    )
    payload["train"]["loss"]["gate_bad_branch_suppression"] = {
        "enabled": True,
        "weight": 0.025,
        "target": "true_class_gate",
        "source": "branch_logits",
        "mode": "margin_below_threshold",
        "margin_mode": "true_vs_hardest_negative",
        "bad_margin_threshold": 0.0,
    }
    config_path = _write_json(
        tmp_path / "gate_bad_branch_suppression_bce.json",
        payload,
    )

    with pytest.raises(
        ValueError,
        match="gate_bad_branch_suppression.*cross_entropy",
    ):
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
        ("power", 0.0, "class_weighting.power"),
        ("power", -0.75, "class_weighting.power"),
        ("power", "strong", "class_weighting.power"),
    ],
)
def test_invalid_class_weighting_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
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
