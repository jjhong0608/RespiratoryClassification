from __future__ import annotations

import json
from dataclasses import asdict
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


def test_repo_example_configs_load() -> None:
    d_e_f_config_paths = [
        "training_d1_c3_3scale_stage1.json",
        "training_d1_c3_3scale_stage2.json",
        "training_d2_stage2_no_rdt_from_stage1.json",
        "training_d3_direct_b3_low_lr.json",
        "training_d4_c3_top1_stage2.json",
        "training_d4_c3_top3_stage2.json",
        "training_d5_c3_stage1_seed0.json",
        "training_d5_c3_stage2_seed0.json",
        "training_d5_c3_stage1_seed1.json",
        "training_d5_c3_stage2_seed1.json",
        "training_d5_c3_stage1_seed2.json",
        "training_d5_c3_stage2_seed2.json",
        "training_e1_c3_attention_logit_stage2.json",
        "training_e2_c3_instance_logit_stage2.json",
        "training_e3_c3_attention_temp05_stage2.json",
        "training_e4_c3_entropy001_stage2.json",
        "training_f1_c3_branch_aux_weights_stage2.json",
        "training_f2_c3_exclude_branch4_evidence_stage2.json",
    ]
    d_e_f_configs = [
        JsonConfigLoader.load_training(ROOT / "configs" / config_path)
        for config_path in d_e_f_config_paths
    ]
    g_config_paths = [
        "training_g1_d3_seed0.json",
        "training_g1_d3_seed1.json",
        "training_g1_d3_seed2.json",
        "training_g1_d3_seed42.json",
        "training_g1_d3_seed43.json",
        "training_g2_direct_low_lr_no_rdt.json",
        "training_g3_direct_low_lr_3scale.json",
        "training_g4_direct_low_lr_aux005.json",
        "training_g5_direct_low_lr_no_aux.json",
        "training_g6_direct_low_lr_focal_gamma1.json",
        "training_g7_direct_low_lr_focal_gamma2.json",
    ]
    g_configs = [
        JsonConfigLoader.load_training(ROOT / "configs" / config_path)
        for config_path in g_config_paths
    ]
    h0_gated_config_paths = [
        "training_h0_branch_gated_seed0.json",
        "training_h0_branch_gated_seed1.json",
        "training_h0_branch_gated_seed2.json",
        "training_h0_branch_gated_seed42.json",
        "training_h0_branch_gated_seed43.json",
    ]
    h0_gated_configs = [
        JsonConfigLoader.load_training(ROOT / "configs" / config_path)
        for config_path in h0_gated_config_paths
    ]
    h_config_paths = [
        "training_h1_no_rdt_seed0.json",
        "training_h1_no_rdt_seed1.json",
        "training_h1_no_rdt_seed2.json",
        "training_h1_no_rdt_seed42.json",
        "training_h1_no_rdt_seed43.json",
        "training_h2_no_aux_seed0.json",
        "training_h2_no_aux_seed1.json",
        "training_h2_no_aux_seed2.json",
        "training_h2_no_aux_seed42.json",
        "training_h2_no_aux_seed43.json",
        "training_h3_3scale_seed0.json",
        "training_h3_3scale_seed1.json",
        "training_h3_3scale_seed2.json",
        "training_h3_3scale_seed42.json",
        "training_h3_3scale_seed43.json",
    ]
    h_configs = [
        JsonConfigLoader.load_training(ROOT / "configs" / config_path)
        for config_path in h_config_paths
    ]
    aug_config_paths = [
        "training_aug0_h0_gated_no_aug.json",
        "training_aug1_h0_gated_waveform_aug.json",
        "training_aug2_h0_gated_fbank_aug.json",
        "training_aug3_h0_gated_waveform_fbank_aug.json",
    ]
    aug_configs = [
        JsonConfigLoader.load_training(ROOT / "configs" / config_path)
        for config_path in aug_config_paths
    ]
    oneof_token_config_paths = [
        "training_aug4_h0_gated_oneof.json",
        "training_pt1_h0_gated_oneof_branch_event_dropout.json",
        "training_pt2_h0_gated_oneof_selected_evidence_dropout.json",
        "training_pt3_h0_gated_oneof_both_token_dropouts.json",
    ]
    oneof_token_configs = [
        JsonConfigLoader.load_training(ROOT / "configs" / config_path)
        for config_path in oneof_token_config_paths
    ]
    c0_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c0_b3_focal_patience8.json"
    )
    c1_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c1_b3_bce_aux01_patience8.json"
    )
    c2_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c2_b1_bce_aux01_patience8.json"
    )
    c3_stage1_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c3_stage1_b1_bce_aux01.json"
    )
    c3_stage2_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c3_stage2_b3_from_stage1.json"
    )
    c4_3scale_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c4_3scale_bce_aux01_rdt3.json"
    )
    c4_4scale_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c4_4scale_bce_aux01_rdt3.json"
    )
    c5_top2_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c5_top2_bce_aux01_rdt3.json"
    )
    c5_top4_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_c5_top4_bce_aux01_rdt3.json"
    )
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
    pretrain_4class_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_4class_pretrain_weighted_ce_branch_bin_aux.json"
    )
    pretrain_4class_cosine_cfg = JsonConfigLoader.load_training(
        ROOT
        / "configs/training_4class_pretrain_branch_bin_cosine_040_010_monitor030.json"
    )
    pretrain_4class_sqrt_sampler_cfg = JsonConfigLoader.load_training(
        ROOT
        / "configs/training_4class_pretrain_branch_bin_cosine_040_010_monitor030_sqrt_sampler.json"
    )
    cv_cfg = JsonConfigLoader.load_cv(ROOT / "configs/cv_multiscale_rdt.json")
    eval_cfg = JsonConfigLoader.load_eval(ROOT / "configs/eval_multiscale_rdt.json")

    assert b0_cfg.model.encoder.architecture.rdt.enabled is False
    assert b1_cfg.train.loss.branch_auxiliary.enabled is True
    assert b2_cfg.model.encoder.architecture.rdt.steps == 2
    assert b3_cfg.model.encoder.architecture.rdt.steps == 3
    assert c0_cfg.train.loss.type == "focal"
    assert c1_cfg.train.loss.type == "bce"
    assert c2_cfg.model.encoder.architecture.rdt.enabled is False
    assert c3_stage1_cfg.model.encoder.architecture.rdt.enabled is False
    assert c3_stage2_cfg.train.initialization.checkpoint_path is None
    assert c3_stage2_cfg.train.initialization.strict is False
    assert c3_stage2_cfg.train.initialization.load_optimizer_state is False
    assert len(c4_3scale_cfg.model.encoder.architecture.patch_branches) == 3
    assert len(c4_4scale_cfg.model.encoder.architecture.patch_branches) == 4
    assert c5_top2_cfg.model.encoder.architecture.rdt.top_tokens_per_branch == 2
    assert c5_top4_cfg.model.encoder.architecture.rdt.top_tokens_per_branch == 4
    assert len(d_e_f_configs) == len(d_e_f_config_paths)
    assert d_e_f_configs[0].model.encoder.architecture.rdt.enabled is False
    assert len(d_e_f_configs[0].model.encoder.architecture.patch_branches) == 3
    assert d_e_f_configs[4].model.encoder.architecture.rdt.top_tokens_per_branch == 1
    assert d_e_f_configs[5].model.encoder.architecture.rdt.top_tokens_per_branch == 3
    assert d_e_f_configs[6].experiment.seed == 0
    assert d_e_f_configs[10].experiment.seed == 2
    assert (
        d_e_f_configs[12].model.encoder.architecture.rdt.evidence_score_source
        == "attention_logit"
    )
    assert (
        d_e_f_configs[13].model.encoder.architecture.rdt.evidence_score_source
        == "instance_logit"
    )
    assert d_e_f_configs[14].model.encoder.architecture.mil.attention_temperature == 0.5
    assert d_e_f_configs[15].train.loss.attention_entropy.enabled is True
    assert d_e_f_configs[16].train.loss.branch_auxiliary.weights == (
        0.1,
        0.1,
        0.1,
        0.03,
    )
    assert d_e_f_configs[
        17
    ].model.encoder.architecture.rdt.exclude_branches_from_evidence == (3,)
    assert len(g_configs) == len(g_config_paths)
    assert [cfg.experiment.seed for cfg in g_configs[:5]] == [0, 1, 2, 42, 43]
    assert all(cfg.train.initialization.checkpoint_path is None for cfg in g_configs)
    assert all(cfg.train.initialization.load_model_state is False for cfg in g_configs)
    assert all(cfg.train.initialization.strict is False for cfg in g_configs)
    assert all(
        cfg.train.initialization.load_optimizer_state is False for cfg in g_configs
    )
    assert g_configs[5].model.encoder.architecture.rdt.enabled is False
    assert len(g_configs[6].model.encoder.architecture.patch_branches) == 3
    assert g_configs[7].train.loss.branch_auxiliary.weight == 0.05
    assert g_configs[8].train.loss.branch_auxiliary.enabled is False
    assert g_configs[8].train.loss.branch_auxiliary.weight == 0.1
    assert g_configs[9].train.loss.type == "focal"
    assert g_configs[9].train.loss.gamma == 1.0
    assert g_configs[10].train.loss.type == "focal"
    assert g_configs[10].train.loss.gamma == 2.0
    assert len(h0_gated_configs) == len(h0_gated_config_paths)
    assert [cfg.experiment.seed for cfg in h0_gated_configs] == [0, 1, 2, 42, 43]
    assert all(
        cfg.model.encoder.architecture.evidence_pooling.type == "branch_gated"
        for cfg in h0_gated_configs
    )
    assert all(
        cfg.model.encoder.architecture.evidence_pooling.temperature == 1.0
        for cfg in h0_gated_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.enabled is True for cfg in h0_gated_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.steps == 3 for cfg in h0_gated_configs
    )
    assert all(
        cfg.train.loss.branch_auxiliary.enabled is True for cfg in h0_gated_configs
    )
    assert all(
        cfg.train.loss.branch_auxiliary.weight == 0.1 for cfg in h0_gated_configs
    )
    assert all(
        cfg.train.initialization.load_model_state is False for cfg in h0_gated_configs
    )
    assert len(h_configs) == len(h_config_paths)
    assert [cfg.experiment.seed for cfg in h_configs[:5]] == [0, 1, 2, 42, 43]
    assert [cfg.experiment.seed for cfg in h_configs[5:10]] == [0, 1, 2, 42, 43]
    assert [cfg.experiment.seed for cfg in h_configs[10:]] == [0, 1, 2, 42, 43]
    assert all(cfg.train.initialization.checkpoint_path is None for cfg in h_configs)
    assert all(cfg.train.initialization.load_model_state is False for cfg in h_configs)
    assert all(cfg.train.initialization.strict is False for cfg in h_configs)
    assert all(
        cfg.train.initialization.load_optimizer_state is False for cfg in h_configs
    )
    assert all(cfg.train.loss.type == "bce" for cfg in h_configs)
    assert all(cfg.train.optimizer.encoder_lr == 1e-5 for cfg in h_configs)
    assert all(cfg.train.optimizer.head_lr == 3e-4 for cfg in h_configs)
    assert all(cfg.train.early_stopping.patience == 8 for cfg in h_configs)
    assert all(
        cfg.model.encoder.architecture.rdt.top_tokens_per_branch == 2
        for cfg in h_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.enabled is False for cfg in h_configs[:5]
    )
    assert all(
        cfg.train.loss.branch_auxiliary.enabled is False for cfg in h_configs[5:10]
    )
    assert all(
        len(cfg.model.encoder.architecture.patch_branches) == 3
        for cfg in h_configs[10:]
    )
    assert len(aug_configs) == len(aug_config_paths)
    assert [cfg.experiment.name for cfg in aug_configs] == [
        "respiratory_aug0_h0_gated_no_aug",
        "respiratory_aug1_h0_gated_waveform_aug",
        "respiratory_aug2_h0_gated_fbank_aug",
        "respiratory_aug3_h0_gated_waveform_fbank_aug",
    ]
    assert aug_configs[0].data.augmentation.enabled is False
    assert aug_configs[1].data.augmentation.waveform.enabled is True
    assert aug_configs[1].data.augmentation.fbank.enabled is False
    assert aug_configs[2].data.augmentation.waveform.enabled is False
    assert aug_configs[2].data.augmentation.fbank.enabled is True
    assert aug_configs[3].data.augmentation.waveform.enabled is True
    assert aug_configs[3].data.augmentation.fbank.enabled is True
    assert all(
        cfg.model.encoder.architecture.evidence_pooling.type == "branch_gated"
        for cfg in aug_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.enabled is True for cfg in aug_configs
    )
    assert all(cfg.model.encoder.architecture.rdt.steps == 3 for cfg in aug_configs)
    assert all(
        cfg.model.encoder.architecture.rdt.top_tokens_per_branch == 2
        for cfg in aug_configs
    )
    assert all(cfg.train.loss.type == "bce" for cfg in aug_configs)
    assert all(cfg.train.loss.branch_auxiliary.enabled is True for cfg in aug_configs)
    assert all(cfg.train.loss.branch_auxiliary.weight == 0.1 for cfg in aug_configs)
    assert all(cfg.train.optimizer.encoder_lr == 1e-5 for cfg in aug_configs)
    assert all(cfg.train.optimizer.head_lr == 3e-4 for cfg in aug_configs)
    assert all(cfg.train.initialization.checkpoint_path is None for cfg in aug_configs)
    assert all(
        cfg.train.initialization.load_model_state is False for cfg in aug_configs
    )
    assert len(oneof_token_configs) == len(oneof_token_config_paths)
    assert [cfg.experiment.name for cfg in oneof_token_configs] == [
        "respiratory_aug4_h0_gated_oneof",
        "respiratory_pt1_h0_gated_oneof_branch_event_dropout",
        "respiratory_pt2_h0_gated_oneof_selected_evidence_dropout",
        "respiratory_pt3_h0_gated_oneof_both_token_dropouts",
    ]
    assert all(
        cfg.data.augmentation.policy.type == "one_of" for cfg in oneof_token_configs
    )
    assert all(
        cfg.model.encoder.architecture.evidence_pooling.type == "branch_gated"
        for cfg in oneof_token_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.enabled is True
        for cfg in oneof_token_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.steps == 3 for cfg in oneof_token_configs
    )
    assert all(
        cfg.model.encoder.architecture.rdt.top_tokens_per_branch == 2
        for cfg in oneof_token_configs
    )
    assert all(cfg.train.loss.type == "bce" for cfg in oneof_token_configs)
    assert all(
        cfg.train.loss.branch_auxiliary.enabled is True for cfg in oneof_token_configs
    )
    assert (
        oneof_token_configs[
            0
        ].model.encoder.architecture.token_augmentation.branch_event_dropout.enabled
        is False
    )
    assert (
        oneof_token_configs[
            0
        ].model.encoder.architecture.token_augmentation.selected_evidence_dropout.enabled
        is False
    )
    assert (
        oneof_token_configs[
            1
        ].model.encoder.architecture.token_augmentation.branch_event_dropout.enabled
        is True
    )
    assert (
        oneof_token_configs[
            1
        ].model.encoder.architecture.token_augmentation.selected_evidence_dropout.enabled
        is False
    )
    assert (
        oneof_token_configs[
            2
        ].model.encoder.architecture.token_augmentation.branch_event_dropout.enabled
        is False
    )
    assert (
        oneof_token_configs[
            2
        ].model.encoder.architecture.token_augmentation.selected_evidence_dropout.enabled
        is True
    )
    assert (
        oneof_token_configs[
            3
        ].model.encoder.architecture.token_augmentation.branch_event_dropout.enabled
        is True
    )
    assert (
        oneof_token_configs[
            3
        ].model.encoder.architecture.token_augmentation.selected_evidence_dropout.enabled
        is True
    )
    assert training_cfg.model.encoder.type == "multiscale_rdt_ast"
    assert multiclass_cfg.train.loss.type == "cross_entropy"
    assert pretrain_4class_cfg.data.label_to_index == {
        "normal": 0,
        "crackle": 1,
        "wheeze": 2,
        "rhonchi": 3,
    }
    assert pretrain_4class_cfg.train.epochs == 120
    assert pretrain_4class_cfg.train.loss.class_weighting.enabled is True
    assert pretrain_4class_cfg.train.loss.label_smoothing.enabled is True
    assert pretrain_4class_cfg.train.loss.label_smoothing.value == 0.05
    assert pretrain_4class_cfg.train.loss.branch_auxiliary.enabled is False
    assert pretrain_4class_cfg.train.loss.branch_binary_auxiliary.enabled is True
    assert pretrain_4class_cfg.train.loss.branch_binary_auxiliary.weight == 0.3
    assert (
        pretrain_4class_cfg.train.loss.branch_binary_auxiliary.label_to_index["normal"]
        == 0
    )
    assert (
        pretrain_4class_cfg.model.encoder.architecture.evidence_pooling.type
        == "branch_gated"
    )
    assert pretrain_4class_cfg.train.early_stopping.enabled is False
    assert pretrain_4class_cfg.checkpointing is not None
    assert [monitor.name for monitor in pretrain_4class_cfg.checkpointing.monitors] == [
        "val_macro_f1",
        "val_macro_recall",
        "val_loss",
        "last",
    ]
    assert pretrain_4class_cosine_cfg.train.loss.branch_binary_auxiliary.enabled is True
    assert pretrain_4class_cosine_cfg.train.loss.label_smoothing.enabled is True
    assert pretrain_4class_cosine_cfg.train.loss.label_smoothing.value == 0.05
    assert (
        pretrain_4class_cosine_cfg.train.loss.branch_binary_auxiliary.schedule.enabled
        is True
    )
    assert (
        pretrain_4class_cosine_cfg.train.loss.branch_binary_auxiliary.schedule.type
        == "cosine_floor"
    )
    assert (
        pretrain_4class_cosine_cfg.train.loss.branch_binary_auxiliary.monitor.loss_weight
        == 0.3
    )
    assert pretrain_4class_cosine_cfg.checkpointing is not None
    assert [
        (monitor.name, monitor.mode, monitor.top_k, monitor.filename_prefix)
        for monitor in pretrain_4class_cosine_cfg.checkpointing.monitors
    ] == [
        ("val_macro_f1", "max", 3, "best_macro_f1"),
        ("val_macro_recall", "max", 3, "best_macro_recall"),
        ("val_loss_total_monitor", "min", 3, "best_loss"),
        ("last", "last", 3, "last"),
    ]
    assert pretrain_4class_sqrt_sampler_cfg.train.sampler.enabled is True
    assert pretrain_4class_sqrt_sampler_cfg.train.loss.label_smoothing.enabled is True
    assert pretrain_4class_sqrt_sampler_cfg.train.loss.label_smoothing.value == 0.05
    assert pretrain_4class_sqrt_sampler_cfg.train.sampler.type == "sqrt_inverse_class"
    assert pretrain_4class_sqrt_sampler_cfg.train.sampler.replacement is True
    assert pretrain_4class_sqrt_sampler_cfg.train.sampler.num_samples == "dataset_size"
    assert pretrain_4class_sqrt_sampler_cfg.train.sampler.source == "train"
    assert (
        pretrain_4class_sqrt_sampler_cfg.train.loss.branch_binary_auxiliary.schedule.enabled
        is True
    )
    assert (
        pretrain_4class_sqrt_sampler_cfg.train.loss.branch_binary_auxiliary.monitor.loss_weight
        == 0.3
    )
    cosine_payload = asdict(pretrain_4class_cosine_cfg)
    sqrt_sampler_payload = asdict(pretrain_4class_sqrt_sampler_cfg)
    cosine_payload["experiment"]["name"] = sqrt_sampler_payload["experiment"]["name"]
    cosine_payload["train"]["sampler"] = sqrt_sampler_payload["train"]["sampler"]
    assert sqrt_sampler_payload == cosine_payload
    assert cv_cfg.folds[0].name == "fold_0"
    assert eval_cfg.threshold_optimization.metric == "f1"


def test_g1_configs_differ_from_d3_only_by_name_and_seed() -> None:
    d3_cfg = JsonConfigLoader.load_training(
        ROOT / "configs/training_d3_direct_b3_low_lr.json"
    )
    expected_seeds = [0, 1, 2, 42, 43]

    for seed in expected_seeds:
        g1_cfg = JsonConfigLoader.load_training(
            ROOT / "configs" / f"training_g1_d3_seed{seed}.json"
        )
        d3_payload = asdict(d3_cfg)
        g1_payload = asdict(g1_cfg)
        d3_payload["experiment"]["name"] = g1_payload["experiment"]["name"]
        d3_payload["experiment"]["seed"] = g1_payload["experiment"]["seed"]

        assert g1_cfg.experiment.name == f"respiratory_g1_d3_seed{seed}"
        assert g1_cfg.experiment.seed == seed
        assert g1_payload == d3_payload


def test_load_training_config_uses_event_mil_schema(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "train.json", _base_payload())

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.experiment.mode == "clip"
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
    assert cfg.train.loss.class_gate_diversity_regularization.enabled is False
    assert cfg.train.loss.class_gate_diversity_regularization.weight == 0.0
    assert (
        cfg.train.loss.class_gate_diversity_regularization.target
        == "class_evidence_gate"
    )
    assert cfg.train.loss.class_gate_diversity_regularization.metric == "js_divergence"
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
    assert cfg.train.loss.gate_branch_regret.margin_mode == ("true_vs_hardest_negative")
    assert cfg.train.loss.gate_branch_regret.positive_threshold == 0.0
    assert cfg.train.loss.gate_branch_regret.tolerance == 0.0
    assert cfg.train.loss.gate_branch_regret.warmup_epochs == 0
    assert cfg.train.loss.branch_auxiliary.enabled is False
    assert cfg.train.sampler.weighted_random is True
    assert cfg.train.sampler.enabled is False
    assert cfg.train.sampler.type == "none"
    assert cfg.train.initialization.checkpoint_path is None


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
    assert class_gate.evidence_auxiliary.enabled is False
    assert class_gate.evidence_auxiliary.weight == 0.1
    assert class_gate.branch_logit_feature.mode == "raw"


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


@pytest.mark.parametrize("target", ["class_evidence_gate", "true_class_evidence_gate"])
def test_valid_class_gate_entropy_regularization_targets_load(
    tmp_path: Path,
    target: str,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["gate_entropy_regularization"] = {
        "enabled": True,
        "weight": 0.001,
        "target": target,
    }
    config_path = _write_json(
        tmp_path / f"gate_entropy_regularization_{target}.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    assert cfg.train.loss.gate_entropy_regularization.target == target


def test_valid_class_gate_diversity_regularization_config_loads(
    tmp_path: Path,
) -> None:
    payload = _fourclass_branch_binary_payload()
    payload["train"]["loss"]["class_gate_diversity_regularization"] = {
        "enabled": True,
        "weight": 0.0003,
        "target": "class_evidence_gate",
        "metric": "js_divergence",
    }
    config_path = _write_json(
        tmp_path / "class_gate_diversity_regularization.json",
        payload,
    )

    cfg = JsonConfigLoader.load_training(config_path)

    diversity_cfg = cfg.train.loss.class_gate_diversity_regularization
    assert diversity_cfg.enabled is True
    assert diversity_cfg.weight == 0.0003
    assert diversity_cfg.target == "class_evidence_gate"
    assert diversity_cfg.metric == "js_divergence"


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
        "tolerance": 0.05,
        "warmup_epochs": 10,
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
    assert regret_cfg.tolerance == 0.05
    assert regret_cfg.warmup_epochs == 10


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", "yes", "class_gate_diversity_regularization.enabled"),
        ("weight", 0.0, "class_gate_diversity_regularization.weight"),
        ("weight", -0.001, "class_gate_diversity_regularization.weight"),
        ("target", "evidence_gate", "class_gate_diversity_regularization.target"),
        ("metric", "kl_divergence", "class_gate_diversity_regularization.metric"),
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
