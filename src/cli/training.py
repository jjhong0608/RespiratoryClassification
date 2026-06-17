from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.data.loaders import build_clip_loader, build_dataset
from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
    summarize_model_architecture,
)
from src.training.imbalance import (
    build_sqrt_inverse_class_sampler,
    build_weighted_sampler,
    collect_targets,
    resolve_imbalance,
    resolve_loss_weights,
)
from src.training.initialization import initialize_from_checkpoint
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import JsonConfigLoader, resolve_label_float_overrides
from src.utils.fs import Fs
from src.utils.logging import configure_rich_logging, enable_file_logging, logger
from src.utils.reproducibility import Reproducibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_training(args.config)
    configure_rich_logging(cfg.experiment.logging.terminal_width)
    generators = Reproducibility.seed_everything(cfg.experiment.seed)

    run_dir = Fs.ensure_dir(Path(cfg.experiment.output_dir) / cfg.experiment.name)
    Fs.copy_file(args.config, run_dir)
    enable_file_logging(run_dir / "run.log", mode="w")

    train_dataset = build_dataset(cfg.data, split="train")
    val_dataset = build_dataset(cfg.data, split="val")
    augmentation = cfg.data.augmentation
    logger.info(
        "Data augmentation | enabled=%s | train_apply=%s | waveform=%s | "
        "waveform_probability=%.4f | gain=%s | noise=%s | time_shift=%s | "
        "fbank=%s | fbank_probability=%.4f | time_mask=%s | freq_mask=%s",
        augmentation.enabled,
        train_dataset.apply_augmentation,
        augmentation.waveform.enabled,
        augmentation.waveform.probability,
        augmentation.waveform.gain.enabled,
        augmentation.waveform.noise.enabled,
        augmentation.waveform.time_shift.enabled,
        augmentation.fbank.enabled,
        augmentation.fbank.probability,
        augmentation.fbank.time_mask.enabled,
        augmentation.fbank.freq_mask.enabled,
    )
    num_classes = len(cfg.data.label_to_index)
    model = build_ast_model(
        cfg.model,
        num_mel_bins=train_dataset.num_mel_bins,
        max_length=train_dataset.max_length,
        num_classes=num_classes,
    )
    architecture_summary = summarize_model_architecture(model)
    adaptation_summary = apply_encoder_adaptation(model, model.cfg.encoder.adaptation)
    logger.info(
        "Model architecture | encoder_type=%s | hidden_size=%d | num_heads=%d | "
        "branch_token_counts=%s | branch_time_lengths=%s | total_patch_tokens=%d | "
        "total_temporal_length=%d | rdt_enabled=%s | rdt_steps=%d | "
        "top_tokens_per_branch=%d | evidence_score_source=%s | "
        "exclude_branches_from_evidence=%s | attention_temperature=%.4f | "
        "evidence_pooling=%s | branch_auxiliary=%s",
        architecture_summary.encoder_type,
        architecture_summary.hidden_size,
        architecture_summary.num_attention_heads,
        list(architecture_summary.branch_token_counts),
        list(architecture_summary.branch_time_lengths),
        architecture_summary.total_patch_token_count,
        architecture_summary.total_temporal_length,
        architecture_summary.rdt_enabled,
        architecture_summary.rdt_steps,
        architecture_summary.rdt_top_tokens_per_branch,
        architecture_summary.rdt_evidence_score_source,
        list(architecture_summary.rdt_exclude_branches_from_evidence),
        architecture_summary.mil_attention_temperature,
        architecture_summary.evidence_pooling_type,
        cfg.train.loss.branch_auxiliary.enabled,
    )
    logger.info(
        "Encoder adaptation | mode=%s | num_layers=%d | trainable_params=%d | frozen_params=%d",
        adaptation_summary.mode,
        adaptation_summary.num_layers,
        adaptation_summary.trainable_parameters,
        adaptation_summary.frozen_parameters,
    )

    train_targets = collect_targets(train_dataset)
    resolved_loss_weights = resolve_loss_weights(
        targets=train_targets,
        num_classes=num_classes,
        label_to_index=cfg.data.label_to_index,
        loss_cfg=cfg.train.loss,
    )
    imbalance = resolve_imbalance(
        targets=train_targets,
        num_classes=num_classes,
        loss_type=cfg.train.loss.type,
        pos_weight=cfg.train.loss.pos_weight,
        auto_pos_weight=cfg.train.loss.auto_pos_weight,
        weighted_random=cfg.train.sampler.weighted_random,
    )
    logger.info("Train class counts: %s", dict(imbalance.class_counts))
    logger.info(
        "Train loss weights | class_counts=%s | class_weights=%s | "
        "main_to_binary=%s | binary_counts=%s | branch_binary_pos_weight=%s",
        list(resolved_loss_weights.class_counts),
        list(resolved_loss_weights.class_weights)
        if resolved_loss_weights.class_weights is not None
        else None,
        list(resolved_loss_weights.main_index_to_binary_target)
        if resolved_loss_weights.main_index_to_binary_target is not None
        else None,
        dict(resolved_loss_weights.binary_counts)
        if resolved_loss_weights.binary_counts is not None
        else None,
        resolved_loss_weights.branch_binary_pos_weight,
    )
    if imbalance.pos_weight is not None:
        if cfg.train.loss.auto_pos_weight:
            logger.info(
                "Using auto-computed positive-class weight=%.6f from training clips (loss=%s)",
                imbalance.pos_weight,
                cfg.train.loss.type,
            )
        else:
            logger.info(
                "Using positive-class weight=%.6f (loss=%s)",
                imbalance.pos_weight,
                cfg.train.loss.type,
            )

    sampler = None
    if cfg.train.sampler.enabled:
        sampler, sampler_summary = build_sqrt_inverse_class_sampler(
            train_targets,
            num_classes=num_classes,
            cfg=cfg.train.sampler,
            generator=generators.sampler,
        )
        index_to_label = {
            int(index): label_name
            for label_name, index in cfg.data.label_to_index.items()
        }
        class_counts_by_label = {
            index_to_label.get(index, str(index)): count
            for index, count in enumerate(sampler_summary.class_counts)
        }
        class_weights_by_label = {
            index_to_label.get(index, str(index)): weight
            for index, weight in enumerate(sampler_summary.class_weights)
        }
        expected_probabilities_by_label = {
            index_to_label.get(index, str(index)): probability
            for index, probability in enumerate(
                sampler_summary.expected_class_probabilities
            )
        }
        logger.info(
            "Sqrt-inverse class sampler enabled | type=%s | replacement=%s | "
            "num_samples=%d | source=%s",
            cfg.train.sampler.type,
            cfg.train.sampler.replacement,
            sampler_summary.num_samples,
            cfg.train.sampler.source,
        )
        logger.info("Sampler class counts: %s", class_counts_by_label)
        logger.info("Sampler class weights: %s", class_weights_by_label)
        logger.info(
            "Sampler expected class probabilities: %s",
            expected_probabilities_by_label,
        )
    elif imbalance.weighted_random:
        sampler = build_weighted_sampler(train_targets, generator=generators.sampler)
        logger.info("Using legacy weighted random sampler for clip training")

    train_loader = build_clip_loader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generators.train_loader,
    )
    val_loader = build_clip_loader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        generator=generators.val_loader,
    )
    logger.info("Train clips: %d | Val clips: %d", len(train_dataset), len(val_dataset))
    optimizer, optimizer_summary = build_grouped_optimizer(
        model,
        encoder_lr=cfg.train.optimizer.encoder_lr,
        head_lr=cfg.train.optimizer.head_lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    logger.info(
        "Optimizer groups | encoder_lr=%.8f | head_lr=%.8f | encoder_params=%d | head_params=%d | groups=%d",
        optimizer_summary.encoder_lr,
        optimizer_summary.head_lr,
        optimizer_summary.encoder_trainable_parameters,
        optimizer_summary.head_trainable_parameters,
        optimizer_summary.param_group_count,
    )
    initialization_summary = initialize_from_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=cfg.train.initialization,
        map_location="cpu",
    )
    if initialization_summary.checkpoint_path is not None:
        logger.info(
            "Training initialization | checkpoint=%s | strict=%s | "
            "loaded_model_state=%s | loaded_optimizer_state=%s | "
            "missing_keys=%s | unexpected_keys=%s",
            initialization_summary.checkpoint_path,
            initialization_summary.strict,
            initialization_summary.loaded_model_state,
            initialization_summary.loaded_optimizer_state,
            list(initialization_summary.missing_keys),
            list(initialization_summary.unexpected_keys),
        )

    class_evidence_margin_cfg = cfg.train.loss.class_evidence_margin
    class_evidence_margin_major_index = (
        cfg.data.label_to_index[class_evidence_margin_cfg.major_class]
        if (
            class_evidence_margin_cfg.enabled
            and class_evidence_margin_cfg.mode == "minority_vs_major"
            and class_evidence_margin_cfg.major_class is not None
        )
        else None
    )
    class_names = tuple(
        label_name
        for label_name, _ in sorted(
            cfg.data.label_to_index.items(),
            key=lambda item: item[1],
        )
    )
    calibrated_teacher_margin_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.calibrated_teacher_margin.margin,
        overrides=cfg.train.loss.calibrated_teacher_margin.margin_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_branch_margin_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_branch_margin.margin,
        overrides=cfg.train.loss.top_branch_margin.margin_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_branch_margin_phase_start_multiplier_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=(
            cfg.train.loss.top_branch_margin.phase_weight_schedule.start_multiplier_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    top_branch_margin_phase_label_multiplier_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=(
            cfg.train.loss.top_branch_margin.phase_weight_schedule.label_multiplier_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    top_support_score_margin_label_weight_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=cfg.train.loss.top_support_score_margin.label_weight_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_support_gap_min_base_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_support_gap_min_constraint.base_min_gap,
        overrides=cfg.train.loss.top_support_gap_min_constraint.base_min_gap_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_margin_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.margin,
        overrides=cfg.train.loss.class_top_branch_relative_margin.margin_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_hardness_gain_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.hardness_weighting.gain,
        overrides={
            label: item.gain
            for label, item in cfg.train.loss.class_top_branch_relative_margin.hardness_weighting_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_hardness_cap_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.hardness_weighting.cap,
        overrides={
            label: item.cap
            for label, item in cfg.train.loss.class_top_branch_relative_margin.hardness_weighting_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_weak_support_min_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.weak_positive_support_weighting.min_support,
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.class_top_branch_relative_margin.weak_positive_support_weighting.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_weak_support_max_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.weak_positive_support_weighting.max_support,
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.class_top_branch_relative_margin.weak_positive_support_weighting.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_margin_boost_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.weak_positive_margin_boost.boost,
        overrides=cfg.train.loss.class_top_branch_relative_margin.weak_positive_margin_boost.boost_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_margin_boost_min_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.weak_positive_margin_boost.min_support,
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.class_top_branch_relative_margin.weak_positive_margin_boost.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    class_top_branch_relative_margin_boost_max_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_top_branch_relative_margin.weak_positive_margin_boost.max_support,
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.class_top_branch_relative_margin.weak_positive_margin_boost.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_base_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.base_min_gap,
        overrides=cfg.train.loss.top_teacher_gap_min_constraint.base_min_gap_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_support_gain_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.support_gain,
        overrides=cfg.train.loss.top_teacher_gap_min_constraint.support_gain_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_support_cap_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.support_cap,
        overrides=cfg.train.loss.top_teacher_gap_min_constraint.support_cap_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_weak_support_min_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_support_weighting.min_support,
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_support_weighting.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_weak_support_max_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_support_weighting.max_support,
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_support_weighting.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_target_boost_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost.boost,
        overrides=cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost.boost_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_target_boost_min_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost.min_support,
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    top_teacher_gap_min_target_boost_max_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost.max_support,
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    true_top_floor_base_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.true_top_floor_constraint.base_floor,
        overrides=cfg.train.loss.true_top_floor_constraint.base_floor_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    true_top_floor_support_gain_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.true_top_floor_constraint.support_gain,
        overrides=cfg.train.loss.true_top_floor_constraint.support_gain_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    true_top_floor_support_cap_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.true_top_floor_constraint.support_cap,
        overrides=cfg.train.loss.true_top_floor_constraint.support_cap_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    true_top_floor_weak_boost_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.true_top_floor_constraint.weak_positive_boost.boost,
        overrides=(
            cfg.train.loss.true_top_floor_constraint.weak_positive_boost.boost_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    true_top_floor_weak_min_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.true_top_floor_constraint.weak_positive_boost.min_support
        ),
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.true_top_floor_constraint.weak_positive_boost.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    true_top_floor_weak_max_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.true_top_floor_constraint.weak_positive_boost.max_support
        ),
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.true_top_floor_constraint.weak_positive_boost.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_base_required_gap_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.hard_negative_top_teacher_suppression.base_required_gap,
        overrides=(
            cfg.train.loss.hard_negative_top_teacher_suppression.base_required_gap_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_label_weight_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=(
            cfg.train.loss.hard_negative_top_teacher_suppression.label_weight_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_weak_boost_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.hard_negative_top_teacher_suppression.weak_positive_band.boost
        ),
        overrides=(
            cfg.train.loss.hard_negative_top_teacher_suppression.weak_positive_band.boost_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_weak_min_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.hard_negative_top_teacher_suppression.weak_positive_band.min_support
        ),
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.hard_negative_top_teacher_suppression.weak_positive_band.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_weak_max_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.hard_negative_top_teacher_suppression.weak_positive_band.max_support
        ),
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.hard_negative_top_teacher_suppression.weak_positive_band.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_moderate_boost_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.hard_negative_top_teacher_suppression.moderate_positive_band.boost
        ),
        overrides=(
            cfg.train.loss.hard_negative_top_teacher_suppression.moderate_positive_band.boost_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_moderate_min_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.hard_negative_top_teacher_suppression.moderate_positive_band.min_support
        ),
        overrides={
            label: item.min_support
            for label, item in cfg.train.loss.hard_negative_top_teacher_suppression.moderate_positive_band.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    hard_negative_top_teacher_moderate_max_by_class = resolve_label_float_overrides(
        default=(
            cfg.train.loss.hard_negative_top_teacher_suppression.moderate_positive_band.max_support
        ),
        overrides={
            label: item.max_support
            for label, item in cfg.train.loss.hard_negative_top_teacher_suppression.moderate_positive_band.support_band_by_label.items()
        },
        label_to_index=cfg.data.label_to_index,
    )
    class_evidence_gap_cap_negative_cap_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.class_evidence_gap_cap_regularization.negative_gap_cap,
        overrides=(
            cfg.train.loss.class_evidence_gap_cap_regularization.negative_gap_cap_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    class_evidence_gap_cap_label_weight_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=(
            cfg.train.loss.class_evidence_gap_cap_regularization.label_weight_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    branch_path_dominance_allowed_drop_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.branch_path_dominance_constraint.allowed_drop,
        overrides=(
            cfg.train.loss.branch_path_dominance_constraint.allowed_drop_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    branch_path_dominance_label_weight_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=(
            cfg.train.loss.branch_path_dominance_constraint.label_weight_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    branch_to_evidence_teacher_floor_by_class = resolve_label_float_overrides(
        default=0.0,
        overrides=(
            cfg.train.loss.branch_to_evidence_ranking_consistency.teacher_floor_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    gate_branch_regret_positive_threshold_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.gate_branch_regret.positive_threshold,
        overrides=cfg.train.loss.gate_branch_regret.positive_threshold_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    gate_bad_branch_suppression_threshold_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.gate_bad_branch_suppression.bad_margin_threshold,
        overrides=(
            cfg.train.loss.gate_bad_branch_suppression.bad_margin_threshold_by_label
        ),
        label_to_index=cfg.data.label_to_index,
    )
    gate_best_branch_alignment_label_weight_by_class = resolve_label_float_overrides(
        default=1.0,
        overrides=cfg.train.loss.gate_best_branch_alignment.label_weight_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    gate_best_branch_alignment_min_best_margin_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.gate_best_branch_alignment.min_best_margin,
        overrides=cfg.train.loss.gate_best_branch_alignment.min_best_margin_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    gate_best_branch_alignment_max_best_margin_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.gate_best_branch_alignment.max_best_margin,
        overrides=cfg.train.loss.gate_best_branch_alignment.max_best_margin_by_label,
        label_to_index=cfg.data.label_to_index,
    )
    gate_best_branch_alignment_mismatch_drop_by_class = resolve_label_float_overrides(
        default=cfg.train.loss.gate_best_branch_alignment.mismatch_margin_drop,
        overrides=cfg.train.loss.gate_best_branch_alignment.mismatch_margin_drop_by_label,
        label_to_index=cfg.data.label_to_index,
    )

    trainer = Trainer(
        TrainerConfig(
            device=cfg.experiment.device,
            epochs=cfg.train.epochs,
            encoder_lr=cfg.train.optimizer.encoder_lr,
            head_lr=cfg.train.optimizer.head_lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            warmup_ratio=cfg.train.scheduler.warmup_ratio,
            max_grad_norm=cfg.train.max_grad_norm,
            top_k=cfg.train.top_k,
            run_dir=run_dir,
            num_classes=num_classes,
            class_names=class_names,
            loss_type=cfg.train.loss.type,
            gamma=cfg.train.loss.gamma,
            pos_weight=imbalance.pos_weight,
            class_weights=resolved_loss_weights.class_weights,
            label_smoothing=cfg.train.loss.label_smoothing,
            branch_auxiliary=cfg.train.loss.branch_auxiliary,
            branch_binary_auxiliary=cfg.train.loss.branch_binary_auxiliary,
            branch_binary_pos_weight=(resolved_loss_weights.branch_binary_pos_weight),
            main_index_to_binary_target=(
                resolved_loss_weights.main_index_to_binary_target
            ),
            attention_entropy=cfg.train.loss.attention_entropy,
            gate_entropy_regularization=(cfg.train.loss.gate_entropy_regularization),
            class_gate_evidence_auxiliary=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_auxiliary
            ),
            class_gate_diversity_regularization=(
                cfg.train.loss.class_gate_diversity_regularization
            ),
            class_evidence_margin=cfg.train.loss.class_evidence_margin,
            class_evidence_margin_major_index=class_evidence_margin_major_index,
            calibrated_teacher_margin=cfg.train.loss.calibrated_teacher_margin,
            calibrated_teacher_margin_by_class=calibrated_teacher_margin_by_class,
            class_evidence_gap_cap_regularization=(
                cfg.train.loss.class_evidence_gap_cap_regularization
            ),
            class_evidence_gap_cap_negative_cap_by_class=(
                class_evidence_gap_cap_negative_cap_by_class
            ),
            class_evidence_gap_cap_label_weight_by_class=(
                class_evidence_gap_cap_label_weight_by_class
            ),
            class_evidence_positive_gap_cap_regularization=(
                cfg.train.loss.class_evidence_positive_gap_cap_regularization
            ),
            interaction_gap_cap_regularization=(
                cfg.train.loss.interaction_gap_cap_regularization
            ),
            top_support_score_margin=cfg.train.loss.top_support_score_margin,
            top_support_score_margin_label_weight_by_class=(
                top_support_score_margin_label_weight_by_class
            ),
            top_support_gap_min_constraint=(
                cfg.train.loss.top_support_gap_min_constraint
            ),
            top_support_gap_min_base_by_class=top_support_gap_min_base_by_class,
            class_top_branch_relative_margin=(
                cfg.train.loss.class_top_branch_relative_margin
            ),
            class_top_branch_relative_margin_by_class=(
                class_top_branch_relative_margin_by_class
            ),
            class_top_branch_relative_hardness_gain_by_class=(
                class_top_branch_relative_hardness_gain_by_class
            ),
            class_top_branch_relative_hardness_cap_by_class=(
                class_top_branch_relative_hardness_cap_by_class
            ),
            class_top_branch_relative_weak_support_min_by_class=(
                class_top_branch_relative_weak_support_min_by_class
            ),
            class_top_branch_relative_weak_support_max_by_class=(
                class_top_branch_relative_weak_support_max_by_class
            ),
            class_top_branch_relative_margin_boost_by_class=(
                class_top_branch_relative_margin_boost_by_class
            ),
            class_top_branch_relative_margin_boost_min_by_class=(
                class_top_branch_relative_margin_boost_min_by_class
            ),
            class_top_branch_relative_margin_boost_max_by_class=(
                class_top_branch_relative_margin_boost_max_by_class
            ),
            top_teacher_gap_min_constraint=(
                cfg.train.loss.top_teacher_gap_min_constraint
            ),
            top_teacher_gap_min_base_by_class=top_teacher_gap_min_base_by_class,
            top_teacher_gap_min_support_gain_by_class=(
                top_teacher_gap_min_support_gain_by_class
            ),
            top_teacher_gap_min_support_cap_by_class=(
                top_teacher_gap_min_support_cap_by_class
            ),
            top_teacher_gap_min_weak_support_min_by_class=(
                top_teacher_gap_min_weak_support_min_by_class
            ),
            top_teacher_gap_min_weak_support_max_by_class=(
                top_teacher_gap_min_weak_support_max_by_class
            ),
            top_teacher_gap_min_target_boost_by_class=(
                top_teacher_gap_min_target_boost_by_class
            ),
            top_teacher_gap_min_target_boost_min_by_class=(
                top_teacher_gap_min_target_boost_min_by_class
            ),
            top_teacher_gap_min_target_boost_max_by_class=(
                top_teacher_gap_min_target_boost_max_by_class
            ),
            true_top_floor_constraint=cfg.train.loss.true_top_floor_constraint,
            true_top_floor_base_by_class=true_top_floor_base_by_class,
            true_top_floor_support_gain_by_class=(true_top_floor_support_gain_by_class),
            true_top_floor_support_cap_by_class=true_top_floor_support_cap_by_class,
            true_top_floor_weak_boost_by_class=true_top_floor_weak_boost_by_class,
            true_top_floor_weak_min_by_class=true_top_floor_weak_min_by_class,
            true_top_floor_weak_max_by_class=true_top_floor_weak_max_by_class,
            hard_negative_top_teacher_suppression=(
                cfg.train.loss.hard_negative_top_teacher_suppression
            ),
            hard_negative_top_teacher_base_required_gap_by_class=(
                hard_negative_top_teacher_base_required_gap_by_class
            ),
            hard_negative_top_teacher_weak_boost_by_class=(
                hard_negative_top_teacher_weak_boost_by_class
            ),
            hard_negative_top_teacher_weak_min_by_class=(
                hard_negative_top_teacher_weak_min_by_class
            ),
            hard_negative_top_teacher_weak_max_by_class=(
                hard_negative_top_teacher_weak_max_by_class
            ),
            hard_negative_top_teacher_moderate_boost_by_class=(
                hard_negative_top_teacher_moderate_boost_by_class
            ),
            hard_negative_top_teacher_moderate_min_by_class=(
                hard_negative_top_teacher_moderate_min_by_class
            ),
            hard_negative_top_teacher_moderate_max_by_class=(
                hard_negative_top_teacher_moderate_max_by_class
            ),
            hard_negative_top_teacher_label_weight_by_class=(
                hard_negative_top_teacher_label_weight_by_class
            ),
            branch_direct_score_margin=cfg.train.loss.branch_direct_score_margin,
            branch_support_score_margin=(cfg.train.loss.branch_support_score_margin),
            branch_path_dominance_constraint=(
                cfg.train.loss.branch_path_dominance_constraint
            ),
            branch_path_dominance_allowed_drop_by_class=(
                branch_path_dominance_allowed_drop_by_class
            ),
            branch_path_dominance_label_weight_by_class=(
                branch_path_dominance_label_weight_by_class
            ),
            branch_support_disagreement_cap_regularization=(
                cfg.train.loss.branch_support_disagreement_cap_regularization
            ),
            class_gated_branch_logit_margin=(
                cfg.train.loss.class_gated_branch_logit_margin
            ),
            branch_to_evidence_ranking_consistency=(
                cfg.train.loss.branch_to_evidence_ranking_consistency
            ),
            branch_to_evidence_teacher_floor_by_class=(
                branch_to_evidence_teacher_floor_by_class
            ),
            global_residual_anti_veto=cfg.train.loss.global_residual_anti_veto,
            residual_contradiction_regularization=(
                cfg.train.loss.residual_contradiction_regularization
            ),
            gate_weighted_branch_margin=cfg.train.loss.gate_weighted_branch_margin,
            gate_best_branch_alignment=cfg.train.loss.gate_best_branch_alignment,
            gate_best_branch_alignment_label_weight_by_class=(
                gate_best_branch_alignment_label_weight_by_class
            ),
            gate_best_branch_alignment_min_best_margin_by_class=(
                gate_best_branch_alignment_min_best_margin_by_class
            ),
            gate_best_branch_alignment_max_best_margin_by_class=(
                gate_best_branch_alignment_max_best_margin_by_class
            ),
            gate_best_branch_alignment_mismatch_drop_by_class=(
                gate_best_branch_alignment_mismatch_drop_by_class
            ),
            gate_branch_regret=cfg.train.loss.gate_branch_regret,
            gate_branch_regret_positive_threshold_by_class=(
                gate_branch_regret_positive_threshold_by_class
            ),
            gate_bad_branch_suppression=(cfg.train.loss.gate_bad_branch_suppression),
            gate_bad_branch_suppression_threshold_by_class=(
                gate_bad_branch_suppression_threshold_by_class
            ),
            top_branch_margin=cfg.train.loss.top_branch_margin,
            top_branch_margin_by_class=top_branch_margin_by_class,
            top_branch_margin_phase_start_multiplier_by_class=(
                top_branch_margin_phase_start_multiplier_by_class
            ),
            top_branch_margin_phase_label_multiplier_by_class=(
                top_branch_margin_phase_label_multiplier_by_class
            ),
            analysis=cfg.analysis,
            early_stopping=cfg.train.early_stopping,
            checkpointing=cfg.checkpointing,
        )
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        extra_state={
            "run_config": asdict(cfg),
            "model_cfg": asdict(model.cfg),
            "label_to_index": dict(cfg.data.label_to_index),
            "architecture_summary": asdict(architecture_summary),
            "adaptation_summary": asdict(adaptation_summary),
            "optimizer_summary": asdict(optimizer_summary),
            "initialization_summary": asdict(initialization_summary),
            "loss_weight_summary": asdict(resolved_loss_weights),
        },
    )


if __name__ == "__main__":
    main()
