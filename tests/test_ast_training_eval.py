from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from src.cli.cv import main as cv_main
from src.cli.evaluate import evaluate_checkpoint
from src.data.loaders import ClipBatch, build_clip_loader, build_dataset
from src.evaluation.metrics import EvalMetrics
from src.evaluation.thresholds import ThresholdOptimizationResult
from src.models.model import AstModelOutput, ClassGateEvidenceAuxiliaryConfig
from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
)
from src.training.epoch_logging import (
    EpochLogContext,
    append_epoch_jsonl_logs,
    format_epoch_log_block,
)
from src.training.trainer import (
    CheckpointManager,
    Trainer,
    TrainerConfig,
    resolve_branch_binary_aux_weight,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.config import (
    AnalysisConfig,
    AnalysisOutputConfig,
    AstFbankConfig,
    AttentionEntropyLossConfig,
    AudioConfig,
    AutoBadBranchThresholdByTrainStatsConfig,
    AutoMarginByTrainStatsConfig,
    AutoPositiveThresholdByTrainStatsConfig,
    BandPassConfig,
    BranchAuxiliaryLossConfig,
    BranchBinaryAuxiliaryLossConfig,
    BranchBinaryAuxiliaryMonitorConfig,
    BranchBinaryAuxiliaryScheduleConfig,
    BranchBinaryPosWeightConfig,
    BranchDirectScoreMarginConfig,
    BranchPathDominanceConstraintConfig,
    BranchSupportDisagreementCapRegularizationConfig,
    BranchSupportScoreMarginConfig,
    BranchToEvidenceRankingConsistencyConfig,
    CheckpointingConfig,
    CheckpointMonitorConfig,
    ClassEvidenceGapCapRegularizationConfig,
    ClassEvidenceMarginConfig,
    ClassEvidencePositiveGapCapRegularizationConfig,
    ClassGatedBranchLogitMarginConfig,
    ClassGateDiversityRegularizationConfig,
    ClassifierConfig,
    ClassTopBranchRelativeMarginConfig,
    DataConfig,
    EarlyStoppingConfig,
    EncoderAdaptationConfig,
    EvalConfig,
    EvidencePoolingConfig,
    ExperimentConfig,
    GateBadBranchSuppressionConfig,
    GateBestBranchAlignmentConfig,
    GateBranchRegretConfig,
    GateBranchRegretWeightScheduleConfig,
    GateEntropyRegularizationConfig,
    GateWeightedBranchMarginConfig,
    GlobalResidualAntiVetoConfig,
    InteractionGapCapRegularizationConfig,
    LabelSmoothingConfig,
    MarginHardnessWeightingConfig,
    MarginSupportWeightingConfig,
    ModelConfig,
    ModelEncoderConfig,
    MultiScaleRdtArchitectureConfig,
    PreprocessingConfig,
    RdtConfig,
    ResidualContradictionRegularizationConfig,
    TopBranchMarginConfig,
    TopBranchMarginHardnessWeightingConfig,
    TopBranchMarginPhaseWeightScheduleConfig,
    TopSupportGapMinConstraintConfig,
    TopSupportScoreMarginConfig,
    TopTeacherGapMinConstraintConfig,
    WeakPositiveMarginBoostConfig,
    WeakPositiveSupportWeightingConfig,
    WeakPositiveTargetBoostConfig,
)

from conftest import small_patch_branches


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 320.0 * t)
    sf.write(path, tone.astype(np.float32), sample_rate)


def _analysis_cfg() -> AnalysisConfig:
    return AnalysisConfig(
        outputs=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=True,
            save_clip_metadata=True,
        )
    )


def _small_architecture(
    *,
    rdt_enabled: bool,
    rdt_steps: int,
    evidence_pooling: EvidencePoolingConfig | None = None,
) -> MultiScaleRdtArchitectureConfig:
    return MultiScaleRdtArchitectureConfig(
        hidden_size=32,
        num_attention_heads=4,
        mlp_ratio=2.0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        shared_stem_depth=1,
        adapter_depth=1,
        patch_branches=small_patch_branches(),
        rdt=RdtConfig(
            enabled=rdt_enabled,
            steps=rdt_steps,
            top_tokens_per_branch=2,
            gated_residual=True,
            layerscale_init=0.01,
        ),
        evidence_pooling=evidence_pooling or EvidencePoolingConfig(),
    )


def _model_cfg(
    *,
    classifier_type: Literal["linear", "mlp"] = "linear",
    rdt_enabled: bool = True,
    rdt_steps: int = 3,
    evidence_pooling: EvidencePoolingConfig | None = None,
) -> ModelConfig:
    return ModelConfig(
        encoder=ModelEncoderConfig(
            adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
            architecture=_small_architecture(
                rdt_enabled=rdt_enabled,
                rdt_steps=rdt_steps,
                evidence_pooling=evidence_pooling,
            ),
        ),
        classifier=ClassifierConfig(
            type=classifier_type,
            hidden_dim=32,
            dropout=0.0,
            pooling="latent_mean",
        ),
    )


def _binary_data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.5),
        preprocessing=PreprocessingConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=32,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
    )


def _multiclass_data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1, "crackle": 2},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.5),
        preprocessing=PreprocessingConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=32,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
    )


def _prepare_binary_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 0.7, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 1.3, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 0.9, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 1.4, 16000)
    return _binary_data_cfg(root)


def _prepare_multiclass_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze", "crackle"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 0.7, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 1.3, 16000)
    _write_wav(root / "train" / "crackle" / "crackle_a.wav", 1.1, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 0.9, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 1.4, 16000)
    _write_wav(root / "val" / "crackle" / "crackle_b.wav", 1.2, 16000)
    return _multiclass_data_cfg(root)


def _trainer_cfg(
    *,
    num_classes: int,
    run_dir: Path,
    loss_type: str,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    branch_auxiliary_enabled: bool = False,
    branch_auxiliary_weight: float = 0.3,
    branch_auxiliary_weights: tuple[float, ...] | None = None,
    branch_binary_auxiliary_enabled: bool = False,
    branch_binary_auxiliary_weight: float = 0.3,
    branch_binary_auxiliary_schedule: BranchBinaryAuxiliaryScheduleConfig | None = None,
    branch_binary_auxiliary_monitor: BranchBinaryAuxiliaryMonitorConfig | None = None,
    branch_binary_pos_weight: float | None = None,
    main_index_to_binary_target: tuple[int, ...] | None = None,
    class_weights: tuple[float, ...] | None = None,
    label_smoothing: LabelSmoothingConfig | None = None,
    attention_entropy_enabled: bool = False,
    attention_entropy_weight: float = 0.0,
    gate_entropy_regularization_enabled: bool = False,
    gate_entropy_regularization_weight: float = 0.0,
    gate_entropy_regularization_target: Literal[
        "evidence_gate",
        "class_evidence_gate",
        "true_class_evidence_gate",
        "class_evidence_learned_gate",
    ] = "evidence_gate",
    gate_entropy_regularization_start_epoch: int = 1,
    gate_entropy_regularization_end_epoch: int | None = None,
    class_gate_evidence_auxiliary_enabled: bool = False,
    class_gate_evidence_auxiliary_weight: float = 0.1,
    class_gate_diversity_regularization_enabled: bool = False,
    class_gate_diversity_regularization_weight: float = 0.0,
    class_gate_diversity_regularization_target: Literal[
        "class_evidence_gate",
        "class_evidence_learned_gate",
    ] = "class_evidence_gate",
    class_gate_diversity_regularization_start_epoch: int = 1,
    class_gate_diversity_regularization_end_epoch: int | None = None,
    class_evidence_margin_enabled: bool = False,
    class_evidence_margin_weight: float = 0.0,
    class_evidence_margin_value: float = 0.0,
    class_evidence_margin_mode: Literal[
        "minority_vs_major",
        "true_vs_hardest_negative",
        "softplus_true_vs_hardest_negative",
    ] = "minority_vs_major",
    class_evidence_margin_major_index: int | None = None,
    class_evidence_margin_class_weighted: bool = False,
    class_evidence_margin_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    class_evidence_margin_temperature: float = 1.0,
    class_evidence_margin_support_weighting: MarginSupportWeightingConfig | None = None,
    class_evidence_gap_cap_enabled: bool = False,
    class_evidence_gap_cap_weight: float = 0.0,
    class_evidence_gap_cap_negative_gap_cap: float = 3.0,
    class_evidence_gap_cap_negative_cap_by_class: (tuple[float, ...] | None) = None,
    class_evidence_gap_cap_label_weight_by_class: tuple[float, ...] | None = None,
    class_evidence_gap_cap_class_weighted: bool = False,
    class_evidence_gap_cap_warmup_epochs: int = 0,
    class_evidence_positive_gap_cap_enabled: bool = False,
    class_evidence_positive_gap_cap_weight: float = 0.0,
    class_evidence_positive_gap_cap_positive_gap_cap: float = 8.0,
    class_evidence_positive_gap_cap_class_weighted: bool = False,
    class_evidence_positive_gap_cap_warmup_epochs: int = 0,
    interaction_gap_cap_enabled: bool = False,
    interaction_gap_cap_weight: float = 0.0,
    interaction_gap_cap_gap_cap: float = 6.0,
    interaction_gap_cap_class_weighted: bool = False,
    interaction_gap_cap_warmup_epochs: int = 0,
    top_support_score_margin_enabled: bool = False,
    top_support_score_margin_weight: float = 0.0,
    top_support_score_margin_temperature: float = 1.0,
    top_support_score_margin_class_weighted: bool = False,
    top_support_score_margin_warmup_epochs: int = 0,
    top_support_score_margin_label_weight_by_class: tuple[float, ...] | None = None,
    top_support_score_margin_support_conditioned_multiplier: (
        MarginSupportWeightingConfig | None
    ) = None,
    top_support_score_margin_hardness_weighting: (
        MarginHardnessWeightingConfig | None
    ) = None,
    top_support_gap_min_enabled: bool = False,
    top_support_gap_min_weight: float = 0.0,
    top_support_gap_min_base_min_gap: float = 0.0,
    top_support_gap_min_base_by_class: tuple[float, ...] | None = None,
    top_support_gap_min_support_gain: float = 0.5,
    top_support_gap_min_support_cap: float = 2.0,
    top_support_gap_min_class_weighted: bool = False,
    top_support_gap_min_warmup_epochs: int = 0,
    class_top_branch_relative_margin_enabled: bool = False,
    class_top_branch_relative_margin_weight: float = 0.0,
    class_top_branch_relative_margin_value: float = 0.3,
    class_top_branch_relative_margin_class_weighted: bool = False,
    class_top_branch_relative_margin_warmup_epochs: int = 0,
    class_top_branch_relative_margin_support_weighting: (
        MarginSupportWeightingConfig | None
    ) = None,
    class_top_branch_relative_margin_hardness_weighting: (
        MarginHardnessWeightingConfig | None
    ) = None,
    class_top_branch_relative_margin_by_class: tuple[float, ...] | None = None,
    class_top_branch_relative_hardness_gain_by_class: (tuple[float, ...] | None) = None,
    class_top_branch_relative_hardness_cap_by_class: (tuple[float, ...] | None) = None,
    class_top_branch_relative_weak_support_min_by_class: (
        tuple[float, ...] | None
    ) = None,
    class_top_branch_relative_weak_support_max_by_class: (
        tuple[float, ...] | None
    ) = None,
    class_top_branch_relative_margin_boost_by_class: (tuple[float, ...] | None) = None,
    class_top_branch_relative_margin_boost_min_by_class: (
        tuple[float, ...] | None
    ) = None,
    class_top_branch_relative_margin_boost_max_by_class: (
        tuple[float, ...] | None
    ) = None,
    class_top_branch_relative_margin_weak_positive_weighting: (
        WeakPositiveSupportWeightingConfig | None
    ) = None,
    class_top_branch_relative_margin_weak_positive_margin_boost: (
        WeakPositiveMarginBoostConfig | None
    ) = None,
    top_teacher_gap_min_enabled: bool = False,
    top_teacher_gap_min_weight: float = 0.0,
    top_teacher_gap_min_base_min_gap: float = 0.0,
    top_teacher_gap_min_support_gain: float = 0.5,
    top_teacher_gap_min_support_cap: float = 2.0,
    top_teacher_gap_min_base_by_class: tuple[float, ...] | None = None,
    top_teacher_gap_min_support_gain_by_class: tuple[float, ...] | None = None,
    top_teacher_gap_min_support_cap_by_class: tuple[float, ...] | None = None,
    top_teacher_gap_min_weak_support_min_by_class: (tuple[float, ...] | None) = None,
    top_teacher_gap_min_weak_support_max_by_class: (tuple[float, ...] | None) = None,
    top_teacher_gap_min_target_boost_by_class: tuple[float, ...] | None = None,
    top_teacher_gap_min_target_boost_min_by_class: (tuple[float, ...] | None) = None,
    top_teacher_gap_min_target_boost_max_by_class: (tuple[float, ...] | None) = None,
    top_teacher_gap_min_class_weighted: bool = False,
    top_teacher_gap_min_warmup_epochs: int = 0,
    top_teacher_gap_min_weak_positive_weighting: (
        WeakPositiveSupportWeightingConfig | None
    ) = None,
    top_teacher_gap_min_weak_positive_target_boost: (
        WeakPositiveTargetBoostConfig | None
    ) = None,
    branch_support_score_margin_enabled: bool = False,
    branch_support_score_margin_weight: float = 0.0,
    branch_support_score_margin_temperature: float = 1.0,
    branch_support_score_margin_class_weighted: bool = False,
    branch_support_score_margin_warmup_epochs: int = 0,
    branch_direct_score_margin_enabled: bool = False,
    branch_direct_score_margin_weight: float = 0.0,
    branch_direct_score_margin_temperature: float = 1.0,
    branch_direct_score_margin_class_weighted: bool = False,
    branch_direct_score_margin_warmup_epochs: int = 0,
    branch_path_dominance_enabled: bool = False,
    branch_path_dominance_weight: float = 0.0,
    branch_path_dominance_allowed_drop: float = 0.5,
    branch_path_dominance_allowed_drop_by_class: tuple[float, ...] | None = None,
    branch_path_dominance_label_weight_by_class: tuple[float, ...] | None = None,
    branch_path_dominance_support_weighting: (
        MarginSupportWeightingConfig | None
    ) = None,
    branch_path_dominance_class_weighted: bool = False,
    branch_path_dominance_warmup_epochs: int = 0,
    branch_support_disagreement_enabled: bool = False,
    branch_support_disagreement_weight: float = 0.0,
    branch_support_disagreement_threshold: float = 0.0,
    branch_support_disagreement_condition_gain: float = 1.0,
    branch_support_disagreement_condition_cap: float = 3.0,
    branch_support_disagreement_embedding_gap_cap: float = 6.0,
    branch_support_disagreement_interaction_gap_cap: float = 4.0,
    branch_support_disagreement_class_weighted: bool = False,
    branch_support_disagreement_warmup_epochs: int = 0,
    class_gated_branch_logit_margin_enabled: bool = False,
    class_gated_branch_logit_margin_weight: float = 0.0,
    class_gated_branch_logit_margin_value: float = 0.0,
    class_gated_branch_logit_margin_class_weighted: bool = False,
    class_gated_branch_logit_margin_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    branch_to_evidence_enabled: bool = False,
    branch_to_evidence_weight: float = 0.0,
    branch_to_evidence_source: Literal[
        "class_gated_branch_logits",
        "top_branch_margin",
        "class_top_branch_margin_features",
        "class_top_branch_margin_relative_features",
    ] = "class_gated_branch_logits",
    branch_to_evidence_mode: Literal[
        "true_vs_hardest_negative",
        "teacher_distribution_kl",
        "true_label_anchored_softplus",
    ] = "true_vs_hardest_negative",
    branch_to_evidence_teacher_detach: bool = True,
    branch_to_evidence_teacher_temperature: float = 1.0,
    branch_to_evidence_student_temperature: float = 1.0,
    branch_to_evidence_temperature: float = 1.0,
    branch_to_evidence_teacher_gap_cap: float | None = None,
    branch_to_evidence_teacher_floor_by_class: tuple[float, ...] | None = None,
    branch_to_evidence_tolerance: float = 0.0,
    branch_to_evidence_class_weighted: bool = False,
    branch_to_evidence_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    branch_to_evidence_warmup_epochs: int = 0,
    branch_to_evidence_support_weighting: MarginSupportWeightingConfig | None = None,
    branch_to_evidence_hardness_weighting: MarginHardnessWeightingConfig | None = None,
    global_residual_anti_veto_enabled: bool = False,
    global_residual_anti_veto_weight: float = 0.0,
    global_residual_anti_veto_target: Literal[
        "global_residual_logits",
        "final_logits",
    ] = "global_residual_logits",
    global_residual_anti_veto_support_source: Literal[
        "none",
        "top_branch_margin",
    ] = "none",
    global_residual_anti_veto_mode: Literal[
        "true_vs_hardest_negative",
        "final_gap_preservation",
    ] = "true_vs_hardest_negative",
    global_residual_anti_veto_margin_mode: Literal[
        "true_vs_hardest_negative"
    ] = "true_vs_hardest_negative",
    global_residual_anti_veto_evidence_confidence_threshold: float = 0.0,
    global_residual_anti_veto_min_residual_gap: float = -0.5,
    global_residual_anti_veto_support_threshold: float = 0.0,
    global_residual_anti_veto_allowed_gap_drop: float = 0.0,
    global_residual_anti_veto_class_weighted: bool = False,
    global_residual_anti_veto_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    global_residual_anti_veto_warmup_epochs: int = 0,
    residual_contradiction_enabled: bool = False,
    residual_contradiction_weight: float = 0.0,
    residual_contradiction_evidence_gap_threshold: float = 0.0,
    residual_contradiction_min_residual_gap: float = -0.3,
    residual_contradiction_class_weighted: bool = False,
    residual_contradiction_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    residual_contradiction_warmup_epochs: int = 0,
    gate_weighted_branch_margin_enabled: bool = False,
    gate_weighted_branch_margin_weight: float = 0.0,
    gate_weighted_branch_margin_value: float = 0.0,
    gate_weighted_branch_margin_class_weighted: bool = False,
    gate_weighted_branch_margin_warmup_epochs: int = 0,
    gate_weighted_branch_margin_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    gate_weighted_branch_margin_branch_selection: Literal["gate_weighted"] = (
        "gate_weighted"
    ),
    gate_best_branch_alignment_enabled: bool = False,
    gate_best_branch_alignment_weight: float = 0.0,
    gate_best_branch_alignment_min_best_margin: float = 0.2,
    gate_best_branch_alignment_max_best_margin: float = 1.0,
    gate_best_branch_alignment_mismatch_margin_drop: float = 0.5,
    gate_best_branch_alignment_label_weight_by_class: (tuple[float, ...] | None) = None,
    gate_best_branch_alignment_min_best_margin_by_class: (
        tuple[float, ...] | None
    ) = None,
    gate_best_branch_alignment_max_best_margin_by_class: (
        tuple[float, ...] | None
    ) = None,
    gate_best_branch_alignment_mismatch_drop_by_class: (
        tuple[float, ...] | None
    ) = None,
    gate_best_branch_alignment_class_weighted: bool = False,
    gate_best_branch_alignment_warmup_epochs: int = 0,
    gate_branch_regret_enabled: bool = False,
    gate_branch_regret_weight: float = 0.0,
    gate_branch_regret_positive_threshold: float = 0.0,
    gate_branch_regret_positive_threshold_by_class: tuple[float, ...] | None = None,
    gate_branch_regret_tolerance: float = 0.0,
    gate_branch_regret_warmup_epochs: int = 0,
    gate_branch_regret_weight_schedule: (
        GateBranchRegretWeightScheduleConfig | None
    ) = None,
    gate_branch_regret_auto: AutoPositiveThresholdByTrainStatsConfig | None = None,
    gate_bad_branch_suppression_enabled: bool = False,
    gate_bad_branch_suppression_weight: float = 0.0,
    gate_bad_branch_suppression_bad_margin_threshold: float = 0.0,
    gate_bad_branch_suppression_threshold_by_class: tuple[float, ...] | None = None,
    gate_bad_branch_suppression_warmup_epochs: int = 0,
    gate_bad_branch_suppression_weight_schedule: (
        GateBranchRegretWeightScheduleConfig | None
    ) = None,
    gate_bad_branch_suppression_auto: (
        AutoBadBranchThresholdByTrainStatsConfig | None
    ) = None,
    top_branch_margin_enabled: bool = False,
    top_branch_margin_weight: float = 0.0,
    top_branch_margin_value: float = 0.0,
    top_branch_margin_by_class: tuple[float, ...] | None = None,
    top_branch_margin_class_weighted: bool = False,
    top_branch_margin_reduction: Literal[
        "mean",
        "class_balanced_violating_mean",
    ] = "mean",
    top_branch_margin_warmup_epochs: int = 0,
    top_branch_margin_auto: AutoMarginByTrainStatsConfig | None = None,
    top_branch_margin_phase_start_multiplier_by_class: (
        tuple[float, ...] | None
    ) = None,
    top_branch_margin_phase_label_multiplier_by_class: tuple[float, ...] | None = None,
    top_branch_margin_phase_schedule: TopBranchMarginPhaseWeightScheduleConfig
    | None = (None),
    top_branch_margin_hardness_weighting: TopBranchMarginHardnessWeightingConfig
    | None = (None),
) -> TrainerConfig:
    return TrainerConfig(
        device="cpu",
        epochs=1,
        encoder_lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        top_k=1,
        run_dir=run_dir,
        num_classes=num_classes,
        class_names=("normal", "crackle", "wheeze", "rhonchi")[:num_classes],
        loss_type=loss_type,
        gamma=gamma,
        pos_weight=pos_weight,
        class_weights=class_weights,
        label_smoothing=label_smoothing or LabelSmoothingConfig(),
        branch_auxiliary=BranchAuxiliaryLossConfig(
            enabled=branch_auxiliary_enabled,
            weight=branch_auxiliary_weight,
            aggregation="mean",
            weights=branch_auxiliary_weights,
        ),
        branch_binary_auxiliary=BranchBinaryAuxiliaryLossConfig(
            enabled=branch_binary_auxiliary_enabled,
            weight=branch_binary_auxiliary_weight,
            label_to_index={
                "normal": 0,
                "crackle": 1,
                "wheeze": 1,
                "rhonchi": 1,
            },
            pos_weight=BranchBinaryPosWeightConfig(
                enabled=branch_binary_pos_weight is not None
            ),
            schedule=(
                branch_binary_auxiliary_schedule
                or BranchBinaryAuxiliaryScheduleConfig()
            ),
            monitor=(
                branch_binary_auxiliary_monitor or BranchBinaryAuxiliaryMonitorConfig()
            ),
            aggregation="mean",
        ),
        branch_binary_pos_weight=branch_binary_pos_weight,
        main_index_to_binary_target=main_index_to_binary_target,
        attention_entropy=AttentionEntropyLossConfig(
            enabled=attention_entropy_enabled,
            weight=attention_entropy_weight,
        ),
        gate_entropy_regularization=GateEntropyRegularizationConfig(
            enabled=gate_entropy_regularization_enabled,
            weight=gate_entropy_regularization_weight,
            target=gate_entropy_regularization_target,
            start_epoch=gate_entropy_regularization_start_epoch,
            end_epoch=gate_entropy_regularization_end_epoch,
        ),
        class_gate_evidence_auxiliary=ClassGateEvidenceAuxiliaryConfig(
            enabled=class_gate_evidence_auxiliary_enabled,
            weight=class_gate_evidence_auxiliary_weight,
        ),
        class_gate_diversity_regularization=ClassGateDiversityRegularizationConfig(
            enabled=class_gate_diversity_regularization_enabled,
            weight=class_gate_diversity_regularization_weight,
            target=class_gate_diversity_regularization_target,
            metric="js_divergence",
            start_epoch=class_gate_diversity_regularization_start_epoch,
            end_epoch=class_gate_diversity_regularization_end_epoch,
        ),
        class_evidence_margin=ClassEvidenceMarginConfig(
            enabled=class_evidence_margin_enabled,
            weight=class_evidence_margin_weight,
            margin=class_evidence_margin_value,
            target="class_evidence_logits",
            mode=class_evidence_margin_mode,
            major_class="normal"
            if class_evidence_margin_major_index is not None
            else None,
            class_weighted=class_evidence_margin_class_weighted,
            reduction=class_evidence_margin_reduction,
            temperature=class_evidence_margin_temperature,
            support_weighting=(
                class_evidence_margin_support_weighting
                or MarginSupportWeightingConfig()
            ),
        ),
        class_evidence_margin_major_index=class_evidence_margin_major_index,
        class_evidence_gap_cap_regularization=(
            ClassEvidenceGapCapRegularizationConfig(
                enabled=class_evidence_gap_cap_enabled,
                weight=class_evidence_gap_cap_weight,
                target="class_evidence_logits",
                mode="negative_gap_hinge",
                negative_gap_cap=class_evidence_gap_cap_negative_gap_cap,
                class_weighted=class_evidence_gap_cap_class_weighted,
                reduction="mean",
                warmup_epochs=class_evidence_gap_cap_warmup_epochs,
            )
        ),
        class_evidence_gap_cap_negative_cap_by_class=(
            class_evidence_gap_cap_negative_cap_by_class
        ),
        class_evidence_gap_cap_label_weight_by_class=(
            class_evidence_gap_cap_label_weight_by_class
        ),
        class_evidence_positive_gap_cap_regularization=(
            ClassEvidencePositiveGapCapRegularizationConfig(
                enabled=class_evidence_positive_gap_cap_enabled,
                weight=class_evidence_positive_gap_cap_weight,
                target="class_evidence_logits",
                mode="positive_gap_hinge",
                positive_gap_cap=(class_evidence_positive_gap_cap_positive_gap_cap),
                class_weighted=class_evidence_positive_gap_cap_class_weighted,
                reduction="mean",
                warmup_epochs=class_evidence_positive_gap_cap_warmup_epochs,
            )
        ),
        interaction_gap_cap_regularization=InteractionGapCapRegularizationConfig(
            enabled=interaction_gap_cap_enabled,
            weight=interaction_gap_cap_weight,
            target="class_evidence_interaction_scores",
            mode="absolute_gap_hinge",
            gap_cap=interaction_gap_cap_gap_cap,
            class_weighted=interaction_gap_cap_class_weighted,
            reduction="mean",
            warmup_epochs=interaction_gap_cap_warmup_epochs,
        ),
        top_support_score_margin=TopSupportScoreMarginConfig(
            enabled=top_support_score_margin_enabled,
            weight=top_support_score_margin_weight,
            target="class_evidence_top_support_scores",
            mode="softplus_true_vs_hardest_negative",
            temperature=top_support_score_margin_temperature,
            class_weighted=top_support_score_margin_class_weighted,
            reduction="mean",
            warmup_epochs=top_support_score_margin_warmup_epochs,
            label_weight_by_label={},
            support_conditioned_multiplier=(
                top_support_score_margin_support_conditioned_multiplier
                or MarginSupportWeightingConfig()
            ),
            hardness_weighting=(
                top_support_score_margin_hardness_weighting
                or MarginHardnessWeightingConfig()
            ),
        ),
        top_support_score_margin_label_weight_by_class=(
            top_support_score_margin_label_weight_by_class
        ),
        top_support_gap_min_constraint=TopSupportGapMinConstraintConfig(
            enabled=top_support_gap_min_enabled,
            weight=top_support_gap_min_weight,
            target="class_evidence_top_support_scores",
            mode="support_conditioned_min_gap",
            base_min_gap=top_support_gap_min_base_min_gap,
            support_source="top_branch_margin",
            support_gain=top_support_gap_min_support_gain,
            support_cap=top_support_gap_min_support_cap,
            class_weighted=top_support_gap_min_class_weighted,
            reduction="mean",
            warmup_epochs=top_support_gap_min_warmup_epochs,
        ),
        top_support_gap_min_base_by_class=top_support_gap_min_base_by_class,
        class_top_branch_relative_margin=ClassTopBranchRelativeMarginConfig(
            enabled=class_top_branch_relative_margin_enabled,
            weight=class_top_branch_relative_margin_weight,
            target="class_top_branch_margin_features",
            mode="true_vs_hardest_negative_hinge",
            margin=class_top_branch_relative_margin_value,
            class_weighted=class_top_branch_relative_margin_class_weighted,
            reduction="mean",
            warmup_epochs=class_top_branch_relative_margin_warmup_epochs,
            support_weighting=(
                class_top_branch_relative_margin_support_weighting
                or MarginSupportWeightingConfig()
            ),
            hardness_weighting=(
                class_top_branch_relative_margin_hardness_weighting
                or MarginHardnessWeightingConfig()
            ),
            weak_positive_support_weighting=(
                class_top_branch_relative_margin_weak_positive_weighting
                or WeakPositiveSupportWeightingConfig()
            ),
            weak_positive_margin_boost=(
                class_top_branch_relative_margin_weak_positive_margin_boost
                or WeakPositiveMarginBoostConfig()
            ),
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
        top_teacher_gap_min_constraint=TopTeacherGapMinConstraintConfig(
            enabled=top_teacher_gap_min_enabled,
            weight=top_teacher_gap_min_weight,
            target="class_top_branch_margin_features",
            mode="support_conditioned_min_gap",
            base_min_gap=top_teacher_gap_min_base_min_gap,
            support_source="top_branch_margin",
            support_gain=top_teacher_gap_min_support_gain,
            support_cap=top_teacher_gap_min_support_cap,
            class_weighted=top_teacher_gap_min_class_weighted,
            reduction="mean",
            warmup_epochs=top_teacher_gap_min_warmup_epochs,
            weak_positive_support_weighting=(
                top_teacher_gap_min_weak_positive_weighting
                or WeakPositiveSupportWeightingConfig()
            ),
            weak_positive_target_boost=(
                top_teacher_gap_min_weak_positive_target_boost
                or WeakPositiveTargetBoostConfig()
            ),
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
        branch_support_score_margin=BranchSupportScoreMarginConfig(
            enabled=branch_support_score_margin_enabled,
            weight=branch_support_score_margin_weight,
            target="class_evidence_branch_support_scores",
            mode="softplus_true_vs_hardest_negative",
            temperature=branch_support_score_margin_temperature,
            class_weighted=branch_support_score_margin_class_weighted,
            reduction="mean",
            warmup_epochs=branch_support_score_margin_warmup_epochs,
        ),
        branch_direct_score_margin=BranchDirectScoreMarginConfig(
            enabled=branch_direct_score_margin_enabled,
            weight=branch_direct_score_margin_weight,
            target="class_evidence_branch_direct_scores",
            mode="softplus_true_vs_hardest_negative",
            temperature=branch_direct_score_margin_temperature,
            class_weighted=branch_direct_score_margin_class_weighted,
            reduction="mean",
            warmup_epochs=branch_direct_score_margin_warmup_epochs,
        ),
        branch_path_dominance_constraint=BranchPathDominanceConstraintConfig(
            enabled=branch_path_dominance_enabled,
            weight=branch_path_dominance_weight,
            branch_source="class_evidence_branch_support_scores",
            target="class_evidence_logits",
            mode="branch_gap_preservation",
            allowed_drop=branch_path_dominance_allowed_drop,
            support_source="top_branch_margin",
            support_weighting=(
                branch_path_dominance_support_weighting
                or MarginSupportWeightingConfig()
            ),
            class_weighted=branch_path_dominance_class_weighted,
            reduction="mean",
            warmup_epochs=branch_path_dominance_warmup_epochs,
        ),
        branch_path_dominance_allowed_drop_by_class=(
            branch_path_dominance_allowed_drop_by_class
        ),
        branch_path_dominance_label_weight_by_class=(
            branch_path_dominance_label_weight_by_class
        ),
        branch_support_disagreement_cap_regularization=(
            BranchSupportDisagreementCapRegularizationConfig(
                enabled=branch_support_disagreement_enabled,
                weight=branch_support_disagreement_weight,
                branch_source="class_evidence_branch_support_scores",
                embedding_source="class_evidence_embedding_scores",
                interaction_source="class_evidence_interaction_scores",
                mode="branch_gap_conditioned_abs_gap_cap",
                disagreement_threshold=branch_support_disagreement_threshold,
                condition_gain=branch_support_disagreement_condition_gain,
                condition_cap=branch_support_disagreement_condition_cap,
                embedding_gap_cap=branch_support_disagreement_embedding_gap_cap,
                interaction_gap_cap=branch_support_disagreement_interaction_gap_cap,
                class_weighted=branch_support_disagreement_class_weighted,
                reduction="mean",
                warmup_epochs=branch_support_disagreement_warmup_epochs,
            )
        ),
        class_gated_branch_logit_margin=ClassGatedBranchLogitMarginConfig(
            enabled=class_gated_branch_logit_margin_enabled,
            weight=class_gated_branch_logit_margin_weight,
            margin=class_gated_branch_logit_margin_value,
            target="class_gated_branch_logits",
            mode="true_vs_hardest_negative",
            class_weighted=class_gated_branch_logit_margin_class_weighted,
            reduction=class_gated_branch_logit_margin_reduction,
        ),
        branch_to_evidence_ranking_consistency=(
            BranchToEvidenceRankingConsistencyConfig(
                enabled=branch_to_evidence_enabled,
                weight=branch_to_evidence_weight,
                target="class_evidence_logits",
                source=branch_to_evidence_source,
                mode=branch_to_evidence_mode,
                teacher_detach=branch_to_evidence_teacher_detach,
                teacher_temperature=branch_to_evidence_teacher_temperature,
                student_temperature=branch_to_evidence_student_temperature,
                temperature=branch_to_evidence_temperature,
                teacher_gap_cap=branch_to_evidence_teacher_gap_cap,
                tolerance=branch_to_evidence_tolerance,
                class_weighted=branch_to_evidence_class_weighted,
                reduction=branch_to_evidence_reduction,
                warmup_epochs=branch_to_evidence_warmup_epochs,
                support_weighting=(
                    branch_to_evidence_support_weighting
                    or MarginSupportWeightingConfig()
                ),
                hardness_weighting=(
                    branch_to_evidence_hardness_weighting
                    or MarginHardnessWeightingConfig()
                ),
            )
        ),
        branch_to_evidence_teacher_floor_by_class=(
            branch_to_evidence_teacher_floor_by_class
        ),
        global_residual_anti_veto=GlobalResidualAntiVetoConfig(
            enabled=global_residual_anti_veto_enabled,
            weight=global_residual_anti_veto_weight,
            target=global_residual_anti_veto_target,
            reference="class_evidence_logits",
            support_source=global_residual_anti_veto_support_source,
            mode=global_residual_anti_veto_mode,
            margin_mode=global_residual_anti_veto_margin_mode,
            evidence_confidence_threshold=(
                global_residual_anti_veto_evidence_confidence_threshold
            ),
            min_residual_gap=global_residual_anti_veto_min_residual_gap,
            support_threshold=global_residual_anti_veto_support_threshold,
            allowed_gap_drop=global_residual_anti_veto_allowed_gap_drop,
            class_weighted=global_residual_anti_veto_class_weighted,
            reduction=global_residual_anti_veto_reduction,
            warmup_epochs=global_residual_anti_veto_warmup_epochs,
        ),
        residual_contradiction_regularization=(
            ResidualContradictionRegularizationConfig(
                enabled=residual_contradiction_enabled,
                weight=residual_contradiction_weight,
                target="global_residual_logits",
                reference="class_evidence_logits",
                mode="opposite_gap_penalty",
                margin_mode="true_vs_hardest_negative",
                evidence_gap_threshold=residual_contradiction_evidence_gap_threshold,
                min_residual_gap=residual_contradiction_min_residual_gap,
                class_weighted=residual_contradiction_class_weighted,
                reduction=residual_contradiction_reduction,
                warmup_epochs=residual_contradiction_warmup_epochs,
            )
        ),
        gate_weighted_branch_margin=GateWeightedBranchMarginConfig(
            enabled=gate_weighted_branch_margin_enabled,
            weight=gate_weighted_branch_margin_weight,
            margin=gate_weighted_branch_margin_value,
            target="true_class_gate",
            source="branch_logits",
            mode="true_vs_hardest_negative",
            class_weighted=gate_weighted_branch_margin_class_weighted,
            warmup_epochs=gate_weighted_branch_margin_warmup_epochs,
            reduction=gate_weighted_branch_margin_reduction,
            branch_selection=gate_weighted_branch_margin_branch_selection,
        ),
        gate_best_branch_alignment=GateBestBranchAlignmentConfig(
            enabled=gate_best_branch_alignment_enabled,
            weight=gate_best_branch_alignment_weight,
            target="true_class_gate",
            source="branch_logits",
            mode="weak_positive_best_branch_alignment",
            margin_mode="true_vs_hardest_negative",
            min_best_margin=gate_best_branch_alignment_min_best_margin,
            max_best_margin=gate_best_branch_alignment_max_best_margin,
            mismatch_margin_drop=gate_best_branch_alignment_mismatch_margin_drop,
            loss="negative_log_best_gate",
            detach_branch_margin=True,
            class_weighted=gate_best_branch_alignment_class_weighted,
            reduction="mean",
            warmup_epochs=gate_best_branch_alignment_warmup_epochs,
        ),
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
        gate_branch_regret=GateBranchRegretConfig(
            enabled=gate_branch_regret_enabled,
            weight=gate_branch_regret_weight,
            target="true_class_gate",
            source="branch_logits",
            mode="best_margin_regret",
            margin_mode="true_vs_hardest_negative",
            positive_threshold=gate_branch_regret_positive_threshold,
            tolerance=gate_branch_regret_tolerance,
            warmup_epochs=gate_branch_regret_warmup_epochs,
            weight_schedule=(
                gate_branch_regret_weight_schedule
                or GateBranchRegretWeightScheduleConfig()
            ),
            auto_positive_threshold_by_train_stats=(
                gate_branch_regret_auto or AutoPositiveThresholdByTrainStatsConfig()
            ),
        ),
        gate_branch_regret_positive_threshold_by_class=(
            gate_branch_regret_positive_threshold_by_class
        ),
        gate_bad_branch_suppression=GateBadBranchSuppressionConfig(
            enabled=gate_bad_branch_suppression_enabled,
            weight=gate_bad_branch_suppression_weight,
            target="true_class_gate",
            source="branch_logits",
            mode="margin_below_threshold",
            margin_mode="true_vs_hardest_negative",
            bad_margin_threshold=(gate_bad_branch_suppression_bad_margin_threshold),
            warmup_epochs=gate_bad_branch_suppression_warmup_epochs,
            weight_schedule=(
                gate_bad_branch_suppression_weight_schedule
                or GateBranchRegretWeightScheduleConfig()
            ),
            auto_bad_margin_threshold_by_train_stats=(
                gate_bad_branch_suppression_auto
                or AutoBadBranchThresholdByTrainStatsConfig()
            ),
        ),
        gate_bad_branch_suppression_threshold_by_class=(
            gate_bad_branch_suppression_threshold_by_class
        ),
        top_branch_margin=TopBranchMarginConfig(
            enabled=top_branch_margin_enabled,
            weight=top_branch_margin_weight,
            margin=top_branch_margin_value,
            target="branch_logits",
            mode="true_vs_hardest_negative",
            branch_reduction="max",
            class_weighted=top_branch_margin_class_weighted,
            reduction=top_branch_margin_reduction,
            warmup_epochs=top_branch_margin_warmup_epochs,
            auto_margin_by_train_stats=(
                top_branch_margin_auto or AutoMarginByTrainStatsConfig()
            ),
            phase_weight_schedule=(
                top_branch_margin_phase_schedule
                or TopBranchMarginPhaseWeightScheduleConfig()
            ),
            hardness_weighting=(
                top_branch_margin_hardness_weighting
                or TopBranchMarginHardnessWeightingConfig()
            ),
        ),
        top_branch_margin_by_class=top_branch_margin_by_class,
        top_branch_margin_phase_start_multiplier_by_class=(
            top_branch_margin_phase_start_multiplier_by_class
        ),
        top_branch_margin_phase_label_multiplier_by_class=(
            top_branch_margin_phase_label_multiplier_by_class
        ),
        analysis=_analysis_cfg(),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
        ),
    )


def _train_smoke_run(
    tmp_path: Path,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    loss_type: str,
    pos_weight: float | None = None,
    branch_auxiliary_enabled: bool = False,
) -> tuple[Path, DataConfig]:
    torch.manual_seed(0)
    train_dataset = build_dataset(data_cfg, split="train")
    val_dataset = build_dataset(data_cfg, split="val")
    train_loader = build_clip_loader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=False,
    )
    val_loader = build_clip_loader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=False,
    )

    model = build_ast_model(
        model_cfg,
        num_mel_bins=train_dataset.num_mel_bins,
        max_length=train_dataset.max_length,
        num_classes=len(data_cfg.label_to_index),
    )
    adaptation_summary = apply_encoder_adaptation(
        model,
        model.cfg.encoder.adaptation,
    )
    optimizer, optimizer_summary = build_grouped_optimizer(
        model,
        encoder_lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.0,
    )

    run_dir = tmp_path / "run"
    trainer = Trainer(
        _trainer_cfg(
            num_classes=len(data_cfg.label_to_index),
            run_dir=run_dir,
            loss_type=loss_type,
            pos_weight=pos_weight,
            branch_auxiliary_enabled=branch_auxiliary_enabled,
        )
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        extra_state={
            "model_cfg": model.cfg,
            "label_to_index": dict(data_cfg.label_to_index),
            "adaptation_summary": adaptation_summary,
            "optimizer_summary": optimizer_summary,
        },
    )
    return run_dir / "last.pt", data_cfg


def _eval_cfg(
    tmp_path: Path,
    checkpoint_path: Path,
    data_cfg: DataConfig,
) -> EvalConfig:
    return EvalConfig(
        experiment=ExperimentConfig(
            name="eval",
            task="respiratory_classification",
            mode="clip",
            seed=0,
            device="cpu",
            output_dir=str(tmp_path / "eval"),
        ),
        checkpoint_path=str(checkpoint_path),
        data=data_cfg,
        analysis=_analysis_cfg(),
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _cv_payload(root: Path, output_dir: Path) -> dict:
    return {
        "experiment": {
            "name": "cv_smoke",
            "task": "normal_vs_wheeze",
            "mode": "clip",
            "seed": 1,
            "device": "cpu",
            "output_dir": str(output_dir),
        },
        "data": {
            "train_dirs": [],
            "val_dirs": [],
            "eval_dirs": [],
            "label_to_index": {"normal": 0, "wheeze": 1},
            "batch_size": 2,
            "num_workers": 0,
            "audio": {
                "sample_rate": 16000,
                "clip_duration_sec": 1.5,
            },
            "preprocessing": {
                "source_type": "original",
                "bandpass": {"enabled": False},
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
                "adaptation": {"mode": "full", "num_layers": 0},
                "architecture": {
                    "hidden_size": 32,
                    "num_attention_heads": 4,
                    "mlp_ratio": 2.0,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1,
                    "layer_norm_eps": 1e-6,
                    "shared_stem_depth": 1,
                    "adapter_depth": 1,
                    "patch_branches": [
                        {"patch_size": [8, 8], "stride": [4, 8]},
                        {"patch_size": [4, 16], "stride": [2, 16]},
                        {"patch_size": [2, 32], "stride": [1, 32]},
                        {"patch_size": [16, 4], "stride": [8, 4]},
                    ],
                    "rdt": {
                        "enabled": True,
                        "steps": 3,
                        "top_tokens_per_branch": 2,
                        "gated_residual": True,
                        "layerscale_init": 0.01,
                    },
                },
            },
            "classifier": {
                "type": "linear",
                "hidden_dim": 32,
                "dropout": 0.0,
                "pooling": "latent_mean",
            },
        },
        "train": {
            "epochs": 1,
            "top_k": 1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "encoder_lr": 1e-4,
                "head_lr": 1e-3,
                "weight_decay": 0.0,
            },
            "scheduler": {"warmup_ratio": 0.0},
            "loss": {
                "type": "bce",
                "auto_pos_weight": False,
                "pos_weight": None,
                "gamma": 2.0,
                "branch_auxiliary": {
                    "enabled": True,
                    "weight": 0.3,
                    "aggregation": "mean",
                },
            },
            "sampler": {"weighted_random": False},
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
                "save_embeddings": True,
                "save_clip_metadata": True,
            },
        },
        "folds": [
            {
                "name": "fold_0",
                "train_dirs": [str(root / "fold_1")],
                "val_dirs": [str(root / "fold_0")],
            }
        ],
    }


def _prepare_cv_dataset(root: Path) -> Path:
    for fold_name in ("fold_0", "fold_1"):
        for label in ("normal", "wheeze"):
            (root / fold_name / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "fold_0" / "normal" / "normal_fold0.wav", 0.8, 16000)
    _write_wav(root / "fold_0" / "wheeze" / "wheeze_fold0.wav", 1.0, 16000)
    _write_wav(root / "fold_1" / "normal" / "normal_fold1.wav", 0.9, 16000)
    _write_wav(root / "fold_1" / "wheeze" / "wheeze_fold1.wav", 1.2, 16000)
    return root


def test_trainer_total_loss_matches_final_loss_when_branch_auxiliary_disabled() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.3]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_branch_auxiliary_disabled_allows_missing_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_total_loss_includes_branch_auxiliary_when_enabled() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=0.5,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.3]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is not None
    expected = final_loss + (0.5 * components.branch_auxiliary)
    assert torch.isclose(components.total, expected)


def test_trainer_branch_auxiliary_weights_override_scalar_weight() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=9.0,
            branch_auxiliary_weights=(0.1, 0.2, 0.3, 0.4),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.3]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    assert output.branch_logits is not None
    branch_losses = trainer._compute_branch_auxiliary_losses(
        criterion,
        output.branch_logits,
        labels,
    )
    weight_tensor = torch.tensor(
        [0.1, 0.2, 0.3, 0.4],
        dtype=branch_losses.dtype,
    )
    expected_auxiliary = (branch_losses * weight_tensor).sum() / weight_tensor.sum()

    assert components.branch_auxiliary is not None
    assert torch.isclose(components.branch_auxiliary, expected_auxiliary)
    assert torch.isclose(components.total, final_loss + expected_auxiliary)


def test_trainer_raises_when_branch_auxiliary_enabled_without_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="branch auxiliary loss enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_cross_entropy_label_smoothing_matches_torch_loss() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            label_smoothing=LabelSmoothingConfig(enabled=True, value=0.2),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    logits = torch.tensor(
        [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    assert torch.isclose(criterion(logits, labels), expected(logits, labels))


def test_weighted_cross_entropy_label_smoothing_matches_torch_loss() -> None:
    class_weights = torch.tensor([0.5, 1.0, 1.25, 1.25], dtype=torch.float32)
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=tuple(float(value.item()) for value in class_weights),
            label_smoothing=LabelSmoothingConfig(enabled=True, value=0.05),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )
    logits = torch.tensor(
        [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    assert torch.isclose(criterion(logits, labels), expected(logits, labels))


def test_cross_entropy_label_smoothing_disabled_uses_zero() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            label_smoothing=LabelSmoothingConfig(enabled=False, value=0.2),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    logits = torch.tensor(
        [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    assert torch.isclose(criterion(logits, labels), expected(logits, labels))


def test_two_class_cross_entropy_main_loss_uses_ce_targets() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss()
    logits = torch.tensor([[1.2, -0.4], [-0.2, 0.7]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    actual = trainer._compute_main_loss(criterion, logits, labels)

    assert torch.isclose(actual, expected(logits, labels))


def test_two_class_cross_entropy_predicts_with_softmax_argmax() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
        )
    )
    logits = torch.tensor([[1.2, -0.4], [-0.2, 0.7]], dtype=torch.float32)

    probabilities, predictions = trainer._predict(logits)

    assert probabilities.shape == (2, 2)
    assert torch.allclose(probabilities, torch.softmax(logits, dim=-1))
    assert predictions.tolist() == [0, 1]


def test_two_class_cross_entropy_branch_auxiliary_uses_ce_per_branch() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_auxiliary_enabled=True,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    branch_logits = torch.tensor(
        [
            [[1.0, -0.2], [0.1, 0.5], [0.8, -0.4]],
            [[-0.1, 0.7], [0.3, -0.3], [0.2, 0.9]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    expected = torch.stack(
        [
            criterion(branch_logits[:, branch_index, :], labels)
            for branch_index in range(3)
        ]
    )

    actual = trainer._compute_branch_auxiliary_losses(
        criterion,
        branch_logits,
        labels,
    )

    assert torch.allclose(actual, expected)


def test_trainer_total_loss_includes_weighted_ce_and_branch_binary_auxiliary() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(0.5, 1.0, 1.25, 1.25),
            branch_binary_auxiliary_enabled=True,
            branch_binary_auxiliary_weight=0.3,
            branch_binary_pos_weight=1.5,
            main_index_to_binary_target=(0, 1, 1, 1),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    branch_binary_criterion = trainer._branch_binary_criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.zeros(2, 4, 4),
        branch_binary_logits=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, 0.0, 0.5]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    branch_binary_loss = trainer._compute_branch_binary_auxiliary_loss(
        branch_binary_criterion,
        output,
        labels,
    )

    assert components.branch_auxiliary is None
    assert components.branch_binary_auxiliary is not None
    assert torch.isclose(components.main, final_loss)
    assert torch.isclose(components.branch_binary_auxiliary, branch_binary_loss)
    assert torch.isclose(components.total, final_loss + (0.3 * branch_binary_loss))


def test_branch_binary_auxiliary_cosine_schedule_resolves_epoch_weights() -> None:
    cfg = BranchBinaryAuxiliaryLossConfig(
        enabled=True,
        weight=0.3,
        schedule=BranchBinaryAuxiliaryScheduleConfig(
            enabled=True,
            type="cosine_floor",
            max_weight=0.4,
            min_weight=0.1,
            total_epochs=120,
        ),
    )

    weights = [
        resolve_branch_binary_aux_weight(cfg, epoch=epoch, total_epochs=120)
        for epoch in range(1, 121)
    ]
    expected_epoch_60 = 0.1 + (0.4 - 0.1) * 0.5 * (1.0 + np.cos(np.pi * (59.0 / 119.0)))

    assert weights[0] == pytest.approx(0.4)
    assert weights[59] == pytest.approx(expected_epoch_60)
    assert weights[-1] == pytest.approx(0.1)
    assert weights == sorted(weights, reverse=True)


def test_branch_binary_auxiliary_schedule_disabled_uses_fixed_weight() -> None:
    cfg = BranchBinaryAuxiliaryLossConfig(
        enabled=True,
        weight=0.7,
        schedule=BranchBinaryAuxiliaryScheduleConfig(enabled=False),
    )
    disabled_cfg = BranchBinaryAuxiliaryLossConfig(enabled=False, weight=0.7)

    assert resolve_branch_binary_aux_weight(cfg, epoch=10, total_epochs=20) == 0.7
    assert (
        resolve_branch_binary_aux_weight(
            disabled_cfg,
            epoch=10,
            total_epochs=20,
        )
        == 0.0
    )


def test_branch_binary_auxiliary_schedule_and_monitor_totals_are_separate() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_binary_auxiliary_enabled=True,
            branch_binary_auxiliary_weight=0.3,
            branch_binary_auxiliary_schedule=BranchBinaryAuxiliaryScheduleConfig(
                enabled=True,
                type="cosine_floor",
                max_weight=0.4,
                min_weight=0.1,
                total_epochs=120,
            ),
            branch_binary_auxiliary_monitor=BranchBinaryAuxiliaryMonitorConfig(
                loss_weight=0.3
            ),
            main_index_to_binary_target=(0, 1, 1, 1),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    branch_binary_criterion = trainer._branch_binary_criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        branch_binary_logits=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, 0.0, 0.5]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    components = trainer._compute_total_loss(
        criterion,
        output,
        labels,
        epoch=120,
        use_monitor_total=True,
    )
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    branch_binary_loss = trainer._compute_branch_binary_auxiliary_loss(
        branch_binary_criterion,
        output,
        labels,
    )

    assert components.branch_binary_aux_weight == pytest.approx(0.1)
    assert components.branch_binary_aux_monitor_weight == pytest.approx(0.3)
    assert torch.isclose(
        components.total_scheduled,
        final_loss + (0.1 * branch_binary_loss),
    )
    assert torch.isclose(
        components.total_monitor,
        final_loss + (0.3 * branch_binary_loss),
    )
    assert torch.isclose(components.total, components.total_monitor)


def test_trainer_raises_when_branch_binary_auxiliary_missing_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_binary_auxiliary_enabled=True,
            main_index_to_binary_target=(0, 1, 1, 1),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        branch_binary_logits=None,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="branch_binary_logits"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_attention_entropy_disabled_adds_no_term() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
            attention_entropy_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_attention_weights=(
            torch.tensor([[0.75, 0.25], [0.60, 0.40]], dtype=torch.float32),
            torch.tensor([[0.20, 0.80], [0.55, 0.45]], dtype=torch.float32),
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss)


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_trainer_focal_loss_builder_accepts_round_g_gamma_values(
    gamma: float,
) -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="focal",
            gamma=gamma,
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=0.1,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)

    assert trainer.cfg.gamma == gamma
    assert torch.isfinite(components.total)
    assert components.branch_auxiliary is not None
    assert torch.isfinite(components.branch_auxiliary)


def test_trainer_attention_entropy_enabled_adds_weighted_entropy() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
            attention_entropy_enabled=True,
            attention_entropy_weight=0.25,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_attention_weights=(
            torch.tensor([[0.75, 0.25], [0.60, 0.40]], dtype=torch.float32),
            torch.tensor([[0.20, 0.80], [0.55, 0.45]], dtype=torch.float32),
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    entropy_loss = trainer._compute_attention_entropy_loss(output)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss + (0.25 * entropy_loss))


def test_trainer_raises_when_attention_entropy_enabled_without_attention() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            attention_entropy_enabled=True,
            attention_entropy_weight=0.1,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_attention_weights=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="attention entropy loss enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_gate_entropy_regularization_disabled_adds_no_term() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        evidence_gate_entropy=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.gate_entropy is None
    assert components.gate_entropy_regularization is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_evidence_auxiliary_disabled_adds_no_term() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.evidence_auxiliary_loss is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_evidence_auxiliary_enabled_adds_weighted_ce() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=True,
            class_gate_evidence_auxiliary_weight=0.2,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[0.1, 0.8, -0.2], [0.7, 0.1, 0.0]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    assert output.class_evidence_logits is not None
    evidence_loss = trainer._compute_main_loss(
        criterion,
        output.class_evidence_logits,
        labels,
    )

    assert components.evidence_auxiliary_loss is not None
    assert torch.isclose(components.evidence_auxiliary_loss, evidence_loss)
    assert torch.isclose(components.total, final_loss + (0.2 * evidence_loss))


def test_trainer_raises_when_evidence_auxiliary_enabled_without_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=True,
            class_gate_evidence_auxiliary_weight=0.2,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="class gate evidence auxiliary loss enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_gate_entropy_regularization_subtracts_weighted_entropy() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        evidence_gate_entropy=torch.tensor([0.2, 0.6], dtype=torch.float32),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    gate_entropy, regularization = trainer._compute_gate_entropy_regularization(output)

    assert torch.isclose(gate_entropy, torch.tensor(0.4))
    assert torch.isclose(regularization, torch.tensor(-0.1))
    assert components.gate_entropy is not None
    assert torch.isclose(components.gate_entropy, gate_entropy)
    assert components.gate_entropy_regularization is not None
    assert torch.isclose(components.gate_entropy_regularization, regularization)
    assert torch.isclose(components.total, final_loss + regularization)


def test_trainer_gate_entropy_regularization_supports_class_gate_targets() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.5,
            gate_entropy_regularization_target="true_class_evidence_gate",
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_entropy=torch.tensor(
            [[0.2, 0.6, 0.4], [0.9, 0.3, 0.7]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    gate_entropy, regularization = trainer._compute_gate_entropy_regularization(
        output,
        labels,
    )

    assert torch.isclose(gate_entropy, torch.tensor(0.75))
    assert torch.isclose(regularization, torch.tensor(-0.375))
    assert torch.isclose(components.total, final_loss + regularization)


def test_trainer_gate_entropy_regularization_supports_all_class_gate_target() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
            gate_entropy_regularization_target="class_evidence_gate",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_entropy=torch.tensor(
            [[0.2, 0.6, 0.4], [0.9, 0.3, 0.7]],
            dtype=torch.float32,
        ),
    )

    gate_entropy, regularization = trainer._compute_gate_entropy_regularization(output)

    assert torch.isclose(gate_entropy, torch.tensor(0.5166667), atol=1e-6)
    assert torch.isclose(regularization, torch.tensor(-0.1291667), atol=1e-6)


def test_trainer_gate_entropy_epoch_window_supports_learned_gate() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.5,
            gate_entropy_regularization_target="class_evidence_learned_gate",
            gate_entropy_regularization_start_epoch=11,
            gate_entropy_regularization_end_epoch=30,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_learned_gate_entropy=torch.tensor(
            [[0.2, 0.6, 0.4], [0.8, 0.3, 0.7]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    inactive_before = trainer._compute_total_loss(
        criterion,
        output,
        labels,
        epoch=10,
    )
    active = trainer._compute_total_loss(criterion, output, labels, epoch=20)
    inactive_after = trainer._compute_total_loss(
        criterion,
        output,
        labels,
        epoch=31,
    )

    expected_entropy = torch.tensor(0.5)
    expected_regularization = torch.tensor(-0.25)
    assert inactive_before.gate_entropy is None
    assert torch.isclose(active.gate_entropy, expected_entropy)
    assert torch.isclose(active.gate_entropy_regularization, expected_regularization)
    assert inactive_after.gate_entropy_regularization is None


def test_trainer_gate_entropy_regularization_combines_with_branch_auxiliary() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=0.1,
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2]]),
        evidence_gate_entropy=torch.tensor([0.2, 0.6], dtype=torch.float32),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    branch_loss = trainer._compute_branch_auxiliary_loss(
        criterion,
        output.branch_logits,
        labels,
    )
    _, regularization = trainer._compute_gate_entropy_regularization(output)

    assert components.branch_auxiliary is not None
    assert torch.isclose(
        components.total,
        final_loss + (0.1 * branch_loss) + regularization,
    )


def test_trainer_class_gate_diversity_regularization_subtracts_js_divergence() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_weights=torch.tensor(
            [
                [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    diversity, regularization = trainer._compute_class_gate_diversity_regularization(
        output
    )

    assert diversity > 0
    assert torch.isclose(regularization, -0.3 * diversity)
    assert components.class_gate_diversity is not None
    assert components.class_gate_diversity_regularization is not None
    assert torch.isclose(components.total, final_loss + regularization)


def test_trainer_class_gate_diversity_epoch_window_supports_learned_gate() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
            class_gate_diversity_regularization_target="class_evidence_learned_gate",
            class_gate_diversity_regularization_start_epoch=11,
            class_gate_diversity_regularization_end_epoch=30,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_learned_gate_weights=torch.tensor(
            [
                [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    inactive_before = trainer._compute_total_loss(
        criterion,
        output,
        labels,
        epoch=10,
    )
    active = trainer._compute_total_loss(criterion, output, labels, epoch=20)
    inactive_after = trainer._compute_total_loss(
        criterion,
        output,
        labels,
        epoch=31,
    )
    diversity, regularization = trainer._compute_class_gate_diversity_regularization(
        output
    )

    assert inactive_before.class_gate_diversity is None
    assert active.class_gate_diversity is not None
    assert active.class_gate_diversity_regularization is not None
    assert torch.isclose(active.class_gate_diversity, diversity)
    assert torch.isclose(active.class_gate_diversity_regularization, regularization)
    assert inactive_after.class_gate_diversity_regularization is None


def test_trainer_class_evidence_margin_minority_vs_major() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.05,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="minority_vs_major",
            class_evidence_margin_major_index=0,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(3, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(3, 32),
        class_evidence_logits=torch.tensor(
            [
                [1.3, 0.2, -0.1],
                [0.2, 0.4, 0.1],
                [0.9, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert torch.isclose(raw_margin, torch.tensor(0.35))
    assert torch.isclose(weighted_margin, torch.tensor(0.0175))
    assert torch.isclose(components.total, final_loss + weighted_margin)
    assert torch.isclose(components.class_evidence_margin, raw_margin)
    assert torch.isclose(components.class_evidence_margin_loss, weighted_margin)


def test_trainer_class_evidence_margin_minority_vs_major_zero_without_minority() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.05,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="minority_vs_major",
            class_evidence_margin_major_index=0,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[1.3, 0.2, -0.1], [0.9, 0.5, 0.2]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 0], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    assert torch.isclose(raw_margin, torch.tensor(0.0))
    assert torch.isclose(weighted_margin, torch.tensor(0.0))


def test_trainer_class_evidence_margin_true_vs_hardest_negative() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.2,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    assert torch.isclose(raw_margin, torch.tensor(0.5))
    assert torch.isclose(weighted_margin, torch.tensor(0.1))


def test_trainer_class_evidence_margin_softplus_true_vs_hardest_negative() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.05,
            class_evidence_margin_value=0.0,
            class_evidence_margin_mode="softplus_true_vs_hardest_negative",
            class_evidence_margin_class_weighted=True,
            class_evidence_margin_reduction="mean",
            class_evidence_margin_temperature=1.0,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    gaps = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    penalties = F.softplus(-gaps)
    expected_raw = torch.mean(penalties * torch.tensor([1.5, 2.0]))
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)


def test_trainer_branch_support_score_margin_softplus_true_vs_hardest_negative() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            branch_support_score_margin_enabled=True,
            branch_support_score_margin_weight=0.05,
            branch_support_score_margin_temperature=1.0,
            branch_support_score_margin_class_weighted=True,
            branch_support_score_margin_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_branch_support_scores=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup = trainer._compute_branch_support_score_margin_loss(
        output,
        labels,
        epoch=10,
    )
    raw_margin, weighted_margin = trainer._compute_branch_support_score_margin_loss(
        output,
        labels,
        epoch=11,
    )

    gaps = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    penalties = F.softplus(-gaps)
    expected_raw = torch.mean(penalties * torch.tensor([1.5, 2.0]))
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)


def test_trainer_class_evidence_gap_cap_regularization_hinges_large_negative_gap() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            class_evidence_gap_cap_enabled=True,
            class_evidence_gap_cap_weight=0.02,
            class_evidence_gap_cap_negative_gap_cap=3.0,
            class_evidence_gap_cap_negative_cap_by_class=(2.5, 3.0, 3.0),
            class_evidence_gap_cap_label_weight_by_class=(2.0, 1.0, 1.0),
            class_evidence_gap_cap_class_weighted=True,
            class_evidence_gap_cap_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[-2.0, 2.0, 0.0], [1.0, 0.5, 0.2]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup, warmup_label_mult, warmup_cap = (
        trainer._compute_class_evidence_gap_cap_regularization_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_loss, weighted_loss, label_mult, cap_mean = (
        trainer._compute_class_evidence_gap_cap_regularization_loss(
            output,
            labels,
            epoch=11,
        )
    )

    gaps = torch.tensor([-4.0, -0.5], dtype=torch.float32)
    caps = torch.tensor([2.5, 3.0])
    label_multipliers = torch.tensor([2.0, 1.0])
    penalties = torch.relu(-gaps - caps) * label_multipliers
    expected_raw = torch.mean(penalties * torch.tensor([1.5, 2.0]))
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(warmup_label_mult, torch.tensor(0.0))
    assert torch.isclose(warmup_cap, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.02 * expected_raw)
    assert torch.isclose(label_mult, label_multipliers.mean())
    assert torch.isclose(cap_mean, caps.mean())


def test_trainer_class_evidence_positive_gap_cap_regularization_hinges_large_gap() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_positive_gap_cap_enabled=True,
            class_evidence_positive_gap_cap_weight=0.01,
            class_evidence_positive_gap_cap_positive_gap_cap=2.0,
            class_evidence_positive_gap_cap_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[4.0, 1.0, 0.0], [0.2, 1.0, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup = (
        trainer._compute_class_evidence_positive_gap_cap_regularization_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_loss, weighted_loss = (
        trainer._compute_class_evidence_positive_gap_cap_regularization_loss(
            output,
            labels,
            epoch=11,
        )
    )

    gaps = torch.tensor([3.0, 0.4], dtype=torch.float32)
    expected_raw = torch.relu(gaps - 2.0).mean()
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.01 * expected_raw)


def test_trainer_interaction_gap_cap_regularization_hinges_abs_gap() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            interaction_gap_cap_enabled=True,
            interaction_gap_cap_weight=0.01,
            interaction_gap_cap_gap_cap=2.0,
            interaction_gap_cap_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_interaction_scores=torch.tensor(
            [[4.0, 1.0, 0.0], [0.2, 1.0, 4.0]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup = (
        trainer._compute_interaction_gap_cap_regularization_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_loss, weighted_loss = trainer._compute_interaction_gap_cap_regularization_loss(
        output,
        labels,
        epoch=11,
    )

    gaps = torch.tensor([3.0, -3.0], dtype=torch.float32)
    expected_raw = torch.relu(gaps.abs() - 2.0).mean()
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.01 * expected_raw)


def test_trainer_top_support_score_margin_softplus_true_vs_hardest_negative() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            top_support_score_margin_enabled=True,
            top_support_score_margin_weight=0.05,
            top_support_score_margin_temperature=1.0,
            top_support_score_margin_class_weighted=True,
            top_support_score_margin_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_top_support_scores=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup, _, _, _ = (
        trainer._compute_top_support_score_margin_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_margin, weighted_margin, _, _, _ = (
        trainer._compute_top_support_score_margin_loss(
            output,
            labels,
            epoch=11,
        )
    )

    gaps = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    penalties = F.softplus(-gaps)
    expected_raw = torch.mean(penalties * torch.tensor([1.5, 2.0]))
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)


def test_trainer_top_support_score_margin_applies_label_support_and_hardness_multiplier() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_support_score_margin_enabled=True,
            top_support_score_margin_weight=0.05,
            top_support_score_margin_temperature=1.0,
            top_support_score_margin_class_weighted=False,
            top_support_score_margin_warmup_epochs=0,
            top_support_score_margin_label_weight_by_class=(1.0, 2.0, 1.0),
            top_support_score_margin_support_conditioned_multiplier=(
                MarginSupportWeightingConfig(
                    enabled=True,
                    source="top_branch_margin",
                    mode="linear",
                    gain=1.0,
                    cap=2.0,
                )
            ),
            top_support_score_margin_hardness_weighting=(
                MarginHardnessWeightingConfig(
                    enabled=True,
                    source="top_support_gap",
                    mode="negative_gap",
                    gain=1.0,
                    cap=3.0,
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_top_support_scores=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
        branch_logits=torch.tensor(
            [
                [[1.0, 0.5, 0.0], [0.2, 0.1, 0.0]],
                [[0.0, 1.2, 0.4], [0.0, 0.7, 0.2]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin, label_mult, support_mult, hardness_mult = (
        trainer._compute_top_support_score_margin_loss(
            output,
            labels,
            epoch=1,
        )
    )

    gaps = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    base_penalties = F.softplus(-gaps)
    label_multipliers = torch.tensor([1.0, 2.0], dtype=torch.float32)
    support_multipliers = torch.tensor([1.5, 1.8], dtype=torch.float32)
    hardness_multipliers = torch.tensor([1.3, 1.0], dtype=torch.float32)
    expected_raw = torch.mean(
        base_penalties * label_multipliers * support_multipliers * hardness_multipliers
    )
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)
    assert torch.isclose(label_mult, label_multipliers.mean())
    assert torch.isclose(support_mult, support_multipliers.mean())
    assert torch.isclose(hardness_mult, hardness_multipliers.mean())


def test_trainer_top_support_gap_min_constraint_is_support_conditioned() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            top_support_gap_min_enabled=True,
            top_support_gap_min_weight=0.05,
            top_support_gap_min_base_by_class=(0.0, 0.2, 0.0),
            top_support_gap_min_support_gain=0.5,
            top_support_gap_min_support_cap=2.0,
            top_support_gap_min_class_weighted=True,
            top_support_gap_min_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_top_support_scores=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
        class_top_branch_margin_features=torch.tensor(
            [[0.6, 0.0, 1.0], [0.0, 1.4, 0.2]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup, target_warmup = (
        trainer._compute_top_support_gap_min_constraint_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_loss, weighted_loss, target_mean = (
        trainer._compute_top_support_gap_min_constraint_loss(
            output,
            labels,
            epoch=11,
        )
    )

    target_min_gap = torch.tensor([0.3, 0.9], dtype=torch.float32)
    top_support_gap = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    class_weights = torch.tensor([1.5, 2.0], dtype=torch.float32)
    expected_raw = torch.mean(
        torch.relu(target_min_gap - top_support_gap) * class_weights
    )
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(target_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.05 * expected_raw)
    assert torch.isclose(target_mean, target_min_gap.mean())


def test_trainer_class_top_branch_relative_margin_uses_support_and_hardness() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            class_top_branch_relative_margin_enabled=True,
            class_top_branch_relative_margin_weight=0.05,
            class_top_branch_relative_margin_value=0.3,
            class_top_branch_relative_margin_class_weighted=True,
            class_top_branch_relative_margin_warmup_epochs=10,
            class_top_branch_relative_margin_support_weighting=(
                MarginSupportWeightingConfig(
                    enabled=True,
                    source="top_branch_margin",
                    mode="linear",
                    gain=0.75,
                    cap=2.0,
                )
            ),
            class_top_branch_relative_margin_hardness_weighting=(
                MarginHardnessWeightingConfig(
                    enabled=True,
                    source="teacher_gap_deficit",
                    mode="linear",
                    gain=0.75,
                    cap=2.0,
                )
            ),
            class_top_branch_relative_margin_weak_positive_weighting=(
                WeakPositiveSupportWeightingConfig(
                    enabled=True,
                    min_support=0.3,
                    max_support=1.0,
                    multiplier=1.25,
                )
            ),
            class_top_branch_relative_margin_weak_positive_margin_boost=(
                WeakPositiveMarginBoostConfig(
                    enabled=True,
                    min_support=0.3,
                    max_support=1.0,
                    boost=0.2,
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[1.0, 0.2, 0.0], [0.1, 0.0, 0.0]],
                [[0.2, 0.4, 0.1], [0.0, 0.3, 0.1]],
            ],
            dtype=torch.float32,
        ),
        class_top_branch_margin_features=torch.tensor(
            [[0.4, 0.6, 0.1], [0.2, 0.7, 0.3]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    (
        raw_warmup,
        weighted_warmup,
        support_warmup,
        hardness_warmup,
        weak_positive_warmup,
        effective_margin_warmup,
        margin_boost_warmup,
        label_margin_warmup,
        hardness_gain_warmup,
        hardness_cap_warmup,
    ) = trainer._compute_class_top_branch_relative_margin_loss(
        output,
        labels,
        epoch=10,
    )
    (
        raw_loss,
        weighted_loss,
        support_mean,
        hardness_mean,
        weak_positive_mean,
        effective_margin_mean,
        margin_boost_mean,
        label_margin_mean,
        hardness_gain_mean,
        hardness_cap_mean,
    ) = trainer._compute_class_top_branch_relative_margin_loss(
        output,
        labels,
        epoch=11,
    )

    margin_deficit = torch.tensor([0.7, 0.0], dtype=torch.float32)
    support_weights = torch.tensor([1.6, 1.15], dtype=torch.float32)
    hardness_weights = torch.tensor([1.525, 1.0], dtype=torch.float32)
    weak_positive_weights = torch.tensor([1.25, 1.0], dtype=torch.float32)
    effective_margins = torch.tensor([0.5, 0.3], dtype=torch.float32)
    margin_boosts = torch.tensor([0.2, 0.0], dtype=torch.float32)
    class_weights = torch.tensor([1.5, 2.0], dtype=torch.float32)
    expected_raw = torch.mean(
        margin_deficit
        * support_weights
        * hardness_weights
        * weak_positive_weights
        * class_weights
    )
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(support_warmup, torch.tensor(0.0))
    assert torch.isclose(hardness_warmup, torch.tensor(0.0))
    assert torch.isclose(weak_positive_warmup, torch.tensor(0.0))
    assert torch.isclose(effective_margin_warmup, torch.tensor(0.0))
    assert torch.isclose(margin_boost_warmup, torch.tensor(0.0))
    assert torch.isclose(label_margin_warmup, torch.tensor(0.0))
    assert torch.isclose(hardness_gain_warmup, torch.tensor(0.0))
    assert torch.isclose(hardness_cap_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.05 * expected_raw)
    assert torch.isclose(support_mean, support_weights.mean())
    assert torch.isclose(hardness_mean, hardness_weights.mean())
    assert torch.isclose(weak_positive_mean, weak_positive_weights.mean())
    assert torch.isclose(effective_margin_mean, effective_margins.mean())
    assert torch.isclose(margin_boost_mean, margin_boosts.mean())
    assert torch.isclose(label_margin_mean, torch.tensor(0.3))
    assert torch.isclose(hardness_gain_mean, torch.tensor(0.75))
    assert torch.isclose(hardness_cap_mean, torch.tensor(2.0))


def test_trainer_top_teacher_gap_min_constraint_is_support_conditioned() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            top_teacher_gap_min_enabled=True,
            top_teacher_gap_min_weight=0.1,
            top_teacher_gap_min_base_min_gap=0.0,
            top_teacher_gap_min_support_gain=0.75,
            top_teacher_gap_min_support_cap=2.0,
            top_teacher_gap_min_class_weighted=True,
            top_teacher_gap_min_warmup_epochs=10,
            top_teacher_gap_min_weak_positive_weighting=(
                WeakPositiveSupportWeightingConfig(
                    enabled=True,
                    min_support=0.3,
                    max_support=1.0,
                    multiplier=1.5,
                )
            ),
            top_teacher_gap_min_weak_positive_target_boost=(
                WeakPositiveTargetBoostConfig(
                    enabled=True,
                    min_support=0.3,
                    max_support=1.0,
                    boost=0.3,
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[1.0, 0.2, 0.0], [0.1, 0.0, 0.0]],
                [[0.2, 0.4, 0.1], [0.0, 0.3, 0.1]],
            ],
            dtype=torch.float32,
        ),
        class_top_branch_margin_features=torch.tensor(
            [[0.4, 0.6, 0.1], [0.2, 0.7, 0.3]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    (
        raw_warmup,
        weighted_warmup,
        target_warmup,
        weak_positive_warmup,
        target_boost_warmup,
        base_min_gap_warmup,
        support_gain_warmup,
    ) = trainer._compute_top_teacher_gap_min_constraint_loss(
        output,
        labels,
        epoch=10,
    )
    (
        raw_loss,
        weighted_loss,
        target_mean,
        weak_positive_mean,
        target_boost_mean,
        base_min_gap_mean,
        support_gain_mean,
    ) = trainer._compute_top_teacher_gap_min_constraint_loss(
        output,
        labels,
        epoch=11,
    )

    teacher_gap = torch.tensor([-0.2, 0.4], dtype=torch.float32)
    target_min_gap = torch.tensor([0.9, 0.15], dtype=torch.float32)
    penalties = torch.relu(target_min_gap - teacher_gap)
    weak_positive_weights = torch.tensor([1.5, 1.0], dtype=torch.float32)
    target_boosts = torch.tensor([0.3, 0.0], dtype=torch.float32)
    class_weights = torch.tensor([1.5, 2.0], dtype=torch.float32)
    expected_raw = torch.mean(penalties * weak_positive_weights * class_weights)
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(target_warmup, torch.tensor(0.0))
    assert torch.isclose(weak_positive_warmup, torch.tensor(0.0))
    assert torch.isclose(target_boost_warmup, torch.tensor(0.0))
    assert torch.isclose(base_min_gap_warmup, torch.tensor(0.0))
    assert torch.isclose(support_gain_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.1 * expected_raw)
    assert torch.isclose(target_mean, target_min_gap.mean())
    assert torch.isclose(weak_positive_mean, weak_positive_weights.mean())
    assert torch.isclose(target_boost_mean, target_boosts.mean())
    assert torch.isclose(base_min_gap_mean, torch.tensor(0.0))
    assert torch.isclose(support_gain_mean, torch.tensor(0.75))


def test_trainer_teacher_relative_losses_use_label_specific_weak_positive_band() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_top_branch_relative_margin_enabled=True,
            class_top_branch_relative_margin_weight=0.05,
            class_top_branch_relative_margin_value=0.3,
            class_top_branch_relative_margin_support_weighting=(
                MarginSupportWeightingConfig(
                    enabled=True,
                    source="top_branch_margin",
                    mode="linear",
                    gain=0.75,
                    cap=2.0,
                )
            ),
            class_top_branch_relative_margin_hardness_weighting=(
                MarginHardnessWeightingConfig(
                    enabled=True,
                    source="teacher_gap_deficit",
                    mode="linear",
                    gain=0.5,
                    cap=2.0,
                )
            ),
            class_top_branch_relative_margin_by_class=(0.3, 0.3, 0.5),
            class_top_branch_relative_hardness_gain_by_class=(0.5, 0.75, 1.0),
            class_top_branch_relative_hardness_cap_by_class=(2.0, 2.5, 3.0),
            class_top_branch_relative_weak_support_max_by_class=(1.0, 1.0, 1.2),
            class_top_branch_relative_margin_boost_by_class=(0.1, 0.2, 0.4),
            class_top_branch_relative_margin_boost_max_by_class=(1.0, 1.0, 1.2),
            class_top_branch_relative_margin_weak_positive_weighting=(
                WeakPositiveSupportWeightingConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    multiplier=1.25,
                )
            ),
            class_top_branch_relative_margin_weak_positive_margin_boost=(
                WeakPositiveMarginBoostConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    boost=0.2,
                )
            ),
            top_teacher_gap_min_enabled=True,
            top_teacher_gap_min_weight=0.1,
            top_teacher_gap_min_base_min_gap=0.0,
            top_teacher_gap_min_support_gain=0.75,
            top_teacher_gap_min_support_cap=2.0,
            top_teacher_gap_min_base_by_class=(0.0, 0.0, 0.2),
            top_teacher_gap_min_support_gain_by_class=(0.5, 0.5, 1.0),
            top_teacher_gap_min_weak_support_max_by_class=(1.0, 1.0, 1.2),
            top_teacher_gap_min_target_boost_by_class=(0.2, 0.2, 0.6),
            top_teacher_gap_min_target_boost_max_by_class=(1.0, 1.0, 1.2),
            top_teacher_gap_min_weak_positive_weighting=(
                WeakPositiveSupportWeightingConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    multiplier=1.5,
                )
            ),
            top_teacher_gap_min_weak_positive_target_boost=(
                WeakPositiveTargetBoostConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    boost=0.3,
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[1.2, 0.1, 0.0], [0.1, 0.0, 0.0]],
                [[0.2, 0.1, 0.3], [0.1, 0.1, 1.2]],
            ],
            dtype=torch.float32,
        ),
        class_top_branch_margin_features=torch.tensor(
            [[0.4, 0.6, 0.1], [0.3, 0.7, 0.5]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    (
        rel_raw,
        rel_weighted,
        rel_support_mean,
        rel_hardness_mean,
        rel_weak_mean,
        rel_effective_margin_mean,
        rel_margin_boost_mean,
        rel_label_margin_mean,
        rel_hardness_gain_mean,
        rel_hardness_cap_mean,
    ) = trainer._compute_class_top_branch_relative_margin_loss(
        output,
        labels,
        epoch=11,
    )
    (
        min_raw,
        min_weighted,
        min_target_mean,
        min_weak_mean,
        min_target_boost_mean,
        min_base_mean,
        min_support_gain_mean,
    ) = trainer._compute_top_teacher_gap_min_constraint_loss(
        output,
        labels,
        epoch=11,
    )

    support_weight = torch.tensor([1.825, 1.825], dtype=torch.float32)
    rel_deficit = torch.tensor([0.5, 1.1], dtype=torch.float32)
    rel_hardness = torch.tensor([1.25, 2.1], dtype=torch.float32)
    rel_weak = torch.tensor([1.0, 1.25], dtype=torch.float32)
    expected_rel_raw = torch.mean(
        rel_deficit * support_weight * rel_hardness * rel_weak
    )
    min_targets = torch.tensor([0.55, 1.9], dtype=torch.float32)
    teacher_gap = torch.tensor([-0.2, -0.2], dtype=torch.float32)
    min_weak = torch.tensor([1.0, 1.5], dtype=torch.float32)
    expected_min_raw = torch.mean(torch.relu(min_targets - teacher_gap) * min_weak)

    assert torch.isclose(rel_raw, expected_rel_raw)
    assert torch.isclose(rel_weighted, 0.05 * expected_rel_raw)
    assert torch.isclose(rel_support_mean, support_weight.mean())
    assert torch.isclose(rel_hardness_mean, rel_hardness.mean())
    assert torch.isclose(rel_weak_mean, rel_weak.mean())
    assert torch.isclose(rel_effective_margin_mean, torch.tensor(0.6))
    assert torch.isclose(rel_margin_boost_mean, torch.tensor(0.2))
    assert torch.isclose(rel_label_margin_mean, torch.tensor(0.4))
    assert torch.isclose(rel_hardness_gain_mean, torch.tensor(0.75))
    assert torch.isclose(rel_hardness_cap_mean, torch.tensor(2.5))
    assert torch.isclose(min_raw, expected_min_raw)
    assert torch.isclose(min_weighted, 0.1 * expected_min_raw)
    assert torch.isclose(min_target_mean, min_targets.mean())
    assert torch.isclose(min_weak_mean, min_weak.mean())
    assert torch.isclose(min_target_boost_mean, torch.tensor(0.3))
    assert torch.isclose(min_base_mean, torch.tensor(0.1))
    assert torch.isclose(min_support_gain_mean, torch.tensor(0.75))


def test_trainer_branch_direct_score_margin_softplus_true_vs_hardest_negative() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            branch_direct_score_margin_enabled=True,
            branch_direct_score_margin_weight=0.05,
            branch_direct_score_margin_temperature=1.0,
            branch_direct_score_margin_class_weighted=True,
            branch_direct_score_margin_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_branch_direct_scores=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup = trainer._compute_branch_direct_score_margin_loss(
        output,
        labels,
        epoch=10,
    )
    raw_margin, weighted_margin = trainer._compute_branch_direct_score_margin_loss(
        output,
        labels,
        epoch=11,
    )

    gaps = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    penalties = F.softplus(-gaps)
    expected_raw = torch.mean(penalties * torch.tensor([1.5, 2.0]))
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)


def test_trainer_branch_support_disagreement_cap_regularization_is_conditioned() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_support_disagreement_enabled=True,
            branch_support_disagreement_weight=0.02,
            branch_support_disagreement_threshold=0.0,
            branch_support_disagreement_condition_gain=1.0,
            branch_support_disagreement_condition_cap=3.0,
            branch_support_disagreement_embedding_gap_cap=6.0,
            branch_support_disagreement_interaction_gap_cap=4.0,
            branch_support_disagreement_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_branch_support_scores=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
        class_evidence_embedding_scores=torch.tensor(
            [[8.0, 0.5, 0.0], [1.0, 0.0, 7.0]],
            dtype=torch.float32,
        ),
        class_evidence_interaction_scores=torch.tensor(
            [[4.5, 0.0, 0.0], [0.0, 5.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_warmup, weighted_warmup, weight_warmup = (
        trainer._compute_branch_support_disagreement_cap_regularization_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_loss, weighted_loss, weight_mean = (
        trainer._compute_branch_support_disagreement_cap_regularization_loss(
            output,
            labels,
            epoch=11,
        )
    )

    disagreement = torch.tensor([1.3, 1.0], dtype=torch.float32)
    embedding_gap = torch.tensor([7.5, -7.0], dtype=torch.float32)
    interaction_gap = torch.tensor([4.5, 5.0], dtype=torch.float32)
    expected_raw = torch.mean(
        disagreement
        * (
            torch.relu(embedding_gap.abs() - 6.0)
            + torch.relu(interaction_gap.abs() - 4.0)
        )
    )
    assert torch.isclose(raw_warmup, torch.tensor(0.0))
    assert torch.isclose(weighted_warmup, torch.tensor(0.0))
    assert torch.isclose(weight_warmup, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.02 * expected_raw)
    assert torch.isclose(weight_mean, disagreement.mean())


def test_trainer_class_evidence_margin_applies_class_weights() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.5, 2.0, 0.5),
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.2,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
            class_evidence_margin_class_weighted=True,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[1.0, 0.4, 1.3], [0.2, 0.9, 0.6]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    assert torch.isclose(raw_margin, torch.tensor(0.8))
    assert torch.isclose(weighted_margin, torch.tensor(0.16))


def test_trainer_class_evidence_margin_applies_support_weighting() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.2,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
            class_evidence_margin_support_weighting=MarginSupportWeightingConfig(
                enabled=True,
                gain=2.0,
                cap=0.5,
            ),
        )
    )
    branch_logits = torch.tensor(
        [
            [[1.0, 0.0, 0.2], [0.3, 0.1, 0.2]],
            [[0.0, 0.4, 0.1], [0.0, 0.2, 0.6]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=branch_logits,
        class_evidence_logits=torch.tensor(
            [[0.4, 0.5, 0.0], [0.1, 0.2, 0.6]],
            dtype=torch.float32,
            requires_grad=True,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    expected_raw = torch.tensor(((0.6 * 2.0) + (0.9 * 1.6)) / 2.0)
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.2 * expected_raw)
    weighted_margin.backward()
    assert branch_logits.grad is None


def test_trainer_class_evidence_margin_class_balanced_violating_mean() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(2.0, 3.0, 5.0),
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.2,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
            class_evidence_margin_class_weighted=True,
            class_evidence_margin_reduction="class_balanced_violating_mean",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(4, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(4, 32),
        class_evidence_logits=torch.tensor(
            [
                [1.0, 0.4, 0.2],
                [0.2, 0.7, 0.1],
                [0.3, 0.2, 0.1],
                [0.0, 0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    expected_raw = torch.tensor((2.0 * 1.0 + 3.0 * 0.6) / 2)
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.2 * expected_raw)


def test_trainer_class_evidence_margin_class_balanced_zero_without_violations() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.2,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
            class_evidence_margin_reduction="class_balanced_violating_mean",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[1.0, 0.2, 0.1], [0.0, 1.0, 0.2]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_evidence_margin_loss(
        output,
        labels,
    )

    assert torch.isclose(raw_margin, torch.tensor(0.0))
    assert torch.isclose(weighted_margin, torch.tensor(0.0))


def test_trainer_class_evidence_margin_requires_weights_when_class_weighted() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.2,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
            class_evidence_margin_class_weighted=True,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        class_evidence_logits=torch.tensor([[1.0, 0.4, 1.3]], dtype=torch.float32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="class_evidence_margin.class_weighted"):
        trainer._compute_class_evidence_margin_loss(output, labels)


def test_trainer_class_gated_branch_logit_margin_true_vs_hardest_negative() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.0, 2.0, 3.0),
            class_gated_branch_logit_margin_enabled=True,
            class_gated_branch_logit_margin_weight=0.25,
            class_gated_branch_logit_margin_value=0.3,
            class_gated_branch_logit_margin_class_weighted=True,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_gated_branch_logits=torch.tensor(
            [[0.6, 0.1, 0.2], [0.2, 0.4, 0.9]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    raw_margin, weighted_margin = trainer._compute_class_gated_branch_logit_margin_loss(
        output, labels
    )
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert torch.isclose(raw_margin, torch.tensor(0.8))
    assert torch.isclose(weighted_margin, torch.tensor(0.2))
    assert torch.isclose(components.total, final_loss + weighted_margin)
    assert torch.isclose(components.class_gated_branch_logit_margin, raw_margin)
    assert torch.isclose(
        components.class_gated_branch_logit_margin_loss,
        weighted_margin,
    )


def test_trainer_class_gated_branch_logit_margin_class_balanced_reduction() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gated_branch_logit_margin_enabled=True,
            class_gated_branch_logit_margin_weight=0.25,
            class_gated_branch_logit_margin_value=0.5,
            class_gated_branch_logit_margin_reduction=("class_balanced_violating_mean"),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(4, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(4, 32),
        class_gated_branch_logits=torch.tensor(
            [
                [1.0, 0.4, 0.2],
                [0.2, 0.7, 0.1],
                [0.3, 0.2, 0.1],
                [0.0, 0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    raw_margin, weighted_margin = trainer._compute_class_gated_branch_logit_margin_loss(
        output,
        labels,
    )

    expected_raw = torch.tensor((1.0 + 0.6) / 2)
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.25 * expected_raw)


def test_trainer_raises_when_branch_logit_margin_enabled_without_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gated_branch_logit_margin_enabled=True,
            class_gated_branch_logit_margin_weight=0.25,
            class_gated_branch_logit_margin_value=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="class-gated branch logit margin enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_branch_to_evidence_consistency_uses_detached_branch_teacher() -> None:
    source_logits = torch.tensor(
        [[1.0, 0.2, 0.5], [0.3, 1.0, 0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    evidence_logits = torch.tensor(
        [[0.7, 0.1, 0.6], [0.1, 0.8, 0.6]],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.05,
            branch_to_evidence_teacher_detach=True,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=evidence_logits,
        class_gated_branch_logits=source_logits,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=1,
        )
    )
    weighted_loss.backward()

    assert torch.isclose(raw_loss, torch.tensor(0.4))
    assert torch.isclose(weighted_loss, torch.tensor(0.02))
    assert torch.isclose(
        components.branch_to_evidence_ranking_consistency,
        raw_loss,
    )
    assert torch.isclose(
        components.branch_to_evidence_ranking_consistency_loss,
        weighted_loss.detach(),
    )
    assert source_logits.grad is None
    assert evidence_logits.grad is not None


def test_trainer_branch_to_evidence_consistency_warmup_returns_zero() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.05,
            branch_to_evidence_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        class_evidence_logits=torch.tensor([[0.1, 0.0, 0.2]], dtype=torch.float32),
        class_gated_branch_logits=torch.tensor([[1.0, 0.0, 0.2]], dtype=torch.float32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=10,
        )
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))


def test_trainer_branch_to_evidence_consistency_uses_top_branch_teacher() -> None:
    branch_logits = torch.tensor(
        [[[1.0, 0.2, 0.1], [0.4, 0.3, 0.2]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    evidence_logits = torch.tensor(
        [[0.2, 0.4, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="top_branch_margin",
            branch_to_evidence_teacher_detach=True,
            branch_to_evidence_teacher_gap_cap=0.7,
            branch_to_evidence_teacher_floor_by_class=(0.5, 0.0, 0.0),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=branch_logits,
        class_evidence_logits=evidence_logits,
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=1,
        )
    )
    weighted_loss.backward()

    assert torch.isclose(raw_loss, torch.tensor(0.9))
    assert torch.isclose(weighted_loss, torch.tensor(0.09))
    assert branch_logits.grad is None
    assert evidence_logits.grad is not None


def test_trainer_branch_to_evidence_consistency_applies_support_and_hardness_weights() -> (
    None
):
    branch_logits = torch.tensor(
        [
            [[1.0, 0.0, 0.2], [0.3, 0.1, 0.2]],
            [[0.0, 0.4, 0.1], [0.0, 0.2, 0.6]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    evidence_logits = torch.tensor(
        [[0.4, 0.5, 0.0], [0.1, 0.2, 0.6]],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="top_branch_margin",
            branch_to_evidence_teacher_detach=True,
            branch_to_evidence_support_weighting=MarginSupportWeightingConfig(
                enabled=True,
                gain=2.0,
                cap=0.5,
            ),
            branch_to_evidence_hardness_weighting=MarginHardnessWeightingConfig(
                enabled=True,
                gain=3.0,
                cap=0.2,
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=branch_logits,
        class_evidence_logits=evidence_logits,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=1,
        )
    )

    expected_raw = torch.tensor(((0.9 * 2.0 * 1.3) + (0.7 * 1.6 * 1.6)) / 2.0)
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.1 * expected_raw)
    weighted_loss.backward()
    assert branch_logits.grad is None
    assert evidence_logits.grad is not None


def test_trainer_branch_to_evidence_teacher_distribution_kl() -> None:
    teacher_logits = torch.tensor(
        [[1.2, 0.2, -0.4], [0.1, 1.1, 0.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    evidence_logits = torch.tensor(
        [[0.5, 0.3, -0.2], [0.4, 0.2, 0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.0, 2.0, 3.0),
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="class_top_branch_margin_features",
            branch_to_evidence_mode="teacher_distribution_kl",
            branch_to_evidence_teacher_detach=True,
            branch_to_evidence_teacher_temperature=0.7,
            branch_to_evidence_student_temperature=1.0,
            branch_to_evidence_class_weighted=True,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=evidence_logits,
        class_top_branch_margin_features=teacher_logits,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=1,
        )
    )

    teacher_probs = F.softmax(teacher_logits.detach() / 0.7, dim=-1)
    student_log_probs = F.log_softmax(evidence_logits / 1.0, dim=-1)
    per_sample = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    expected_raw = (per_sample * torch.tensor([1.0, 2.0])).mean()
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.1 * expected_raw)
    weighted_loss.backward()
    assert teacher_logits.grad is None
    assert evidence_logits.grad is not None


def test_trainer_branch_to_evidence_teacher_distribution_kl_warmup_returns_zero() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="class_top_branch_margin_features",
            branch_to_evidence_mode="teacher_distribution_kl",
            branch_to_evidence_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        class_evidence_logits=torch.tensor([[0.1, 0.0, 0.2]], dtype=torch.float32),
        class_top_branch_margin_features=torch.tensor(
            [[0.3, 0.2, 0.1]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=10,
        )
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))


def test_trainer_branch_to_evidence_teacher_distribution_kl_requires_teacher() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="class_top_branch_margin_features",
            branch_to_evidence_mode="teacher_distribution_kl",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        class_evidence_logits=torch.tensor([[0.1, 0.0, 0.2]], dtype=torch.float32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="class_top_branch_margin_features"):
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=1,
        )


def test_trainer_branch_to_evidence_true_label_anchored_softplus() -> None:
    support_features = torch.tensor(
        [[0.8, -0.2, 0.1], [-0.1, 0.5, 0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    evidence_logits = torch.tensor(
        [[0.5, 0.3, -0.2], [0.4, 0.2, 0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.0, 2.0, 3.0),
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="class_top_branch_margin_relative_features",
            branch_to_evidence_mode="true_label_anchored_softplus",
            branch_to_evidence_temperature=1.0,
            branch_to_evidence_class_weighted=True,
            branch_to_evidence_support_weighting=MarginSupportWeightingConfig(
                enabled=True,
                source="same_as_source",
                mode="positive_linear",
                gain=1.0,
                cap=2.0,
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=evidence_logits,
        class_top_branch_margin_relative_features=support_features,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_loss, weighted_loss = (
        trainer._compute_branch_to_evidence_ranking_consistency_loss(
            output,
            labels,
            epoch=1,
        )
    )

    gap = torch.tensor([0.2, -0.6])
    support_weight = torch.tensor([1.8, 1.5])
    class_weights = torch.tensor([1.0, 2.0])
    expected_raw = (F.softplus(-gap) * support_weight * class_weights).mean()
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.1 * expected_raw)
    weighted_loss.backward()
    assert support_features.grad is None
    assert evidence_logits.grad is not None


def test_trainer_global_residual_anti_veto_filters_by_evidence_confidence() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            global_residual_anti_veto_enabled=True,
            global_residual_anti_veto_weight=0.02,
            global_residual_anti_veto_evidence_confidence_threshold=0.2,
            global_residual_anti_veto_min_residual_gap=-0.5,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[1.0, 0.2, 0.4], [0.5, 0.6, 0.2]],
            dtype=torch.float32,
        ),
        global_residual_logits=torch.tensor(
            [[0.0, 0.1, 0.7], [0.1, 0.2, 0.0]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    raw_loss, weighted_loss, eligible_fraction = (
        trainer._compute_global_residual_anti_veto_loss(
            output,
            labels,
            epoch=1,
        )
    )

    assert torch.isclose(raw_loss, torch.tensor(0.2))
    assert torch.isclose(weighted_loss, torch.tensor(0.004))
    assert torch.isclose(eligible_fraction, torch.tensor(0.5))
    assert torch.isclose(components.global_residual_anti_veto, raw_loss)
    assert torch.isclose(components.global_residual_anti_veto_loss, weighted_loss)
    assert torch.isclose(
        components.global_residual_anti_veto_eligible_fraction,
        eligible_fraction,
    )


def test_trainer_global_residual_anti_veto_warmup_returns_zero() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            global_residual_anti_veto_enabled=True,
            global_residual_anti_veto_weight=0.02,
            global_residual_anti_veto_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        class_evidence_logits=torch.tensor([[1.0, 0.2, 0.4]], dtype=torch.float32),
        global_residual_logits=torch.tensor([[0.0, 0.1, 0.7]], dtype=torch.float32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss, eligible_fraction = (
        trainer._compute_global_residual_anti_veto_loss(
            output,
            labels,
            epoch=10,
        )
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))
    assert torch.isclose(eligible_fraction, torch.tensor(0.0))


def test_trainer_global_residual_anti_veto_preserves_final_gap() -> None:
    final_logits = torch.tensor(
        [[0.3, 0.1, 0.4], [0.1, 0.2, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    evidence_logits = torch.tensor(
        [[1.0, 0.2, 0.5], [0.4, 0.9, 0.6]],
        dtype=torch.float32,
        requires_grad=True,
    )
    branch_logits = torch.tensor(
        [
            [[1.1, 0.2, 0.3], [0.1, 0.2, 0.0]],
            [[0.1, 0.2, 0.1], [0.0, 0.1, 0.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            global_residual_anti_veto_enabled=True,
            global_residual_anti_veto_weight=0.05,
            global_residual_anti_veto_target="final_logits",
            global_residual_anti_veto_support_source="top_branch_margin",
            global_residual_anti_veto_mode="final_gap_preservation",
            global_residual_anti_veto_support_threshold=0.3,
            global_residual_anti_veto_allowed_gap_drop=0.3,
        )
    )
    output = AstModelOutput(
        logits=final_logits,
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=branch_logits,
        class_evidence_logits=evidence_logits,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_loss, weighted_loss, eligible_fraction = (
        trainer._compute_global_residual_anti_veto_loss(
            output,
            labels,
            epoch=1,
        )
    )
    weighted_loss.backward()

    assert torch.isclose(raw_loss, torch.tensor(0.3))
    assert torch.isclose(weighted_loss, torch.tensor(0.015))
    assert torch.isclose(eligible_fraction, torch.tensor(0.5))
    assert final_logits.grad is not None
    assert evidence_logits.grad is None
    assert branch_logits.grad is None


def test_trainer_residual_contradiction_regularization_penalizes_opposite_gap() -> None:
    evidence_logits = torch.tensor(
        [[1.0, 0.4, 0.0], [0.1, 0.8, 0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    residual_logits = torch.tensor(
        [[0.0, 0.6, 0.1], [0.2, 0.0, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            residual_contradiction_enabled=True,
            residual_contradiction_weight=0.02,
            residual_contradiction_evidence_gap_threshold=0.0,
            residual_contradiction_min_residual_gap=-0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=evidence_logits,
        global_residual_logits=residual_logits,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    raw_loss, weighted_loss, eligible_fraction = (
        trainer._compute_residual_contradiction_regularization_loss(
            output,
            labels,
            epoch=1,
        )
    )

    assert torch.isclose(raw_loss, torch.tensor(0.15))
    assert torch.isclose(weighted_loss, torch.tensor(0.003))
    assert torch.isclose(eligible_fraction, torch.tensor(1.0))
    assert torch.isclose(components.residual_contradiction_regularization, raw_loss)
    assert torch.isclose(
        components.residual_contradiction_regularization_loss,
        weighted_loss,
    )
    weighted_loss.backward()
    assert residual_logits.grad is not None
    assert evidence_logits.grad is None


def test_trainer_residual_contradiction_regularization_warmup_returns_zero() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            residual_contradiction_enabled=True,
            residual_contradiction_weight=0.02,
            residual_contradiction_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        class_evidence_logits=torch.tensor([[1.0, 0.4, 0.0]], dtype=torch.float32),
        global_residual_logits=torch.tensor([[0.0, 0.6, 0.1]], dtype=torch.float32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss, eligible_fraction = (
        trainer._compute_residual_contradiction_regularization_loss(
            output,
            labels,
            epoch=10,
        )
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))
    assert torch.isclose(eligible_fraction, torch.tensor(0.0))


def test_trainer_gate_weighted_branch_margin_is_zero_during_warmup() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_weighted_branch_margin_enabled=True,
            gate_weighted_branch_margin_weight=0.01,
            gate_weighted_branch_margin_value=0.3,
            gate_weighted_branch_margin_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss = trainer._compute_gate_weighted_branch_margin_loss(
        output,
        labels,
        epoch=10,
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))


def test_trainer_gate_weighted_branch_margin_uses_detached_true_class_gate() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_weighted_branch_margin_enabled=True,
            gate_weighted_branch_margin_weight=0.01,
            gate_weighted_branch_margin_value=0.3,
            gate_weighted_branch_margin_warmup_epochs=10,
        )
    )
    branch_logits = torch.tensor(
        [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gate_weights = torch.tensor(
        [[[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=branch_logits,
        class_evidence_gate_weights=gate_weights,
    )
    labels = torch.tensor([0], dtype=torch.long)

    raw_loss, weighted_loss = trainer._compute_gate_weighted_branch_margin_loss(
        output,
        labels,
        epoch=11,
    )
    branch_margin = torch.tensor([[0.7, -0.6]])
    branch_penalty = torch.relu(torch.tensor(0.3) - branch_margin)
    expected_raw = (torch.tensor([[0.6, 0.4]]) * branch_penalty).sum(dim=1).mean()

    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.01 * expected_raw)

    weighted_loss.backward()
    assert branch_logits.grad is not None
    assert gate_weights.grad is None


def test_trainer_gate_weighted_branch_margin_applies_class_weights() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(1.0, 3.0, 2.0),
            gate_weighted_branch_margin_enabled=True,
            gate_weighted_branch_margin_weight=0.5,
            gate_weighted_branch_margin_value=0.4,
            gate_weighted_branch_margin_class_weighted=True,
        )
    )
    branch_logits = torch.tensor(
        [
            [[0.8, 0.1, 0.2], [0.2, 0.3, 0.9]],
            [[0.1, 0.7, 0.3], [0.6, 0.2, 0.1]],
        ],
        dtype=torch.float32,
    )
    gate_weights = torch.tensor(
        [
            [[0.75, 0.25], [0.2, 0.8], [0.5, 0.5]],
            [[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=branch_logits,
        class_evidence_gate_weights=gate_weights,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_loss, weighted_loss = trainer._compute_gate_weighted_branch_margin_loss(
        output,
        labels,
        epoch=1,
    )

    expected_sample0 = 0.75 * max(0.0, 0.4 - (0.8 - 0.2)) + 0.25 * max(
        0.0,
        0.4 - (0.2 - 0.9),
    )
    expected_sample1 = 0.6 * max(0.0, 0.4 - (0.7 - 0.3)) + 0.4 * max(
        0.0,
        0.4 - (0.2 - 0.6),
    )
    expected_raw = torch.tensor((1.0 * expected_sample0 + 3.0 * expected_sample1) / 2)

    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.5 * expected_raw)


def test_trainer_gate_branch_regret_is_zero_during_warmup() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.01,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_tolerance=0.05,
            gate_branch_regret_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        weight_multiplier,
        effective_weight,
    ) = trainer._compute_gate_branch_regret_loss(
        output,
        labels,
        epoch=10,
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))
    assert torch.isclose(eligible_fraction, torch.tensor(0.0))
    assert torch.isclose(weight_multiplier, torch.tensor(0.0))
    assert torch.isclose(effective_weight, torch.tensor(0.0))


def test_trainer_gate_branch_regret_updates_gate_not_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.5,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_tolerance=0.05,
        )
    )
    branch_logits = torch.tensor(
        [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gate_weights = torch.tensor(
        [[[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=branch_logits,
        class_evidence_gate_weights=gate_weights,
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        weight_multiplier,
        effective_weight,
    ) = trainer._compute_gate_branch_regret_loss(
        output,
        labels,
        epoch=1,
    )

    branch_margin = torch.tensor([[0.7, -0.6]])
    best_margin = torch.tensor(0.7)
    gate_expected_margin = (torch.tensor([[0.2, 0.8]]) * branch_margin).sum()
    expected_raw = torch.relu(best_margin - gate_expected_margin - 0.05)

    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.5 * expected_raw)
    assert torch.isclose(eligible_fraction, torch.tensor(1.0))
    assert torch.isclose(weight_multiplier, torch.tensor(1.0))
    assert torch.isclose(effective_weight, torch.tensor(0.5))

    weighted_loss.backward()
    assert branch_logits.grad is None
    assert gate_weights.grad is not None


def test_trainer_gate_best_branch_alignment_updates_gate_not_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(2.0, 1.0, 1.0),
            gate_best_branch_alignment_enabled=True,
            gate_best_branch_alignment_weight=0.03,
            gate_best_branch_alignment_min_best_margin=0.2,
            gate_best_branch_alignment_max_best_margin=1.0,
            gate_best_branch_alignment_mismatch_margin_drop=0.5,
            gate_best_branch_alignment_class_weighted=True,
        )
    )
    branch_logits = torch.tensor(
        [[[1.0, 0.4, 0.0], [0.2, 0.7, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gate_weights = torch.tensor(
        [[[0.2, 0.8], [0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=branch_logits,
        class_evidence_gate_weights=gate_weights,
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        mismatch_fraction,
        best_gate_mean,
        label_multiplier_mean,
        min_best_margin_mean,
        max_best_margin_mean,
        mismatch_drop_mean,
    ) = trainer._compute_gate_best_branch_alignment_loss(output, labels, epoch=1)

    expected_raw = -torch.log(torch.tensor(0.2)) * 2.0
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.03 * expected_raw)
    assert torch.isclose(eligible_fraction, torch.tensor(1.0))
    assert torch.isclose(mismatch_fraction, torch.tensor(1.0))
    assert torch.isclose(best_gate_mean, torch.tensor(0.2))
    assert torch.isclose(label_multiplier_mean, torch.tensor(1.0))
    assert torch.isclose(min_best_margin_mean, torch.tensor(0.2))
    assert torch.isclose(max_best_margin_mean, torch.tensor(1.0))
    assert torch.isclose(mismatch_drop_mean, torch.tensor(0.5))

    weighted_loss.backward()
    assert branch_logits.grad is None
    assert gate_weights.grad is not None


@pytest.mark.parametrize(
    ("gate_row", "branch_logits", "expected_mismatch"),
    [
        (
            [[0.8, 0.2], [0.5, 0.5], [0.5, 0.5]],
            [[[1.0, 0.4, 0.0], [0.2, 0.7, 0.0]]],
            False,
        ),
        (
            [[0.2, 0.8], [0.5, 0.5], [0.5, 0.5]],
            [[[1.5, 0.1, 0.0], [0.2, 0.7, 0.0]]],
            True,
        ),
    ],
)
def test_trainer_gate_best_branch_alignment_ineligible_conditions(
    gate_row: list[list[float]],
    branch_logits: list[list[list[float]]],
    expected_mismatch: bool,
) -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_best_branch_alignment_enabled=True,
            gate_best_branch_alignment_weight=0.03,
            gate_best_branch_alignment_min_best_margin=0.2,
            gate_best_branch_alignment_max_best_margin=1.0,
            gate_best_branch_alignment_mismatch_margin_drop=0.5,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(branch_logits, dtype=torch.float32),
        class_evidence_gate_weights=torch.tensor([gate_row], dtype=torch.float32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        mismatch_fraction,
        _best_gate_mean,
        _label_multiplier_mean,
        _min_best_margin_mean,
        _max_best_margin_mean,
        _mismatch_drop_mean,
    ) = trainer._compute_gate_best_branch_alignment_loss(output, labels, epoch=1)

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))
    assert torch.isclose(eligible_fraction, torch.tensor(0.0))
    assert torch.isclose(mismatch_fraction, torch.tensor(float(expected_mismatch)))


def test_trainer_gate_best_branch_alignment_uses_label_specific_band() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_best_branch_alignment_enabled=True,
            gate_best_branch_alignment_weight=0.03,
            gate_best_branch_alignment_min_best_margin=0.2,
            gate_best_branch_alignment_max_best_margin=1.0,
            gate_best_branch_alignment_mismatch_margin_drop=0.5,
            gate_best_branch_alignment_label_weight_by_class=(1.0, 0.75, 0.25),
            gate_best_branch_alignment_max_best_margin_by_class=(1.0, 1.0, 1.2),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(
            [[[0.2, 0.1, 0.3], [0.1, 0.1, 1.2]]],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5], [0.8, 0.2]]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([2], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        mismatch_fraction,
        best_gate_mean,
        label_multiplier_mean,
        min_best_margin_mean,
        max_best_margin_mean,
        mismatch_drop_mean,
    ) = trainer._compute_gate_best_branch_alignment_loss(output, labels, epoch=1)

    expected_raw = -torch.log(torch.tensor(0.2)) * 0.25
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.03 * expected_raw)
    assert torch.isclose(eligible_fraction, torch.tensor(1.0))
    assert torch.isclose(mismatch_fraction, torch.tensor(1.0))
    assert torch.isclose(best_gate_mean, torch.tensor(0.2))
    assert torch.isclose(label_multiplier_mean, torch.tensor(0.25))
    assert torch.isclose(min_best_margin_mean, torch.tensor(0.2))
    assert torch.isclose(max_best_margin_mean, torch.tensor(1.2))
    assert torch.isclose(mismatch_drop_mean, torch.tensor(0.5))


def test_trainer_gate_branch_regret_is_zero_when_no_positive_branch_margin() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.5,
            gate_branch_regret_positive_threshold=0.8,
            gate_branch_regret_tolerance=0.05,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(
            [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [[[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        weight_multiplier,
        effective_weight,
    ) = trainer._compute_gate_branch_regret_loss(
        output,
        labels,
        epoch=1,
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))
    assert torch.isclose(eligible_fraction, torch.tensor(0.0))
    assert torch.isclose(weight_multiplier, torch.tensor(1.0))
    assert torch.isclose(effective_weight, torch.tensor(0.5))


def test_trainer_raises_when_gate_branch_regret_enabled_without_outputs() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.01,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="gate-branch regret enabled"):
        trainer._compute_total_loss(criterion, output, labels, epoch=1)


def test_trainer_gate_branch_regret_is_recorded_in_loss_components() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.5,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_tolerance=0.05,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(
            [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [[[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels, epoch=1)

    assert components.gate_branch_regret is not None
    assert components.gate_branch_regret_loss is not None
    assert components.gate_branch_regret_eligible_fraction is not None
    assert components.gate_branch_regret_weight_multiplier is not None
    assert components.gate_branch_regret_effective_weight is not None
    assert torch.isclose(
        components.gate_branch_regret_eligible_fraction,
        torch.tensor(1.0),
    )
    assert torch.isclose(
        components.gate_branch_regret_weight_multiplier,
        torch.tensor(1.0),
    )
    assert torch.isclose(
        components.gate_branch_regret_effective_weight,
        torch.tensor(0.5),
    )


def test_trainer_gate_branch_regret_weight_schedule() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.01,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_tolerance=0.05,
            gate_branch_regret_warmup_epochs=15,
            gate_branch_regret_weight_schedule=(
                GateBranchRegretWeightScheduleConfig(
                    enabled=True,
                    start_epoch=16,
                    end_epoch=30,
                    start_multiplier=0.2,
                    end_multiplier=1.0,
                )
            ),
        )
    )
    branch_logits = torch.tensor(
        [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
        dtype=torch.float32,
    )
    gate_weights = torch.tensor(
        [[[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]],
        dtype=torch.float32,
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=branch_logits,
        class_evidence_gate_weights=gate_weights,
    )
    labels = torch.tensor([0], dtype=torch.long)

    expected_raw = torch.tensor(0.7 - ((0.2 * 0.7) + (0.8 * -0.6)) - 0.05)
    cases = (
        (15, 0.0, 0.0),
        (16, 0.2, 0.002),
        (23, 0.2 + ((23 - 16) / (30 - 16)) * 0.8, 0.006),
        (30, 1.0, 0.01),
        (31, 1.0, 0.01),
    )
    for epoch, expected_multiplier, expected_weight in cases:
        (
            raw_loss,
            weighted_loss,
            eligible_fraction,
            weight_multiplier,
            effective_weight,
        ) = trainer._compute_gate_branch_regret_loss(output, labels, epoch=epoch)

        if epoch <= 15:
            assert torch.isclose(raw_loss, torch.tensor(0.0))
            assert torch.isclose(eligible_fraction, torch.tensor(0.0))
        else:
            assert torch.isclose(raw_loss, expected_raw)
            assert torch.isclose(eligible_fraction, torch.tensor(1.0))
        assert weight_multiplier.item() == pytest.approx(expected_multiplier)
        assert effective_weight.item() == pytest.approx(expected_weight)
        assert weighted_loss.item() == pytest.approx(
            (expected_raw.item() if epoch > 15 else 0.0) * expected_weight
        )


def test_trainer_gate_branch_regret_uses_label_specific_thresholds() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.5,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_positive_threshold_by_class=(0.8, 0.3, 0.0),
            gate_branch_regret_tolerance=0.0,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]],
                [[0.4, 0.1, 0.5], [0.2, 0.4, 0.2]],
            ],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [
                [[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]],
                [[0.5, 0.5], [0.4, 0.6], [0.5, 0.5]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        eligible_fraction,
        weight_multiplier,
        effective_weight,
    ) = trainer._compute_gate_branch_regret_loss(output, labels, epoch=1)

    expected_raw = torch.tensor(0.15)
    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.5 * expected_raw)
    assert torch.isclose(eligible_fraction, torch.tensor(0.5))
    assert torch.isclose(weight_multiplier, torch.tensor(1.0))
    assert torch.isclose(effective_weight, torch.tensor(0.5))


def test_trainer_gate_bad_branch_suppression_is_zero_during_warmup() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.025,
            gate_bad_branch_suppression_warmup_epochs=15,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        bad_gate_mass,
        weight_multiplier,
        effective_weight,
    ) = trainer._compute_gate_bad_branch_suppression_loss(
        output,
        labels,
        epoch=15,
    )

    assert torch.isclose(raw_loss, torch.tensor(0.0))
    assert torch.isclose(weighted_loss, torch.tensor(0.0))
    assert torch.isclose(bad_gate_mass, torch.tensor(0.0))
    assert torch.isclose(weight_multiplier, torch.tensor(0.0))
    assert torch.isclose(effective_weight, torch.tensor(0.0))


def test_trainer_gate_bad_branch_suppression_updates_gate_not_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.5,
            gate_bad_branch_suppression_bad_margin_threshold=0.0,
        )
    )
    branch_logits = torch.tensor(
        [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gate_weights = torch.tensor(
        [[[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=branch_logits,
        class_evidence_gate_weights=gate_weights,
    )
    labels = torch.tensor([0], dtype=torch.long)

    (
        raw_loss,
        weighted_loss,
        bad_gate_mass,
        weight_multiplier,
        effective_weight,
    ) = trainer._compute_gate_bad_branch_suppression_loss(
        output,
        labels,
        epoch=1,
    )

    branch_margin = torch.tensor([[0.7, -0.6]])
    expected_raw = (
        torch.tensor([[0.2, 0.8]]) * torch.relu(torch.tensor(0.0) - branch_margin)
    ).sum()

    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.5 * expected_raw)
    assert torch.isclose(bad_gate_mass, torch.tensor(0.8))
    assert torch.isclose(weight_multiplier, torch.tensor(1.0))
    assert torch.isclose(effective_weight, torch.tensor(0.5))

    weighted_loss.backward()
    assert branch_logits.grad is None
    assert gate_weights.grad is not None


def test_trainer_gate_bad_branch_suppression_weight_schedule() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.025,
            gate_bad_branch_suppression_bad_margin_threshold=0.0,
            gate_bad_branch_suppression_warmup_epochs=15,
            gate_bad_branch_suppression_weight_schedule=(
                GateBranchRegretWeightScheduleConfig(
                    enabled=True,
                    start_epoch=16,
                    end_epoch=25,
                    start_multiplier=0.3,
                    end_multiplier=1.0,
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(
            [[[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]]],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [[[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0], dtype=torch.long)

    expected = {
        15: 0.0,
        16: 0.3,
        25: 1.0,
        26: 1.0,
    }
    for epoch, expected_multiplier in expected.items():
        (
            _raw_loss,
            _weighted_loss,
            _bad_gate_mass,
            weight_multiplier,
            effective_weight,
        ) = trainer._compute_gate_bad_branch_suppression_loss(
            output,
            labels,
            epoch=epoch,
        )
        assert torch.isclose(weight_multiplier, torch.tensor(expected_multiplier))
        assert torch.isclose(
            effective_weight,
            torch.tensor(0.025 * expected_multiplier),
        )


def test_trainer_gate_bad_branch_suppression_uses_label_specific_thresholds() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.5,
            gate_bad_branch_suppression_bad_margin_threshold=0.0,
            gate_bad_branch_suppression_threshold_by_class=(-0.2, 0.1, 0.1),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[1.2, 0.2, 0.5], [0.3, 0.1, 0.9]],
                [[0.2, 0.6, 0.1], [0.1, 0.5, 0.0]],
            ],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [
                [[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]],
                [[0.4, 0.6], [0.75, 0.25], [0.5, 0.5]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_loss, weighted_loss, bad_gate_mass, _, _ = (
        trainer._compute_gate_bad_branch_suppression_loss(
            output,
            labels,
            epoch=1,
        )
    )

    expected_sample0 = 0.8 * max(0.0, -0.2 - (-0.6))
    expected_sample1 = 0.25 * max(0.0, 0.1 - 0.4)
    expected_raw = torch.tensor((expected_sample0 + expected_sample1) / 2)

    assert torch.isclose(raw_loss, expected_raw)
    assert torch.isclose(weighted_loss, 0.5 * expected_raw)
    assert torch.isclose(bad_gate_mass, torch.tensor((0.8 + 0.0) / 2))


def test_trainer_gate_weighted_branch_margin_rejects_unsupported_selection() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_weighted_branch_margin_enabled=True,
            gate_weighted_branch_margin_weight=0.5,
            gate_weighted_branch_margin_value=0.4,
        )
    )
    object.__setattr__(
        trainer.cfg.gate_weighted_branch_margin,
        "branch_selection",
        "topk_gate",
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.zeros(1, 2, 3, dtype=torch.float32),
        class_evidence_gate_weights=torch.tensor(
            [[[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="branch_selection"):
        trainer._compute_gate_weighted_branch_margin_loss(
            output,
            labels,
            epoch=1,
        )


def test_trainer_raises_when_gate_weighted_branch_margin_enabled_without_outputs() -> (
    None
):
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_weighted_branch_margin_enabled=True,
            gate_weighted_branch_margin_weight=0.01,
            gate_weighted_branch_margin_value=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="gate-weighted branch margin enabled"):
        trainer._compute_total_loss(criterion, output, labels, epoch=1)


def test_trainer_top_branch_margin_uses_best_branch_margin() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.05,
            top_branch_margin_value=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[0.8, 0.6, 0.5], [0.4, 0.3, 0.2]],
                [[0.7, 0.6, 0.2], [0.4, 0.5, 0.3]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels, epoch=1)
    raw_margin, weighted_margin, _, _ = trainer._compute_top_branch_margin_loss(
        output,
        labels,
        epoch=1,
    )
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    expected_raw = torch.tensor((0.1 + 0.2) / 2)
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)
    assert torch.isclose(components.total, final_loss + weighted_margin)
    assert torch.isclose(components.top_branch_margin, raw_margin)
    assert torch.isclose(components.top_branch_margin_loss, weighted_margin)


def test_trainer_top_branch_margin_uses_label_specific_margins() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.1,
            top_branch_margin_value=0.3,
            top_branch_margin_by_class=(0.3, 0.3, 0.6),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[0.8, 0.6, 0.5], [0.4, 0.3, 0.2]],
                [[0.0, 0.1, 0.4], [0.1, 0.0, 0.5]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    raw_margin, weighted_margin, _, _ = trainer._compute_top_branch_margin_loss(
        output,
        labels,
        epoch=1,
    )

    expected_raw = torch.tensor((0.1 + 0.2) / 2)
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.1 * expected_raw)


def test_trainer_top_branch_margin_applies_hardness_weighting() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.1,
            top_branch_margin_value=0.5,
            top_branch_margin_hardness_weighting=(
                TopBranchMarginHardnessWeightingConfig(
                    enabled=True,
                    source="margin_deficit",
                    mode="linear",
                    gain=1.0,
                    cap=3.0,
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[0.5, 0.3, 0.0]],
                [[0.2, 0.4, 0.3]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_margin, weighted_margin, _, hardness_mean = (
        trainer._compute_top_branch_margin_loss(
            output,
            labels,
            epoch=1,
        )
    )

    penalties = torch.tensor([0.3, 0.4])
    hardness = 1.0 + penalties
    assert torch.isclose(raw_margin, penalties.mean())
    assert torch.isclose(
        weighted_margin, torch.tensor(0.1) * (penalties * hardness).mean()
    )
    assert torch.isclose(hardness_mean, hardness.mean())


def test_trainer_top_branch_margin_applies_phase_label_multiplier() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.1,
            top_branch_margin_value=0.3,
            top_branch_margin_phase_label_multiplier_by_class=(1.0, 2.0, 1.0),
            top_branch_margin_phase_schedule=(
                TopBranchMarginPhaseWeightScheduleConfig(
                    enabled=True,
                    start_epoch=31,
                    label_multiplier_by_label={},
                )
            ),
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor(
            [
                [[0.8, 0.6, 0.5], [0.4, 0.3, 0.2]],
                [[0.7, 0.6, 0.2], [0.4, 0.5, 0.3]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    raw_epoch30, weighted_epoch30, effective_epoch30, _ = (
        trainer._compute_top_branch_margin_loss(
            output,
            labels,
            epoch=30,
        )
    )
    raw_epoch31, weighted_epoch31, effective_epoch31, _ = (
        trainer._compute_top_branch_margin_loss(
            output,
            labels,
            epoch=31,
        )
    )

    expected_raw = torch.tensor((0.1 + 0.2) / 2)
    expected_phase_raw = torch.tensor((0.1 + (0.2 * 2.0)) / 2)
    assert torch.isclose(raw_epoch30, expected_raw)
    assert torch.isclose(weighted_epoch30, 0.1 * expected_raw)
    assert torch.isclose(effective_epoch30, torch.tensor(0.1))
    assert torch.isclose(raw_epoch31, expected_raw)
    assert torch.isclose(weighted_epoch31, 0.1 * expected_phase_raw)
    assert torch.isclose(effective_epoch31, torch.tensor(0.15))


def test_trainer_top_branch_margin_phase_multiplier_can_ramp() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.1,
            top_branch_margin_value=0.3,
            top_branch_margin_phase_start_multiplier_by_class=(1.0, 1.0, 1.0),
            top_branch_margin_phase_label_multiplier_by_class=(1.0, 1.0, 2.0),
            top_branch_margin_phase_schedule=(
                TopBranchMarginPhaseWeightScheduleConfig(
                    enabled=True,
                    start_epoch=21,
                    end_epoch=31,
                    start_multiplier_by_label={},
                    label_multiplier_by_label={},
                )
            ),
        )
    )
    labels = torch.tensor([0, 2], dtype=torch.long)
    reference = torch.ones(2, dtype=torch.float32)

    before = trainer._top_branch_margin_phase_multipliers(
        labels,
        reference,
        epoch=20,
    )
    start = trainer._top_branch_margin_phase_multipliers(
        labels,
        reference,
        epoch=21,
    )
    middle = trainer._top_branch_margin_phase_multipliers(
        labels,
        reference,
        epoch=26,
    )
    end = trainer._top_branch_margin_phase_multipliers(
        labels,
        reference,
        epoch=31,
    )
    after = trainer._top_branch_margin_phase_multipliers(
        labels,
        reference,
        epoch=32,
    )

    assert torch.allclose(before, torch.tensor([1.0, 1.0]))
    assert torch.allclose(start, torch.tensor([1.0, 1.0]))
    assert torch.allclose(middle, torch.tensor([1.0, 1.5]))
    assert torch.allclose(end, torch.tensor([1.0, 2.0]))
    assert torch.allclose(after, torch.tensor([1.0, 2.0]))


def test_trainer_branch_objective_diagnostics_include_dynamic_targets() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.1,
            top_branch_margin_value=0.3,
            top_branch_margin_by_class=(0.3, 0.4, 0.6),
            class_evidence_gap_cap_enabled=True,
            class_evidence_gap_cap_weight=0.02,
            class_evidence_gap_cap_negative_gap_cap=3.0,
            top_support_score_margin_enabled=True,
            top_support_score_margin_weight=0.05,
            top_support_score_margin_hardness_weighting=(
                MarginHardnessWeightingConfig(
                    enabled=True,
                    source="top_support_gap",
                    mode="negative_gap",
                    gain=1.0,
                    cap=3.0,
                )
            ),
            class_top_branch_relative_margin_enabled=True,
            class_top_branch_relative_margin_weight=0.05,
            class_top_branch_relative_margin_value=0.3,
            class_top_branch_relative_margin_support_weighting=(
                MarginSupportWeightingConfig(
                    enabled=True,
                    source="top_branch_margin",
                    mode="linear",
                    gain=0.5,
                    cap=3.0,
                )
            ),
            class_top_branch_relative_margin_hardness_weighting=(
                MarginHardnessWeightingConfig(
                    enabled=True,
                    source="teacher_gap_deficit",
                    mode="linear",
                    gain=0.5,
                    cap=2.0,
                )
            ),
            class_top_branch_relative_margin_weak_positive_weighting=(
                WeakPositiveSupportWeightingConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    multiplier=1.25,
                )
            ),
            class_top_branch_relative_margin_weak_positive_margin_boost=(
                WeakPositiveMarginBoostConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    boost=0.2,
                )
            ),
            top_teacher_gap_min_enabled=True,
            top_teacher_gap_min_weight=0.075,
            top_teacher_gap_min_base_min_gap=0.0,
            top_teacher_gap_min_support_gain=0.75,
            top_teacher_gap_min_support_cap=2.0,
            top_teacher_gap_min_weak_positive_weighting=(
                WeakPositiveSupportWeightingConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    multiplier=1.5,
                )
            ),
            top_teacher_gap_min_weak_positive_target_boost=(
                WeakPositiveTargetBoostConfig(
                    enabled=True,
                    min_support=0.2,
                    max_support=1.0,
                    boost=0.3,
                )
            ),
            gate_best_branch_alignment_enabled=True,
            gate_best_branch_alignment_weight=0.03,
            gate_best_branch_alignment_min_best_margin=0.2,
            gate_best_branch_alignment_max_best_margin=1.0,
            gate_best_branch_alignment_mismatch_margin_drop=0.5,
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.1,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_positive_threshold_by_class=(0.3, 0.2, 0.1),
            gate_branch_regret_tolerance=0.05,
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.025,
            gate_bad_branch_suppression_bad_margin_threshold=0.0,
            gate_bad_branch_suppression_threshold_by_class=(-0.2, 0.1, 0.1),
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.05,
            global_residual_anti_veto_enabled=True,
            global_residual_anti_veto_weight=0.02,
            global_residual_anti_veto_evidence_confidence_threshold=0.0,
            global_residual_anti_veto_min_residual_gap=-0.5,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(
            [[[0.8, 0.6, 0.5], [0.4, 0.7, 0.2]]],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [[[0.5, 0.5], [0.25, 0.75], [0.5, 0.5]]],
            dtype=torch.float32,
        ),
        class_gated_branch_logits=torch.tensor(
            [[0.4, 0.7, 0.2]],
            dtype=torch.float32,
        ),
        class_evidence_logits=torch.tensor(
            [[0.4, 0.55, 0.2]],
            dtype=torch.float32,
        ),
        class_evidence_top_support_scores=torch.tensor(
            [[0.2, 0.0, 0.6]],
            dtype=torch.float32,
        ),
        class_top_branch_margin_features=torch.tensor(
            [[0.2, 0.3, 0.5]],
            dtype=torch.float32,
        ),
        global_residual_logits=torch.tensor(
            [[0.2, -0.6, 0.3]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1], dtype=torch.long)

    rows = trainer._branch_objective_diagnostic_rows(output, labels, epoch=1)

    row = rows[0]
    assert row["top_branch_margin_target"] == pytest.approx(0.4)
    assert row["top_branch_margin_value"] == pytest.approx(0.3)
    assert row["top_branch_margin_violation"] is True
    assert row["gate_branch_regret_positive_threshold"] == pytest.approx(0.2)
    assert row["gate_branch_regret_best_margin"] == pytest.approx(0.3)
    assert row["gate_branch_regret_gate_expected_margin"] == pytest.approx(0.175)
    assert row["gate_branch_regret_eligible"] is True
    assert row["gate_branch_regret_weight_multiplier"] == pytest.approx(1.0)
    assert row["gate_branch_regret_effective_weight"] == pytest.approx(0.1)
    assert row["gate_bad_branch_suppression_threshold"] == pytest.approx(0.1)
    assert row["gate_bad_branch_suppression_bad_gate_mass"] == pytest.approx(0.25)
    assert row["gate_bad_branch_suppression_penalty"] == pytest.approx(0.075)
    assert row["gate_bad_branch_suppression_weight_multiplier"] == pytest.approx(1.0)
    assert row["gate_bad_branch_suppression_effective_weight"] == pytest.approx(0.025)
    assert row["class_evidence_gap_cap_gap"] == pytest.approx(0.15)
    assert row["class_evidence_gap_cap_penalty"] == pytest.approx(0.0)
    assert row["class_evidence_gap_cap_eligible"] is False
    assert row["top_support_score_margin_gap"] == pytest.approx(-0.6)
    assert row["top_support_score_margin_hardness"] == pytest.approx(0.6)
    assert row["top_support_score_margin_hardness_weight"] == pytest.approx(1.6)
    assert row["top_support_score_margin_total_multiplier"] == pytest.approx(1.6)
    assert row["class_top_branch_relative_gap"] == pytest.approx(-0.2)
    assert row["class_top_branch_relative_negative_class"] == 2
    assert row["class_top_branch_relative_margin_target"] == pytest.approx(0.3)
    assert row["class_top_branch_relative_margin_label_margin"] == pytest.approx(0.3)
    assert row["class_top_branch_relative_margin_effective_target"] == pytest.approx(
        0.5
    )
    assert row[
        "class_top_branch_relative_margin_weak_positive_margin_boost"
    ] == pytest.approx(0.2)
    assert row["class_top_branch_relative_margin_penalty"] == pytest.approx(
        0.7 * 1.15 * 1.35 * 1.25
    )
    assert row["class_top_branch_relative_margin_support_weight"] == pytest.approx(1.15)
    assert row["class_top_branch_relative_margin_hardness"] == pytest.approx(0.7)
    assert row["class_top_branch_relative_margin_hardness_weight"] == pytest.approx(
        1.35
    )
    assert row["class_top_branch_relative_margin_hardness_gain"] == pytest.approx(0.5)
    assert row["class_top_branch_relative_margin_hardness_cap"] == pytest.approx(2.0)
    assert row[
        "class_top_branch_relative_margin_weak_positive_weight"
    ] == pytest.approx(1.25)
    assert row["class_top_branch_relative_margin_weak_positive_eligible"] is True
    assert row["class_top_branch_relative_margin_eligible"] is True
    assert row["top_teacher_gap_min_gap"] == pytest.approx(-0.2)
    assert row["top_teacher_gap_min_negative_class"] == 2
    assert row["top_teacher_gap_min_target"] == pytest.approx(0.525)
    assert row["top_teacher_gap_min_base_min_gap"] == pytest.approx(0.0)
    assert row["top_teacher_gap_min_support_gain"] == pytest.approx(0.75)
    assert row["top_teacher_gap_min_effective_target"] == pytest.approx(0.525)
    assert row["top_teacher_gap_min_weak_positive_target_boost"] == pytest.approx(0.3)
    assert row["top_teacher_gap_min_support_value"] == pytest.approx(0.3)
    assert row["top_teacher_gap_min_penalty"] == pytest.approx(0.725 * 1.5)
    assert row["top_teacher_gap_min_weak_positive_weight"] == pytest.approx(1.5)
    assert row["top_teacher_gap_min_weak_positive_eligible"] is True
    assert row["top_teacher_gap_min_eligible"] is True
    assert row["gate_best_branch_alignment_best_branch"] == 1
    assert row["gate_best_branch_alignment_selected_branch"] == 1
    assert row["gate_best_branch_alignment_best_margin"] == pytest.approx(0.3)
    assert row["gate_best_branch_alignment_selected_margin"] == pytest.approx(0.3)
    assert row["gate_best_branch_alignment_margin_drop"] == pytest.approx(0.0)
    assert row["gate_best_branch_alignment_label_multiplier"] == pytest.approx(1.0)
    assert row["gate_best_branch_alignment_min_best_margin"] == pytest.approx(0.2)
    assert row["gate_best_branch_alignment_max_best_margin"] == pytest.approx(1.0)
    assert row["gate_best_branch_alignment_mismatch_margin_drop"] == pytest.approx(0.5)
    assert row["gate_best_branch_alignment_best_gate_weight"] == pytest.approx(0.75)
    assert row["gate_best_branch_alignment_mismatch"] is False
    assert row["gate_best_branch_alignment_eligible"] is False
    assert row["gate_best_branch_alignment_penalty"] == pytest.approx(0.0)
    assert row["branch_to_evidence_branch_gap"] == pytest.approx(0.3)
    assert row["branch_to_evidence_evidence_gap"] == pytest.approx(0.15)
    assert row["branch_to_evidence_penalty"] == pytest.approx(0.15)
    assert row["global_residual_anti_veto_evidence_gap"] == pytest.approx(0.15)
    assert row["global_residual_anti_veto_residual_gap"] == pytest.approx(-0.8)
    assert row["global_residual_anti_veto_eligible"] is True
    assert row["global_residual_anti_veto_penalty"] == pytest.approx(0.3)


def test_trainer_branch_objective_diagnostics_include_b2e_kl_fields() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.1,
            branch_to_evidence_source="class_top_branch_margin_features",
            branch_to_evidence_mode="teacher_distribution_kl",
            branch_to_evidence_teacher_temperature=0.7,
            branch_to_evidence_student_temperature=1.0,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
        branch_logits=torch.tensor(
            [[[0.8, 0.6, 0.5], [0.4, 0.7, 0.2]]],
            dtype=torch.float32,
        ),
        class_evidence_logits=torch.tensor(
            [[0.4, 0.55, 0.2]],
            dtype=torch.float32,
        ),
        class_top_branch_margin_features=torch.tensor(
            [[0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1], dtype=torch.long)

    rows = trainer._branch_objective_diagnostic_rows(output, labels, epoch=1)

    row = rows[0]
    assert row["branch_to_evidence_mode"] == "teacher_distribution_kl"
    assert row["branch_to_evidence_source"] == "class_top_branch_margin_features"
    assert row["branch_to_evidence_teacher_temperature"] == pytest.approx(0.7)
    assert row["branch_to_evidence_student_temperature"] == pytest.approx(1.0)
    assert len(row["branch_to_evidence_teacher_probs"]) == 3
    assert len(row["branch_to_evidence_student_probs"]) == 3
    assert row["branch_to_evidence_teacher_top_class"] == 1
    assert row["branch_to_evidence_student_top_class"] == 1
    assert row["branch_to_evidence_true_teacher_prob"] == pytest.approx(
        row["branch_to_evidence_teacher_probs"][1]
    )
    assert row["branch_to_evidence_true_student_prob"] == pytest.approx(
        row["branch_to_evidence_student_probs"][1]
    )
    assert row["branch_to_evidence_kl"] >= 0.0


def test_trainer_top_branch_margin_class_balanced_and_warmup() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(2.0, 3.0, 5.0),
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.05,
            top_branch_margin_value=0.3,
            top_branch_margin_class_weighted=True,
            top_branch_margin_reduction="class_balanced_violating_mean",
            top_branch_margin_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(3, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(3, 32),
        branch_logits=torch.tensor(
            [
                [[0.8, 0.6, 0.5], [0.4, 0.3, 0.2]],
                [[0.7, 0.6, 0.2], [0.4, 0.5, 0.3]],
                [[0.1, 0.2, 0.9], [0.0, 0.1, 1.0]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    warmup_raw, warmup_weighted, _, _ = trainer._compute_top_branch_margin_loss(
        output,
        labels,
        epoch=10,
    )
    raw_margin, weighted_margin, _, _ = trainer._compute_top_branch_margin_loss(
        output,
        labels,
        epoch=11,
    )

    expected_raw = torch.tensor(((2.0 * 0.1) + (3.0 * 0.2)) / 2)
    assert torch.isclose(warmup_raw, torch.tensor(0.0))
    assert torch.isclose(warmup_weighted, torch.tensor(0.0))
    assert torch.isclose(raw_margin, expected_raw)
    assert torch.isclose(weighted_margin, 0.05 * expected_raw)


def test_trainer_branch_path_dominance_preserves_branch_gap() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_path_dominance_enabled=True,
            branch_path_dominance_weight=0.05,
            branch_path_dominance_allowed_drop=0.5,
            branch_path_dominance_allowed_drop_by_class=(0.25, 0.5, 0.5),
            branch_path_dominance_label_weight_by_class=(2.0, 1.0, 1.0),
            branch_path_dominance_support_weighting=MarginSupportWeightingConfig(
                enabled=True,
                source="top_branch_margin",
                mode="linear",
                gain=0.5,
                cap=3.0,
            ),
            branch_path_dominance_warmup_epochs=10,
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.4],
            ],
            dtype=torch.float32,
        ),
        class_evidence_branch_support_scores=torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 0.9, 0.4],
            ],
            dtype=torch.float32,
        ),
        branch_logits=torch.tensor(
            [
                [[2.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.4]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    warmup_raw, warmup_weighted, _, _, _ = (
        trainer._compute_branch_path_dominance_constraint_loss(
            output,
            labels,
            epoch=10,
        )
    )
    raw_loss, weighted_loss, support_mean, label_mean, drop_mean = (
        trainer._compute_branch_path_dominance_constraint_loss(
            output,
            labels,
            epoch=11,
        )
    )

    expected_penalties = torch.tensor([0.75 * 1.5 * 2.0, 0.0])
    assert torch.isclose(warmup_raw, torch.tensor(0.0))
    assert torch.isclose(warmup_weighted, torch.tensor(0.0))
    assert torch.isclose(raw_loss, expected_penalties.mean())
    assert torch.isclose(weighted_loss, torch.tensor(0.05) * expected_penalties.mean())
    assert torch.isclose(support_mean, torch.tensor((1.5 + 1.3) / 2.0))
    assert torch.isclose(label_mean, torch.tensor(1.5))
    assert torch.isclose(drop_mean, torch.tensor(0.375))


def test_trainer_adaptive_branch_objectives_update_train_state_only() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.1,
            top_branch_margin_value=0.3,
            top_branch_margin_by_class=(0.3, 0.3, 0.5),
            top_branch_margin_auto=AutoMarginByTrainStatsConfig(
                enabled=True,
                strategy="ema_violation_controller",
                start_epoch=1,
                update_interval_epochs=1,
                ema=0.9,
                step=0.02,
                target_violation_rate_by_label={
                    "normal": 0.2,
                    "crackle": 0.45,
                    "wheeze": 0.8,
                },
                min_margin_by_label={
                    "normal": 0.3,
                    "crackle": 0.3,
                    "wheeze": 0.45,
                },
                max_margin_by_label={
                    "normal": 0.3,
                    "crackle": 0.3,
                    "wheeze": 0.75,
                },
            ),
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.1,
            gate_branch_regret_positive_threshold=0.3,
            gate_branch_regret_positive_threshold_by_class=(0.3, 0.3, 0.1),
            gate_branch_regret_tolerance=0.05,
            gate_branch_regret_auto=AutoPositiveThresholdByTrainStatsConfig(
                enabled=True,
                strategy="ema_eligible_controller",
                start_epoch=1,
                update_interval_epochs=1,
                ema=0.9,
                step=0.02,
                target_eligible_rate_by_label={
                    "normal": 0.7,
                    "crackle": 0.7,
                    "wheeze": 0.5,
                },
                min_threshold_by_label={
                    "normal": 0.3,
                    "crackle": 0.3,
                    "wheeze": 0.0,
                },
                max_threshold_by_label={
                    "normal": 0.3,
                    "crackle": 0.3,
                    "wheeze": 0.2,
                },
            ),
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.025,
            gate_bad_branch_suppression_bad_margin_threshold=0.0,
            gate_bad_branch_suppression_threshold_by_class=(-0.2, 0.1, 0.1),
            gate_bad_branch_suppression_auto=(
                AutoBadBranchThresholdByTrainStatsConfig(
                    enabled=True,
                    strategy="ema_bad_gate_mass_controller",
                    start_epoch=1,
                    update_interval_epochs=1,
                    ema=0.9,
                    step=0.01,
                    target_bad_gate_mass_by_label={
                        "normal": 0.05,
                        "crackle": 0.15,
                        "wheeze": 0.1,
                    },
                    min_threshold_by_label={
                        "normal": -0.2,
                        "crackle": 0.0,
                        "wheeze": 0.0,
                    },
                    max_threshold_by_label={
                        "normal": -0.2,
                        "crackle": 0.35,
                        "wheeze": 0.3,
                    },
                )
            ),
        )
    )
    stats = trainer._new_branch_objective_epoch_stats()
    output = AstModelOutput(
        logits=torch.zeros(3, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(3, 32),
        branch_logits=torch.tensor(
            [
                [[0.8, 0.6, 0.5], [0.4, 0.3, 0.2]],
                [[0.7, 0.6, 0.2], [0.4, 0.5, 0.3]],
                [[0.0, 0.1, 0.4], [0.1, 0.0, 0.5]],
            ],
            dtype=torch.float32,
        ),
        class_evidence_gate_weights=torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    trainer._observe_branch_objective_stats(output, labels, stats)
    trainer._update_adaptive_branch_objective_state(epoch=1, stats=stats)

    state = trainer._branch_objective_state_dict()
    assert state["top_branch_margin_by_label"]["normal"] == pytest.approx(0.3)
    assert state["top_branch_margin_by_label"]["crackle"] == pytest.approx(0.3)
    assert state["top_branch_margin_by_label"]["wheeze"] == pytest.approx(0.48)
    assert state["gate_branch_regret_positive_threshold_by_label"][
        "normal"
    ] == pytest.approx(0.3)
    assert state["gate_branch_regret_positive_threshold_by_label"][
        "crackle"
    ] == pytest.approx(0.3)
    assert state["gate_branch_regret_positive_threshold_by_label"][
        "wheeze"
    ] == pytest.approx(0.12)
    assert state["gate_bad_branch_suppression_threshold_by_label"][
        "normal"
    ] == pytest.approx(-0.2)
    assert state["gate_bad_branch_suppression_threshold_by_label"][
        "crackle"
    ] == pytest.approx(0.09)
    assert state["gate_bad_branch_suppression_threshold_by_label"][
        "wheeze"
    ] == pytest.approx(0.11)
    assert state["gate_bad_branch_suppression_bad_gate_mass_ema_by_label"][
        "crackle"
    ] == pytest.approx(1.0)


def test_trainer_raises_when_top_branch_margin_enabled_without_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.05,
            top_branch_margin_value=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(1, 3, dtype=torch.float32),
        pooled_embedding=torch.zeros(1, 32),
    )
    labels = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="top branch margin enabled"):
        trainer._compute_total_loss(criterion, output, labels, epoch=1)


def test_trainer_raises_when_class_evidence_margin_enabled_without_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.05,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="minority_vs_major",
            class_evidence_margin_major_index=0,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="class evidence margin enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_raises_when_class_evidence_margin_label_is_outside_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.05,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="true_vs_hardest_negative",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.zeros(2, 3),
    )
    labels = torch.tensor([1, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="outside class_evidence_logits"):
        trainer._compute_class_evidence_margin_loss(output, labels)


def test_trainer_raises_when_class_gate_diversity_enabled_without_weights() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_weights=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="class gate diversity regularization enabled",
    ):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_raises_when_gate_entropy_enabled_without_evidence_entropy() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.1,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        evidence_gate_entropy=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="gate entropy regularization enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_epoch_loss_components_include_gate_entropy_regularization() -> None:
    class GateEntropyModel(torch.nn.Module):
        def forward(self, input_values: torch.Tensor) -> AstModelOutput:
            batch_size = input_values.shape[0]
            return AstModelOutput(
                logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
                pooled_embedding=torch.zeros(batch_size, 32),
                evidence_gate_entropy=torch.tensor([0.2, 0.6], dtype=torch.float32),
            )

    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
        )
    )
    batch = ClipBatch(
        input_values=torch.zeros(2, 1),
        labels=torch.tensor([1, 0], dtype=torch.long),
        audio_paths=("a.wav", "b.wav"),
        label_names=("wheeze", "normal"),
    )

    result = trainer._epoch(
        GateEntropyModel(),
        [batch],
        optimizer=None,
        scheduler=None,
        device=torch.device("cpu"),
        epoch=1,
    )

    assert result.loss_components["gate_entropy"] == pytest.approx(0.4)
    assert result.loss_components["gate_entropy_regularization"] == pytest.approx(-0.1)


def test_epoch_loss_components_include_class_gate_terms() -> None:
    class ClassGateModel(torch.nn.Module):
        def forward(self, input_values: torch.Tensor) -> AstModelOutput:
            batch_size = input_values.shape[0]
            return AstModelOutput(
                logits=torch.zeros(batch_size, 3, dtype=torch.float32),
                pooled_embedding=torch.zeros(batch_size, 32),
                class_evidence_logits=torch.tensor(
                    [[0.1, 0.4, -0.2], [0.7, 0.1, 0.0]],
                    dtype=torch.float32,
                ),
                class_evidence_gate_weights=torch.tensor(
                    [
                        [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
                        [[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]],
                    ],
                    dtype=torch.float32,
                ),
                branch_logits=torch.tensor(
                    [
                        [[0.2, 0.6, 0.1], [0.4, 0.3, 0.2]],
                        [[0.8, 0.2, 0.1], [0.5, 0.4, 0.3]],
                    ],
                    dtype=torch.float32,
                ),
                class_gated_branch_logits=torch.tensor(
                    [[0.3, 0.5, 0.2], [0.7, 0.3, 0.2]],
                    dtype=torch.float32,
                ),
                global_residual_logits=torch.tensor(
                    [[0.1, -0.8, 0.2], [-0.7, 0.1, 0.2]],
                    dtype=torch.float32,
                ),
            )

    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=True,
            class_gate_evidence_auxiliary_weight=0.2,
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
            class_evidence_margin_enabled=True,
            class_evidence_margin_weight=0.05,
            class_evidence_margin_value=0.5,
            class_evidence_margin_mode="minority_vs_major",
            class_evidence_margin_major_index=0,
            class_gated_branch_logit_margin_enabled=True,
            class_gated_branch_logit_margin_weight=0.03,
            class_gated_branch_logit_margin_value=0.3,
            branch_to_evidence_enabled=True,
            branch_to_evidence_weight=0.05,
            global_residual_anti_veto_enabled=True,
            global_residual_anti_veto_weight=0.02,
            global_residual_anti_veto_evidence_confidence_threshold=0.0,
            global_residual_anti_veto_min_residual_gap=-0.5,
            gate_weighted_branch_margin_enabled=True,
            gate_weighted_branch_margin_weight=0.01,
            gate_weighted_branch_margin_value=0.3,
            gate_branch_regret_enabled=True,
            gate_branch_regret_weight=0.01,
            gate_branch_regret_positive_threshold=0.1,
            gate_branch_regret_tolerance=0.05,
            gate_bad_branch_suppression_enabled=True,
            gate_bad_branch_suppression_weight=0.025,
            gate_bad_branch_suppression_bad_margin_threshold=0.0,
            top_branch_margin_enabled=True,
            top_branch_margin_weight=0.05,
            top_branch_margin_value=0.3,
        )
    )
    batch = ClipBatch(
        input_values=torch.zeros(2, 1),
        labels=torch.tensor([1, 0], dtype=torch.long),
        audio_paths=("a.wav", "b.wav"),
        label_names=("wheeze", "normal"),
    )

    result = trainer._epoch(
        ClassGateModel(),
        [batch],
        optimizer=None,
        scheduler=None,
        device=torch.device("cpu"),
        epoch=1,
    )

    assert "evidence_auxiliary_loss" in result.loss_components
    assert "class_gate_diversity" in result.loss_components
    assert "class_gate_diversity_regularization" in result.loss_components
    assert "class_evidence_margin" in result.loss_components
    assert "class_evidence_margin_loss" in result.loss_components
    assert "class_gated_branch_logit_margin" in result.loss_components
    assert "class_gated_branch_logit_margin_loss" in result.loss_components
    assert "branch_to_evidence_ranking_consistency" in result.loss_components
    assert "branch_to_evidence_ranking_consistency_loss" in result.loss_components
    assert "global_residual_anti_veto" in result.loss_components
    assert "global_residual_anti_veto_loss" in result.loss_components
    assert "global_residual_anti_veto_eligible_fraction" in result.loss_components
    assert "gate_weighted_branch_margin" in result.loss_components
    assert "gate_weighted_branch_margin_loss" in result.loss_components
    assert "gate_branch_regret" in result.loss_components
    assert "gate_branch_regret_loss" in result.loss_components
    assert "gate_branch_regret_eligible_fraction" in result.loss_components
    assert "gate_bad_branch_suppression" in result.loss_components
    assert "gate_bad_branch_suppression_loss" in result.loss_components
    assert "gate_bad_branch_suppression_bad_gate_mass" in result.loss_components
    assert "top_branch_margin" in result.loss_components
    assert "top_branch_margin_loss" in result.loss_components


def _sample_epoch_log_context(
    *,
    diagnostics_path: Path | None = None,
) -> EpochLogContext:
    class_names = ("normal", "crackle", "wheeze")
    train_metrics = EvalMetrics(
        accuracy=0.65,
        precision=0.61,
        recall=0.62,
        specificity=0.70,
        balanced_accuracy=0.66,
        f1_score=0.60,
        roc_auc=None,
        pr_auc=None,
        brier_score=None,
        confusion_matrix=[[7, 1, 3], [2, 5, 1], [4, 1, 3]],
        macro_f1=0.60,
        macro_recall=0.62,
        weighted_f1=0.63,
        per_class_precision=[0.54, 0.71, 0.43],
        per_class_recall=[0.70, 0.50, 0.30],
        per_class_f1=[0.61, 0.59, 0.35],
    )
    val_metrics = EvalMetrics(
        accuracy=0.55,
        precision=0.52,
        recall=0.50,
        specificity=0.64,
        balanced_accuracy=0.57,
        f1_score=0.49,
        roc_auc=None,
        pr_auc=None,
        brier_score=None,
        confusion_matrix=[[6, 0, 3], [1, 4, 2], [4, 0, 2]],
        macro_f1=0.49,
        macro_recall=0.50,
        weighted_f1=0.52,
        per_class_precision=[0.55, 1.00, 0.29],
        per_class_recall=[0.67, 0.57, 0.33],
        per_class_f1=[0.60, 0.73, 0.31],
    )
    train_components = {
        "main": 1.1,
        "total_scheduled": 1.62,
        "evidence_auxiliary_loss": 0.12,
        "class_evidence_margin": 0.40,
        "class_evidence_margin_loss": 0.08,
        "top_support_score_margin": 0.34,
        "top_support_score_margin_loss": 0.017,
        "branch_support_score_margin": 0.36,
        "branch_support_score_margin_loss": 0.018,
        "class_gated_branch_logit_margin": 0.31,
        "class_gated_branch_logit_margin_loss": 0.0465,
        "branch_to_evidence_ranking_consistency": 0.27,
        "branch_to_evidence_ranking_consistency_loss": 0.0135,
        "global_residual_anti_veto": 0.19,
        "global_residual_anti_veto_loss": 0.0038,
        "global_residual_anti_veto_eligible_fraction": 0.40,
        "gate_weighted_branch_margin": 0.25,
        "gate_weighted_branch_margin_loss": 0.025,
        "top_branch_margin": 0.50,
        "top_branch_margin_loss": 0.10,
        "gate_branch_regret": 0.22,
        "gate_branch_regret_loss": 0.0011,
        "gate_branch_regret_eligible_fraction": 0.43,
        "gate_branch_regret_weight_multiplier": 0.5,
        "gate_branch_regret_effective_weight": 0.005,
        "gate_bad_branch_suppression": 0.18,
        "gate_bad_branch_suppression_loss": 0.0045,
        "gate_bad_branch_suppression_bad_gate_mass": 0.22,
        "gate_bad_branch_suppression_weight_multiplier": 0.3,
        "gate_bad_branch_suppression_effective_weight": 0.0075,
        "top_branch_violation_rate_by_label": {
            "normal": 0.20,
            "crackle": 0.50,
            "wheeze": 0.75,
        },
        "gate_branch_regret_eligible_rate_by_label": {
            "normal": 0.70,
            "crackle": 0.62,
            "wheeze": 0.48,
        },
        "gate_bad_branch_suppression_bad_gate_mass_by_label": {
            "normal": 0.05,
            "crackle": 0.18,
            "wheeze": 0.11,
        },
    }
    val_components = {
        "main": 1.3,
        "total_scheduled": 1.76,
        "total_monitor": 1.70,
        "evidence_auxiliary_loss": 0.20,
        "class_evidence_margin": 0.45,
        "class_evidence_margin_loss": 0.09,
        "top_support_score_margin": 0.39,
        "top_support_score_margin_loss": 0.0195,
        "branch_support_score_margin": 0.41,
        "branch_support_score_margin_loss": 0.0205,
        "class_gated_branch_logit_margin": 0.38,
        "class_gated_branch_logit_margin_loss": 0.057,
        "branch_to_evidence_ranking_consistency": 0.33,
        "branch_to_evidence_ranking_consistency_loss": 0.0165,
        "global_residual_anti_veto": 0.22,
        "global_residual_anti_veto_loss": 0.0044,
        "global_residual_anti_veto_eligible_fraction": 0.45,
        "gate_weighted_branch_margin": 0.35,
        "gate_weighted_branch_margin_loss": 0.035,
        "top_branch_margin": 0.60,
        "top_branch_margin_loss": 0.12,
        "gate_branch_regret": 0.30,
        "gate_branch_regret_loss": 0.0015,
        "gate_branch_regret_eligible_fraction": 0.50,
        "gate_branch_regret_weight_multiplier": 0.5,
        "gate_branch_regret_effective_weight": 0.005,
        "gate_bad_branch_suppression": 0.21,
        "gate_bad_branch_suppression_loss": 0.00525,
        "gate_bad_branch_suppression_bad_gate_mass": 0.25,
        "gate_bad_branch_suppression_weight_multiplier": 0.3,
        "gate_bad_branch_suppression_effective_weight": 0.0075,
    }
    adaptive_state = {
        "top_branch_margin_by_label": {
            "normal": 0.30,
            "crackle": 0.36,
            "wheeze": 0.52,
        },
        "gate_branch_regret_positive_threshold_by_label": {
            "normal": 0.30,
            "crackle": 0.25,
            "wheeze": 0.10,
        },
        "gate_bad_branch_suppression_threshold_by_label": {
            "normal": -0.20,
            "crackle": 0.10,
            "wheeze": 0.10,
        },
    }
    return EpochLogContext(
        epoch=16,
        total_epochs=120,
        learning_rates=[0.001, 0.0001],
        class_names=class_names,
        train_loss=1.62,
        val_loss=1.76,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_metrics_optimized=val_metrics,
        val_threshold_optimization=ThresholdOptimizationResult.disabled(
            "f1",
            reason="threshold optimization is only supported for one-logit outputs",
        ),
        train_components=train_components,
        val_components=val_components,
        adaptive_state=adaptive_state,
        diagnostics_path=diagnostics_path,
    )


def test_epoch_log_formatter_builds_readable_multiclass_block() -> None:
    context = _sample_epoch_log_context(
        diagnostics_path=Path("diagnostics/val_epoch_016.jsonl")
    )

    block = "\n".join(format_epoch_log_block(context))

    assert "Epoch 016/120" in block
    assert "phase=2" in block
    assert "threshold=disabled" in block
    assert "recall normal=0.6700/crackle=0.5700/wheeze=0.3300" in block
    assert "wheeze->normal=4" in block
    assert "normal->wheeze=3" in block
    assert "class_margin raw=0.4000 loss=0.0800" in block
    assert "top_support raw=0.3400 loss=0.0170" in block
    assert "branch_support raw=0.3600 loss=0.0180" in block
    assert "branch_logit_margin raw=0.3100 loss=0.0465" in block
    assert "b2e raw=0.2700 loss=0.0135" in block
    assert "anti_veto raw=0.1900 loss=0.0038" in block
    assert "top_branch raw=0.5000 loss=0.1000" in block
    assert "regret raw=0.2200/0.3000" in block
    assert "bad_suppress raw=0.1800/0.2100" in block
    assert "bad_eff_w=0.0075" in block
    assert "bad_multiplier=0.3000" in block
    assert "bad_mass=0.2200/0.2500" in block
    assert "eff_w=0.0050" in block
    assert "multiplier=0.5000" in block
    assert "top_margin={normal=0.3000, crackle=0.3600, wheeze=0.5200}" in block
    assert "bad_threshold={normal=-0.2000, crackle=0.1000, wheeze=0.1000}" in block
    assert "bad_gate_mass={normal=0.0500, crackle=0.1800, wheeze=0.1100}" in block
    assert "diagnostics=diagnostics/val_epoch_016.jsonl" in block
    assert "entropy=" not in block
    assert "diversity=" not in block


def test_epoch_jsonl_logs_append_metric_loss_and_adaptive_payloads(
    tmp_path: Path,
) -> None:
    context = _sample_epoch_log_context(
        diagnostics_path=tmp_path / "diagnostics" / "val_epoch_016.jsonl"
    )

    paths = append_epoch_jsonl_logs(tmp_path, context)

    assert set(paths) == {"metrics", "loss_components", "adaptive_state"}
    metrics_payload = json.loads(paths["metrics"].read_text().splitlines()[0])
    loss_payload = json.loads(paths["loss_components"].read_text().splitlines()[0])
    adaptive_payload = json.loads(paths["adaptive_state"].read_text().splitlines()[0])
    assert metrics_payload["epoch"] == 16
    assert metrics_payload["val"]["confusion_summary"]["wheeze->normal"] == 4
    assert loss_payload["train"]["class_evidence_margin_loss"] == pytest.approx(0.08)
    assert loss_payload["train"]["top_support_score_margin_loss"] == pytest.approx(
        0.017
    )
    assert loss_payload["train"]["branch_support_score_margin_loss"] == pytest.approx(
        0.018
    )
    assert loss_payload["train"][
        "branch_to_evidence_ranking_consistency_loss"
    ] == pytest.approx(0.0135)
    assert loss_payload["train"]["global_residual_anti_veto_loss"] == pytest.approx(
        0.0038
    )
    assert adaptive_payload["adaptive_state"]["top_branch_margin_by_label"][
        "wheeze"
    ] == pytest.approx(0.52)
    assert adaptive_payload["train_top_branch_violation_rate_by_label"][
        "wheeze"
    ] == pytest.approx(0.75)
    assert adaptive_payload["train_gate_bad_branch_suppression_bad_gate_mass_by_label"][
        "crackle"
    ] == pytest.approx(0.18)


def test_configured_checkpoint_monitors_keep_top_three(tmp_path: Path) -> None:
    manager = CheckpointManager(
        tmp_path,
        top_k=1,
        checkpointing=CheckpointingConfig(
            monitors=(
                CheckpointMonitorConfig(
                    name="val_macro_f1",
                    mode="max",
                    top_k=3,
                ),
                CheckpointMonitorConfig(
                    name="val_macro_recall",
                    mode="max",
                    top_k=3,
                ),
                CheckpointMonitorConfig(
                    name="val_loss_total_monitor",
                    mode="min",
                    top_k=3,
                    filename_prefix="best_loss",
                ),
                CheckpointMonitorConfig(
                    name="last",
                    mode="last",
                    top_k=3,
                ),
            )
        ),
    )

    for epoch, (macro_f1, macro_recall, loss) in enumerate(
        [
            (0.10, 0.20, 0.90),
            (0.40, 0.30, 0.80),
            (0.20, 0.50, 0.70),
            (0.30, 0.10, 0.60),
        ],
        start=1,
    ):
        state = {"epoch": epoch}
        manager.save_last(state)
        manager.save_configured_monitors(
            {
                "val_macro_f1": macro_f1,
                "val_macro_recall": macro_recall,
                "val_loss_total_monitor": loss,
            },
            state,
            epoch=epoch,
        )

    assert len(sorted(tmp_path.glob("best_macro_f1_rank*.pt"))) == 3
    assert len(sorted(tmp_path.glob("best_macro_recall_rank*.pt"))) == 3
    assert len(sorted(tmp_path.glob("best_loss_rank*.pt"))) == 3
    assert len(sorted(tmp_path.glob("last_epoch*.pt"))) == 3
    assert (tmp_path / "last.pt").exists()
    assert sorted(path.name for path in tmp_path.glob("last_epoch*.pt")) == [
        "last_epoch002.pt",
        "last_epoch003.pt",
        "last_epoch004.pt",
    ]


def test_legacy_checkpoint_behavior_is_preserved(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=1)
    state = {"epoch": 1}

    manager.save_last(state)
    manager.maybe_save_best("loss", 0.5, state, maximize=False)
    manager.maybe_save_best("loss", 0.4, {"epoch": 2}, maximize=False)
    manager.maybe_save_best("f1", 0.1, state, maximize=True)

    assert (tmp_path / "last.pt").exists()
    assert len(sorted(tmp_path.glob("best_loss_*.pt"))) == 1
    assert len(sorted(tmp_path.glob("best_f1_*.pt"))) == 1


def test_binary_trainer_and_evaluator_smoke_b3(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "clips")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(rdt_enabled=True, rdt_steps=3),
        loss_type="focal",
        pos_weight=2.0,
        branch_auxiliary_enabled=True,
    )
    run_dir = checkpoint_path.parent

    assert checkpoint_path.exists()
    assert sorted(run_dir.glob("best_loss_*.pt"))

    checkpoint = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    assert checkpoint["dims"] == {"num_mel_bins": 32, "max_length": 32}

    result = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    metrics, rows, diagnostics = result

    assert metrics["threshold_optimization"]["enabled"] is True
    assert len(rows) == len(build_dataset(data_cfg, split="eval"))
    assert len(diagnostics) == len(rows)
    assert "branch_logits" in diagnostics[0]
    assert "selected_evidence_indices" in diagnostics[0]
    assert "selected_evidence_scores" in diagnostics[0]
    assert "selected_evidence_branch_ids" in diagnostics[0]
    assert diagnostics[0]["evidence_score_source"] == "attention_weight"
    assert "selected_evidence_tokens" in diagnostics[0]


def test_two_class_class_aware_ce_smoke_uses_softmax_without_threshold(
    tmp_path: Path,
) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "class_aware_ce")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(
            classifier_type="mlp",
            rdt_enabled=True,
            rdt_steps=2,
            evidence_pooling=EvidencePoolingConfig(
                type="class_aware_branch_gated",
                dropout=0.0,
            ),
        ),
        loss_type="cross_entropy",
        branch_auxiliary_enabled=True,
    )

    checkpoint = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    assert checkpoint["val_threshold_optimization"]["enabled"] is False

    result = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    metrics, rows, diagnostics = result

    assert metrics["decision_threshold"] is None
    assert metrics["threshold_optimization"]["enabled"] is False
    assert metrics["optimized_metrics"]["decision_threshold"] is None
    assert np.asarray(metrics["predicted_probability"]).shape == (2, 2)
    assert len(rows) == len(build_dataset(data_cfg, split="eval"))
    assert len(diagnostics) == len(rows)
    assert diagnostics[0]["evidence_pooling_type"] == "class_aware_branch_gated"
    assert "class_evidence_gate_weights" in diagnostics[0]
    assert "true_class_gate_weights" in diagnostics[0]
    assert "class_gated_branch_logits" in diagnostics[0]


def test_multiclass_trainer_and_evaluator_smoke_b3(tmp_path: Path) -> None:
    data_cfg = _prepare_multiclass_dataset(tmp_path / "multiclass")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(
            classifier_type="mlp",
            rdt_enabled=True,
            rdt_steps=2,
        ),
        loss_type="cross_entropy",
        branch_auxiliary_enabled=True,
    )

    result = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    metrics, rows, diagnostics = result

    assert metrics["optimized_metrics"]["decision_threshold"] is None
    assert len(rows) == len(build_dataset(data_cfg, split="eval"))
    assert len(diagnostics) == len(rows)
    assert "branch_logits" in diagnostics[0]
    assert "selected_evidence_indices" in diagnostics[0]
    assert "selected_evidence_scores" in diagnostics[0]
    assert "selected_evidence_branch_ids" in diagnostics[0]
    assert diagnostics[0]["evidence_score_source"] == "attention_weight"


def test_evaluator_rejects_frontend_dim_mismatch(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "clips")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(rdt_enabled=False, rdt_steps=3),
        loss_type="bce",
        branch_auxiliary_enabled=False,
    )
    mismatched_cfg = DataConfig(
        train_dirs=data_cfg.train_dirs,
        val_dirs=data_cfg.val_dirs,
        eval_dirs=data_cfg.eval_dirs,
        label_to_index=data_cfg.label_to_index,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        audio=data_cfg.audio,
        preprocessing=PreprocessingConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=40,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
    )

    with pytest.raises(ValueError, match="Evaluation frontend dims do not match"):
        evaluate_checkpoint(
            _eval_cfg(tmp_path, checkpoint_path, mismatched_cfg),
            checkpoint_path,
        )


def test_cv_cli_smoke_b3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = _prepare_cv_dataset(tmp_path / "cv")
    config_path = _write_json(
        tmp_path / "cv.json",
        _cv_payload(dataset_root, tmp_path / "cv_outputs"),
    )

    monkeypatch.setattr(sys, "argv", ["cv", "--config", str(config_path)])
    cv_main()

    fold_dir = tmp_path / "cv_outputs" / "cv_smoke" / "fold_0"
    assert (fold_dir / "last.pt").exists()
