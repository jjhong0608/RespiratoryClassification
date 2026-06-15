from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig
from src.models.model import (
    AstFeatureDims,
    BranchEventDropoutConfig,
    ClassGateBranchDirectScoreConfig,
    ClassGateBranchFeatureTransformConfig,
    ClassGateBranchLogitFeatureConfig,
    ClassGateConfig,
    ClassGateEvidenceAuxiliaryConfig,
    ClassGateEvidenceScorerConfig,
    ClassGateGlobalResidualBoundingConfig,
    ClassGateGlobalResidualConfig,
    ClassGateGlobalResidualCorrectionConfig,
    ClassGateGlobalResidualWarmupConfig,
    ClassGateInteractionScaleScheduleConfig,
    ClassGateMixingConfig,
    ClassGateReliabilityMixtureConfig,
    ClassGateResidualConfidenceAwareGateConfig,
    ClassGateScoreBoundComponentConfig,
    ClassGateScoreBoundingConfig,
    ClassGateScoreDecompositionConfig,
    ClassGateTopRelativeCorrectionConfig,
    ClassGateTopSupportDirectPathConfig,
    ClassGateTopSupportNegativeRelativeCapConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    EvidencePoolingConfig,
    FusionProjectorConfig,
    MilConfig,
    MultiScaleRdtArchitectureConfig,
    PatchBranchConfig,
    RdtConfig,
    SelectedEvidenceDropoutConfig,
    TokenAugmentationConfig,
    compute_token_count,
    compute_token_grid,
)


@dataclass(frozen=True)
class ExperimentLoggingConfig:
    terminal_width: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    mode: Literal["clip"]
    seed: int
    device: str
    output_dir: str
    logging: ExperimentLoggingConfig = field(default_factory=ExperimentLoggingConfig)


@dataclass(frozen=True)
class BandPassConfig:
    enabled: bool = False
    low_hz: float | None = None
    high_hz: float | None = None
    q: float = 0.707


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    clip_duration_sec: float

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_duration_sec))


@dataclass(frozen=True)
class AstFbankConfig:
    num_mel_bins: int = 128
    max_length: int = 1024
    do_normalize: bool = True
    mean: float = -4.2677393
    std: float = 4.5689974


@dataclass(frozen=True)
class PreprocessingConfig:
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)
    ast_fbank: AstFbankConfig = field(default_factory=AstFbankConfig)


@dataclass(frozen=True)
class RandomGainConfig:
    enabled: bool = False
    probability: float = 0.5
    min_db: float = -3.0
    max_db: float = 3.0


@dataclass(frozen=True)
class AdditiveNoiseConfig:
    enabled: bool = False
    probability: float = 0.3
    snr_db_min: float = 15.0
    snr_db_max: float = 30.0


@dataclass(frozen=True)
class TimeShiftConfig:
    enabled: bool = False
    probability: float = 0.5
    max_shift_fraction: float = 0.05
    mode: Literal["zero_pad", "roll"] = "zero_pad"


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    enabled: bool = False
    probability: float = 1.0
    gain: RandomGainConfig = field(default_factory=RandomGainConfig)
    noise: AdditiveNoiseConfig = field(default_factory=AdditiveNoiseConfig)
    time_shift: TimeShiftConfig = field(default_factory=TimeShiftConfig)


@dataclass(frozen=True)
class MaskConfig:
    enabled: bool = False
    num_masks: int = 1
    max_width: int = 32


@dataclass(frozen=True)
class FbankAugmentationConfig:
    enabled: bool = False
    probability: float = 0.5
    time_mask: MaskConfig = field(default_factory=MaskConfig)
    freq_mask: MaskConfig = field(default_factory=MaskConfig)
    mask_value: float = 0.0


@dataclass(frozen=True)
class AugmentationPolicyChoiceConfig:
    name: Literal["none", "waveform", "fbank", "both_light"]
    probability: float


@dataclass(frozen=True)
class AugmentationPolicyConfig:
    type: Literal["independent", "one_of"] = "independent"
    choices: tuple[AugmentationPolicyChoiceConfig, ...] = ()


@dataclass(frozen=True)
class DataAugmentationConfig:
    enabled: bool = False
    policy: AugmentationPolicyConfig = field(default_factory=AugmentationPolicyConfig)
    waveform: WaveformAugmentationConfig = field(
        default_factory=WaveformAugmentationConfig
    )
    fbank: FbankAugmentationConfig = field(default_factory=FbankAugmentationConfig)


@dataclass(frozen=True)
class DataConfig:
    train_dirs: list[str]
    val_dirs: list[str]
    eval_dirs: list[str]
    label_to_index: Mapping[str, int]
    batch_size: int
    num_workers: int
    audio: AudioConfig
    preprocessing: PreprocessingConfig
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)


@dataclass(frozen=True)
class ModelEncoderConfig:
    type: Literal["multiscale_rdt_ast"] = "multiscale_rdt_ast"
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: MultiScaleRdtArchitectureConfig = field(
        default_factory=MultiScaleRdtArchitectureConfig
    )


@dataclass(frozen=True)
class ModelConfig:
    encoder: ModelEncoderConfig
    classifier: ClassifierConfig


@dataclass(frozen=True)
class OptimizerConfig:
    encoder_lr: float
    head_lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_ratio: float = 0.0


@dataclass(frozen=True)
class BranchAuxiliaryLossConfig:
    enabled: bool = False
    weight: float = 0.3
    weights: tuple[float, ...] | None = None
    aggregation: Literal["mean"] = "mean"


@dataclass(frozen=True)
class AttentionEntropyLossConfig:
    enabled: bool = False
    weight: float = 0.0


@dataclass(frozen=True)
class GateEntropyRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal[
        "evidence_gate",
        "class_evidence_gate",
        "true_class_evidence_gate",
        "class_evidence_learned_gate",
    ] = "evidence_gate"
    start_epoch: int = 1
    end_epoch: int | None = None


@dataclass(frozen=True)
class ClassGateDiversityRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal[
        "class_evidence_gate",
        "class_evidence_learned_gate",
    ] = "class_evidence_gate"
    metric: Literal["js_divergence"] = "js_divergence"
    start_epoch: int = 1
    end_epoch: int | None = None


@dataclass(frozen=True)
class MarginSupportWeightingConfig:
    enabled: bool = False
    source: Literal[
        "top_branch_margin",
        "same_as_source",
    ] = "top_branch_margin"
    mode: Literal["linear", "positive_linear"] = "linear"
    gain: float = 1.0
    cap: float = 2.0


@dataclass(frozen=True)
class MarginHardnessWeightingConfig:
    enabled: bool = False
    source: Literal[
        "evidence_gap",
        "top_support_gap",
        "teacher_gap_deficit",
    ] = "evidence_gap"
    mode: Literal["negative_gap", "linear"] = "negative_gap"
    gain: float = 1.0
    cap: float = 3.0


@dataclass(frozen=True)
class ClassEvidenceMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    margin: float = 0.0
    target: Literal["class_evidence_logits"] = "class_evidence_logits"
    mode: Literal[
        "minority_vs_major",
        "true_vs_hardest_negative",
        "softplus_true_vs_hardest_negative",
    ] = "minority_vs_major"
    major_class: str | None = None
    class_weighted: bool = False
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"
    temperature: float = 1.0
    support_weighting: MarginSupportWeightingConfig = field(
        default_factory=MarginSupportWeightingConfig
    )


@dataclass(frozen=True)
class ClassEvidenceGapCapRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_logits"] = "class_evidence_logits"
    mode: Literal["negative_gap_hinge"] = "negative_gap_hinge"
    negative_gap_cap: float = 3.0
    negative_gap_cap_by_label: Mapping[str, float] = field(default_factory=dict)
    label_weight_by_label: Mapping[str, float] = field(default_factory=dict)
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class ClassEvidencePositiveGapCapRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_logits"] = "class_evidence_logits"
    mode: Literal["positive_gap_hinge"] = "positive_gap_hinge"
    positive_gap_cap: float = 8.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class InteractionGapCapRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_interaction_scores"] = (
        "class_evidence_interaction_scores"
    )
    mode: Literal["absolute_gap_hinge"] = "absolute_gap_hinge"
    gap_cap: float = 6.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class TopSupportScoreMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_top_support_scores"] = (
        "class_evidence_top_support_scores"
    )
    mode: Literal["softplus_true_vs_hardest_negative"] = (
        "softplus_true_vs_hardest_negative"
    )
    temperature: float = 1.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0
    label_weight_by_label: Mapping[str, float] = field(default_factory=dict)
    support_conditioned_multiplier: MarginSupportWeightingConfig = field(
        default_factory=MarginSupportWeightingConfig
    )
    hardness_weighting: MarginHardnessWeightingConfig = field(
        default_factory=MarginHardnessWeightingConfig
    )


@dataclass(frozen=True)
class TopSupportGapMinConstraintConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_top_support_scores"] = (
        "class_evidence_top_support_scores"
    )
    mode: Literal["support_conditioned_min_gap"] = "support_conditioned_min_gap"
    base_min_gap: float = 0.0
    base_min_gap_by_label: Mapping[str, float] = field(default_factory=dict)
    support_source: Literal["top_branch_margin"] = "top_branch_margin"
    support_gain: float = 0.5
    support_cap: float = 2.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class WeakPositiveSupportBandConfig:
    min_support: float = 0.2
    max_support: float = 1.0


@dataclass(frozen=True)
class WeakPositiveSupportWeightingConfig:
    enabled: bool = False
    min_support: float = 0.2
    max_support: float = 1.0
    multiplier: float = 1.0
    support_band_by_label: Mapping[str, WeakPositiveSupportBandConfig] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class WeakPositiveTargetBoostConfig:
    enabled: bool = False
    min_support: float = 0.2
    max_support: float = 1.0
    boost: float = 0.0
    boost_by_label: Mapping[str, float] = field(default_factory=dict)
    support_band_by_label: Mapping[str, WeakPositiveSupportBandConfig] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class WeakPositiveMarginBoostConfig:
    enabled: bool = False
    min_support: float = 0.2
    max_support: float = 1.0
    boost: float = 0.0
    boost_by_label: Mapping[str, float] = field(default_factory=dict)
    support_band_by_label: Mapping[str, WeakPositiveSupportBandConfig] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ClassTopBranchRelativeMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_top_branch_margin_features"] = (
        "class_top_branch_margin_features"
    )
    mode: Literal["true_vs_hardest_negative_hinge"] = "true_vs_hardest_negative_hinge"
    margin: float = 0.3
    margin_by_label: Mapping[str, float] = field(default_factory=dict)
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0
    support_weighting: MarginSupportWeightingConfig = field(
        default_factory=MarginSupportWeightingConfig
    )
    hardness_weighting: MarginHardnessWeightingConfig = field(
        default_factory=MarginHardnessWeightingConfig
    )
    hardness_weighting_by_label: Mapping[str, MarginHardnessWeightingConfig] = field(
        default_factory=dict
    )
    weak_positive_support_weighting: WeakPositiveSupportWeightingConfig = field(
        default_factory=WeakPositiveSupportWeightingConfig
    )
    weak_positive_margin_boost: WeakPositiveMarginBoostConfig = field(
        default_factory=WeakPositiveMarginBoostConfig
    )


@dataclass(frozen=True)
class TopTeacherGapMinConstraintConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_top_branch_margin_features"] = (
        "class_top_branch_margin_features"
    )
    mode: Literal["support_conditioned_min_gap"] = "support_conditioned_min_gap"
    base_min_gap: float = 0.0
    base_min_gap_by_label: Mapping[str, float] = field(default_factory=dict)
    support_source: Literal["top_branch_margin"] = "top_branch_margin"
    support_gain: float = 0.5
    support_gain_by_label: Mapping[str, float] = field(default_factory=dict)
    support_cap: float = 2.0
    support_cap_by_label: Mapping[str, float] = field(default_factory=dict)
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0
    weak_positive_support_weighting: WeakPositiveSupportWeightingConfig = field(
        default_factory=WeakPositiveSupportWeightingConfig
    )
    weak_positive_target_boost: WeakPositiveTargetBoostConfig = field(
        default_factory=WeakPositiveTargetBoostConfig
    )


@dataclass(frozen=True)
class BranchSupportScoreMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_branch_support_scores"] = (
        "class_evidence_branch_support_scores"
    )
    mode: Literal["softplus_true_vs_hardest_negative"] = (
        "softplus_true_vs_hardest_negative"
    )
    temperature: float = 1.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class BranchDirectScoreMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_branch_direct_scores"] = (
        "class_evidence_branch_direct_scores"
    )
    mode: Literal["softplus_true_vs_hardest_negative"] = (
        "softplus_true_vs_hardest_negative"
    )
    temperature: float = 1.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class BranchPathDominanceConstraintConfig:
    enabled: bool = False
    weight: float = 0.0
    branch_source: Literal["class_evidence_branch_support_scores"] = (
        "class_evidence_branch_support_scores"
    )
    target: Literal["class_evidence_logits"] = "class_evidence_logits"
    mode: Literal["branch_gap_preservation"] = "branch_gap_preservation"
    allowed_drop: float = 0.5
    allowed_drop_by_label: Mapping[str, float] = field(default_factory=dict)
    label_weight_by_label: Mapping[str, float] = field(default_factory=dict)
    support_source: Literal["top_branch_margin"] = "top_branch_margin"
    support_weighting: MarginSupportWeightingConfig = field(
        default_factory=MarginSupportWeightingConfig
    )
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class BranchSupportDisagreementCapRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    branch_source: Literal["class_evidence_branch_support_scores"] = (
        "class_evidence_branch_support_scores"
    )
    embedding_source: Literal["class_evidence_embedding_scores"] = (
        "class_evidence_embedding_scores"
    )
    interaction_source: Literal["class_evidence_interaction_scores"] = (
        "class_evidence_interaction_scores"
    )
    mode: Literal["branch_gap_conditioned_abs_gap_cap"] = (
        "branch_gap_conditioned_abs_gap_cap"
    )
    disagreement_threshold: float = 0.0
    condition_gain: float = 1.0
    condition_cap: float = 3.0
    embedding_gap_cap: float = 6.0
    interaction_gap_cap: float = 4.0
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class ClassGatedBranchLogitMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    margin: float = 0.0
    target: Literal["class_gated_branch_logits"] = "class_gated_branch_logits"
    mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    class_weighted: bool = False
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"


@dataclass(frozen=True)
class BranchToEvidenceRankingConsistencyConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["class_evidence_logits"] = "class_evidence_logits"
    source: Literal[
        "class_gated_branch_logits",
        "top_branch_margin",
        "class_top_branch_margin_features",
        "class_top_branch_margin_relative_features",
    ] = "class_gated_branch_logits"
    mode: Literal[
        "true_vs_hardest_negative",
        "teacher_distribution_kl",
        "true_label_anchored_softplus",
    ] = "true_vs_hardest_negative"
    teacher_detach: bool = True
    teacher_temperature: float = 1.0
    student_temperature: float = 1.0
    temperature: float = 1.0
    teacher_gap_cap: float | None = None
    teacher_floor_by_label: Mapping[str, float] = field(default_factory=dict)
    tolerance: float = 0.0
    class_weighted: bool = False
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"
    warmup_epochs: int = 0
    support_weighting: MarginSupportWeightingConfig = field(
        default_factory=MarginSupportWeightingConfig
    )
    hardness_weighting: MarginHardnessWeightingConfig = field(
        default_factory=MarginHardnessWeightingConfig
    )


@dataclass(frozen=True)
class GlobalResidualAntiVetoConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["global_residual_logits", "final_logits"] = "global_residual_logits"
    reference: Literal["class_evidence_logits"] = "class_evidence_logits"
    support_source: Literal["none", "top_branch_margin"] = "none"
    mode: Literal[
        "true_vs_hardest_negative",
        "final_gap_preservation",
    ] = "true_vs_hardest_negative"
    margin_mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    evidence_confidence_threshold: float = 0.0
    min_residual_gap: float = -0.5
    support_threshold: float = 0.0
    allowed_gap_drop: float = 0.0
    class_weighted: bool = False
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class ResidualContradictionRegularizationConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["global_residual_logits"] = "global_residual_logits"
    reference: Literal["class_evidence_logits"] = "class_evidence_logits"
    mode: Literal["opposite_gap_penalty"] = "opposite_gap_penalty"
    margin_mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    evidence_gap_threshold: float = 0.0
    min_residual_gap: float = -0.3
    class_weighted: bool = False
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class GateWeightedBranchMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    margin: float = 0.0
    target: Literal["true_class_gate"] = "true_class_gate"
    source: Literal["branch_logits"] = "branch_logits"
    mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    class_weighted: bool = False
    warmup_epochs: int = 0
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"
    branch_selection: Literal["gate_weighted"] = "gate_weighted"


@dataclass(frozen=True)
class AutoMarginByTrainStatsConfig:
    enabled: bool = False
    strategy: Literal["ema_violation_controller"] = "ema_violation_controller"
    start_epoch: int = 1
    update_interval_epochs: int = 1
    ema: float = 0.9
    step: float = 0.0
    target_violation_rate_by_label: Mapping[str, float] = field(default_factory=dict)
    min_margin_by_label: Mapping[str, float] = field(default_factory=dict)
    max_margin_by_label: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoPositiveThresholdByTrainStatsConfig:
    enabled: bool = False
    strategy: Literal["ema_eligible_controller"] = "ema_eligible_controller"
    start_epoch: int = 1
    update_interval_epochs: int = 1
    ema: float = 0.9
    step: float = 0.0
    target_eligible_rate_by_label: Mapping[str, float] = field(default_factory=dict)
    min_threshold_by_label: Mapping[str, float] = field(default_factory=dict)
    max_threshold_by_label: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoBadBranchThresholdByTrainStatsConfig:
    enabled: bool = False
    strategy: Literal["ema_bad_gate_mass_controller"] = "ema_bad_gate_mass_controller"
    start_epoch: int = 1
    update_interval_epochs: int = 1
    ema: float = 0.9
    step: float = 0.0
    target_bad_gate_mass_by_label: Mapping[str, float] = field(default_factory=dict)
    min_threshold_by_label: Mapping[str, float] = field(default_factory=dict)
    max_threshold_by_label: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GateBranchRegretWeightScheduleConfig:
    enabled: bool = False
    start_epoch: int = 1
    end_epoch: int = 1
    start_multiplier: float = 1.0
    end_multiplier: float = 1.0


@dataclass(frozen=True)
class TopBranchMarginPhaseWeightScheduleConfig:
    enabled: bool = False
    start_epoch: int = 1
    end_epoch: int | None = None
    start_multiplier_by_label: Mapping[str, float] = field(default_factory=dict)
    label_multiplier_by_label: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TopBranchMarginHardnessWeightingConfig:
    enabled: bool = False
    source: Literal["margin_deficit"] = "margin_deficit"
    mode: Literal["linear"] = "linear"
    gain: float = 1.0
    cap: float = 3.0


@dataclass(frozen=True)
class GateBranchRegretConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["true_class_gate"] = "true_class_gate"
    source: Literal["branch_logits"] = "branch_logits"
    mode: Literal["best_margin_regret"] = "best_margin_regret"
    margin_mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    positive_threshold: float = 0.0
    positive_threshold_by_label: Mapping[str, float] = field(default_factory=dict)
    tolerance: float = 0.0
    warmup_epochs: int = 0
    weight_schedule: GateBranchRegretWeightScheduleConfig = field(
        default_factory=GateBranchRegretWeightScheduleConfig
    )
    auto_positive_threshold_by_train_stats: AutoPositiveThresholdByTrainStatsConfig = (
        field(default_factory=AutoPositiveThresholdByTrainStatsConfig)
    )


@dataclass(frozen=True)
class GateBestBranchAlignmentConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["true_class_gate"] = "true_class_gate"
    source: Literal["branch_logits"] = "branch_logits"
    mode: Literal["weak_positive_best_branch_alignment"] = (
        "weak_positive_best_branch_alignment"
    )
    margin_mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    min_best_margin: float = 0.2
    min_best_margin_by_label: Mapping[str, float] = field(default_factory=dict)
    max_best_margin: float = 1.0
    max_best_margin_by_label: Mapping[str, float] = field(default_factory=dict)
    mismatch_margin_drop: float = 0.5
    mismatch_margin_drop_by_label: Mapping[str, float] = field(default_factory=dict)
    label_weight_by_label: Mapping[str, float] = field(default_factory=dict)
    loss: Literal["negative_log_best_gate"] = "negative_log_best_gate"
    detach_branch_margin: bool = True
    class_weighted: bool = False
    reduction: Literal["mean"] = "mean"
    warmup_epochs: int = 0


@dataclass(frozen=True)
class GateBadBranchSuppressionConfig:
    enabled: bool = False
    weight: float = 0.0
    target: Literal["true_class_gate"] = "true_class_gate"
    source: Literal["branch_logits"] = "branch_logits"
    mode: Literal["margin_below_threshold"] = "margin_below_threshold"
    margin_mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    bad_margin_threshold: float = 0.0
    bad_margin_threshold_by_label: Mapping[str, float] = field(default_factory=dict)
    warmup_epochs: int = 0
    weight_schedule: GateBranchRegretWeightScheduleConfig = field(
        default_factory=GateBranchRegretWeightScheduleConfig
    )
    auto_bad_margin_threshold_by_train_stats: AutoBadBranchThresholdByTrainStatsConfig = field(
        default_factory=AutoBadBranchThresholdByTrainStatsConfig
    )


@dataclass(frozen=True)
class TopBranchMarginConfig:
    enabled: bool = False
    weight: float = 0.0
    margin: float = 0.0
    target: Literal["branch_logits"] = "branch_logits"
    mode: Literal["true_vs_hardest_negative"] = "true_vs_hardest_negative"
    branch_reduction: Literal["max"] = "max"
    class_weighted: bool = False
    reduction: Literal["mean", "class_balanced_violating_mean"] = "mean"
    warmup_epochs: int = 0
    margin_by_label: Mapping[str, float] = field(default_factory=dict)
    auto_margin_by_train_stats: AutoMarginByTrainStatsConfig = field(
        default_factory=AutoMarginByTrainStatsConfig
    )
    phase_weight_schedule: TopBranchMarginPhaseWeightScheduleConfig = field(
        default_factory=TopBranchMarginPhaseWeightScheduleConfig
    )
    hardness_weighting: TopBranchMarginHardnessWeightingConfig = field(
        default_factory=TopBranchMarginHardnessWeightingConfig
    )


@dataclass(frozen=True)
class ClassWeightingConfig:
    enabled: bool = False
    type: Literal[
        "sqrt_inverse_frequency",
        "power_inverse_frequency",
    ] = "sqrt_inverse_frequency"
    normalize: Literal["mean_one"] = "mean_one"
    source: Literal["train"] = "train"
    power: float = 0.5


@dataclass(frozen=True)
class LabelSmoothingConfig:
    enabled: bool = False
    value: float = 0.0


@dataclass(frozen=True)
class BranchBinaryPosWeightConfig:
    enabled: bool = False
    type: Literal["sqrt_normal_over_abnormal"] = "sqrt_normal_over_abnormal"
    source: Literal["train"] = "train"


@dataclass(frozen=True)
class BranchBinaryAuxiliaryScheduleConfig:
    enabled: bool = False
    type: Literal["none", "cosine_floor"] = "none"
    max_weight: float = 0.4
    min_weight: float = 0.1
    total_epochs: int | None = None


@dataclass(frozen=True)
class BranchBinaryAuxiliaryMonitorConfig:
    loss_weight: float = 0.3


@dataclass(frozen=True)
class BranchBinaryAuxiliaryLossConfig:
    enabled: bool = False
    weight: float = 0.3
    label_to_index: Mapping[str, int] = field(default_factory=dict)
    pos_weight: BranchBinaryPosWeightConfig = field(
        default_factory=BranchBinaryPosWeightConfig
    )
    schedule: BranchBinaryAuxiliaryScheduleConfig = field(
        default_factory=BranchBinaryAuxiliaryScheduleConfig
    )
    monitor: BranchBinaryAuxiliaryMonitorConfig = field(
        default_factory=BranchBinaryAuxiliaryMonitorConfig
    )
    aggregation: Literal["mean"] = "mean"


@dataclass(frozen=True)
class LossConfig:
    type: Literal["bce", "focal", "cross_entropy"] = "bce"
    auto_pos_weight: bool = False
    pos_weight: float | None = None
    gamma: float = 2.0
    class_weighting: ClassWeightingConfig = field(default_factory=ClassWeightingConfig)
    label_smoothing: LabelSmoothingConfig = field(default_factory=LabelSmoothingConfig)
    branch_auxiliary: BranchAuxiliaryLossConfig = field(
        default_factory=BranchAuxiliaryLossConfig
    )
    branch_binary_auxiliary: BranchBinaryAuxiliaryLossConfig = field(
        default_factory=BranchBinaryAuxiliaryLossConfig
    )
    attention_entropy: AttentionEntropyLossConfig = field(
        default_factory=AttentionEntropyLossConfig
    )
    gate_entropy_regularization: GateEntropyRegularizationConfig = field(
        default_factory=GateEntropyRegularizationConfig
    )
    class_gate_diversity_regularization: ClassGateDiversityRegularizationConfig = field(
        default_factory=ClassGateDiversityRegularizationConfig
    )
    class_evidence_margin: ClassEvidenceMarginConfig = field(
        default_factory=ClassEvidenceMarginConfig
    )
    class_evidence_gap_cap_regularization: ClassEvidenceGapCapRegularizationConfig = (
        field(default_factory=ClassEvidenceGapCapRegularizationConfig)
    )
    class_evidence_positive_gap_cap_regularization: ClassEvidencePositiveGapCapRegularizationConfig = field(
        default_factory=ClassEvidencePositiveGapCapRegularizationConfig
    )
    interaction_gap_cap_regularization: InteractionGapCapRegularizationConfig = field(
        default_factory=InteractionGapCapRegularizationConfig
    )
    top_support_score_margin: TopSupportScoreMarginConfig = field(
        default_factory=TopSupportScoreMarginConfig
    )
    top_support_gap_min_constraint: TopSupportGapMinConstraintConfig = field(
        default_factory=TopSupportGapMinConstraintConfig
    )
    class_top_branch_relative_margin: ClassTopBranchRelativeMarginConfig = field(
        default_factory=ClassTopBranchRelativeMarginConfig
    )
    top_teacher_gap_min_constraint: TopTeacherGapMinConstraintConfig = field(
        default_factory=TopTeacherGapMinConstraintConfig
    )
    branch_support_score_margin: BranchSupportScoreMarginConfig = field(
        default_factory=BranchSupportScoreMarginConfig
    )
    branch_direct_score_margin: BranchDirectScoreMarginConfig = field(
        default_factory=BranchDirectScoreMarginConfig
    )
    branch_path_dominance_constraint: BranchPathDominanceConstraintConfig = field(
        default_factory=BranchPathDominanceConstraintConfig
    )
    branch_support_disagreement_cap_regularization: BranchSupportDisagreementCapRegularizationConfig = field(
        default_factory=BranchSupportDisagreementCapRegularizationConfig
    )
    class_gated_branch_logit_margin: ClassGatedBranchLogitMarginConfig = field(
        default_factory=ClassGatedBranchLogitMarginConfig
    )
    branch_to_evidence_ranking_consistency: BranchToEvidenceRankingConsistencyConfig = (
        field(default_factory=BranchToEvidenceRankingConsistencyConfig)
    )
    global_residual_anti_veto: GlobalResidualAntiVetoConfig = field(
        default_factory=GlobalResidualAntiVetoConfig
    )
    residual_contradiction_regularization: ResidualContradictionRegularizationConfig = (
        field(default_factory=ResidualContradictionRegularizationConfig)
    )
    gate_weighted_branch_margin: GateWeightedBranchMarginConfig = field(
        default_factory=GateWeightedBranchMarginConfig
    )
    gate_best_branch_alignment: GateBestBranchAlignmentConfig = field(
        default_factory=GateBestBranchAlignmentConfig
    )
    gate_branch_regret: GateBranchRegretConfig = field(
        default_factory=GateBranchRegretConfig
    )
    gate_bad_branch_suppression: GateBadBranchSuppressionConfig = field(
        default_factory=GateBadBranchSuppressionConfig
    )
    top_branch_margin: TopBranchMarginConfig = field(
        default_factory=TopBranchMarginConfig
    )


@dataclass(frozen=True)
class SamplerConfig:
    weighted_random: bool = False
    enabled: bool = False
    type: Literal["none", "sqrt_inverse_class"] = "none"
    replacement: bool = True
    num_samples: Literal["dataset_size"] | int = "dataset_size"
    source: Literal["train"] = "train"


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: Literal["val_loss"] = "val_loss"
    patience: int = 15
    min_delta: float = 1e-4


@dataclass(frozen=True)
class TrainingInitializationConfig:
    checkpoint_path: str | None = None
    load_model_state: bool = True
    strict: bool = False
    load_optimizer_state: bool = False
    skip_mismatched_shapes: bool = False


@dataclass(frozen=True)
class ValidationLossConfig:
    use_train_loss_config: bool = True
    class_weight_source: Literal["train"] = "train"
    binary_pos_weight_source: Literal["train"] = "train"


@dataclass(frozen=True)
class ValidationConfig:
    loss: ValidationLossConfig = field(default_factory=ValidationLossConfig)


@dataclass(frozen=True)
class CheckpointMonitorConfig:
    name: Literal[
        "val_macro_f1",
        "val_macro_recall",
        "val_loss",
        "val_loss_total_monitor",
        "last",
    ]
    mode: Literal["max", "min", "latest", "last"]
    top_k: int = 3
    filename_prefix: str | None = None


@dataclass(frozen=True)
class CheckpointingConfig:
    monitors: tuple[CheckpointMonitorConfig, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    top_k: int
    max_grad_norm: float
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    initialization: TrainingInitializationConfig = field(
        default_factory=TrainingInitializationConfig
    )


@dataclass(frozen=True)
class AnalysisOutputConfig:
    save_logits: bool = True
    save_probabilities: bool = True
    save_embeddings: bool = False
    save_clip_metadata: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    outputs: AnalysisOutputConfig = field(default_factory=AnalysisOutputConfig)


@dataclass(frozen=True)
class TrainingRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    analysis: AnalysisConfig
    val: ValidationConfig = field(default_factory=ValidationConfig)
    checkpointing: CheckpointingConfig | None = None


@dataclass(frozen=True)
class EvalConfig:
    experiment: ExperimentConfig
    checkpoint_path: str
    data: DataConfig
    analysis: AnalysisConfig
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )


@dataclass(frozen=True)
class CvFoldConfig:
    name: str
    train_dirs: list[str]
    val_dirs: list[str]


@dataclass(frozen=True)
class CvRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    analysis: AnalysisConfig
    folds: list[CvFoldConfig]
    val: ValidationConfig = field(default_factory=ValidationConfig)
    checkpointing: CheckpointingConfig | None = None


def resolve_label_float_overrides(
    *,
    default: float,
    overrides: Mapping[str, float],
    label_to_index: Mapping[str, int],
) -> tuple[float, ...]:
    values = [float(default)] * len(label_to_index)
    for label_name, value in overrides.items():
        if label_name not in label_to_index:
            raise ValueError(f"unknown label override {label_name!r}")
        values[int(label_to_index[label_name])] = float(value)
    return tuple(values)


class JsonConfigLoader:
    _THRESHOLD_METRICS = {"f1", "balanced_accuracy", "youden_j"}
    _EARLY_STOPPING_MONITORS = {"val_loss"}
    _EVIDENCE_SCORE_SOURCES = {
        "attention_weight",
        "attention_logit",
        "instance_logit",
    }
    _EVIDENCE_POOLING_TYPES = {"mean", "branch_gated", "class_aware_branch_gated"}
    _CLASS_GATE_MODES = {"query"}
    _CLASS_GATE_SCORERS = {"diagonal", "normalized_mlp"}
    _CLASS_GATE_EVIDENCE_SCORER_TYPES = {
        "embedding_mlp",
        "two_tower_mlp",
        "class_axis_attention",
    }
    _CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES = {
        "sigmoid_max",
        "bounded_sigmoid",
    }
    _CLASS_GATE_SCORE_BOUNDING_MODES = {"tanh_bound"}
    _CLASS_GATE_BRANCH_DIRECT_SCORE_TOP_SUPPORT_MODES = {
        "learned_weighted_sum",
        "raw_existential_plus_relative_correction",
    }
    _CLASS_GATE_TOP_SUPPORT_DIRECT_PATH_MODES = {"monotonic_raw_relative"}
    _CLASS_GATE_TOP_SUPPORT_DIRECT_PATH_POSITIVE_TRANSFORMS = {"softplus"}
    _CLASS_GATE_TOP_SUPPORT_NEGATIVE_RELATIVE_CAP_MODES = {"raw_fraction_cap"}
    _CLASS_GATE_BRANCH_DIRECT_SCORE_POSITIVE_WEIGHT_MODES = {"softplus"}
    _CLASS_GATE_BRANCH_FEATURE_TRANSFORM_MODES = {"tanh"}
    _CLASS_GATE_BRANCH_LOGIT_FEATURE_MODES = {"raw", "hardest_negative_margin"}
    _CLASS_GATE_MIXING_MODES = {"uniform_to_learned"}
    _GLOBAL_RESIDUAL_WARMUP_MODES = {"zero_to_learned"}
    _GLOBAL_RESIDUAL_CORRECTION_MODES = {"additive", "gated_zero_mean"}
    _GLOBAL_RESIDUAL_CONFIDENCE_GATE_SOURCES = {"evidence_gap"}
    _GLOBAL_RESIDUAL_CONFIDENCE_GATE_MODES = {"damped_sigmoid"}
    _MARGIN_SUPPORT_WEIGHTING_SOURCES = {"top_branch_margin", "same_as_source"}
    _MARGIN_SUPPORT_WEIGHTING_MODES = {"linear", "positive_linear"}
    _MARGIN_HARDNESS_WEIGHTING_SOURCES = {
        "evidence_gap",
        "top_support_gap",
        "teacher_gap_deficit",
    }
    _MARGIN_HARDNESS_WEIGHTING_MODES = {"negative_gap", "linear"}
    _GATE_ENTROPY_TARGETS = {
        "evidence_gate",
        "class_evidence_gate",
        "true_class_evidence_gate",
        "class_evidence_learned_gate",
    }
    _CLASS_GATE_DIVERSITY_TARGETS = {
        "class_evidence_gate",
        "class_evidence_learned_gate",
    }
    _CLASS_GATE_DIVERSITY_METRICS = {"js_divergence"}
    _CLASS_EVIDENCE_MARGIN_TARGETS = {"class_evidence_logits"}
    _CLASS_EVIDENCE_MARGIN_MODES = {
        "minority_vs_major",
        "true_vs_hardest_negative",
        "softplus_true_vs_hardest_negative",
    }
    _CLASS_EVIDENCE_GAP_CAP_REGULARIZATION_TARGETS = {"class_evidence_logits"}
    _CLASS_EVIDENCE_GAP_CAP_REGULARIZATION_MODES = {"negative_gap_hinge"}
    _CLASS_EVIDENCE_GAP_CAP_REGULARIZATION_REDUCTIONS = {"mean"}
    _CLASS_EVIDENCE_POSITIVE_GAP_CAP_REGULARIZATION_TARGETS = {"class_evidence_logits"}
    _CLASS_EVIDENCE_POSITIVE_GAP_CAP_REGULARIZATION_MODES = {"positive_gap_hinge"}
    _CLASS_EVIDENCE_POSITIVE_GAP_CAP_REGULARIZATION_REDUCTIONS = {"mean"}
    _INTERACTION_GAP_CAP_REGULARIZATION_TARGETS = {"class_evidence_interaction_scores"}
    _INTERACTION_GAP_CAP_REGULARIZATION_MODES = {"absolute_gap_hinge"}
    _INTERACTION_GAP_CAP_REGULARIZATION_REDUCTIONS = {"mean"}
    _TOP_SUPPORT_SCORE_MARGIN_TARGETS = {"class_evidence_top_support_scores"}
    _TOP_SUPPORT_SCORE_MARGIN_MODES = {"softplus_true_vs_hardest_negative"}
    _TOP_SUPPORT_SCORE_MARGIN_REDUCTIONS = {"mean"}
    _TOP_SUPPORT_GAP_MIN_CONSTRAINT_TARGETS = {"class_evidence_top_support_scores"}
    _TOP_SUPPORT_GAP_MIN_CONSTRAINT_MODES = {"support_conditioned_min_gap"}
    _TOP_SUPPORT_GAP_MIN_CONSTRAINT_SUPPORT_SOURCES = {"top_branch_margin"}
    _TOP_SUPPORT_GAP_MIN_CONSTRAINT_REDUCTIONS = {"mean"}
    _CLASS_TOP_BRANCH_RELATIVE_MARGIN_TARGETS = {"class_top_branch_margin_features"}
    _CLASS_TOP_BRANCH_RELATIVE_MARGIN_MODES = {"true_vs_hardest_negative_hinge"}
    _CLASS_TOP_BRANCH_RELATIVE_MARGIN_REDUCTIONS = {"mean"}
    _TOP_TEACHER_GAP_MIN_CONSTRAINT_TARGETS = {"class_top_branch_margin_features"}
    _TOP_TEACHER_GAP_MIN_CONSTRAINT_MODES = {"support_conditioned_min_gap"}
    _TOP_TEACHER_GAP_MIN_CONSTRAINT_SUPPORT_SOURCES = {"top_branch_margin"}
    _TOP_TEACHER_GAP_MIN_CONSTRAINT_REDUCTIONS = {"mean"}
    _BRANCH_SUPPORT_SCORE_MARGIN_TARGETS = {"class_evidence_branch_support_scores"}
    _BRANCH_SUPPORT_SCORE_MARGIN_MODES = {"softplus_true_vs_hardest_negative"}
    _BRANCH_SUPPORT_SCORE_MARGIN_REDUCTIONS = {"mean"}
    _BRANCH_DIRECT_SCORE_MARGIN_TARGETS = {"class_evidence_branch_direct_scores"}
    _BRANCH_DIRECT_SCORE_MARGIN_MODES = {"softplus_true_vs_hardest_negative"}
    _BRANCH_DIRECT_SCORE_MARGIN_REDUCTIONS = {"mean"}
    _BRANCH_PATH_DOMINANCE_BRANCH_SOURCES = {"class_evidence_branch_support_scores"}
    _BRANCH_PATH_DOMINANCE_TARGETS = {"class_evidence_logits"}
    _BRANCH_PATH_DOMINANCE_MODES = {"branch_gap_preservation"}
    _BRANCH_PATH_DOMINANCE_SUPPORT_SOURCES = {"top_branch_margin"}
    _BRANCH_PATH_DOMINANCE_REDUCTIONS = {"mean"}
    _BRANCH_SUPPORT_DISAGREEMENT_BRANCH_SOURCES = {
        "class_evidence_branch_support_scores"
    }
    _BRANCH_SUPPORT_DISAGREEMENT_EMBEDDING_SOURCES = {"class_evidence_embedding_scores"}
    _BRANCH_SUPPORT_DISAGREEMENT_INTERACTION_SOURCES = {
        "class_evidence_interaction_scores"
    }
    _BRANCH_SUPPORT_DISAGREEMENT_MODES = {"branch_gap_conditioned_abs_gap_cap"}
    _BRANCH_SUPPORT_DISAGREEMENT_REDUCTIONS = {"mean"}
    _MARGIN_REDUCTIONS = {"mean", "class_balanced_violating_mean"}
    _CLASS_GATED_BRANCH_LOGIT_MARGIN_TARGETS = {"class_gated_branch_logits"}
    _CLASS_GATED_BRANCH_LOGIT_MARGIN_MODES = {"true_vs_hardest_negative"}
    _BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_TARGETS = {"class_evidence_logits"}
    _BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_SOURCES = {
        "class_gated_branch_logits",
        "top_branch_margin",
        "class_top_branch_margin_features",
        "class_top_branch_margin_relative_features",
    }
    _BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_MODES = {
        "true_vs_hardest_negative",
        "teacher_distribution_kl",
        "true_label_anchored_softplus",
    }
    _GLOBAL_RESIDUAL_ANTI_VETO_TARGETS = {
        "global_residual_logits",
        "final_logits",
    }
    _GLOBAL_RESIDUAL_ANTI_VETO_REFERENCES = {"class_evidence_logits"}
    _GLOBAL_RESIDUAL_ANTI_VETO_SUPPORT_SOURCES = {"none", "top_branch_margin"}
    _GLOBAL_RESIDUAL_ANTI_VETO_MODES = {
        "true_vs_hardest_negative",
        "final_gap_preservation",
    }
    _GLOBAL_RESIDUAL_ANTI_VETO_MARGIN_MODES = {"true_vs_hardest_negative"}
    _RESIDUAL_CONTRADICTION_TARGETS = {"global_residual_logits"}
    _RESIDUAL_CONTRADICTION_REFERENCES = {"class_evidence_logits"}
    _RESIDUAL_CONTRADICTION_MODES = {"opposite_gap_penalty"}
    _RESIDUAL_CONTRADICTION_MARGIN_MODES = {"true_vs_hardest_negative"}
    _GATE_WEIGHTED_BRANCH_MARGIN_TARGETS = {"true_class_gate"}
    _GATE_WEIGHTED_BRANCH_MARGIN_SOURCES = {"branch_logits"}
    _GATE_WEIGHTED_BRANCH_MARGIN_MODES = {"true_vs_hardest_negative"}
    _GATE_WEIGHTED_BRANCH_MARGIN_SELECTIONS = {"gate_weighted"}
    _GATE_BEST_BRANCH_ALIGNMENT_TARGETS = {"true_class_gate"}
    _GATE_BEST_BRANCH_ALIGNMENT_SOURCES = {"branch_logits"}
    _GATE_BEST_BRANCH_ALIGNMENT_MODES = {"weak_positive_best_branch_alignment"}
    _GATE_BEST_BRANCH_ALIGNMENT_MARGIN_MODES = {"true_vs_hardest_negative"}
    _GATE_BEST_BRANCH_ALIGNMENT_LOSSES = {"negative_log_best_gate"}
    _GATE_BEST_BRANCH_ALIGNMENT_REDUCTIONS = {"mean"}
    _GATE_BRANCH_REGRET_TARGETS = {"true_class_gate"}
    _GATE_BRANCH_REGRET_SOURCES = {"branch_logits"}
    _GATE_BRANCH_REGRET_MODES = {"best_margin_regret"}
    _GATE_BRANCH_REGRET_MARGIN_MODES = {"true_vs_hardest_negative"}
    _GATE_BAD_BRANCH_SUPPRESSION_TARGETS = {"true_class_gate"}
    _GATE_BAD_BRANCH_SUPPRESSION_SOURCES = {"branch_logits"}
    _GATE_BAD_BRANCH_SUPPRESSION_MODES = {"margin_below_threshold"}
    _GATE_BAD_BRANCH_SUPPRESSION_MARGIN_MODES = {"true_vs_hardest_negative"}
    _TOP_BRANCH_MARGIN_TARGETS = {"branch_logits"}
    _TOP_BRANCH_MARGIN_MODES = {"true_vs_hardest_negative"}
    _TOP_BRANCH_MARGIN_BRANCH_REDUCTIONS = {"max"}
    _TOP_BRANCH_MARGIN_HARDNESS_SOURCES = {"margin_deficit"}
    _TOP_BRANCH_MARGIN_HARDNESS_MODES = {"linear"}
    _AUTO_MARGIN_STRATEGIES = {"ema_violation_controller"}
    _AUTO_POSITIVE_THRESHOLD_STRATEGIES = {"ema_eligible_controller"}
    _AUTO_BAD_BRANCH_THRESHOLD_STRATEGIES = {"ema_bad_gate_mass_controller"}
    _TIME_SHIFT_MODES = {"zero_pad", "roll"}
    _AUGMENTATION_POLICY_TYPES = {"independent", "one_of"}
    _AUGMENTATION_POLICY_CHOICES = {"none", "waveform", "fbank", "both_light"}
    _BRANCH_EVENT_DROPOUT_MODES = {"zero_mask"}
    _SELECTED_EVIDENCE_DROPOUT_MODES = {"zero"}
    _CLASS_WEIGHTING_TYPES = {
        "sqrt_inverse_frequency",
        "power_inverse_frequency",
    }
    _CLASS_WEIGHTING_NORMALIZERS = {"mean_one"}
    _LOSS_WEIGHT_SOURCES = {"train"}
    _BRANCH_BINARY_POS_WEIGHT_TYPES = {"sqrt_normal_over_abnormal"}
    _BRANCH_BINARY_AUXILIARY_SCHEDULE_TYPES = {"none", "cosine_floor"}
    _SAMPLER_TYPES = {"none", "sqrt_inverse_class"}
    _CHECKPOINT_MONITORS = {
        "val_macro_f1": {"max"},
        "val_macro_recall": {"max"},
        "val_loss": {"min"},
        "val_loss_total_monitor": {"min"},
        "last": {"latest", "last"},
    }

    @staticmethod
    def _validate_probability(value: float, *, field_name: str) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{field_name} must be within [0, 1]")

    @staticmethod
    def _validate_epoch_window(
        *,
        start_epoch: int,
        end_epoch: int | None,
        field_name: str,
    ) -> None:
        if not isinstance(start_epoch, int):
            raise TypeError(f"{field_name}.start_epoch must be an integer")
        if start_epoch <= 0:
            raise ValueError(f"{field_name}.start_epoch must be greater than zero")
        if end_epoch is not None and not isinstance(end_epoch, int):
            raise TypeError(f"{field_name}.end_epoch must be an integer or null")
        if end_epoch is not None and end_epoch < start_epoch:
            raise ValueError(
                f"{field_name}.end_epoch must be greater than or equal to start_epoch"
            )

    @staticmethod
    def _validate_numeric_label_mapping(
        values: Mapping[str, float],
        *,
        field_name: str,
        label_to_index: Mapping[str, int],
        min_value: float,
        strict_min: bool,
        require_all_labels: bool = False,
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{field_name} must be an object")
        if require_all_labels and set(values) != set(label_to_index):
            missing = sorted(set(label_to_index) - set(values))
            extra = sorted(set(values) - set(label_to_index))
            raise ValueError(
                f"{field_name} must contain exactly data.label_to_index keys; "
                f"missing={missing}, extra={extra}"
            )
        for label_name, value in values.items():
            if not isinstance(label_name, str):
                raise TypeError(f"{field_name} keys must be label names")
            if label_name not in label_to_index:
                raise ValueError(f"{field_name} contains unknown label {label_name!r}")
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{label_name} must be numeric")
            if strict_min and value <= min_value:
                raise ValueError(
                    f"{field_name}.{label_name} must be greater than {min_value}"
                )
            if not strict_min and value < min_value:
                raise ValueError(
                    f"{field_name}.{label_name} must be greater than or equal to "
                    f"{min_value}"
                )

    @staticmethod
    def _validate_rate_label_mapping(
        values: Mapping[str, float],
        *,
        field_name: str,
        label_to_index: Mapping[str, int],
    ) -> None:
        JsonConfigLoader._validate_numeric_label_mapping(
            values,
            field_name=field_name,
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
            require_all_labels=True,
        )
        for label_name, value in values.items():
            if value > 1.0:
                raise ValueError(f"{field_name}.{label_name} must be within [0, 1]")

    @staticmethod
    def _validate_threshold_label_mapping(
        values: Mapping[str, float],
        *,
        field_name: str,
        label_to_index: Mapping[str, int],
        require_all_labels: bool = False,
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{field_name} must be an object")
        if require_all_labels and set(values) != set(label_to_index):
            missing = sorted(set(label_to_index) - set(values))
            extra = sorted(set(values) - set(label_to_index))
            raise ValueError(
                f"{field_name} must contain exactly data.label_to_index keys; "
                f"missing={missing}, extra={extra}"
            )
        for label_name, value in values.items():
            if not isinstance(label_name, str):
                raise TypeError(f"{field_name} keys must be label names")
            if label_name not in label_to_index:
                raise ValueError(f"{field_name} contains unknown label {label_name!r}")
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{label_name} must be numeric")

    @staticmethod
    def _validate_auto_margin_by_train_stats(
        cfg: AutoMarginByTrainStatsConfig,
        *,
        label_to_index: Mapping[str, int],
    ) -> None:
        field_name = "train.loss.top_branch_margin.auto_margin_by_train_stats"
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.strategy not in JsonConfigLoader._AUTO_MARGIN_STRATEGIES:
            raise ValueError(
                f"{field_name}.strategy must be 'ema_violation_controller'"
            )
        if not isinstance(cfg.start_epoch, int):
            raise TypeError(f"{field_name}.start_epoch must be an integer")
        if cfg.start_epoch <= 0:
            raise ValueError(f"{field_name}.start_epoch must be greater than zero")
        if not isinstance(cfg.update_interval_epochs, int):
            raise TypeError(f"{field_name}.update_interval_epochs must be an integer")
        if cfg.update_interval_epochs <= 0:
            raise ValueError(
                f"{field_name}.update_interval_epochs must be greater than zero"
            )
        if not isinstance(cfg.ema, int | float) or isinstance(cfg.ema, bool):
            raise TypeError(f"{field_name}.ema must be numeric")
        if not (0.0 < float(cfg.ema) < 1.0):
            raise ValueError(f"{field_name}.ema must be within (0, 1)")
        if not isinstance(cfg.step, int | float) or isinstance(cfg.step, bool):
            raise TypeError(f"{field_name}.step must be numeric")
        if cfg.enabled and cfg.step <= 0:
            raise ValueError(f"{field_name}.step must be greater than zero")
        has_label_settings = any(
            (
                cfg.target_violation_rate_by_label,
                cfg.min_margin_by_label,
                cfg.max_margin_by_label,
            )
        )
        if not cfg.enabled and not has_label_settings:
            return
        JsonConfigLoader._validate_rate_label_mapping(
            cfg.target_violation_rate_by_label,
            field_name=f"{field_name}.target_violation_rate_by_label",
            label_to_index=label_to_index,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            cfg.min_margin_by_label,
            field_name=f"{field_name}.min_margin_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
            require_all_labels=True,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            cfg.max_margin_by_label,
            field_name=f"{field_name}.max_margin_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
            require_all_labels=True,
        )
        for label_name, min_value in cfg.min_margin_by_label.items():
            max_value = cfg.max_margin_by_label[label_name]
            if min_value > max_value:
                raise ValueError(
                    f"{field_name}.min_margin_by_label.{label_name} must be less "
                    "than or equal to max_margin_by_label"
                )

    @staticmethod
    def _validate_auto_positive_threshold_by_train_stats(
        cfg: AutoPositiveThresholdByTrainStatsConfig,
        *,
        label_to_index: Mapping[str, int],
    ) -> None:
        field_name = (
            "train.loss.gate_branch_regret.auto_positive_threshold_by_train_stats"
        )
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.strategy not in JsonConfigLoader._AUTO_POSITIVE_THRESHOLD_STRATEGIES:
            raise ValueError(f"{field_name}.strategy must be 'ema_eligible_controller'")
        if not isinstance(cfg.start_epoch, int):
            raise TypeError(f"{field_name}.start_epoch must be an integer")
        if cfg.start_epoch <= 0:
            raise ValueError(f"{field_name}.start_epoch must be greater than zero")
        if not isinstance(cfg.update_interval_epochs, int):
            raise TypeError(f"{field_name}.update_interval_epochs must be an integer")
        if cfg.update_interval_epochs <= 0:
            raise ValueError(
                f"{field_name}.update_interval_epochs must be greater than zero"
            )
        if not isinstance(cfg.ema, int | float) or isinstance(cfg.ema, bool):
            raise TypeError(f"{field_name}.ema must be numeric")
        if not (0.0 < float(cfg.ema) < 1.0):
            raise ValueError(f"{field_name}.ema must be within (0, 1)")
        if not isinstance(cfg.step, int | float) or isinstance(cfg.step, bool):
            raise TypeError(f"{field_name}.step must be numeric")
        if cfg.enabled and cfg.step <= 0:
            raise ValueError(f"{field_name}.step must be greater than zero")
        has_label_settings = any(
            (
                cfg.target_eligible_rate_by_label,
                cfg.min_threshold_by_label,
                cfg.max_threshold_by_label,
            )
        )
        if not cfg.enabled and not has_label_settings:
            return
        JsonConfigLoader._validate_rate_label_mapping(
            cfg.target_eligible_rate_by_label,
            field_name=f"{field_name}.target_eligible_rate_by_label",
            label_to_index=label_to_index,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            cfg.min_threshold_by_label,
            field_name=f"{field_name}.min_threshold_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
            require_all_labels=True,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            cfg.max_threshold_by_label,
            field_name=f"{field_name}.max_threshold_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
            require_all_labels=True,
        )
        for label_name, min_value in cfg.min_threshold_by_label.items():
            max_value = cfg.max_threshold_by_label[label_name]
            if min_value > max_value:
                raise ValueError(
                    f"{field_name}.min_threshold_by_label.{label_name} must be "
                    "less than or equal to max_threshold_by_label"
                )

    @staticmethod
    def _validate_auto_bad_branch_threshold_by_train_stats(
        cfg: AutoBadBranchThresholdByTrainStatsConfig,
        *,
        label_to_index: Mapping[str, int],
    ) -> None:
        field_name = (
            "train.loss.gate_bad_branch_suppression."
            "auto_bad_margin_threshold_by_train_stats"
        )
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.strategy not in JsonConfigLoader._AUTO_BAD_BRANCH_THRESHOLD_STRATEGIES:
            raise ValueError(
                f"{field_name}.strategy must be 'ema_bad_gate_mass_controller'"
            )
        if not isinstance(cfg.start_epoch, int):
            raise TypeError(f"{field_name}.start_epoch must be an integer")
        if cfg.start_epoch <= 0:
            raise ValueError(f"{field_name}.start_epoch must be greater than zero")
        if not isinstance(cfg.update_interval_epochs, int):
            raise TypeError(f"{field_name}.update_interval_epochs must be an integer")
        if cfg.update_interval_epochs <= 0:
            raise ValueError(
                f"{field_name}.update_interval_epochs must be greater than zero"
            )
        if not isinstance(cfg.ema, int | float) or isinstance(cfg.ema, bool):
            raise TypeError(f"{field_name}.ema must be numeric")
        if not (0.0 < float(cfg.ema) < 1.0):
            raise ValueError(f"{field_name}.ema must be within (0, 1)")
        if not isinstance(cfg.step, int | float) or isinstance(cfg.step, bool):
            raise TypeError(f"{field_name}.step must be numeric")
        if cfg.enabled and cfg.step <= 0:
            raise ValueError(f"{field_name}.step must be greater than zero")
        has_label_settings = any(
            (
                cfg.target_bad_gate_mass_by_label,
                cfg.min_threshold_by_label,
                cfg.max_threshold_by_label,
            )
        )
        if not cfg.enabled and not has_label_settings:
            return
        JsonConfigLoader._validate_rate_label_mapping(
            cfg.target_bad_gate_mass_by_label,
            field_name=f"{field_name}.target_bad_gate_mass_by_label",
            label_to_index=label_to_index,
        )
        JsonConfigLoader._validate_threshold_label_mapping(
            cfg.min_threshold_by_label,
            field_name=f"{field_name}.min_threshold_by_label",
            label_to_index=label_to_index,
            require_all_labels=True,
        )
        JsonConfigLoader._validate_threshold_label_mapping(
            cfg.max_threshold_by_label,
            field_name=f"{field_name}.max_threshold_by_label",
            label_to_index=label_to_index,
            require_all_labels=True,
        )
        for label_name, min_value in cfg.min_threshold_by_label.items():
            max_value = cfg.max_threshold_by_label[label_name]
            if min_value > max_value:
                raise ValueError(
                    f"{field_name}.min_threshold_by_label.{label_name} must be "
                    "less than or equal to max_threshold_by_label"
                )

    @staticmethod
    def _validate_gate_branch_regret_weight_schedule(
        cfg: GateBranchRegretWeightScheduleConfig,
        *,
        field_name: str = "train.loss.gate_branch_regret.weight_schedule",
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if not isinstance(cfg.start_epoch, int):
            raise TypeError(f"{field_name}.start_epoch must be an integer")
        if cfg.start_epoch <= 0:
            raise ValueError(f"{field_name}.start_epoch must be greater than zero")
        if not isinstance(cfg.end_epoch, int):
            raise TypeError(f"{field_name}.end_epoch must be an integer")
        if cfg.end_epoch < cfg.start_epoch:
            raise ValueError(
                f"{field_name}.end_epoch must be greater than or equal to start_epoch"
            )
        if not isinstance(cfg.start_multiplier, int | float) or isinstance(
            cfg.start_multiplier,
            bool,
        ):
            raise TypeError(f"{field_name}.start_multiplier must be numeric")
        if cfg.start_multiplier < 0:
            raise ValueError(
                f"{field_name}.start_multiplier must be greater than or equal to zero"
            )
        if not isinstance(cfg.end_multiplier, int | float) or isinstance(
            cfg.end_multiplier,
            bool,
        ):
            raise TypeError(f"{field_name}.end_multiplier must be numeric")
        if cfg.end_multiplier < 0:
            raise ValueError(
                f"{field_name}.end_multiplier must be greater than or equal to zero"
            )

    @staticmethod
    def _validate_margin_support_weighting(
        cfg: MarginSupportWeightingConfig,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.source not in JsonConfigLoader._MARGIN_SUPPORT_WEIGHTING_SOURCES:
            raise ValueError(
                f"{field_name}.source must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_SUPPORT_WEIGHTING_SOURCES)}"
            )
        if cfg.mode not in JsonConfigLoader._MARGIN_SUPPORT_WEIGHTING_MODES:
            raise ValueError(
                f"{field_name}.mode must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_SUPPORT_WEIGHTING_MODES)}"
            )
        for value, suffix in ((cfg.gain, "gain"), (cfg.cap, "cap")):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{suffix} must be numeric")
        if cfg.gain < 0:
            raise ValueError(f"{field_name}.gain must be greater than or equal to zero")
        if cfg.cap <= 0:
            raise ValueError(f"{field_name}.cap must be greater than zero")

    @staticmethod
    def _validate_margin_hardness_weighting(
        cfg: MarginHardnessWeightingConfig,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.source not in JsonConfigLoader._MARGIN_HARDNESS_WEIGHTING_SOURCES:
            raise ValueError(
                f"{field_name}.source must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_HARDNESS_WEIGHTING_SOURCES)}"
            )
        if cfg.mode not in JsonConfigLoader._MARGIN_HARDNESS_WEIGHTING_MODES:
            raise ValueError(
                f"{field_name}.mode must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_HARDNESS_WEIGHTING_MODES)}"
            )
        for value, suffix in ((cfg.gain, "gain"), (cfg.cap, "cap")):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{suffix} must be numeric")
        if cfg.gain < 0:
            raise ValueError(f"{field_name}.gain must be greater than or equal to zero")
        if cfg.cap <= 0:
            raise ValueError(f"{field_name}.cap must be greater than zero")

    @staticmethod
    def _validate_weak_positive_support_weighting(
        cfg: WeakPositiveSupportWeightingConfig,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        for value, suffix in (
            (cfg.min_support, "min_support"),
            (cfg.max_support, "max_support"),
            (cfg.multiplier, "multiplier"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{suffix} must be numeric")
        if float(cfg.min_support) < 0.0:
            raise ValueError(
                f"{field_name}.min_support must be greater than or equal to zero"
            )
        if float(cfg.max_support) <= float(cfg.min_support):
            raise ValueError(
                f"{field_name}.max_support must be greater than min_support"
            )
        if float(cfg.multiplier) < 1.0:
            raise ValueError(
                f"{field_name}.multiplier must be greater than or equal to 1.0"
            )

    @staticmethod
    def _validate_weak_positive_support_bands(
        values: Mapping[str, WeakPositiveSupportBandConfig],
        *,
        field_name: str,
        label_to_index: Mapping[str, int],
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{field_name} must be an object")
        for label_name, band in values.items():
            if not isinstance(label_name, str):
                raise TypeError(f"{field_name} keys must be label names")
            if label_name not in label_to_index:
                raise ValueError(f"{field_name} contains unknown label {label_name!r}")
            if not isinstance(band, WeakPositiveSupportBandConfig):
                raise TypeError(f"{field_name}.{label_name} must be a support band")
            for value, suffix in (
                (band.min_support, "min_support"),
                (band.max_support, "max_support"),
            ):
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise TypeError(
                        f"{field_name}.{label_name}.{suffix} must be numeric"
                    )
            if float(band.min_support) < 0.0:
                raise ValueError(
                    f"{field_name}.{label_name}.min_support must be greater than "
                    "or equal to zero"
                )
            if float(band.max_support) <= float(band.min_support):
                raise ValueError(
                    f"{field_name}.{label_name}.max_support must be greater than "
                    "min_support"
                )

    @staticmethod
    def _validate_weak_positive_boost(
        cfg: WeakPositiveTargetBoostConfig | WeakPositiveMarginBoostConfig,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        for value, suffix in (
            (cfg.min_support, "min_support"),
            (cfg.max_support, "max_support"),
            (cfg.boost, "boost"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{suffix} must be numeric")
        if float(cfg.min_support) < 0.0:
            raise ValueError(
                f"{field_name}.min_support must be greater than or equal to zero"
            )
        if float(cfg.max_support) <= float(cfg.min_support):
            raise ValueError(
                f"{field_name}.max_support must be greater than min_support"
            )
        if float(cfg.boost) < 0.0:
            raise ValueError(
                f"{field_name}.boost must be greater than or equal to zero"
            )

    @staticmethod
    def _validate_top_branch_margin_hardness_weighting(
        cfg: TopBranchMarginHardnessWeightingConfig,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.source not in JsonConfigLoader._TOP_BRANCH_MARGIN_HARDNESS_SOURCES:
            raise ValueError(
                f"{field_name}.source must be one of "
                f"{sorted(JsonConfigLoader._TOP_BRANCH_MARGIN_HARDNESS_SOURCES)}"
            )
        if cfg.mode not in JsonConfigLoader._TOP_BRANCH_MARGIN_HARDNESS_MODES:
            raise ValueError(
                f"{field_name}.mode must be one of "
                f"{sorted(JsonConfigLoader._TOP_BRANCH_MARGIN_HARDNESS_MODES)}"
            )
        for value, suffix in ((cfg.gain, "gain"), (cfg.cap, "cap")):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{suffix} must be numeric")
        if cfg.gain < 0:
            raise ValueError(f"{field_name}.gain must be greater than or equal to zero")
        if cfg.cap <= 0:
            raise ValueError(f"{field_name}.cap must be greater than zero")

    @staticmethod
    def _validate_score_bound_component(
        cfg: ClassGateScoreBoundComponentConfig,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if cfg.mode not in JsonConfigLoader._CLASS_GATE_SCORE_BOUNDING_MODES:
            raise ValueError(
                f"{field_name}.mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_SCORE_BOUNDING_MODES)}"
            )
        for value, suffix in ((cfg.bound, "bound"), (cfg.temperature, "temperature")):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}.{suffix} must be numeric")
            if float(value) <= 0.0:
                raise ValueError(f"{field_name}.{suffix} must be greater than zero")

    @staticmethod
    def _validate_scale_bounds(
        *,
        mode: str,
        min_value: float,
        init_value: float,
        max_value: float,
        field_name: str,
    ) -> None:
        if mode not in JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES:
            raise ValueError(
                f"{field_name}_mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES)}"
            )
        for value, suffix in (
            (min_value, "min"),
            (init_value, "init"),
            (max_value, "max"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{field_name}_{suffix} must be numeric")
        if mode == "sigmoid_max":
            if float(max_value) <= 0.0:
                raise ValueError(f"{field_name}_max must be greater than zero")
            if not (0.0 <= float(init_value) <= float(max_value)):
                raise ValueError(f"{field_name}_init must be within [0, max]")
        if mode == "bounded_sigmoid":
            if float(min_value) < 0.0:
                raise ValueError(
                    f"{field_name}_min must be non-negative when mode='bounded_sigmoid'"
                )
            if float(max_value) <= float(min_value):
                raise ValueError(
                    f"{field_name}_max must be greater than min when "
                    "mode='bounded_sigmoid'"
                )
            if not (float(min_value) <= float(init_value) <= float(max_value)):
                raise ValueError(
                    f"{field_name}_init must be within [min, max] when "
                    "mode='bounded_sigmoid'"
                )

    @staticmethod
    def _validate_hold_decay_schedule(
        *,
        enabled: bool,
        hold_epochs: int,
        decay_epochs: int,
        field_name: str,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError(f"{field_name}.enabled must be a boolean")
        if not isinstance(hold_epochs, int):
            raise TypeError(f"{field_name}.hold_epochs must be an integer")
        if not isinstance(decay_epochs, int):
            raise TypeError(f"{field_name}.decay_epochs must be an integer")
        if hold_epochs < 0:
            raise ValueError(
                f"{field_name}.hold_epochs must be greater than or equal to zero"
            )
        if decay_epochs < 0:
            raise ValueError(
                f"{field_name}.decay_epochs must be greater than or equal to zero"
            )
        if enabled and hold_epochs == 0 and decay_epochs == 0:
            raise ValueError(
                f"{field_name} requires hold_epochs or decay_epochs when enabled"
            )

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Config root must be a JSON object")
        return payload

    @staticmethod
    def _validate_contiguous_labels(label_to_index: Mapping[str, int]) -> None:
        indices = sorted(int(value) for value in label_to_index.values())
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(f"label_to_index must be contiguous 0..N-1; got {indices}")
        if len(indices) < 2:
            raise ValueError("label_to_index must contain at least two classes")

    @staticmethod
    def _validate_experiment(cfg: ExperimentConfig) -> None:
        if cfg.mode != "clip":
            raise ValueError("experiment.mode must be 'clip' for this branch")
        if not cfg.name:
            raise ValueError("experiment.name must not be empty")
        if not cfg.task:
            raise ValueError("experiment.task must not be empty")
        terminal_width = cfg.logging.terminal_width
        if terminal_width is None:
            return
        if isinstance(terminal_width, bool) or not isinstance(terminal_width, int):
            raise TypeError(
                "experiment.logging.terminal_width must be an integer or null"
            )
        if terminal_width <= 0:
            raise ValueError(
                "experiment.logging.terminal_width must be greater than zero"
            )

    @staticmethod
    def _validate_bandpass(cfg: BandPassConfig, sample_rate: int) -> None:
        if not cfg.enabled:
            return
        if cfg.low_hz is None or cfg.high_hz is None:
            raise ValueError(
                "data.preprocessing.bandpass requires low_hz and high_hz when enabled"
            )
        if cfg.q <= 0:
            raise ValueError("data.preprocessing.bandpass.q must be greater than zero")
        nyquist = float(sample_rate) / 2.0
        if not (0.0 < cfg.low_hz < cfg.high_hz < nyquist):
            raise ValueError(
                "data.preprocessing.bandpass must satisfy "
                "0 < low_hz < high_hz < sample_rate / 2"
            )

    @staticmethod
    def _validate_audio(cfg: AudioConfig) -> None:
        if cfg.sample_rate <= 0:
            raise ValueError("data.audio.sample_rate must be greater than zero")
        if cfg.clip_duration_sec <= 0:
            raise ValueError("data.audio.clip_duration_sec must be greater than zero")

    @staticmethod
    def _validate_preprocessing(cfg: PreprocessingConfig, sample_rate: int) -> None:
        if cfg.ast_fbank.num_mel_bins <= 0:
            raise ValueError(
                "data.preprocessing.ast_fbank.num_mel_bins must be greater than zero"
            )
        if cfg.ast_fbank.max_length <= 0:
            raise ValueError(
                "data.preprocessing.ast_fbank.max_length must be greater than zero"
            )
        if cfg.ast_fbank.do_normalize and cfg.ast_fbank.std <= 0:
            raise ValueError(
                "data.preprocessing.ast_fbank.std must be greater than zero when normalization is enabled"
            )
        if sample_rate != 16000:
            raise ValueError(
                "data.audio.sample_rate must be 16000 for AST clip classification"
            )
        JsonConfigLoader._validate_bandpass(cfg.bandpass, sample_rate)

    @staticmethod
    def _validate_mask(cfg: MaskConfig, *, field_name: str) -> None:
        if cfg.num_masks < 0:
            raise ValueError(f"{field_name}.num_masks must be non-negative")
        if cfg.max_width < 0:
            raise ValueError(f"{field_name}.max_width must be non-negative")

    @staticmethod
    def _validate_augmentation_policy(cfg: AugmentationPolicyConfig) -> None:
        if cfg.type not in JsonConfigLoader._AUGMENTATION_POLICY_TYPES:
            raise ValueError(
                "data.augmentation.policy.type must be one of "
                f"{sorted(JsonConfigLoader._AUGMENTATION_POLICY_TYPES)}"
            )
        if cfg.type == "independent":
            return
        if not cfg.choices:
            raise ValueError(
                "data.augmentation.policy.choices must not be empty for one_of policy"
            )
        total_probability = 0.0
        positive_count = 0
        seen_names: set[str] = set()
        for choice in cfg.choices:
            if choice.name not in JsonConfigLoader._AUGMENTATION_POLICY_CHOICES:
                raise ValueError(
                    "data.augmentation.policy.choices.name must be one of "
                    f"{sorted(JsonConfigLoader._AUGMENTATION_POLICY_CHOICES)}"
                )
            if choice.name in seen_names:
                raise ValueError(
                    "data.augmentation.policy.choices must not contain duplicate names"
                )
            seen_names.add(choice.name)
            if choice.probability < 0:
                raise ValueError(
                    "data.augmentation.policy.choices.probability must be non-negative"
                )
            total_probability += float(choice.probability)
            if choice.probability > 0:
                positive_count += 1
        if positive_count == 0:
            raise ValueError(
                "data.augmentation.policy.choices must contain at least one positive probability"
            )
        if abs(total_probability - 1.0) > 1e-6:
            raise ValueError(
                "data.augmentation.policy.choices probabilities must sum to 1.0"
            )

    @staticmethod
    def _validate_augmentation(cfg: DataAugmentationConfig) -> None:
        if not isinstance(cfg.enabled, bool):
            raise ValueError("data.augmentation.enabled must be a boolean")
        if not isinstance(cfg.waveform.enabled, bool):
            raise ValueError("data.augmentation.waveform.enabled must be a boolean")
        if not isinstance(cfg.fbank.enabled, bool):
            raise ValueError("data.augmentation.fbank.enabled must be a boolean")
        JsonConfigLoader._validate_augmentation_policy(cfg.policy)
        JsonConfigLoader._validate_probability(
            cfg.waveform.probability,
            field_name="data.augmentation.waveform.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.waveform.gain.probability,
            field_name="data.augmentation.waveform.gain.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.waveform.noise.probability,
            field_name="data.augmentation.waveform.noise.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.waveform.time_shift.probability,
            field_name="data.augmentation.waveform.time_shift.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.fbank.probability,
            field_name="data.augmentation.fbank.probability",
        )
        if cfg.waveform.gain.min_db > cfg.waveform.gain.max_db:
            raise ValueError(
                "data.augmentation.waveform.gain.min_db must be less than or equal to max_db"
            )
        if cfg.waveform.noise.snr_db_min <= 0:
            raise ValueError(
                "data.augmentation.waveform.noise.snr_db_min must be greater than zero"
            )
        if cfg.waveform.noise.snr_db_min > cfg.waveform.noise.snr_db_max:
            raise ValueError(
                "data.augmentation.waveform.noise.snr_db_min must be less than or equal to snr_db_max"
            )
        if not (0.0 <= cfg.waveform.time_shift.max_shift_fraction < 1.0):
            raise ValueError(
                "data.augmentation.waveform.time_shift.max_shift_fraction must be within [0, 1)"
            )
        if cfg.waveform.time_shift.mode not in JsonConfigLoader._TIME_SHIFT_MODES:
            raise ValueError(
                "data.augmentation.waveform.time_shift.mode must be one of "
                f"{sorted(JsonConfigLoader._TIME_SHIFT_MODES)}"
            )
        JsonConfigLoader._validate_mask(
            cfg.fbank.time_mask,
            field_name="data.augmentation.fbank.time_mask",
        )
        JsonConfigLoader._validate_mask(
            cfg.fbank.freq_mask,
            field_name="data.augmentation.fbank.freq_mask",
        )

    @staticmethod
    def _validate_data(cfg: DataConfig) -> None:
        JsonConfigLoader._validate_contiguous_labels(cfg.label_to_index)
        JsonConfigLoader._validate_audio(cfg.audio)
        JsonConfigLoader._validate_preprocessing(
            cfg.preprocessing, cfg.audio.sample_rate
        )
        JsonConfigLoader._validate_augmentation(cfg.augmentation)
        if cfg.batch_size <= 0:
            raise ValueError("data.batch_size must be greater than zero")
        if cfg.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")

    @staticmethod
    def _validate_encoder(cfg: ModelEncoderConfig, data_cfg: DataConfig) -> None:
        if cfg.type != "multiscale_rdt_ast":
            raise ValueError(
                "Only model.encoder.type='multiscale_rdt_ast' is supported"
            )
        if cfg.adaptation.mode == "partial":
            raise ValueError(
                "model.encoder.adaptation.mode='partial' is not supported; "
                "use 'full' or 'frozen'"
            )
        if cfg.adaptation.mode not in {"frozen", "full", "partial"}:
            raise ValueError(
                "model.encoder.adaptation.mode must be one of "
                "['frozen', 'full', 'partial']"
            )
        if cfg.adaptation.num_layers != 0:
            raise ValueError("model.encoder.adaptation.num_layers must be 0")

        architecture = cfg.architecture
        if architecture.hidden_size <= 0:
            raise ValueError(
                "model.encoder.architecture.hidden_size must be greater than zero"
            )
        if architecture.num_attention_heads <= 0:
            raise ValueError(
                "model.encoder.architecture.num_attention_heads must be greater than zero"
            )
        if architecture.hidden_size % architecture.num_attention_heads != 0:
            raise ValueError(
                "model.encoder.architecture.hidden_size must be divisible by "
                "model.encoder.architecture.num_attention_heads"
            )
        if architecture.mlp_ratio <= 0:
            raise ValueError(
                "model.encoder.architecture.mlp_ratio must be greater than zero"
            )
        if not (0.0 <= architecture.hidden_dropout_prob < 1.0):
            raise ValueError(
                "model.encoder.architecture.hidden_dropout_prob must be within [0, 1)"
            )
        if not (0.0 <= architecture.attention_probs_dropout_prob < 1.0):
            raise ValueError(
                "model.encoder.architecture.attention_probs_dropout_prob must be within [0, 1)"
            )
        if architecture.shared_stem_depth <= 0:
            raise ValueError(
                "model.encoder.architecture.shared_stem_depth must be greater than zero"
            )
        if architecture.adapter_depth <= 0:
            raise ValueError(
                "model.encoder.architecture.adapter_depth must be greater than zero"
            )
        if not architecture.patch_branches:
            raise ValueError(
                "model.encoder.architecture.patch_branches must not be empty"
            )
        if not isinstance(architecture.rdt.enabled, bool):
            raise ValueError("model.encoder.architecture.rdt.enabled must be a boolean")
        if architecture.rdt.enabled and architecture.rdt.steps <= 0:
            raise ValueError(
                "model.encoder.architecture.rdt.steps must be greater than zero when rdt.enabled is true"
            )
        if architecture.rdt.top_tokens_per_branch <= 0:
            raise ValueError(
                "model.encoder.architecture.rdt.top_tokens_per_branch must be greater than zero"
            )
        if architecture.rdt.layerscale_init <= 0:
            raise ValueError(
                "model.encoder.architecture.rdt.layerscale_init must be greater than zero"
            )
        if (
            architecture.rdt.evidence_score_source
            not in JsonConfigLoader._EVIDENCE_SCORE_SOURCES
        ):
            raise ValueError(
                "model.encoder.architecture.rdt.evidence_score_source must be one of "
                f"{sorted(JsonConfigLoader._EVIDENCE_SCORE_SOURCES)}"
            )
        if architecture.mil.attention_temperature <= 0:
            raise ValueError(
                "model.encoder.architecture.mil.attention_temperature must be greater than zero"
            )
        if architecture.evidence_pooling.type not in (
            JsonConfigLoader._EVIDENCE_POOLING_TYPES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.type must be one of "
                f"{sorted(JsonConfigLoader._EVIDENCE_POOLING_TYPES)}"
            )
        if architecture.evidence_pooling.temperature <= 0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.temperature must be greater than zero"
            )
        if not (0.0 <= architecture.evidence_pooling.dropout < 1.0):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.dropout must be within [0, 1)"
            )
        if (
            architecture.evidence_pooling.gate_hidden_size is not None
            and architecture.evidence_pooling.gate_hidden_size <= 0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.gate_hidden_size must be null or greater than zero"
            )
        class_gate = architecture.evidence_pooling.class_gate
        if class_gate.mode not in JsonConfigLoader._CLASS_GATE_MODES:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate.mode "
                "must be 'query'"
            )
        if class_gate.scorer not in JsonConfigLoader._CLASS_GATE_SCORERS:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate.scorer "
                "must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_SCORERS)}"
            )
        if class_gate.scorer_hidden_size is not None:
            if isinstance(class_gate.scorer_hidden_size, bool) or not isinstance(
                class_gate.scorer_hidden_size,
                int,
            ):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "scorer_hidden_size must be an integer or null"
                )
            if class_gate.scorer_hidden_size <= 0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "scorer_hidden_size must be greater than zero"
                )
        if isinstance(class_gate.scorer_dropout, bool) or not isinstance(
            class_gate.scorer_dropout,
            int | float,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "scorer_dropout must be numeric"
            )
        if not (0.0 <= float(class_gate.scorer_dropout) < 1.0):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "scorer_dropout must be within [0, 1)"
            )
        evidence_scorer = class_gate.evidence_scorer
        if (
            evidence_scorer.type
            not in JsonConfigLoader._CLASS_GATE_EVIDENCE_SCORER_TYPES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.type must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_EVIDENCE_SCORER_TYPES)}"
            )
        for field_name in (
            "embedding_hidden_size",
            "branch_hidden_size",
            "fusion_hidden_size",
            "num_attention_heads",
            "num_attention_layers",
        ):
            value = getattr(evidence_scorer, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.{field_name} must be an integer"
                )
            if value <= 0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.{field_name} must be greater than zero"
                )
        if (
            evidence_scorer.type == "class_axis_attention"
            and evidence_scorer.fusion_hidden_size % evidence_scorer.num_attention_heads
            != 0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.fusion_hidden_size must be divisible by "
                "num_attention_heads when type='class_axis_attention'"
            )
        if not isinstance(evidence_scorer.use_class_embedding, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.use_class_embedding must be a boolean"
            )
        if not isinstance(evidence_scorer.logit_centering, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.logit_centering must be a boolean"
            )
        score_decomposition = evidence_scorer.score_decomposition
        if not isinstance(score_decomposition.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.enabled must be a boolean"
            )
        if (
            score_decomposition.branch_scale_mode
            not in JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.branch_scale_mode must be "
                "one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES)}"
            )
        if (
            score_decomposition.interaction_scale_mode
            not in JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.interaction_scale_mode must be "
                "one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES)}"
            )
        for prefix, mode, min_value, init_value, max_value in (
            (
                "branch_scale",
                score_decomposition.branch_scale_mode,
                score_decomposition.branch_scale_min,
                score_decomposition.branch_scale_init,
                score_decomposition.branch_scale_max,
            ),
            (
                "interaction_scale",
                score_decomposition.interaction_scale_mode,
                score_decomposition.interaction_scale_min,
                score_decomposition.interaction_scale_init,
                score_decomposition.interaction_scale_max,
            ),
        ):
            for value, suffix in (
                (min_value, "min"),
                (init_value, "init"),
                (max_value, "max"),
            ):
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise TypeError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.score_decomposition.{prefix}_{suffix} "
                        "must be numeric"
                    )
            if mode == "sigmoid_max" and float(max_value) <= 0.0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.score_decomposition.{prefix}_max must be "
                    "greater than zero"
                )
            if mode == "sigmoid_max" and not (
                0.0 <= float(init_value) <= float(max_value)
            ):
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.score_decomposition.{prefix}_init must be "
                    "within [0, max]"
                )
            if mode == "bounded_sigmoid":
                if float(min_value) < 0.0:
                    raise ValueError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.score_decomposition.{prefix}_min must be "
                        "non-negative when mode='bounded_sigmoid'"
                    )
                if float(max_value) <= float(min_value):
                    raise ValueError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.score_decomposition.{prefix}_max must be "
                        "greater than min when mode='bounded_sigmoid'"
                    )
                if not (float(min_value) <= float(init_value) <= float(max_value)):
                    raise ValueError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.score_decomposition.{prefix}_init must be "
                        "within [min, max] when mode='bounded_sigmoid'"
                    )
        interaction_schedule = score_decomposition.interaction_scale_schedule
        if not isinstance(interaction_schedule.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.interaction_scale_schedule."
                "enabled must be a boolean"
            )
        for field_name in ("start_epoch", "end_epoch"):
            value = getattr(interaction_schedule, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.score_decomposition.interaction_scale_schedule."
                    f"{field_name} must be an integer"
                )
        if int(interaction_schedule.start_epoch) <= 0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.interaction_scale_schedule."
                "start_epoch must be greater than zero"
            )
        if int(interaction_schedule.end_epoch) < int(interaction_schedule.start_epoch):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.interaction_scale_schedule."
                "end_epoch must be greater than or equal to start_epoch"
            )
        for field_name in ("start_multiplier", "end_multiplier"):
            value = getattr(interaction_schedule, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.score_decomposition.interaction_scale_schedule."
                    f"{field_name} must be numeric"
                )
            if float(value) < 0.0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.score_decomposition.interaction_scale_schedule."
                    f"{field_name} must be non-negative"
                )
        score_bounding = score_decomposition.score_bounding
        if not isinstance(score_bounding.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.score_bounding.enabled "
                "must be a boolean"
            )
        JsonConfigLoader._validate_score_bound_component(
            score_bounding.embedding,
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.score_bounding.embedding"
            ),
        )
        JsonConfigLoader._validate_score_bound_component(
            score_bounding.interaction,
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.score_bounding.interaction"
            ),
        )
        if score_bounding.enabled and not score_decomposition.enabled:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition.score_bounding requires "
                "score_decomposition.enabled=true"
            )
        if (
            score_decomposition.enabled
            and evidence_scorer.type != "class_axis_attention"
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.score_decomposition requires "
                "evidence_scorer.type='class_axis_attention'"
            )
        direct_score = evidence_scorer.branch_direct_score
        if not isinstance(direct_score.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.enabled must be a boolean"
            )
        if (
            direct_score.positive_weight_mode
            not in JsonConfigLoader._CLASS_GATE_BRANCH_DIRECT_SCORE_POSITIVE_WEIGHT_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.positive_weight_mode must "
                "be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_BRANCH_DIRECT_SCORE_POSITIVE_WEIGHT_MODES)}"
            )
        if (
            direct_score.top_support_mode
            not in JsonConfigLoader._CLASS_GATE_BRANCH_DIRECT_SCORE_TOP_SUPPORT_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_mode must be "
                "one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_BRANCH_DIRECT_SCORE_TOP_SUPPORT_MODES)}"
            )
        top_relative_correction = direct_score.top_relative_correction
        if not isinstance(top_relative_correction.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_relative_correction."
                "enabled must be a boolean"
            )
        for field_name in ("positive_scale", "negative_scale"):
            value = getattr(top_relative_correction, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score.top_relative_correction."
                    f"{field_name} must be numeric"
                )
            if float(value) < 0.0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score.top_relative_correction."
                    f"{field_name} must be greater than or equal to zero"
                )
        if (
            not isinstance(top_relative_correction.negative_clip, int | float)
            or isinstance(top_relative_correction.negative_clip, bool)
            or float(top_relative_correction.negative_clip) <= 0.0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_relative_correction."
                "negative_clip must be greater than zero"
            )
        top_support_direct_path = direct_score.top_support_direct_path
        if not isinstance(top_support_direct_path.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "enabled must be a boolean"
            )
        if (
            top_support_direct_path.mode
            not in JsonConfigLoader._CLASS_GATE_TOP_SUPPORT_DIRECT_PATH_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_TOP_SUPPORT_DIRECT_PATH_MODES)}"
            )
        if (
            top_support_direct_path.positive_transform
            not in JsonConfigLoader._CLASS_GATE_TOP_SUPPORT_DIRECT_PATH_POSITIVE_TRANSFORMS
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "positive_transform must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_TOP_SUPPORT_DIRECT_PATH_POSITIVE_TRANSFORMS)}"
            )
        for prefix, mode, min_value, init_value, max_value in (
            (
                "raw_scale",
                top_support_direct_path.raw_scale_mode,
                top_support_direct_path.raw_scale_min,
                top_support_direct_path.raw_scale_init,
                top_support_direct_path.raw_scale_max,
            ),
            (
                "relative_positive_scale",
                top_support_direct_path.relative_positive_scale_mode,
                top_support_direct_path.relative_positive_scale_min,
                top_support_direct_path.relative_positive_scale_init,
                top_support_direct_path.relative_positive_scale_max,
            ),
            (
                "relative_negative_scale",
                top_support_direct_path.relative_negative_scale_mode,
                top_support_direct_path.relative_negative_scale_min,
                top_support_direct_path.relative_negative_scale_init,
                top_support_direct_path.relative_negative_scale_max,
            ),
            (
                "residual_scale",
                top_support_direct_path.residual_scale_mode,
                top_support_direct_path.residual_scale_min,
                top_support_direct_path.residual_scale_init,
                top_support_direct_path.residual_scale_max,
            ),
        ):
            JsonConfigLoader._validate_scale_bounds(
                mode=mode,
                min_value=min_value,
                init_value=init_value,
                max_value=max_value,
                field_name=(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score.top_support_direct_path."
                    f"{prefix}"
                ),
            )
        if (
            not isinstance(top_support_direct_path.residual_hidden_size, int)
            or isinstance(top_support_direct_path.residual_hidden_size, bool)
            or int(top_support_direct_path.residual_hidden_size) <= 0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "residual_hidden_size must be a positive integer"
            )
        for field_name in ("residual_bound", "residual_temperature"):
            value = getattr(top_support_direct_path, field_name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or float(value) <= 0.0
            ):
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score.top_support_direct_path."
                    f"{field_name} must be greater than zero"
                )
        negative_cap_cfg = top_support_direct_path.negative_relative_cap
        if not isinstance(negative_cap_cfg.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "negative_relative_cap.enabled must be a boolean"
            )
        if negative_cap_cfg.enabled and not top_support_direct_path.enabled:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "negative_relative_cap requires top_support_direct_path.enabled=true"
            )
        if (
            negative_cap_cfg.mode
            not in JsonConfigLoader._CLASS_GATE_TOP_SUPPORT_NEGATIVE_RELATIVE_CAP_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "negative_relative_cap.mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_TOP_SUPPORT_NEGATIVE_RELATIVE_CAP_MODES)}"
            )
        for value, field_name in (
            (
                negative_cap_cfg.max_negative_fraction,
                "max_negative_fraction",
            ),
            (negative_cap_cfg.negative_cap, "negative_cap"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score.top_support_direct_path."
                    f"negative_relative_cap.{field_name} must be numeric"
                )
            if float(value) < 0.0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score.top_support_direct_path."
                    f"negative_relative_cap.{field_name} must be greater than "
                    "or equal to zero"
                )
        JsonConfigLoader._validate_numeric_label_mapping(
            negative_cap_cfg.max_negative_fraction_by_label,
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "negative_relative_cap.max_negative_fraction_by_label"
            ),
            label_to_index=data_cfg.label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            negative_cap_cfg.negative_cap_by_label,
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.top_support_direct_path."
                "negative_relative_cap.negative_cap_by_label"
            ),
            label_to_index=data_cfg.label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        for prefix, mode, min_value, init_value, max_value in (
            (
                "top_scale",
                direct_score.top_scale_mode,
                direct_score.top_scale_min,
                direct_score.top_scale_init,
                direct_score.top_scale_max,
            ),
            (
                "gated_scale",
                direct_score.gated_scale_mode,
                direct_score.gated_scale_min,
                direct_score.gated_scale_init,
                direct_score.gated_scale_max,
            ),
            (
                "residual_scale",
                direct_score.residual_scale_mode,
                direct_score.residual_scale_min,
                direct_score.residual_scale_init,
                direct_score.residual_scale_max,
            ),
        ):
            if mode not in JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.branch_direct_score.{prefix}_mode must be "
                    "one of "
                    f"{sorted(JsonConfigLoader._CLASS_GATE_SCORE_DECOMPOSITION_SCALE_MODES)}"
                )
            for value, suffix in (
                (min_value, "min"),
                (init_value, "init"),
                (max_value, "max"),
            ):
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise TypeError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.branch_direct_score.{prefix}_{suffix} "
                        "must be numeric"
                    )
            if mode == "sigmoid_max" and float(max_value) <= 0.0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.branch_direct_score.{prefix}_max must be "
                    "greater than zero"
                )
            if mode == "sigmoid_max" and not (
                0.0 <= float(init_value) <= float(max_value)
            ):
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.branch_direct_score.{prefix}_init must be "
                    "within [0, max]"
                )
            if mode == "bounded_sigmoid":
                if float(min_value) < 0.0:
                    raise ValueError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.branch_direct_score.{prefix}_min must "
                        "be non-negative when mode='bounded_sigmoid'"
                    )
                if float(max_value) <= float(min_value):
                    raise ValueError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.branch_direct_score.{prefix}_max must "
                        "be greater than min when mode='bounded_sigmoid'"
                    )
                if not (float(min_value) <= float(init_value) <= float(max_value)):
                    raise ValueError(
                        "model.encoder.architecture.evidence_pooling.class_gate."
                        f"evidence_scorer.branch_direct_score.{prefix}_init must "
                        "be within [min, max] when mode='bounded_sigmoid'"
                    )
        if not isinstance(direct_score.residual_hidden_size, int) or isinstance(
            direct_score.residual_hidden_size,
            bool,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.residual_hidden_size must "
                "be an integer"
            )
        if int(direct_score.residual_hidden_size) <= 0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.residual_hidden_size must "
                "be greater than zero"
            )
        for field_name in ("residual_bound", "residual_temperature"):
            value = getattr(direct_score, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.branch_direct_score.{field_name} must be "
                    "numeric"
                )
            if float(value) <= 0.0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"evidence_scorer.branch_direct_score.{field_name} must be "
                    "greater than zero"
                )
        reliability = direct_score.gate_reliability_mixture
        if not isinstance(reliability.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.gate_reliability_mixture."
                "enabled must be a boolean"
            )
        if reliability.source != "top_vs_gated_margin_regret":
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.gate_reliability_mixture."
                "source must be 'top_vs_gated_margin_regret'"
            )
        if reliability.mode != "exp_neg_regret":
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.gate_reliability_mixture."
                "mode must be 'exp_neg_regret'"
            )
        if (
            not isinstance(reliability.temperature, int | float)
            or isinstance(reliability.temperature, bool)
            or float(reliability.temperature) <= 0.0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.gate_reliability_mixture."
                "temperature must be greater than zero"
            )
        if (
            not isinstance(reliability.tolerance, int | float)
            or isinstance(reliability.tolerance, bool)
            or float(reliability.tolerance) < 0.0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.gate_reliability_mixture."
                "tolerance must be non-negative"
            )
        if not isinstance(reliability.detach, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_direct_score.gate_reliability_mixture."
                "detach must be a boolean"
            )
        if direct_score.enabled:
            if evidence_scorer.type != "class_axis_attention":
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score requires "
                    "evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition.enabled:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    "evidence_scorer.branch_direct_score requires "
                    "evidence_scorer.score_decomposition.enabled=true"
                )
        if isinstance(evidence_scorer.dropout, bool) or not isinstance(
            evidence_scorer.dropout,
            int | float,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.dropout must be numeric"
            )
        if not (0.0 <= float(evidence_scorer.dropout) < 1.0):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.dropout must be within [0, 1)"
            )
        transform = evidence_scorer.branch_feature_transform
        if (
            transform.mode
            not in JsonConfigLoader._CLASS_GATE_BRANCH_FEATURE_TRANSFORM_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_feature_transform.mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_BRANCH_FEATURE_TRANSFORM_MODES)}"
            )
        if isinstance(transform.temperature, bool) or not isinstance(
            transform.temperature,
            int | float,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_feature_transform.temperature must be numeric"
            )
        if float(transform.temperature) <= 0.0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_scorer.branch_feature_transform.temperature must be "
                "greater than zero"
            )
        if (
            class_gate.branch_logit_feature.mode
            not in JsonConfigLoader._CLASS_GATE_BRANCH_LOGIT_FEATURE_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "branch_logit_feature.mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_GATE_BRANCH_LOGIT_FEATURE_MODES)}"
            )
        JsonConfigLoader._validate_hold_decay_schedule(
            enabled=class_gate.gate_mixing.enabled,
            hold_epochs=class_gate.gate_mixing.hold_epochs,
            decay_epochs=class_gate.gate_mixing.decay_epochs,
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate.gate_mixing"
            ),
        )
        if class_gate.gate_mixing.mode not in JsonConfigLoader._CLASS_GATE_MIXING_MODES:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "gate_mixing.mode must be 'uniform_to_learned'"
            )
        JsonConfigLoader._validate_probability(
            float(class_gate.gate_mixing.start_alpha),
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "gate_mixing.start_alpha"
            ),
        )
        JsonConfigLoader._validate_probability(
            float(class_gate.gate_mixing.end_alpha),
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "gate_mixing.end_alpha"
            ),
        )
        if not isinstance(class_gate.global_residual.enabled, bool):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.enabled must be a boolean"
            )
        if class_gate.global_residual.init_scale < 0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.init_scale must be greater than or equal to zero"
            )
        if not isinstance(class_gate.global_residual.learnable, bool):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.learnable must be a boolean"
            )
        warmup = class_gate.global_residual.warmup
        JsonConfigLoader._validate_hold_decay_schedule(
            enabled=warmup.enabled,
            hold_epochs=warmup.hold_epochs,
            decay_epochs=warmup.decay_epochs,
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.warmup"
            ),
        )
        if warmup.mode not in JsonConfigLoader._GLOBAL_RESIDUAL_WARMUP_MODES:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.warmup.mode must be 'zero_to_learned'"
            )
        JsonConfigLoader._validate_probability(
            float(warmup.start_multiplier),
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.warmup.start_multiplier"
            ),
        )
        JsonConfigLoader._validate_probability(
            float(warmup.end_multiplier),
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.warmup.end_multiplier"
            ),
        )
        bounding = class_gate.global_residual.bounding
        if not isinstance(bounding.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.bounding.enabled must be a boolean"
            )
        for value, field_name in (
            (bounding.bound, "bound"),
            (bounding.temperature, "temperature"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"global_residual.bounding.{field_name} must be numeric"
                )
            if value <= 0:
                raise ValueError(
                    "model.encoder.architecture.evidence_pooling.class_gate."
                    f"global_residual.bounding.{field_name} must be greater "
                    "than zero"
                )
        correction = class_gate.global_residual.correction
        if correction.mode not in JsonConfigLoader._GLOBAL_RESIDUAL_CORRECTION_MODES:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.mode must be one of "
                f"{sorted(JsonConfigLoader._GLOBAL_RESIDUAL_CORRECTION_MODES)}"
            )
        if isinstance(correction.gate_hidden_size, bool) or not isinstance(
            correction.gate_hidden_size,
            int,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.gate_hidden_size must be an integer"
            )
        if correction.gate_hidden_size <= 0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.gate_hidden_size must be greater "
                "than zero"
            )
        if not isinstance(correction.dropout, int | float) or isinstance(
            correction.dropout,
            bool,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.dropout must be numeric"
            )
        if not (0.0 <= float(correction.dropout) < 1.0):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.dropout must be within [0, 1)"
            )
        if not isinstance(correction.zero_mean, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.zero_mean must be a boolean"
            )
        if not isinstance(correction.rebound, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.rebound must be a boolean"
            )
        confidence_gate = correction.confidence_aware_gate
        if not isinstance(confidence_gate.enabled, bool):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.enabled "
                "must be a boolean"
            )
        if (
            confidence_gate.source
            not in JsonConfigLoader._GLOBAL_RESIDUAL_CONFIDENCE_GATE_SOURCES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.source must be "
                "'evidence_gap'"
            )
        if (
            confidence_gate.mode
            not in JsonConfigLoader._GLOBAL_RESIDUAL_CONFIDENCE_GATE_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.mode must be "
                "'damped_sigmoid'"
            )
        if not isinstance(confidence_gate.temperature, int | float) or isinstance(
            confidence_gate.temperature,
            bool,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.temperature "
                "must be numeric"
            )
        if float(confidence_gate.temperature) <= 0.0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.temperature "
                "must be greater than zero"
            )
        if not isinstance(confidence_gate.damping, int | float) or isinstance(
            confidence_gate.damping,
            bool,
        ):
            raise TypeError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.damping "
                "must be numeric"
            )
        JsonConfigLoader._validate_probability(
            float(confidence_gate.damping),
            field_name=(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate.damping"
            ),
        )
        if confidence_gate.enabled and correction.mode != "gated_zero_mean":
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "global_residual.correction.confidence_aware_gate requires "
                "global_residual.correction.mode='gated_zero_mean'"
            )
        if not isinstance(class_gate.evidence_auxiliary.enabled, bool):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_auxiliary.enabled must be a boolean"
            )
        if (
            class_gate.evidence_auxiliary.enabled
            and class_gate.evidence_auxiliary.weight <= 0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.class_gate."
                "evidence_auxiliary.weight must be greater than zero when enabled"
            )
        token_augmentation = architecture.token_augmentation
        if not isinstance(token_augmentation.branch_event_dropout.enabled, bool):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.enabled must be a boolean"
            )
        if not (0.0 <= token_augmentation.branch_event_dropout.probability < 1.0):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.probability must be within [0, 1)"
            )
        if (
            token_augmentation.branch_event_dropout.mode
            not in JsonConfigLoader._BRANCH_EVENT_DROPOUT_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.mode must be 'zero_mask'"
            )
        if token_augmentation.branch_event_dropout.min_keep_tokens < 1:
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.min_keep_tokens must be at least 1"
            )
        if not isinstance(token_augmentation.selected_evidence_dropout.enabled, bool):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.enabled must be a boolean"
            )
        if not (0.0 <= token_augmentation.selected_evidence_dropout.probability < 1.0):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.probability must be within [0, 1)"
            )
        if (
            token_augmentation.selected_evidence_dropout.mode
            not in JsonConfigLoader._SELECTED_EVIDENCE_DROPOUT_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.mode must be 'zero'"
            )
        if token_augmentation.selected_evidence_dropout.min_keep_per_branch < 1:
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.min_keep_per_branch must be at least 1"
            )

        feature_dims = AstFeatureDims(
            num_mel_bins=data_cfg.preprocessing.ast_fbank.num_mel_bins,
            max_length=data_cfg.preprocessing.ast_fbank.max_length,
        )
        branch_count = len(architecture.patch_branches)
        excluded_branches = architecture.rdt.exclude_branches_from_evidence
        if len(set(excluded_branches)) != len(excluded_branches):
            raise ValueError(
                "model.encoder.architecture.rdt.exclude_branches_from_evidence "
                "must not contain duplicate branch indices"
            )
        for excluded_branch in excluded_branches:
            if excluded_branch < 0 or excluded_branch >= branch_count:
                raise ValueError(
                    "model.encoder.architecture.rdt.exclude_branches_from_evidence "
                    f"contains invalid branch index {excluded_branch}"
                )
        if len(excluded_branches) >= branch_count:
            raise ValueError(
                "model.encoder.architecture.rdt.exclude_branches_from_evidence "
                "must leave at least one evidence branch"
            )
        for branch_index, branch in enumerate(architecture.patch_branches):
            patch_t, patch_f = branch.patch_size
            stride_t, stride_f = branch.stride
            if patch_t <= 0 or patch_f <= 0:
                raise ValueError(
                    "model.encoder.architecture.patch_branches.patch_size values "
                    "must be greater than zero"
                )
            if stride_t <= 0 or stride_f <= 0:
                raise ValueError(
                    "model.encoder.architecture.patch_branches.stride values must "
                    "be greater than zero"
                )
            if patch_t > feature_dims.max_length or patch_f > feature_dims.num_mel_bins:
                raise ValueError(
                    "model.encoder.architecture.patch_branches contains a patch "
                    "that does not fit within data.preprocessing.ast_fbank"
                )
            token_count = compute_token_count(
                feature_dims=feature_dims,
                patch_branch=branch,
            )
            if token_count <= 0:
                raise ValueError(
                    "model.encoder.architecture.patch_branches must yield a positive "
                    f"token count; branch_index={branch_index}"
                )
            time_steps, _ = compute_token_grid(
                feature_dims=feature_dims,
                patch_branch=branch,
            )
            if (
                branch_index not in excluded_branches
                and architecture.rdt.top_tokens_per_branch > time_steps
            ):
                raise ValueError(
                    "model.encoder.architecture.rdt.top_tokens_per_branch exceeds "
                    f"branch time length for branch_index={branch_index}"
                )

    @staticmethod
    def _validate_classifier(cfg: ClassifierConfig) -> None:
        if cfg.hidden_dim <= 0:
            raise ValueError("model.classifier.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.classifier.dropout must be within [0, 1)")
        if cfg.pooling != "latent_mean":
            raise ValueError(
                "model.classifier.pooling must be 'latent_mean' for this branch"
            )
        if cfg.fusion_projector.type not in {"linear", "mlp"}:
            raise ValueError(
                "model.classifier.fusion_projector.type must be one of "
                "['linear', 'mlp']"
            )
        if isinstance(cfg.fusion_projector.hidden_dim, bool) or not isinstance(
            cfg.fusion_projector.hidden_dim,
            int,
        ):
            raise TypeError(
                "model.classifier.fusion_projector.hidden_dim must be an integer"
            )
        if cfg.fusion_projector.hidden_dim <= 0:
            raise ValueError(
                "model.classifier.fusion_projector.hidden_dim must be greater than zero"
            )
        if not isinstance(cfg.fusion_projector.layer_norm, bool):
            raise TypeError(
                "model.classifier.fusion_projector.layer_norm must be a boolean"
            )

    @staticmethod
    def _validate_model(cfg: ModelConfig, data_cfg: DataConfig) -> None:
        JsonConfigLoader._validate_encoder(cfg.encoder, data_cfg)
        JsonConfigLoader._validate_classifier(cfg.classifier)

    @staticmethod
    def _validate_train(
        cfg: TrainConfig,
        *,
        num_classes: int,
        num_branches: int,
        label_to_index: Mapping[str, int],
        evidence_pooling_type: str,
        evidence_scorer_type: str,
        score_decomposition_enabled: bool,
        branch_direct_score_enabled: bool,
    ) -> None:
        if cfg.epochs <= 0:
            raise ValueError("train.epochs must be greater than zero")
        if cfg.top_k <= 0:
            raise ValueError("train.top_k must be greater than zero")
        if cfg.max_grad_norm <= 0:
            raise ValueError("train.max_grad_norm must be greater than zero")
        if cfg.optimizer.encoder_lr <= 0:
            raise ValueError("train.optimizer.encoder_lr must be greater than zero")
        if cfg.optimizer.head_lr <= 0:
            raise ValueError("train.optimizer.head_lr must be greater than zero")
        if cfg.optimizer.weight_decay < 0:
            raise ValueError("train.optimizer.weight_decay must be non-negative")
        if not (0.0 <= cfg.scheduler.warmup_ratio <= 1.0):
            raise ValueError("train.scheduler.warmup_ratio must be within [0, 1]")
        JsonConfigLoader._validate_sampler(cfg.sampler)
        if cfg.early_stopping.monitor not in JsonConfigLoader._EARLY_STOPPING_MONITORS:
            raise ValueError(
                "train.early_stopping.monitor must be one of "
                f"{sorted(JsonConfigLoader._EARLY_STOPPING_MONITORS)}"
            )
        if cfg.early_stopping.patience <= 0:
            raise ValueError("train.early_stopping.patience must be greater than zero")
        if cfg.early_stopping.min_delta < 0:
            raise ValueError("train.early_stopping.min_delta must be non-negative")
        if cfg.initialization.checkpoint_path is not None and not isinstance(
            cfg.initialization.checkpoint_path, str
        ):
            raise TypeError(
                "train.initialization.checkpoint_path must be a string or null"
            )
        if not isinstance(cfg.initialization.load_model_state, bool):
            raise TypeError("train.initialization.load_model_state must be a boolean")
        if not isinstance(cfg.initialization.strict, bool):
            raise TypeError("train.initialization.strict must be a boolean")
        if not isinstance(cfg.initialization.load_optimizer_state, bool):
            raise TypeError(
                "train.initialization.load_optimizer_state must be a boolean"
            )
        if not isinstance(cfg.initialization.skip_mismatched_shapes, bool):
            raise TypeError(
                "train.initialization.skip_mismatched_shapes must be a boolean"
            )
        if (
            cfg.initialization.skip_mismatched_shapes
            and cfg.initialization.load_optimizer_state
        ):
            raise ValueError(
                "train.initialization.load_optimizer_state must be false when "
                "skip_mismatched_shapes is true"
            )
        if not isinstance(cfg.loss.branch_auxiliary.enabled, bool):
            raise ValueError("train.loss.branch_auxiliary.enabled must be a boolean")
        if cfg.loss.branch_auxiliary.aggregation != "mean":
            raise ValueError("train.loss.branch_auxiliary.aggregation must be 'mean'")
        if cfg.loss.branch_auxiliary.weights is not None:
            if len(cfg.loss.branch_auxiliary.weights) != num_branches:
                raise ValueError(
                    "train.loss.branch_auxiliary.weights length must match the number "
                    "of model.encoder.architecture.patch_branches"
                )
            if any(weight <= 0 for weight in cfg.loss.branch_auxiliary.weights):
                raise ValueError(
                    "train.loss.branch_auxiliary.weights values must be greater than zero"
                )
        if (
            cfg.loss.branch_auxiliary.enabled
            and cfg.loss.branch_auxiliary.weights is None
            and cfg.loss.branch_auxiliary.weight <= 0
        ):
            raise ValueError(
                "train.loss.branch_auxiliary.weight must be greater than zero when branch auxiliary loss is enabled"
            )
        if not isinstance(cfg.loss.attention_entropy.enabled, bool):
            raise ValueError("train.loss.attention_entropy.enabled must be a boolean")
        if (
            cfg.loss.attention_entropy.enabled
            and cfg.loss.attention_entropy.weight <= 0
        ):
            raise ValueError(
                "train.loss.attention_entropy.weight must be greater than zero when enabled"
            )
        if not isinstance(cfg.loss.gate_entropy_regularization.enabled, bool):
            raise ValueError(
                "train.loss.gate_entropy_regularization.enabled must be a boolean"
            )
        if (
            cfg.loss.gate_entropy_regularization.target
            not in JsonConfigLoader._GATE_ENTROPY_TARGETS
        ):
            raise ValueError(
                "train.loss.gate_entropy_regularization.target must be one of "
                f"{sorted(JsonConfigLoader._GATE_ENTROPY_TARGETS)}"
            )
        JsonConfigLoader._validate_epoch_window(
            start_epoch=cfg.loss.gate_entropy_regularization.start_epoch,
            end_epoch=cfg.loss.gate_entropy_regularization.end_epoch,
            field_name="train.loss.gate_entropy_regularization",
        )
        if (
            cfg.loss.gate_entropy_regularization.enabled
            and cfg.loss.gate_entropy_regularization.weight <= 0
        ):
            raise ValueError(
                "train.loss.gate_entropy_regularization.weight must be greater than zero when enabled"
            )
        if not isinstance(cfg.loss.class_gate_diversity_regularization.enabled, bool):
            raise ValueError(
                "train.loss.class_gate_diversity_regularization.enabled must be a boolean"
            )
        if (
            cfg.loss.class_gate_diversity_regularization.target
            not in JsonConfigLoader._CLASS_GATE_DIVERSITY_TARGETS
        ):
            raise ValueError(
                "train.loss.class_gate_diversity_regularization.target must be "
                f"one of {sorted(JsonConfigLoader._CLASS_GATE_DIVERSITY_TARGETS)}"
            )
        if (
            cfg.loss.class_gate_diversity_regularization.metric
            not in JsonConfigLoader._CLASS_GATE_DIVERSITY_METRICS
        ):
            raise ValueError(
                "train.loss.class_gate_diversity_regularization.metric must be "
                "'js_divergence'"
            )
        JsonConfigLoader._validate_epoch_window(
            start_epoch=cfg.loss.class_gate_diversity_regularization.start_epoch,
            end_epoch=cfg.loss.class_gate_diversity_regularization.end_epoch,
            field_name="train.loss.class_gate_diversity_regularization",
        )
        if (
            cfg.loss.class_gate_diversity_regularization.enabled
            and cfg.loss.class_gate_diversity_regularization.weight <= 0
        ):
            raise ValueError(
                "train.loss.class_gate_diversity_regularization.weight must be "
                "greater than zero when enabled"
            )
        if not isinstance(cfg.loss.class_evidence_margin.enabled, bool):
            raise ValueError(
                "train.loss.class_evidence_margin.enabled must be a boolean"
            )
        if not isinstance(cfg.loss.class_evidence_margin.class_weighted, bool):
            raise ValueError(
                "train.loss.class_evidence_margin.class_weighted must be a boolean"
            )
        if (
            cfg.loss.class_evidence_margin.target
            not in JsonConfigLoader._CLASS_EVIDENCE_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.target must be "
                "'class_evidence_logits'"
            )
        if (
            cfg.loss.class_evidence_margin.mode
            not in JsonConfigLoader._CLASS_EVIDENCE_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.mode must be one of "
                f"{sorted(JsonConfigLoader._CLASS_EVIDENCE_MARGIN_MODES)}"
            )
        if (
            cfg.loss.class_evidence_margin.reduction
            not in JsonConfigLoader._MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.reduction must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        JsonConfigLoader._validate_margin_support_weighting(
            cfg.loss.class_evidence_margin.support_weighting,
            field_name="train.loss.class_evidence_margin.support_weighting",
        )
        if cfg.loss.class_evidence_margin.support_weighting.enabled and (
            cfg.loss.class_evidence_margin.support_weighting.source
            != "top_branch_margin"
            or cfg.loss.class_evidence_margin.support_weighting.mode != "linear"
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.support_weighting supports only "
                "source='top_branch_margin' and mode='linear'"
            )
        if (
            not isinstance(cfg.loss.class_evidence_margin.temperature, (int, float))
            or isinstance(cfg.loss.class_evidence_margin.temperature, bool)
            or cfg.loss.class_evidence_margin.temperature <= 0
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.temperature must be greater than zero"
            )
        if cfg.loss.class_evidence_margin.major_class is not None and not isinstance(
            cfg.loss.class_evidence_margin.major_class, str
        ):
            raise TypeError(
                "train.loss.class_evidence_margin.major_class must be a string or null"
            )
        if (
            cfg.loss.class_evidence_margin.major_class is not None
            and cfg.loss.class_evidence_margin.major_class not in label_to_index
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.major_class must exist in "
                "data.label_to_index"
            )
        if cfg.loss.class_evidence_margin.enabled:
            if cfg.loss.class_evidence_margin.weight <= 0:
                raise ValueError(
                    "train.loss.class_evidence_margin.weight must be greater than "
                    "zero when enabled"
                )
            if (
                cfg.loss.class_evidence_margin.mode
                == "softplus_true_vs_hardest_negative"
            ):
                if cfg.loss.class_evidence_margin.margin < 0:
                    raise ValueError(
                        "train.loss.class_evidence_margin.margin must be "
                        "non-negative when mode="
                        "'softplus_true_vs_hardest_negative'"
                    )
                if cfg.loss.class_evidence_margin.reduction != "mean":
                    raise ValueError(
                        "train.loss.class_evidence_margin.reduction must be 'mean' "
                        "when mode='softplus_true_vs_hardest_negative'"
                    )
            elif cfg.loss.class_evidence_margin.margin <= 0:
                raise ValueError(
                    "train.loss.class_evidence_margin.margin must be greater than "
                    "zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.class_evidence_margin is supported only for "
                    "cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.class_evidence_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if (
                cfg.loss.class_evidence_margin.mode == "minority_vs_major"
                and cfg.loss.class_evidence_margin.major_class is None
            ):
                raise ValueError(
                    "train.loss.class_evidence_margin.major_class is required "
                    "when mode='minority_vs_major'"
                )
        gap_cap_cfg = cfg.loss.class_evidence_gap_cap_regularization
        if not isinstance(gap_cap_cfg.enabled, bool):
            raise TypeError(
                "train.loss.class_evidence_gap_cap_regularization.enabled "
                "must be a boolean"
            )
        if not isinstance(gap_cap_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.class_evidence_gap_cap_regularization.class_weighted "
                "must be a boolean"
            )
        if (
            gap_cap_cfg.target
            not in JsonConfigLoader._CLASS_EVIDENCE_GAP_CAP_REGULARIZATION_TARGETS
        ):
            raise ValueError(
                "train.loss.class_evidence_gap_cap_regularization.target must be "
                "'class_evidence_logits'"
            )
        if (
            gap_cap_cfg.mode
            not in JsonConfigLoader._CLASS_EVIDENCE_GAP_CAP_REGULARIZATION_MODES
        ):
            raise ValueError(
                "train.loss.class_evidence_gap_cap_regularization.mode must be "
                "'negative_gap_hinge'"
            )
        if (
            gap_cap_cfg.reduction
            not in JsonConfigLoader._CLASS_EVIDENCE_GAP_CAP_REGULARIZATION_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.class_evidence_gap_cap_regularization.reduction "
                "must be 'mean'"
            )
        if (
            not isinstance(gap_cap_cfg.negative_gap_cap, int | float)
            or isinstance(gap_cap_cfg.negative_gap_cap, bool)
            or float(gap_cap_cfg.negative_gap_cap) <= 0.0
        ):
            raise ValueError(
                "train.loss.class_evidence_gap_cap_regularization."
                "negative_gap_cap must be greater than zero"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            gap_cap_cfg.negative_gap_cap_by_label,
            field_name=(
                "train.loss.class_evidence_gap_cap_regularization."
                "negative_gap_cap_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            gap_cap_cfg.label_weight_by_label,
            field_name=(
                "train.loss.class_evidence_gap_cap_regularization.label_weight_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        if not isinstance(gap_cap_cfg.warmup_epochs, int) or isinstance(
            gap_cap_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.class_evidence_gap_cap_regularization.warmup_epochs "
                "must be an integer"
            )
        if int(gap_cap_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.class_evidence_gap_cap_regularization.warmup_epochs "
                "must be non-negative"
            )
        if gap_cap_cfg.enabled:
            if gap_cap_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.class_evidence_gap_cap_regularization.weight "
                    "must be greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.class_evidence_gap_cap_regularization is supported "
                    "only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.class_evidence_gap_cap_regularization requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        positive_gap_cap_cfg = cfg.loss.class_evidence_positive_gap_cap_regularization
        if not isinstance(positive_gap_cap_cfg.enabled, bool):
            raise TypeError(
                "train.loss.class_evidence_positive_gap_cap_regularization.enabled "
                "must be a boolean"
            )
        if not isinstance(positive_gap_cap_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.class_evidence_positive_gap_cap_regularization."
                "class_weighted must be a boolean"
            )
        if (
            positive_gap_cap_cfg.target
            not in JsonConfigLoader._CLASS_EVIDENCE_POSITIVE_GAP_CAP_REGULARIZATION_TARGETS
        ):
            raise ValueError(
                "train.loss.class_evidence_positive_gap_cap_regularization.target "
                "must be 'class_evidence_logits'"
            )
        if (
            positive_gap_cap_cfg.mode
            not in JsonConfigLoader._CLASS_EVIDENCE_POSITIVE_GAP_CAP_REGULARIZATION_MODES
        ):
            raise ValueError(
                "train.loss.class_evidence_positive_gap_cap_regularization.mode "
                "must be 'positive_gap_hinge'"
            )
        if (
            positive_gap_cap_cfg.reduction
            not in JsonConfigLoader._CLASS_EVIDENCE_POSITIVE_GAP_CAP_REGULARIZATION_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.class_evidence_positive_gap_cap_regularization."
                "reduction must be 'mean'"
            )
        if (
            not isinstance(positive_gap_cap_cfg.positive_gap_cap, int | float)
            or isinstance(positive_gap_cap_cfg.positive_gap_cap, bool)
            or float(positive_gap_cap_cfg.positive_gap_cap) <= 0.0
        ):
            raise ValueError(
                "train.loss.class_evidence_positive_gap_cap_regularization."
                "positive_gap_cap must be greater than zero"
            )
        if not isinstance(
            positive_gap_cap_cfg.warmup_epochs,
            int,
        ) or isinstance(positive_gap_cap_cfg.warmup_epochs, bool):
            raise TypeError(
                "train.loss.class_evidence_positive_gap_cap_regularization."
                "warmup_epochs must be an integer"
            )
        if int(positive_gap_cap_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.class_evidence_positive_gap_cap_regularization."
                "warmup_epochs must be non-negative"
            )
        if positive_gap_cap_cfg.enabled:
            if positive_gap_cap_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.class_evidence_positive_gap_cap_regularization."
                    "weight must be greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.class_evidence_positive_gap_cap_regularization "
                    "is supported only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.class_evidence_positive_gap_cap_regularization "
                    "requires model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        interaction_gap_cap_cfg = cfg.loss.interaction_gap_cap_regularization
        if not isinstance(interaction_gap_cap_cfg.enabled, bool):
            raise TypeError(
                "train.loss.interaction_gap_cap_regularization.enabled must be "
                "a boolean"
            )
        if not isinstance(interaction_gap_cap_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.interaction_gap_cap_regularization.class_weighted "
                "must be a boolean"
            )
        if (
            interaction_gap_cap_cfg.target
            not in JsonConfigLoader._INTERACTION_GAP_CAP_REGULARIZATION_TARGETS
        ):
            raise ValueError(
                "train.loss.interaction_gap_cap_regularization.target must be "
                "'class_evidence_interaction_scores'"
            )
        if (
            interaction_gap_cap_cfg.mode
            not in JsonConfigLoader._INTERACTION_GAP_CAP_REGULARIZATION_MODES
        ):
            raise ValueError(
                "train.loss.interaction_gap_cap_regularization.mode must be "
                "'absolute_gap_hinge'"
            )
        if (
            interaction_gap_cap_cfg.reduction
            not in JsonConfigLoader._INTERACTION_GAP_CAP_REGULARIZATION_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.interaction_gap_cap_regularization.reduction must be 'mean'"
            )
        if (
            not isinstance(interaction_gap_cap_cfg.gap_cap, int | float)
            or isinstance(interaction_gap_cap_cfg.gap_cap, bool)
            or float(interaction_gap_cap_cfg.gap_cap) <= 0.0
        ):
            raise ValueError(
                "train.loss.interaction_gap_cap_regularization.gap_cap must be "
                "greater than zero"
            )
        if not isinstance(
            interaction_gap_cap_cfg.warmup_epochs,
            int,
        ) or isinstance(interaction_gap_cap_cfg.warmup_epochs, bool):
            raise TypeError(
                "train.loss.interaction_gap_cap_regularization.warmup_epochs "
                "must be an integer"
            )
        if int(interaction_gap_cap_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.interaction_gap_cap_regularization.warmup_epochs "
                "must be non-negative"
            )
        if interaction_gap_cap_cfg.enabled:
            if interaction_gap_cap_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.interaction_gap_cap_regularization.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.interaction_gap_cap_regularization is supported "
                    "only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.interaction_gap_cap_regularization requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if evidence_scorer_type != "class_axis_attention":
                raise ValueError(
                    "train.loss.interaction_gap_cap_regularization requires "
                    "class_gate.evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.interaction_gap_cap_regularization requires "
                    "class_gate.evidence_scorer.score_decomposition.enabled=true"
                )
        top_support_cfg = cfg.loss.top_support_score_margin
        if not isinstance(top_support_cfg.enabled, bool):
            raise ValueError(
                "train.loss.top_support_score_margin.enabled must be a boolean"
            )
        if not isinstance(top_support_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.top_support_score_margin.class_weighted must be a boolean"
            )
        if (
            top_support_cfg.target
            not in JsonConfigLoader._TOP_SUPPORT_SCORE_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.top_support_score_margin.target must be "
                "'class_evidence_top_support_scores'"
            )
        if top_support_cfg.mode not in JsonConfigLoader._TOP_SUPPORT_SCORE_MARGIN_MODES:
            raise ValueError(
                "train.loss.top_support_score_margin.mode must be "
                "'softplus_true_vs_hardest_negative'"
            )
        if (
            top_support_cfg.reduction
            not in JsonConfigLoader._TOP_SUPPORT_SCORE_MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.top_support_score_margin.reduction must be 'mean'"
            )
        if (
            not isinstance(top_support_cfg.temperature, int | float)
            or isinstance(top_support_cfg.temperature, bool)
            or float(top_support_cfg.temperature) <= 0.0
        ):
            raise ValueError(
                "train.loss.top_support_score_margin.temperature must be "
                "greater than zero"
            )
        if not isinstance(top_support_cfg.warmup_epochs, int) or isinstance(
            top_support_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.top_support_score_margin.warmup_epochs must be an integer"
            )
        if int(top_support_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.top_support_score_margin.warmup_epochs must be non-negative"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            top_support_cfg.label_weight_by_label,
            field_name="train.loss.top_support_score_margin.label_weight_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_margin_support_weighting(
            top_support_cfg.support_conditioned_multiplier,
            field_name=(
                "train.loss.top_support_score_margin.support_conditioned_multiplier"
            ),
        )
        if top_support_cfg.support_conditioned_multiplier.enabled and (
            top_support_cfg.support_conditioned_multiplier.source != "top_branch_margin"
            or top_support_cfg.support_conditioned_multiplier.mode != "linear"
        ):
            raise ValueError(
                "train.loss.top_support_score_margin."
                "support_conditioned_multiplier supports only "
                "source='top_branch_margin' and mode='linear'"
            )
        JsonConfigLoader._validate_margin_hardness_weighting(
            top_support_cfg.hardness_weighting,
            field_name="train.loss.top_support_score_margin.hardness_weighting",
        )
        if top_support_cfg.hardness_weighting.enabled and (
            top_support_cfg.hardness_weighting.source != "top_support_gap"
            or top_support_cfg.hardness_weighting.mode != "negative_gap"
        ):
            raise ValueError(
                "train.loss.top_support_score_margin.hardness_weighting "
                "supports only source='top_support_gap' and mode='negative_gap'"
            )
        if top_support_cfg.enabled:
            if top_support_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.top_support_score_margin.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.top_support_score_margin is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.top_support_score_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if evidence_scorer_type != "class_axis_attention":
                raise ValueError(
                    "train.loss.top_support_score_margin requires "
                    "class_gate.evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.top_support_score_margin requires "
                    "class_gate.evidence_scorer.score_decomposition.enabled=true"
                )
            if not branch_direct_score_enabled:
                raise ValueError(
                    "train.loss.top_support_score_margin requires "
                    "class_gate.evidence_scorer.branch_direct_score.enabled=true"
                )
        top_support_min_cfg = cfg.loss.top_support_gap_min_constraint
        if not isinstance(top_support_min_cfg.enabled, bool):
            raise TypeError(
                "train.loss.top_support_gap_min_constraint.enabled must be a boolean"
            )
        if not isinstance(top_support_min_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.top_support_gap_min_constraint.class_weighted "
                "must be a boolean"
            )
        if (
            top_support_min_cfg.target
            not in JsonConfigLoader._TOP_SUPPORT_GAP_MIN_CONSTRAINT_TARGETS
        ):
            raise ValueError(
                "train.loss.top_support_gap_min_constraint.target must be "
                "'class_evidence_top_support_scores'"
            )
        if (
            top_support_min_cfg.mode
            not in JsonConfigLoader._TOP_SUPPORT_GAP_MIN_CONSTRAINT_MODES
        ):
            raise ValueError(
                "train.loss.top_support_gap_min_constraint.mode must be "
                "'support_conditioned_min_gap'"
            )
        if (
            top_support_min_cfg.support_source
            not in JsonConfigLoader._TOP_SUPPORT_GAP_MIN_CONSTRAINT_SUPPORT_SOURCES
        ):
            raise ValueError(
                "train.loss.top_support_gap_min_constraint.support_source must be "
                "'top_branch_margin'"
            )
        if (
            top_support_min_cfg.reduction
            not in JsonConfigLoader._TOP_SUPPORT_GAP_MIN_CONSTRAINT_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.top_support_gap_min_constraint.reduction must be 'mean'"
            )
        for field_name, value, min_value, strict_min in (
            (
                "base_min_gap",
                top_support_min_cfg.base_min_gap,
                0.0,
                False,
            ),
            (
                "support_gain",
                top_support_min_cfg.support_gain,
                0.0,
                False,
            ),
            (
                "support_cap",
                top_support_min_cfg.support_cap,
                0.0,
                True,
            ),
        ):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or (
                    float(value) <= min_value
                    if strict_min
                    else float(value) < min_value
                )
            ):
                comparator = (
                    "greater than" if strict_min else "greater than or equal to"
                )
                raise ValueError(
                    f"train.loss.top_support_gap_min_constraint.{field_name} "
                    f"must be {comparator} {min_value}"
                )
        if not isinstance(top_support_min_cfg.warmup_epochs, int) or isinstance(
            top_support_min_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.top_support_gap_min_constraint.warmup_epochs "
                "must be an integer"
            )
        if int(top_support_min_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.top_support_gap_min_constraint.warmup_epochs "
                "must be non-negative"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            top_support_min_cfg.base_min_gap_by_label,
            field_name=(
                "train.loss.top_support_gap_min_constraint.base_min_gap_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        if top_support_min_cfg.enabled:
            if top_support_min_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if evidence_scorer_type != "class_axis_attention":
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint requires "
                    "class_gate.evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint requires "
                    "class_gate.evidence_scorer.score_decomposition.enabled=true"
                )
            if not branch_direct_score_enabled:
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint requires "
                    "class_gate.evidence_scorer.branch_direct_score.enabled=true"
                )
            if (
                top_support_min_cfg.class_weighted
                and not cfg.loss.class_weighting.enabled
            ):
                raise ValueError(
                    "train.loss.top_support_gap_min_constraint.class_weighted "
                    "requires train.loss.class_weighting.enabled=true"
                )
        class_top_branch_relative_cfg = cfg.loss.class_top_branch_relative_margin
        if not isinstance(class_top_branch_relative_cfg.enabled, bool):
            raise TypeError(
                "train.loss.class_top_branch_relative_margin.enabled must be a boolean"
            )
        if not isinstance(class_top_branch_relative_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.class_top_branch_relative_margin.class_weighted "
                "must be a boolean"
            )
        if (
            class_top_branch_relative_cfg.target
            not in JsonConfigLoader._CLASS_TOP_BRANCH_RELATIVE_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.target must be "
                "'class_top_branch_margin_features'"
            )
        if (
            class_top_branch_relative_cfg.mode
            not in JsonConfigLoader._CLASS_TOP_BRANCH_RELATIVE_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.mode must be "
                "'true_vs_hardest_negative_hinge'"
            )
        if (
            class_top_branch_relative_cfg.reduction
            not in JsonConfigLoader._CLASS_TOP_BRANCH_RELATIVE_MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.reduction must be 'mean'"
            )
        if (
            not isinstance(class_top_branch_relative_cfg.margin, int | float)
            or isinstance(class_top_branch_relative_cfg.margin, bool)
            or float(class_top_branch_relative_cfg.margin) <= 0.0
        ):
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.margin must be "
                "greater than zero"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            class_top_branch_relative_cfg.margin_by_label,
            field_name=("train.loss.class_top_branch_relative_margin.margin_by_label"),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
        )
        if not isinstance(
            class_top_branch_relative_cfg.warmup_epochs,
            int,
        ) or isinstance(class_top_branch_relative_cfg.warmup_epochs, bool):
            raise TypeError(
                "train.loss.class_top_branch_relative_margin.warmup_epochs "
                "must be an integer"
            )
        if int(class_top_branch_relative_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.warmup_epochs "
                "must be non-negative"
            )
        JsonConfigLoader._validate_margin_support_weighting(
            class_top_branch_relative_cfg.support_weighting,
            field_name=(
                "train.loss.class_top_branch_relative_margin.support_weighting"
            ),
        )
        if class_top_branch_relative_cfg.support_weighting.enabled and (
            class_top_branch_relative_cfg.support_weighting.source
            != "top_branch_margin"
            or class_top_branch_relative_cfg.support_weighting.mode != "linear"
        ):
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.support_weighting "
                "supports only source='top_branch_margin' and mode='linear'"
            )
        JsonConfigLoader._validate_margin_hardness_weighting(
            class_top_branch_relative_cfg.hardness_weighting,
            field_name=(
                "train.loss.class_top_branch_relative_margin.hardness_weighting"
            ),
        )
        if class_top_branch_relative_cfg.hardness_weighting.enabled and (
            class_top_branch_relative_cfg.hardness_weighting.source
            != "teacher_gap_deficit"
            or class_top_branch_relative_cfg.hardness_weighting.mode != "linear"
        ):
            raise ValueError(
                "train.loss.class_top_branch_relative_margin.hardness_weighting "
                "supports only source='teacher_gap_deficit' and mode='linear'"
            )
        if not isinstance(
            class_top_branch_relative_cfg.hardness_weighting_by_label,
            Mapping,
        ):
            raise TypeError(
                "train.loss.class_top_branch_relative_margin."
                "hardness_weighting_by_label must be an object"
            )
        for (
            label_name,
            hardness_cfg,
        ) in class_top_branch_relative_cfg.hardness_weighting_by_label.items():
            if label_name not in label_to_index:
                raise ValueError(
                    "train.loss.class_top_branch_relative_margin."
                    f"hardness_weighting_by_label contains unknown label {label_name!r}"
                )
            JsonConfigLoader._validate_margin_hardness_weighting(
                hardness_cfg,
                field_name=(
                    "train.loss.class_top_branch_relative_margin."
                    f"hardness_weighting_by_label.{label_name}"
                ),
            )
            if hardness_cfg.source != "teacher_gap_deficit" or (
                hardness_cfg.mode != "linear"
            ):
                raise ValueError(
                    "train.loss.class_top_branch_relative_margin."
                    f"hardness_weighting_by_label.{label_name} supports only "
                    "source='teacher_gap_deficit' and mode='linear'"
                )
        JsonConfigLoader._validate_weak_positive_support_weighting(
            class_top_branch_relative_cfg.weak_positive_support_weighting,
            field_name=(
                "train.loss.class_top_branch_relative_margin."
                "weak_positive_support_weighting"
            ),
        )
        JsonConfigLoader._validate_weak_positive_support_bands(
            class_top_branch_relative_cfg.weak_positive_support_weighting.support_band_by_label,
            field_name=(
                "train.loss.class_top_branch_relative_margin."
                "weak_positive_support_weighting.support_band_by_label"
            ),
            label_to_index=label_to_index,
        )
        JsonConfigLoader._validate_weak_positive_boost(
            class_top_branch_relative_cfg.weak_positive_margin_boost,
            field_name=(
                "train.loss.class_top_branch_relative_margin.weak_positive_margin_boost"
            ),
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            class_top_branch_relative_cfg.weak_positive_margin_boost.boost_by_label,
            field_name=(
                "train.loss.class_top_branch_relative_margin."
                "weak_positive_margin_boost.boost_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_weak_positive_support_bands(
            class_top_branch_relative_cfg.weak_positive_margin_boost.support_band_by_label,
            field_name=(
                "train.loss.class_top_branch_relative_margin."
                "weak_positive_margin_boost.support_band_by_label"
            ),
            label_to_index=label_to_index,
        )
        if class_top_branch_relative_cfg.enabled:
            if class_top_branch_relative_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.class_top_branch_relative_margin.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.class_top_branch_relative_margin is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.class_top_branch_relative_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if (
                class_top_branch_relative_cfg.class_weighted
                and not cfg.loss.class_weighting.enabled
            ):
                raise ValueError(
                    "train.loss.class_top_branch_relative_margin.class_weighted "
                    "requires train.loss.class_weighting.enabled=true"
                )
        top_teacher_min_cfg = cfg.loss.top_teacher_gap_min_constraint
        if not isinstance(top_teacher_min_cfg.enabled, bool):
            raise TypeError(
                "train.loss.top_teacher_gap_min_constraint.enabled must be a boolean"
            )
        if not isinstance(top_teacher_min_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.top_teacher_gap_min_constraint.class_weighted "
                "must be a boolean"
            )
        if (
            top_teacher_min_cfg.target
            not in JsonConfigLoader._TOP_TEACHER_GAP_MIN_CONSTRAINT_TARGETS
        ):
            raise ValueError(
                "train.loss.top_teacher_gap_min_constraint.target must be "
                "'class_top_branch_margin_features'"
            )
        if (
            top_teacher_min_cfg.mode
            not in JsonConfigLoader._TOP_TEACHER_GAP_MIN_CONSTRAINT_MODES
        ):
            raise ValueError(
                "train.loss.top_teacher_gap_min_constraint.mode must be "
                "'support_conditioned_min_gap'"
            )
        if (
            top_teacher_min_cfg.support_source
            not in JsonConfigLoader._TOP_TEACHER_GAP_MIN_CONSTRAINT_SUPPORT_SOURCES
        ):
            raise ValueError(
                "train.loss.top_teacher_gap_min_constraint.support_source must be "
                "'top_branch_margin'"
            )
        if (
            top_teacher_min_cfg.reduction
            not in JsonConfigLoader._TOP_TEACHER_GAP_MIN_CONSTRAINT_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.top_teacher_gap_min_constraint.reduction must be 'mean'"
            )
        for field_name, value, min_value, strict_min in (
            ("base_min_gap", top_teacher_min_cfg.base_min_gap, 0.0, False),
            ("support_gain", top_teacher_min_cfg.support_gain, 0.0, False),
            ("support_cap", top_teacher_min_cfg.support_cap, 0.0, True),
        ):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or (
                    float(value) <= min_value
                    if strict_min
                    else float(value) < min_value
                )
            ):
                comparator = (
                    "greater than" if strict_min else "greater than or equal to"
                )
                raise ValueError(
                    f"train.loss.top_teacher_gap_min_constraint.{field_name} "
                    f"must be {comparator} {min_value}"
                )
        JsonConfigLoader._validate_numeric_label_mapping(
            top_teacher_min_cfg.base_min_gap_by_label,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint.base_min_gap_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            top_teacher_min_cfg.support_gain_by_label,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint.support_gain_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            top_teacher_min_cfg.support_cap_by_label,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint.support_cap_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
        )
        if not isinstance(top_teacher_min_cfg.warmup_epochs, int) or isinstance(
            top_teacher_min_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.top_teacher_gap_min_constraint.warmup_epochs "
                "must be an integer"
            )
        if int(top_teacher_min_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.top_teacher_gap_min_constraint.warmup_epochs "
                "must be non-negative"
            )
        JsonConfigLoader._validate_weak_positive_support_weighting(
            top_teacher_min_cfg.weak_positive_support_weighting,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint."
                "weak_positive_support_weighting"
            ),
        )
        JsonConfigLoader._validate_weak_positive_support_bands(
            top_teacher_min_cfg.weak_positive_support_weighting.support_band_by_label,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint."
                "weak_positive_support_weighting.support_band_by_label"
            ),
            label_to_index=label_to_index,
        )
        JsonConfigLoader._validate_weak_positive_boost(
            top_teacher_min_cfg.weak_positive_target_boost,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint.weak_positive_target_boost"
            ),
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            top_teacher_min_cfg.weak_positive_target_boost.boost_by_label,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint."
                "weak_positive_target_boost.boost_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_weak_positive_support_bands(
            top_teacher_min_cfg.weak_positive_target_boost.support_band_by_label,
            field_name=(
                "train.loss.top_teacher_gap_min_constraint."
                "weak_positive_target_boost.support_band_by_label"
            ),
            label_to_index=label_to_index,
        )
        if top_teacher_min_cfg.enabled:
            if top_teacher_min_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.top_teacher_gap_min_constraint.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.top_teacher_gap_min_constraint is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.top_teacher_gap_min_constraint requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if (
                top_teacher_min_cfg.class_weighted
                and not cfg.loss.class_weighting.enabled
            ):
                raise ValueError(
                    "train.loss.top_teacher_gap_min_constraint.class_weighted "
                    "requires train.loss.class_weighting.enabled=true"
                )
        branch_support_cfg = cfg.loss.branch_support_score_margin
        if not isinstance(branch_support_cfg.enabled, bool):
            raise ValueError(
                "train.loss.branch_support_score_margin.enabled must be a boolean"
            )
        if not isinstance(branch_support_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.branch_support_score_margin.class_weighted must be "
                "a boolean"
            )
        if (
            branch_support_cfg.target
            not in JsonConfigLoader._BRANCH_SUPPORT_SCORE_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.branch_support_score_margin.target must be "
                "'class_evidence_branch_support_scores'"
            )
        if (
            branch_support_cfg.mode
            not in JsonConfigLoader._BRANCH_SUPPORT_SCORE_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.branch_support_score_margin.mode must be "
                "'softplus_true_vs_hardest_negative'"
            )
        if (
            branch_support_cfg.reduction
            not in JsonConfigLoader._BRANCH_SUPPORT_SCORE_MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.branch_support_score_margin.reduction must be 'mean'"
            )
        if (
            not isinstance(branch_support_cfg.temperature, int | float)
            or isinstance(branch_support_cfg.temperature, bool)
            or float(branch_support_cfg.temperature) <= 0.0
        ):
            raise ValueError(
                "train.loss.branch_support_score_margin.temperature must be "
                "greater than zero"
            )
        if not isinstance(branch_support_cfg.warmup_epochs, int) or isinstance(
            branch_support_cfg.warmup_epochs, bool
        ):
            raise TypeError(
                "train.loss.branch_support_score_margin.warmup_epochs must be "
                "an integer"
            )
        if int(branch_support_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.branch_support_score_margin.warmup_epochs must be "
                "non-negative"
            )
        if branch_support_cfg.enabled:
            if branch_support_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.branch_support_score_margin.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.branch_support_score_margin is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.branch_support_score_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if evidence_scorer_type != "class_axis_attention":
                raise ValueError(
                    "train.loss.branch_support_score_margin requires "
                    "class_gate.evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.branch_support_score_margin requires "
                    "class_gate.evidence_scorer.score_decomposition.enabled=true"
                )
        branch_direct_cfg = cfg.loss.branch_direct_score_margin
        if not isinstance(branch_direct_cfg.enabled, bool):
            raise ValueError(
                "train.loss.branch_direct_score_margin.enabled must be a boolean"
            )
        if not isinstance(branch_direct_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.branch_direct_score_margin.class_weighted must be a boolean"
            )
        if (
            branch_direct_cfg.target
            not in JsonConfigLoader._BRANCH_DIRECT_SCORE_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.branch_direct_score_margin.target must be "
                "'class_evidence_branch_direct_scores'"
            )
        if (
            branch_direct_cfg.mode
            not in JsonConfigLoader._BRANCH_DIRECT_SCORE_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.branch_direct_score_margin.mode must be "
                "'softplus_true_vs_hardest_negative'"
            )
        if (
            branch_direct_cfg.reduction
            not in JsonConfigLoader._BRANCH_DIRECT_SCORE_MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.branch_direct_score_margin.reduction must be 'mean'"
            )
        if (
            not isinstance(branch_direct_cfg.temperature, int | float)
            or isinstance(branch_direct_cfg.temperature, bool)
            or float(branch_direct_cfg.temperature) <= 0.0
        ):
            raise ValueError(
                "train.loss.branch_direct_score_margin.temperature must be "
                "greater than zero"
            )
        if not isinstance(branch_direct_cfg.warmup_epochs, int) or isinstance(
            branch_direct_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_direct_score_margin.warmup_epochs must be an integer"
            )
        if int(branch_direct_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.branch_direct_score_margin.warmup_epochs must be "
                "non-negative"
            )
        if branch_direct_cfg.enabled:
            if branch_direct_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.branch_direct_score_margin.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.branch_direct_score_margin is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.branch_direct_score_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if evidence_scorer_type != "class_axis_attention":
                raise ValueError(
                    "train.loss.branch_direct_score_margin requires "
                    "class_gate.evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.branch_direct_score_margin requires "
                    "class_gate.evidence_scorer.score_decomposition.enabled=true"
                )
            if not branch_direct_score_enabled:
                raise ValueError(
                    "train.loss.branch_direct_score_margin requires "
                    "class_gate.evidence_scorer.branch_direct_score.enabled=true"
                )
        branch_dominance_cfg = cfg.loss.branch_path_dominance_constraint
        if not isinstance(branch_dominance_cfg.enabled, bool):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.enabled must be a boolean"
            )
        if not isinstance(branch_dominance_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.class_weighted "
                "must be a boolean"
            )
        if (
            branch_dominance_cfg.branch_source
            not in JsonConfigLoader._BRANCH_PATH_DOMINANCE_BRANCH_SOURCES
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.branch_source must be "
                "'class_evidence_branch_support_scores'"
            )
        if (
            branch_dominance_cfg.target
            not in JsonConfigLoader._BRANCH_PATH_DOMINANCE_TARGETS
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.target must be "
                "'class_evidence_logits'"
            )
        if (
            branch_dominance_cfg.mode
            not in JsonConfigLoader._BRANCH_PATH_DOMINANCE_MODES
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.mode must be "
                "'branch_gap_preservation'"
            )
        if (
            branch_dominance_cfg.support_source
            not in JsonConfigLoader._BRANCH_PATH_DOMINANCE_SUPPORT_SOURCES
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.support_source must be "
                "'top_branch_margin'"
            )
        if (
            branch_dominance_cfg.reduction
            not in JsonConfigLoader._BRANCH_PATH_DOMINANCE_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.reduction must be 'mean'"
            )
        if (
            not isinstance(branch_dominance_cfg.allowed_drop, int | float)
            or isinstance(branch_dominance_cfg.allowed_drop, bool)
            or float(branch_dominance_cfg.allowed_drop) < 0.0
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.allowed_drop must be "
                "greater than or equal to zero"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            branch_dominance_cfg.allowed_drop_by_label,
            field_name=(
                "train.loss.branch_path_dominance_constraint.allowed_drop_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            branch_dominance_cfg.label_weight_by_label,
            field_name=(
                "train.loss.branch_path_dominance_constraint.label_weight_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        if not isinstance(branch_dominance_cfg.warmup_epochs, int) or isinstance(
            branch_dominance_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_path_dominance_constraint.warmup_epochs "
                "must be an integer"
            )
        if int(branch_dominance_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.warmup_epochs "
                "must be non-negative"
            )
        JsonConfigLoader._validate_margin_support_weighting(
            branch_dominance_cfg.support_weighting,
            field_name=(
                "train.loss.branch_path_dominance_constraint.support_weighting"
            ),
        )
        if branch_dominance_cfg.support_weighting.enabled and (
            branch_dominance_cfg.support_weighting.source != "top_branch_margin"
            or branch_dominance_cfg.support_weighting.mode != "linear"
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.support_weighting "
                "supports only source='top_branch_margin' and mode='linear'"
            )
        if branch_dominance_cfg.enabled:
            if branch_dominance_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.branch_path_dominance_constraint.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.branch_path_dominance_constraint is supported "
                    "only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.branch_path_dominance_constraint requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.branch_path_dominance_constraint requires "
                    "class_gate.evidence_scorer.score_decomposition.enabled=true"
                )
        disagreement_cap_cfg = cfg.loss.branch_support_disagreement_cap_regularization
        if not isinstance(disagreement_cap_cfg.enabled, bool):
            raise TypeError(
                "train.loss.branch_support_disagreement_cap_regularization.enabled "
                "must be a boolean"
            )
        if not isinstance(disagreement_cap_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "class_weighted must be a boolean"
            )
        if (
            disagreement_cap_cfg.branch_source
            not in JsonConfigLoader._BRANCH_SUPPORT_DISAGREEMENT_BRANCH_SOURCES
        ):
            raise ValueError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "branch_source must be 'class_evidence_branch_support_scores'"
            )
        if (
            disagreement_cap_cfg.embedding_source
            not in JsonConfigLoader._BRANCH_SUPPORT_DISAGREEMENT_EMBEDDING_SOURCES
        ):
            raise ValueError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "embedding_source must be 'class_evidence_embedding_scores'"
            )
        if (
            disagreement_cap_cfg.interaction_source
            not in JsonConfigLoader._BRANCH_SUPPORT_DISAGREEMENT_INTERACTION_SOURCES
        ):
            raise ValueError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "interaction_source must be 'class_evidence_interaction_scores'"
            )
        if (
            disagreement_cap_cfg.mode
            not in JsonConfigLoader._BRANCH_SUPPORT_DISAGREEMENT_MODES
        ):
            raise ValueError(
                "train.loss.branch_support_disagreement_cap_regularization.mode "
                "must be 'branch_gap_conditioned_abs_gap_cap'"
            )
        if (
            disagreement_cap_cfg.reduction
            not in JsonConfigLoader._BRANCH_SUPPORT_DISAGREEMENT_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "reduction must be 'mean'"
            )
        for field_name, value, min_value, strict_min in (
            (
                "disagreement_threshold",
                disagreement_cap_cfg.disagreement_threshold,
                0.0,
                False,
            ),
            ("condition_gain", disagreement_cap_cfg.condition_gain, 0.0, False),
            ("condition_cap", disagreement_cap_cfg.condition_cap, 0.0, True),
            ("embedding_gap_cap", disagreement_cap_cfg.embedding_gap_cap, 0.0, True),
            (
                "interaction_gap_cap",
                disagreement_cap_cfg.interaction_gap_cap,
                0.0,
                True,
            ),
        ):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or (
                    float(value) <= min_value
                    if strict_min
                    else float(value) < min_value
                )
            ):
                comparator = (
                    "greater than" if strict_min else "greater than or equal to"
                )
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization."
                    f"{field_name} must be {comparator} {min_value}"
                )
        if not isinstance(disagreement_cap_cfg.warmup_epochs, int) or isinstance(
            disagreement_cap_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "warmup_epochs must be an integer"
            )
        if int(disagreement_cap_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.branch_support_disagreement_cap_regularization."
                "warmup_epochs must be non-negative"
            )
        if disagreement_cap_cfg.enabled:
            if disagreement_cap_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization."
                    "weight must be greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization "
                    "is supported only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization "
                    "requires model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if evidence_scorer_type != "class_axis_attention":
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization "
                    "requires class_gate.evidence_scorer.type='class_axis_attention'"
                )
            if not score_decomposition_enabled:
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization "
                    "requires class_gate.evidence_scorer.score_decomposition."
                    "enabled=true"
                )
            if not branch_direct_score_enabled:
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization "
                    "requires class_gate.evidence_scorer.branch_direct_score."
                    "enabled=true"
                )
            if (
                disagreement_cap_cfg.class_weighted
                and not cfg.loss.class_weighting.enabled
            ):
                raise ValueError(
                    "train.loss.branch_support_disagreement_cap_regularization."
                    "class_weighted requires train.loss.class_weighting.enabled=true"
                )
        if not isinstance(cfg.loss.class_gated_branch_logit_margin.enabled, bool):
            raise ValueError(
                "train.loss.class_gated_branch_logit_margin.enabled must be a boolean"
            )
        if not isinstance(
            cfg.loss.class_gated_branch_logit_margin.class_weighted, bool
        ):
            raise ValueError(
                "train.loss.class_gated_branch_logit_margin.class_weighted "
                "must be a boolean"
            )
        if (
            cfg.loss.class_gated_branch_logit_margin.target
            not in JsonConfigLoader._CLASS_GATED_BRANCH_LOGIT_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.class_gated_branch_logit_margin.target must be "
                "'class_gated_branch_logits'"
            )
        if (
            cfg.loss.class_gated_branch_logit_margin.mode
            not in JsonConfigLoader._CLASS_GATED_BRANCH_LOGIT_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.class_gated_branch_logit_margin.mode must be "
                "'true_vs_hardest_negative'"
            )
        if (
            cfg.loss.class_gated_branch_logit_margin.reduction
            not in JsonConfigLoader._MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.class_gated_branch_logit_margin.reduction must be "
                f"one of {sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        if cfg.loss.class_gated_branch_logit_margin.enabled:
            if cfg.loss.class_gated_branch_logit_margin.weight <= 0:
                raise ValueError(
                    "train.loss.class_gated_branch_logit_margin.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.class_gated_branch_logit_margin.margin <= 0:
                raise ValueError(
                    "train.loss.class_gated_branch_logit_margin.margin must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.class_gated_branch_logit_margin is supported "
                    "only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.class_gated_branch_logit_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        b2e_cfg = cfg.loss.branch_to_evidence_ranking_consistency
        if not isinstance(b2e_cfg.enabled, bool):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.enabled "
                "must be a boolean"
            )
        if not isinstance(b2e_cfg.teacher_detach, bool):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.teacher_detach "
                "must be a boolean"
            )
        if not isinstance(b2e_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.class_weighted "
                "must be a boolean"
            )
        if (
            b2e_cfg.target
            not in JsonConfigLoader._BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_TARGETS
        ):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.target must be "
                "'class_evidence_logits'"
            )
        if (
            b2e_cfg.source
            not in JsonConfigLoader._BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_SOURCES
        ):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.source must be "
                "one of "
                f"{sorted(JsonConfigLoader._BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_SOURCES)}"
            )
        if (
            b2e_cfg.mode
            not in JsonConfigLoader._BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_MODES
        ):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.mode must be "
                "one of "
                f"{sorted(JsonConfigLoader._BRANCH_TO_EVIDENCE_RANKING_CONSISTENCY_MODES)}"
            )
        if b2e_cfg.reduction not in JsonConfigLoader._MARGIN_REDUCTIONS:
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.reduction "
                f"must be one of {sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        if not isinstance(b2e_cfg.teacher_temperature, int | float) or isinstance(
            b2e_cfg.teacher_temperature,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_to_evidence_ranking_consistency."
                "teacher_temperature must be numeric"
            )
        if b2e_cfg.teacher_temperature <= 0:
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency."
                "teacher_temperature must be greater than zero"
            )
        if not isinstance(b2e_cfg.student_temperature, int | float) or isinstance(
            b2e_cfg.student_temperature,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_to_evidence_ranking_consistency."
                "student_temperature must be numeric"
            )
        if b2e_cfg.student_temperature <= 0:
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency."
                "student_temperature must be greater than zero"
            )
        if not isinstance(b2e_cfg.temperature, int | float) or isinstance(
            b2e_cfg.temperature,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_to_evidence_ranking_consistency."
                "temperature must be numeric"
            )
        if b2e_cfg.temperature <= 0:
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency."
                "temperature must be greater than zero"
            )
        if b2e_cfg.mode == "teacher_distribution_kl":
            if b2e_cfg.source != "class_top_branch_margin_features":
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency.source "
                    "must be 'class_top_branch_margin_features' when "
                    "mode='teacher_distribution_kl'"
                )
            if b2e_cfg.teacher_gap_cap is not None:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_gap_cap is not used when "
                    "mode='teacher_distribution_kl'"
                )
            if b2e_cfg.teacher_floor_by_label:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_floor_by_label is not used when "
                    "mode='teacher_distribution_kl'"
                )
            if b2e_cfg.support_weighting.enabled:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "support_weighting is not used when "
                    "mode='teacher_distribution_kl'"
                )
            if b2e_cfg.hardness_weighting.enabled:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "hardness_weighting is not used when "
                    "mode='teacher_distribution_kl'"
                )
            if b2e_cfg.tolerance != 0:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "tolerance is not used when mode='teacher_distribution_kl'"
                )
            if b2e_cfg.reduction != "mean":
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "reduction must be 'mean' when "
                    "mode='teacher_distribution_kl'"
                )
        elif b2e_cfg.mode == "true_label_anchored_softplus":
            if b2e_cfg.source != "class_top_branch_margin_relative_features":
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency.source "
                    "must be 'class_top_branch_margin_relative_features' when "
                    "mode='true_label_anchored_softplus'"
                )
            if not b2e_cfg.teacher_detach:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_detach must be true when "
                    "mode='true_label_anchored_softplus'"
                )
            if b2e_cfg.teacher_gap_cap is not None:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_gap_cap is not used when "
                    "mode='true_label_anchored_softplus'"
                )
            if b2e_cfg.teacher_floor_by_label:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_floor_by_label is not used when "
                    "mode='true_label_anchored_softplus'"
                )
            if b2e_cfg.hardness_weighting.enabled:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "hardness_weighting is not used when "
                    "mode='true_label_anchored_softplus'"
                )
            if b2e_cfg.tolerance != 0:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "tolerance is not used when "
                    "mode='true_label_anchored_softplus'"
                )
            if b2e_cfg.reduction != "mean":
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "reduction must be 'mean' when "
                    "mode='true_label_anchored_softplus'"
                )
            if b2e_cfg.support_weighting.enabled:
                if b2e_cfg.support_weighting.source != "same_as_source":
                    raise ValueError(
                        "train.loss.branch_to_evidence_ranking_consistency."
                        "support_weighting.source must be 'same_as_source' "
                        "when mode='true_label_anchored_softplus'"
                    )
                if b2e_cfg.support_weighting.mode != "positive_linear":
                    raise ValueError(
                        "train.loss.branch_to_evidence_ranking_consistency."
                        "support_weighting.mode must be 'positive_linear' "
                        "when mode='true_label_anchored_softplus'"
                    )
        elif b2e_cfg.source == "class_top_branch_margin_features":
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.source "
                "'class_top_branch_margin_features' requires "
                "mode='teacher_distribution_kl'"
            )
        elif b2e_cfg.source == "class_top_branch_margin_relative_features":
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.source "
                "'class_top_branch_margin_relative_features' requires "
                "mode='true_label_anchored_softplus'"
            )
        JsonConfigLoader._validate_margin_support_weighting(
            b2e_cfg.support_weighting,
            field_name=(
                "train.loss.branch_to_evidence_ranking_consistency.support_weighting"
            ),
        )
        JsonConfigLoader._validate_margin_hardness_weighting(
            b2e_cfg.hardness_weighting,
            field_name=(
                "train.loss.branch_to_evidence_ranking_consistency.hardness_weighting"
            ),
        )
        if not isinstance(b2e_cfg.warmup_epochs, int):
            raise TypeError(
                "train.loss.branch_to_evidence_ranking_consistency.warmup_epochs "
                "must be an integer"
            )
        if b2e_cfg.warmup_epochs < 0:
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.warmup_epochs "
                "must be greater than or equal to zero"
            )
        if not isinstance(b2e_cfg.tolerance, int | float) or isinstance(
            b2e_cfg.tolerance,
            bool,
        ):
            raise TypeError(
                "train.loss.branch_to_evidence_ranking_consistency.tolerance "
                "must be numeric"
            )
        if b2e_cfg.tolerance < 0:
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.tolerance "
                "must be greater than or equal to zero"
            )
        if b2e_cfg.teacher_gap_cap is not None:
            if not isinstance(
                b2e_cfg.teacher_gap_cap,
                int | float,
            ) or isinstance(b2e_cfg.teacher_gap_cap, bool):
                raise TypeError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_gap_cap must be numeric or null"
                )
            if b2e_cfg.teacher_gap_cap <= 0:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency."
                    "teacher_gap_cap must be greater than zero"
                )
        JsonConfigLoader._validate_numeric_label_mapping(
            b2e_cfg.teacher_floor_by_label,
            field_name=(
                "train.loss.branch_to_evidence_ranking_consistency."
                "teacher_floor_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        if b2e_cfg.teacher_gap_cap is not None:
            for label_name, floor_value in b2e_cfg.teacher_floor_by_label.items():
                if floor_value > b2e_cfg.teacher_gap_cap:
                    raise ValueError(
                        "train.loss.branch_to_evidence_ranking_consistency."
                        f"teacher_floor_by_label.{label_name} must be less than "
                        "or equal to teacher_gap_cap"
                    )
        if b2e_cfg.enabled:
            if b2e_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency.weight "
                    "must be greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency is "
                    "supported only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.branch_to_evidence_ranking_consistency requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        anti_veto_cfg = cfg.loss.global_residual_anti_veto
        if not isinstance(anti_veto_cfg.enabled, bool):
            raise ValueError(
                "train.loss.global_residual_anti_veto.enabled must be a boolean"
            )
        if not isinstance(anti_veto_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.global_residual_anti_veto.class_weighted must be a boolean"
            )
        if (
            anti_veto_cfg.target
            not in JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_TARGETS
        ):
            raise ValueError(
                "train.loss.global_residual_anti_veto.target must be "
                "one of "
                f"{sorted(JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_TARGETS)}"
            )
        if (
            anti_veto_cfg.reference
            not in JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_REFERENCES
        ):
            raise ValueError(
                "train.loss.global_residual_anti_veto.reference must be "
                "'class_evidence_logits'"
            )
        if anti_veto_cfg.mode not in JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_MODES:
            raise ValueError(
                "train.loss.global_residual_anti_veto.mode must be "
                f"one of {sorted(JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_MODES)}"
            )
        if (
            anti_veto_cfg.support_source
            not in JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_SUPPORT_SOURCES
        ):
            raise ValueError(
                "train.loss.global_residual_anti_veto.support_source must be "
                "one of "
                f"{sorted(JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_SUPPORT_SOURCES)}"
            )
        if (
            anti_veto_cfg.margin_mode
            not in JsonConfigLoader._GLOBAL_RESIDUAL_ANTI_VETO_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.global_residual_anti_veto.margin_mode must be "
                "'true_vs_hardest_negative'"
            )
        if anti_veto_cfg.reduction not in JsonConfigLoader._MARGIN_REDUCTIONS:
            raise ValueError(
                "train.loss.global_residual_anti_veto.reduction must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        if not isinstance(anti_veto_cfg.warmup_epochs, int):
            raise TypeError(
                "train.loss.global_residual_anti_veto.warmup_epochs must be an integer"
            )
        if anti_veto_cfg.warmup_epochs < 0:
            raise ValueError(
                "train.loss.global_residual_anti_veto.warmup_epochs must be "
                "greater than or equal to zero"
            )
        for value, field_name in (
            (
                anti_veto_cfg.evidence_confidence_threshold,
                "evidence_confidence_threshold",
            ),
            (anti_veto_cfg.min_residual_gap, "min_residual_gap"),
            (anti_veto_cfg.support_threshold, "support_threshold"),
            (anti_veto_cfg.allowed_gap_drop, "allowed_gap_drop"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    f"train.loss.global_residual_anti_veto.{field_name} must be numeric"
                )
        if anti_veto_cfg.allowed_gap_drop < 0:
            raise ValueError(
                "train.loss.global_residual_anti_veto.allowed_gap_drop must be "
                "greater than or equal to zero"
            )
        if anti_veto_cfg.mode == "true_vs_hardest_negative":
            if anti_veto_cfg.target != "global_residual_logits":
                raise ValueError(
                    "train.loss.global_residual_anti_veto.target must be "
                    "'global_residual_logits' when mode='true_vs_hardest_negative'"
                )
            if anti_veto_cfg.support_source != "none":
                raise ValueError(
                    "train.loss.global_residual_anti_veto.support_source must be "
                    "'none' when mode='true_vs_hardest_negative'"
                )
        if anti_veto_cfg.mode == "final_gap_preservation":
            if anti_veto_cfg.target != "final_logits":
                raise ValueError(
                    "train.loss.global_residual_anti_veto.target must be "
                    "'final_logits' when mode='final_gap_preservation'"
                )
            if anti_veto_cfg.support_source != "top_branch_margin":
                raise ValueError(
                    "train.loss.global_residual_anti_veto.support_source must be "
                    "'top_branch_margin' when mode='final_gap_preservation'"
                )
        if anti_veto_cfg.enabled:
            if anti_veto_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.global_residual_anti_veto.weight must be greater "
                    "than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.global_residual_anti_veto is supported only for "
                    "cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.global_residual_anti_veto requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        residual_contradiction_cfg = cfg.loss.residual_contradiction_regularization
        if not isinstance(residual_contradiction_cfg.enabled, bool):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.enabled "
                "must be a boolean"
            )
        if not isinstance(residual_contradiction_cfg.class_weighted, bool):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.class_weighted "
                "must be a boolean"
            )
        if (
            residual_contradiction_cfg.target
            not in JsonConfigLoader._RESIDUAL_CONTRADICTION_TARGETS
        ):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.target must be "
                "'global_residual_logits'"
            )
        if (
            residual_contradiction_cfg.reference
            not in JsonConfigLoader._RESIDUAL_CONTRADICTION_REFERENCES
        ):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.reference must be "
                "'class_evidence_logits'"
            )
        if (
            residual_contradiction_cfg.mode
            not in JsonConfigLoader._RESIDUAL_CONTRADICTION_MODES
        ):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.mode must be "
                "'opposite_gap_penalty'"
            )
        if (
            residual_contradiction_cfg.margin_mode
            not in JsonConfigLoader._RESIDUAL_CONTRADICTION_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.margin_mode "
                "must be 'true_vs_hardest_negative'"
            )
        if (
            residual_contradiction_cfg.reduction
            not in JsonConfigLoader._MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.reduction "
                f"must be one of {sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        if not isinstance(residual_contradiction_cfg.warmup_epochs, int):
            raise TypeError(
                "train.loss.residual_contradiction_regularization.warmup_epochs "
                "must be an integer"
            )
        if residual_contradiction_cfg.warmup_epochs < 0:
            raise ValueError(
                "train.loss.residual_contradiction_regularization.warmup_epochs "
                "must be greater than or equal to zero"
            )
        for value, field_name in (
            (
                residual_contradiction_cfg.evidence_gap_threshold,
                "evidence_gap_threshold",
            ),
            (residual_contradiction_cfg.min_residual_gap, "min_residual_gap"),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    "train.loss.residual_contradiction_regularization."
                    f"{field_name} must be numeric"
                )
        if residual_contradiction_cfg.enabled:
            if residual_contradiction_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.residual_contradiction_regularization.weight "
                    "must be greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.residual_contradiction_regularization is "
                    "supported only for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.residual_contradiction_regularization requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        if not isinstance(cfg.loss.gate_weighted_branch_margin.enabled, bool):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.enabled must be a boolean"
            )
        if not isinstance(cfg.loss.gate_weighted_branch_margin.class_weighted, bool):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.class_weighted "
                "must be a boolean"
            )
        if (
            cfg.loss.gate_weighted_branch_margin.target
            not in JsonConfigLoader._GATE_WEIGHTED_BRANCH_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.target must be "
                "'true_class_gate'"
            )
        if (
            cfg.loss.gate_weighted_branch_margin.source
            not in JsonConfigLoader._GATE_WEIGHTED_BRANCH_MARGIN_SOURCES
        ):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.source must be 'branch_logits'"
            )
        if (
            cfg.loss.gate_weighted_branch_margin.mode
            not in JsonConfigLoader._GATE_WEIGHTED_BRANCH_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.mode must be "
                "'true_vs_hardest_negative'"
            )
        if (
            cfg.loss.gate_weighted_branch_margin.reduction
            not in JsonConfigLoader._MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.reduction must be one "
                f"of {sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        if (
            cfg.loss.gate_weighted_branch_margin.branch_selection
            not in JsonConfigLoader._GATE_WEIGHTED_BRANCH_MARGIN_SELECTIONS
        ):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.branch_selection must "
                "be one of "
                f"{sorted(JsonConfigLoader._GATE_WEIGHTED_BRANCH_MARGIN_SELECTIONS)}"
            )
        if not isinstance(cfg.loss.gate_weighted_branch_margin.warmup_epochs, int):
            raise TypeError(
                "train.loss.gate_weighted_branch_margin.warmup_epochs "
                "must be an integer"
            )
        if cfg.loss.gate_weighted_branch_margin.warmup_epochs < 0:
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.warmup_epochs must be "
                "greater than or equal to zero"
            )
        if cfg.loss.gate_weighted_branch_margin.enabled:
            if cfg.loss.gate_weighted_branch_margin.weight <= 0:
                raise ValueError(
                    "train.loss.gate_weighted_branch_margin.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.gate_weighted_branch_margin.margin <= 0:
                raise ValueError(
                    "train.loss.gate_weighted_branch_margin.margin must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.gate_weighted_branch_margin is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.gate_weighted_branch_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        gate_best_cfg = cfg.loss.gate_best_branch_alignment
        if not isinstance(gate_best_cfg.enabled, bool):
            raise TypeError(
                "train.loss.gate_best_branch_alignment.enabled must be a boolean"
            )
        if not isinstance(gate_best_cfg.class_weighted, bool):
            raise TypeError(
                "train.loss.gate_best_branch_alignment.class_weighted must be a boolean"
            )
        if not isinstance(gate_best_cfg.detach_branch_margin, bool):
            raise TypeError(
                "train.loss.gate_best_branch_alignment.detach_branch_margin "
                "must be a boolean"
            )
        if (
            gate_best_cfg.target
            not in JsonConfigLoader._GATE_BEST_BRANCH_ALIGNMENT_TARGETS
        ):
            raise ValueError(
                "train.loss.gate_best_branch_alignment.target must be 'true_class_gate'"
            )
        if (
            gate_best_cfg.source
            not in JsonConfigLoader._GATE_BEST_BRANCH_ALIGNMENT_SOURCES
        ):
            raise ValueError(
                "train.loss.gate_best_branch_alignment.source must be 'branch_logits'"
            )
        if gate_best_cfg.mode not in JsonConfigLoader._GATE_BEST_BRANCH_ALIGNMENT_MODES:
            raise ValueError(
                "train.loss.gate_best_branch_alignment.mode must be "
                "'weak_positive_best_branch_alignment'"
            )
        if (
            gate_best_cfg.margin_mode
            not in JsonConfigLoader._GATE_BEST_BRANCH_ALIGNMENT_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.gate_best_branch_alignment.margin_mode must be "
                "'true_vs_hardest_negative'"
            )
        if (
            gate_best_cfg.loss
            not in JsonConfigLoader._GATE_BEST_BRANCH_ALIGNMENT_LOSSES
        ):
            raise ValueError(
                "train.loss.gate_best_branch_alignment.loss must be "
                "'negative_log_best_gate'"
            )
        if (
            gate_best_cfg.reduction
            not in JsonConfigLoader._GATE_BEST_BRANCH_ALIGNMENT_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.gate_best_branch_alignment.reduction must be 'mean'"
            )
        if not isinstance(gate_best_cfg.weight, int | float) or isinstance(
            gate_best_cfg.weight,
            bool,
        ):
            raise TypeError(
                "train.loss.gate_best_branch_alignment.weight must be numeric"
            )
        for field_name, value, min_value, strict_min in (
            ("min_best_margin", gate_best_cfg.min_best_margin, 0.0, False),
            ("max_best_margin", gate_best_cfg.max_best_margin, 0.0, True),
            (
                "mismatch_margin_drop",
                gate_best_cfg.mismatch_margin_drop,
                0.0,
                False,
            ),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(
                    f"train.loss.gate_best_branch_alignment.{field_name} "
                    "must be numeric"
                )
            if float(value) <= min_value if strict_min else float(value) < min_value:
                comparator = (
                    "greater than" if strict_min else "greater than or equal to"
                )
                raise ValueError(
                    f"train.loss.gate_best_branch_alignment.{field_name} "
                    f"must be {comparator} {min_value}"
                )
        if float(gate_best_cfg.max_best_margin) <= float(gate_best_cfg.min_best_margin):
            raise ValueError(
                "train.loss.gate_best_branch_alignment.max_best_margin must be "
                "greater than min_best_margin"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            gate_best_cfg.label_weight_by_label,
            field_name="train.loss.gate_best_branch_alignment.label_weight_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            gate_best_cfg.min_best_margin_by_label,
            field_name=(
                "train.loss.gate_best_branch_alignment.min_best_margin_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            gate_best_cfg.max_best_margin_by_label,
            field_name=(
                "train.loss.gate_best_branch_alignment.max_best_margin_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            gate_best_cfg.mismatch_margin_drop_by_label,
            field_name=(
                "train.loss.gate_best_branch_alignment.mismatch_margin_drop_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        for label_name in set(gate_best_cfg.min_best_margin_by_label) | set(
            gate_best_cfg.max_best_margin_by_label
        ):
            min_value = gate_best_cfg.min_best_margin_by_label.get(
                label_name,
                gate_best_cfg.min_best_margin,
            )
            max_value = gate_best_cfg.max_best_margin_by_label.get(
                label_name,
                gate_best_cfg.max_best_margin,
            )
            if float(max_value) <= float(min_value):
                raise ValueError(
                    "train.loss.gate_best_branch_alignment max_best_margin for "
                    f"{label_name!r} must be greater than min_best_margin"
                )
        if not isinstance(gate_best_cfg.warmup_epochs, int) or isinstance(
            gate_best_cfg.warmup_epochs,
            bool,
        ):
            raise TypeError(
                "train.loss.gate_best_branch_alignment.warmup_epochs must be an integer"
            )
        if int(gate_best_cfg.warmup_epochs) < 0:
            raise ValueError(
                "train.loss.gate_best_branch_alignment.warmup_epochs must be "
                "non-negative"
            )
        if gate_best_cfg.enabled:
            if gate_best_cfg.weight <= 0:
                raise ValueError(
                    "train.loss.gate_best_branch_alignment.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.gate_best_branch_alignment is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.gate_best_branch_alignment requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
            if gate_best_cfg.class_weighted and not cfg.loss.class_weighting.enabled:
                raise ValueError(
                    "train.loss.gate_best_branch_alignment.class_weighted "
                    "requires train.loss.class_weighting.enabled=true"
                )
        if not isinstance(cfg.loss.gate_branch_regret.enabled, bool):
            raise ValueError("train.loss.gate_branch_regret.enabled must be a boolean")
        if (
            cfg.loss.gate_branch_regret.target
            not in JsonConfigLoader._GATE_BRANCH_REGRET_TARGETS
        ):
            raise ValueError(
                "train.loss.gate_branch_regret.target must be 'true_class_gate'"
            )
        if (
            cfg.loss.gate_branch_regret.source
            not in JsonConfigLoader._GATE_BRANCH_REGRET_SOURCES
        ):
            raise ValueError(
                "train.loss.gate_branch_regret.source must be 'branch_logits'"
            )
        if (
            cfg.loss.gate_branch_regret.mode
            not in JsonConfigLoader._GATE_BRANCH_REGRET_MODES
        ):
            raise ValueError(
                "train.loss.gate_branch_regret.mode must be 'best_margin_regret'"
            )
        if (
            cfg.loss.gate_branch_regret.margin_mode
            not in JsonConfigLoader._GATE_BRANCH_REGRET_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.gate_branch_regret.margin_mode must be "
                "'true_vs_hardest_negative'"
            )
        if not isinstance(cfg.loss.gate_branch_regret.warmup_epochs, int):
            raise TypeError(
                "train.loss.gate_branch_regret.warmup_epochs must be an integer"
            )
        if cfg.loss.gate_branch_regret.warmup_epochs < 0:
            raise ValueError(
                "train.loss.gate_branch_regret.warmup_epochs must be greater "
                "than or equal to zero"
            )
        JsonConfigLoader._validate_gate_branch_regret_weight_schedule(
            cfg.loss.gate_branch_regret.weight_schedule
        )
        if (
            cfg.loss.gate_branch_regret.weight_schedule.enabled
            and not cfg.loss.gate_branch_regret.enabled
        ):
            raise ValueError(
                "train.loss.gate_branch_regret.weight_schedule requires "
                "train.loss.gate_branch_regret.enabled=true"
            )
        if not isinstance(
            cfg.loss.gate_branch_regret.positive_threshold,
            int | float,
        ) or isinstance(cfg.loss.gate_branch_regret.positive_threshold, bool):
            raise TypeError(
                "train.loss.gate_branch_regret.positive_threshold must be numeric"
            )
        if cfg.loss.gate_branch_regret.positive_threshold < 0:
            raise ValueError(
                "train.loss.gate_branch_regret.positive_threshold must be "
                "greater than or equal to zero"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            cfg.loss.gate_branch_regret.positive_threshold_by_label,
            field_name="train.loss.gate_branch_regret.positive_threshold_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_auto_positive_threshold_by_train_stats(
            cfg.loss.gate_branch_regret.auto_positive_threshold_by_train_stats,
            label_to_index=label_to_index,
        )
        if (
            cfg.loss.gate_branch_regret.auto_positive_threshold_by_train_stats.enabled
            and not cfg.loss.gate_branch_regret.enabled
        ):
            raise ValueError(
                "train.loss.gate_branch_regret.auto_positive_threshold_by_train_stats "
                "requires train.loss.gate_branch_regret.enabled=true"
            )
        if cfg.loss.gate_branch_regret.auto_positive_threshold_by_train_stats.enabled:
            threshold_auto_cfg = (
                cfg.loss.gate_branch_regret.auto_positive_threshold_by_train_stats
            )
            for (
                label_name,
                min_value,
            ) in threshold_auto_cfg.min_threshold_by_label.items():
                initial_value = (
                    cfg.loss.gate_branch_regret.positive_threshold_by_label.get(
                        label_name,
                        cfg.loss.gate_branch_regret.positive_threshold,
                    )
                )
                max_value = threshold_auto_cfg.max_threshold_by_label[label_name]
                if initial_value < min_value or initial_value > max_value:
                    raise ValueError(
                        "train.loss.gate_branch_regret initial threshold for "
                        f"{label_name!r} must be within auto min/max bounds"
                    )
        if not isinstance(
            cfg.loss.gate_branch_regret.tolerance,
            int | float,
        ) or isinstance(cfg.loss.gate_branch_regret.tolerance, bool):
            raise TypeError("train.loss.gate_branch_regret.tolerance must be numeric")
        if cfg.loss.gate_branch_regret.tolerance < 0:
            raise ValueError(
                "train.loss.gate_branch_regret.tolerance must be greater than "
                "or equal to zero"
            )
        if cfg.loss.gate_branch_regret.enabled:
            if cfg.loss.gate_branch_regret.weight <= 0:
                raise ValueError(
                    "train.loss.gate_branch_regret.weight must be greater than "
                    "zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.gate_branch_regret is supported only for "
                    "cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.gate_branch_regret requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        if not isinstance(cfg.loss.gate_bad_branch_suppression.enabled, bool):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.enabled must be a boolean"
            )
        if (
            cfg.loss.gate_bad_branch_suppression.target
            not in JsonConfigLoader._GATE_BAD_BRANCH_SUPPRESSION_TARGETS
        ):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.target must be "
                "'true_class_gate'"
            )
        if (
            cfg.loss.gate_bad_branch_suppression.source
            not in JsonConfigLoader._GATE_BAD_BRANCH_SUPPRESSION_SOURCES
        ):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.source must be 'branch_logits'"
            )
        if (
            cfg.loss.gate_bad_branch_suppression.mode
            not in JsonConfigLoader._GATE_BAD_BRANCH_SUPPRESSION_MODES
        ):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.mode must be "
                "'margin_below_threshold'"
            )
        if (
            cfg.loss.gate_bad_branch_suppression.margin_mode
            not in JsonConfigLoader._GATE_BAD_BRANCH_SUPPRESSION_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.margin_mode must be "
                "'true_vs_hardest_negative'"
            )
        if not isinstance(
            cfg.loss.gate_bad_branch_suppression.bad_margin_threshold,
            int | float,
        ) or isinstance(
            cfg.loss.gate_bad_branch_suppression.bad_margin_threshold,
            bool,
        ):
            raise TypeError(
                "train.loss.gate_bad_branch_suppression.bad_margin_threshold "
                "must be numeric"
            )
        JsonConfigLoader._validate_threshold_label_mapping(
            cfg.loss.gate_bad_branch_suppression.bad_margin_threshold_by_label,
            field_name=(
                "train.loss.gate_bad_branch_suppression.bad_margin_threshold_by_label"
            ),
            label_to_index=label_to_index,
        )
        if not isinstance(cfg.loss.gate_bad_branch_suppression.warmup_epochs, int):
            raise TypeError(
                "train.loss.gate_bad_branch_suppression.warmup_epochs "
                "must be an integer"
            )
        if cfg.loss.gate_bad_branch_suppression.warmup_epochs < 0:
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.warmup_epochs must be "
                "greater than or equal to zero"
            )
        JsonConfigLoader._validate_gate_branch_regret_weight_schedule(
            cfg.loss.gate_bad_branch_suppression.weight_schedule,
            field_name=("train.loss.gate_bad_branch_suppression.weight_schedule"),
        )
        if (
            cfg.loss.gate_bad_branch_suppression.weight_schedule.enabled
            and not cfg.loss.gate_bad_branch_suppression.enabled
        ):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression.weight_schedule requires "
                "train.loss.gate_bad_branch_suppression.enabled=true"
            )
        JsonConfigLoader._validate_auto_bad_branch_threshold_by_train_stats(
            cfg.loss.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats,
            label_to_index=label_to_index,
        )
        if (
            cfg.loss.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats.enabled
            and not cfg.loss.gate_bad_branch_suppression.enabled
        ):
            raise ValueError(
                "train.loss.gate_bad_branch_suppression."
                "auto_bad_margin_threshold_by_train_stats requires "
                "train.loss.gate_bad_branch_suppression.enabled=true"
            )
        if cfg.loss.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats.enabled:
            bad_auto_cfg = cfg.loss.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats
            for label_name, min_value in bad_auto_cfg.min_threshold_by_label.items():
                initial_value = cfg.loss.gate_bad_branch_suppression.bad_margin_threshold_by_label.get(
                    label_name,
                    cfg.loss.gate_bad_branch_suppression.bad_margin_threshold,
                )
                max_value = bad_auto_cfg.max_threshold_by_label[label_name]
                if initial_value < min_value or initial_value > max_value:
                    raise ValueError(
                        "train.loss.gate_bad_branch_suppression initial threshold "
                        f"for {label_name!r} must be within auto min/max bounds"
                    )
        if cfg.loss.gate_bad_branch_suppression.enabled:
            if cfg.loss.gate_bad_branch_suppression.weight <= 0:
                raise ValueError(
                    "train.loss.gate_bad_branch_suppression.weight must be "
                    "greater than zero when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.gate_bad_branch_suppression is supported only "
                    "for cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.gate_bad_branch_suppression requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        if not isinstance(cfg.loss.top_branch_margin.enabled, bool):
            raise ValueError("train.loss.top_branch_margin.enabled must be a boolean")
        if not isinstance(cfg.loss.top_branch_margin.class_weighted, bool):
            raise ValueError(
                "train.loss.top_branch_margin.class_weighted must be a boolean"
            )
        if (
            cfg.loss.top_branch_margin.target
            not in JsonConfigLoader._TOP_BRANCH_MARGIN_TARGETS
        ):
            raise ValueError(
                "train.loss.top_branch_margin.target must be 'branch_logits'"
            )
        if (
            cfg.loss.top_branch_margin.mode
            not in JsonConfigLoader._TOP_BRANCH_MARGIN_MODES
        ):
            raise ValueError(
                "train.loss.top_branch_margin.mode must be 'true_vs_hardest_negative'"
            )
        if (
            cfg.loss.top_branch_margin.branch_reduction
            not in JsonConfigLoader._TOP_BRANCH_MARGIN_BRANCH_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.top_branch_margin.branch_reduction must be 'max'"
            )
        if (
            cfg.loss.top_branch_margin.reduction
            not in JsonConfigLoader._MARGIN_REDUCTIONS
        ):
            raise ValueError(
                "train.loss.top_branch_margin.reduction must be one of "
                f"{sorted(JsonConfigLoader._MARGIN_REDUCTIONS)}"
            )
        if not isinstance(cfg.loss.top_branch_margin.warmup_epochs, int):
            raise TypeError(
                "train.loss.top_branch_margin.warmup_epochs must be an integer"
            )
        if cfg.loss.top_branch_margin.warmup_epochs < 0:
            raise ValueError(
                "train.loss.top_branch_margin.warmup_epochs must be greater than "
                "or equal to zero"
            )
        JsonConfigLoader._validate_numeric_label_mapping(
            cfg.loss.top_branch_margin.margin_by_label,
            field_name="train.loss.top_branch_margin.margin_by_label",
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=True,
        )
        JsonConfigLoader._validate_auto_margin_by_train_stats(
            cfg.loss.top_branch_margin.auto_margin_by_train_stats,
            label_to_index=label_to_index,
        )
        phase_cfg = cfg.loss.top_branch_margin.phase_weight_schedule
        if not isinstance(phase_cfg.enabled, bool):
            raise TypeError(
                "train.loss.top_branch_margin.phase_weight_schedule.enabled "
                "must be a boolean"
            )
        if not isinstance(phase_cfg.start_epoch, int) or isinstance(
            phase_cfg.start_epoch,
            bool,
        ):
            raise TypeError(
                "train.loss.top_branch_margin.phase_weight_schedule.start_epoch "
                "must be an integer"
            )
        if int(phase_cfg.start_epoch) <= 0:
            raise ValueError(
                "train.loss.top_branch_margin.phase_weight_schedule.start_epoch "
                "must be greater than zero"
            )
        if phase_cfg.end_epoch is not None:
            if not isinstance(phase_cfg.end_epoch, int) or isinstance(
                phase_cfg.end_epoch,
                bool,
            ):
                raise TypeError(
                    "train.loss.top_branch_margin.phase_weight_schedule.end_epoch "
                    "must be an integer or null"
                )
            if int(phase_cfg.end_epoch) < int(phase_cfg.start_epoch):
                raise ValueError(
                    "train.loss.top_branch_margin.phase_weight_schedule.end_epoch "
                    "must be greater than or equal to start_epoch"
                )
        JsonConfigLoader._validate_numeric_label_mapping(
            phase_cfg.start_multiplier_by_label,
            field_name=(
                "train.loss.top_branch_margin.phase_weight_schedule."
                "start_multiplier_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_numeric_label_mapping(
            phase_cfg.label_multiplier_by_label,
            field_name=(
                "train.loss.top_branch_margin.phase_weight_schedule."
                "label_multiplier_by_label"
            ),
            label_to_index=label_to_index,
            min_value=0.0,
            strict_min=False,
        )
        JsonConfigLoader._validate_top_branch_margin_hardness_weighting(
            cfg.loss.top_branch_margin.hardness_weighting,
            field_name="train.loss.top_branch_margin.hardness_weighting",
        )
        if (
            cfg.loss.top_branch_margin.auto_margin_by_train_stats.enabled
            and not cfg.loss.top_branch_margin.enabled
        ):
            raise ValueError(
                "train.loss.top_branch_margin.auto_margin_by_train_stats requires "
                "train.loss.top_branch_margin.enabled=true"
            )
        if cfg.loss.top_branch_margin.auto_margin_by_train_stats.enabled:
            margin_auto_cfg = cfg.loss.top_branch_margin.auto_margin_by_train_stats
            for label_name, min_value in margin_auto_cfg.min_margin_by_label.items():
                initial_value = cfg.loss.top_branch_margin.margin_by_label.get(
                    label_name,
                    cfg.loss.top_branch_margin.margin,
                )
                max_value = margin_auto_cfg.max_margin_by_label[label_name]
                if initial_value < min_value or initial_value > max_value:
                    raise ValueError(
                        "train.loss.top_branch_margin initial margin for "
                        f"{label_name!r} must be within auto min/max bounds"
                    )
        if cfg.loss.top_branch_margin.enabled:
            if cfg.loss.top_branch_margin.weight <= 0:
                raise ValueError(
                    "train.loss.top_branch_margin.weight must be greater than zero "
                    "when enabled"
                )
            if cfg.loss.top_branch_margin.margin <= 0:
                raise ValueError(
                    "train.loss.top_branch_margin.margin must be greater than zero "
                    "when enabled"
                )
            if cfg.loss.type != "cross_entropy":
                raise ValueError(
                    "train.loss.top_branch_margin is supported only for "
                    "cross_entropy runs"
                )
            if evidence_pooling_type != "class_aware_branch_gated":
                raise ValueError(
                    "train.loss.top_branch_margin requires "
                    "model.encoder.architecture.evidence_pooling.type="
                    "'class_aware_branch_gated'"
                )
        if not isinstance(cfg.loss.class_weighting.enabled, bool):
            raise ValueError("train.loss.class_weighting.enabled must be a boolean")
        if cfg.loss.class_weighting.type not in JsonConfigLoader._CLASS_WEIGHTING_TYPES:
            raise ValueError(
                "train.loss.class_weighting.type must be one of "
                f"{sorted(JsonConfigLoader._CLASS_WEIGHTING_TYPES)}"
            )
        if (
            not isinstance(cfg.loss.class_weighting.power, (int, float))
            or isinstance(cfg.loss.class_weighting.power, bool)
            or cfg.loss.class_weighting.power <= 0
        ):
            raise ValueError(
                "train.loss.class_weighting.power must be greater than zero"
            )
        if (
            cfg.loss.class_weighting.normalize
            not in JsonConfigLoader._CLASS_WEIGHTING_NORMALIZERS
        ):
            raise ValueError("train.loss.class_weighting.normalize must be 'mean_one'")
        if cfg.loss.class_weighting.source not in JsonConfigLoader._LOSS_WEIGHT_SOURCES:
            raise ValueError("train.loss.class_weighting.source must be 'train'")
        if cfg.loss.class_weighting.enabled and cfg.loss.type != "cross_entropy":
            raise ValueError(
                "train.loss.class_weighting is supported only for cross_entropy runs"
            )
        if (
            cfg.loss.class_evidence_margin.enabled
            and cfg.loss.class_evidence_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.class_evidence_margin.class_weighted requires "
                "train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.class_gated_branch_logit_margin.enabled
            and cfg.loss.class_gated_branch_logit_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.class_gated_branch_logit_margin.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.branch_support_score_margin.enabled
            and cfg.loss.branch_support_score_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.branch_support_score_margin.class_weighted requires "
                "train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.top_support_score_margin.enabled
            and cfg.loss.top_support_score_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.top_support_score_margin.class_weighted requires "
                "train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.class_evidence_gap_cap_regularization.enabled
            and cfg.loss.class_evidence_gap_cap_regularization.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.class_evidence_gap_cap_regularization.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.class_evidence_positive_gap_cap_regularization.enabled
            and cfg.loss.class_evidence_positive_gap_cap_regularization.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.class_evidence_positive_gap_cap_regularization."
                "class_weighted requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.interaction_gap_cap_regularization.enabled
            and cfg.loss.interaction_gap_cap_regularization.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.interaction_gap_cap_regularization.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.branch_direct_score_margin.enabled
            and cfg.loss.branch_direct_score_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.branch_direct_score_margin.class_weighted requires "
                "train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.branch_path_dominance_constraint.enabled
            and cfg.loss.branch_path_dominance_constraint.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.branch_path_dominance_constraint.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.branch_to_evidence_ranking_consistency.enabled
            and cfg.loss.branch_to_evidence_ranking_consistency.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.branch_to_evidence_ranking_consistency.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.global_residual_anti_veto.enabled
            and cfg.loss.global_residual_anti_veto.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.global_residual_anti_veto.class_weighted requires "
                "train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.residual_contradiction_regularization.enabled
            and cfg.loss.residual_contradiction_regularization.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.residual_contradiction_regularization.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.gate_weighted_branch_margin.enabled
            and cfg.loss.gate_weighted_branch_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.class_weighted "
                "requires train.loss.class_weighting.enabled=true"
            )
        if (
            cfg.loss.top_branch_margin.enabled
            and cfg.loss.top_branch_margin.class_weighted
            and not cfg.loss.class_weighting.enabled
        ):
            raise ValueError(
                "train.loss.top_branch_margin.class_weighted requires "
                "train.loss.class_weighting.enabled=true"
            )
        if not isinstance(cfg.loss.label_smoothing.enabled, bool):
            raise ValueError("train.loss.label_smoothing.enabled must be a boolean")
        if not (0.0 <= cfg.loss.label_smoothing.value < 1.0):
            raise ValueError("train.loss.label_smoothing.value must be within [0, 1)")
        JsonConfigLoader._validate_branch_binary_auxiliary(
            cfg.loss.branch_binary_auxiliary,
            label_to_index=label_to_index,
        )
        is_class_aware_pooling = evidence_pooling_type == "class_aware_branch_gated"
        if num_classes == 2:
            if cfg.loss.type in {"bce", "focal"}:
                if is_class_aware_pooling:
                    raise ValueError(
                        "Two-label class_aware_branch_gated AST runs require "
                        "train.loss.type='cross_entropy'"
                    )
                if cfg.loss.auto_pos_weight and cfg.loss.pos_weight is not None:
                    raise ValueError(
                        "train.loss.auto_pos_weight and train.loss.pos_weight cannot both be set"
                    )
                if cfg.loss.pos_weight is not None and cfg.loss.pos_weight <= 0:
                    raise ValueError("train.loss.pos_weight must be greater than zero")
                return
            if cfg.loss.type == "cross_entropy":
                if not is_class_aware_pooling:
                    raise ValueError(
                        "Two-label cross_entropy AST runs require "
                        "model.encoder.architecture.evidence_pooling.type="
                        "'class_aware_branch_gated'"
                    )
                if cfg.loss.auto_pos_weight:
                    raise ValueError(
                        "train.loss.auto_pos_weight is only supported for one-logit "
                        "binary bce/focal runs"
                    )
                if cfg.loss.pos_weight is not None:
                    raise ValueError(
                        "train.loss.pos_weight is only supported for one-logit "
                        "binary bce/focal runs"
                    )
                return
            raise ValueError(
                "Two-label AST runs support train.loss.type='bce', 'focal', "
                "or class-aware 'cross_entropy'"
            )
        if cfg.loss.type != "cross_entropy":
            raise ValueError(
                "Multi-class AST runs require train.loss.type='cross_entropy'"
            )
        if cfg.loss.auto_pos_weight:
            raise ValueError(
                "train.loss.auto_pos_weight is only supported for one-logit "
                "binary bce/focal runs"
            )
        if cfg.loss.pos_weight is not None:
            raise ValueError(
                "train.loss.pos_weight is only supported for one-logit binary "
                "bce/focal runs"
            )

    @staticmethod
    def _validate_sampler(cfg: SamplerConfig) -> None:
        if not isinstance(cfg.weighted_random, bool):
            raise ValueError("train.sampler.weighted_random must be a boolean")
        if not isinstance(cfg.enabled, bool):
            raise ValueError("train.sampler.enabled must be a boolean")
        if cfg.type not in JsonConfigLoader._SAMPLER_TYPES:
            raise ValueError(
                "train.sampler.type must be one of "
                f"{sorted(JsonConfigLoader._SAMPLER_TYPES)}"
            )
        if not isinstance(cfg.replacement, bool):
            raise ValueError("train.sampler.replacement must be a boolean")
        if cfg.source != "train":
            raise ValueError("train.sampler.source must be 'train'")
        if isinstance(cfg.num_samples, str):
            if cfg.num_samples != "dataset_size":
                raise ValueError(
                    "train.sampler.num_samples must be 'dataset_size' or a positive integer"
                )
        elif isinstance(cfg.num_samples, bool) or not isinstance(cfg.num_samples, int):
            raise TypeError(
                "train.sampler.num_samples must be 'dataset_size' or a positive integer"
            )
        elif cfg.num_samples <= 0:
            raise ValueError(
                "train.sampler.num_samples must be 'dataset_size' or a positive integer"
            )
        if not cfg.enabled:
            return
        if cfg.type != "sqrt_inverse_class":
            raise ValueError(
                "train.sampler.type must be 'sqrt_inverse_class' when sampler is enabled"
            )
        if not cfg.replacement:
            raise ValueError(
                "train.sampler.replacement must be true for sqrt_inverse_class"
            )

    @staticmethod
    def _validate_branch_binary_auxiliary(
        cfg: BranchBinaryAuxiliaryLossConfig,
        *,
        label_to_index: Mapping[str, int],
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.enabled must be a boolean"
            )
        if cfg.aggregation != "mean":
            raise ValueError(
                "train.loss.branch_binary_auxiliary.aggregation must be 'mean'"
            )
        if cfg.weight <= 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.weight must be greater than zero"
            )
        if not isinstance(cfg.schedule.enabled, bool):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.enabled must be a boolean"
            )
        if (
            cfg.schedule.type
            not in JsonConfigLoader._BRANCH_BINARY_AUXILIARY_SCHEDULE_TYPES
        ):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.type must be "
                "'none' or 'cosine_floor'"
            )
        if cfg.schedule.enabled and cfg.schedule.type != "cosine_floor":
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.type must be "
                "'cosine_floor' when schedule is enabled"
            )
        if cfg.schedule.max_weight <= 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.max_weight must be "
                "greater than zero"
            )
        if cfg.schedule.min_weight < 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.min_weight must be "
                "greater than or equal to zero"
            )
        if cfg.schedule.max_weight < cfg.schedule.min_weight:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.max_weight must be "
                "greater than or equal to min_weight"
            )
        if cfg.schedule.total_epochs is not None and cfg.schedule.total_epochs <= 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.total_epochs must be "
                "greater than zero when provided"
            )
        if cfg.monitor.loss_weight < 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.monitor.loss_weight must be "
                "greater than or equal to zero"
            )
        if not isinstance(cfg.pos_weight.enabled, bool):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.pos_weight.enabled must be a boolean"
            )
        if cfg.pos_weight.type not in JsonConfigLoader._BRANCH_BINARY_POS_WEIGHT_TYPES:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.pos_weight.type must be "
                "'sqrt_normal_over_abnormal'"
            )
        if cfg.pos_weight.source not in JsonConfigLoader._LOSS_WEIGHT_SOURCES:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.pos_weight.source must be 'train'"
            )
        if not cfg.enabled:
            return
        configured_labels = set(cfg.label_to_index.keys())
        expected_labels = set(label_to_index.keys())
        missing = sorted(expected_labels - configured_labels)
        extra = sorted(configured_labels - expected_labels)
        if missing or extra:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index must contain "
                "exactly the same labels as data.label_to_index; "
                f"missing={missing} extra={extra}"
            )
        values = [int(value) for value in cfg.label_to_index.values()]
        if any(value not in {0, 1} for value in values):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index values must be 0 or 1"
            )
        if all(value == 0 for value in values):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index must include at least one abnormal label"
            )
        if all(value == 1 for value in values):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index must include at least one normal label"
            )

    @staticmethod
    def _validate_val(cfg: ValidationConfig) -> None:
        if not isinstance(cfg.loss.use_train_loss_config, bool):
            raise ValueError("val.loss.use_train_loss_config must be a boolean")
        if not cfg.loss.use_train_loss_config:
            raise ValueError(
                "val.loss.use_train_loss_config=false is not supported; validation "
                "loss uses train-derived weighting semantics"
            )
        if cfg.loss.class_weight_source != "train":
            raise ValueError("val.loss.class_weight_source must be 'train'")
        if cfg.loss.binary_pos_weight_source != "train":
            raise ValueError("val.loss.binary_pos_weight_source must be 'train'")

    @staticmethod
    def _validate_checkpointing(cfg: CheckpointingConfig | None) -> None:
        if cfg is None:
            return
        seen: set[str] = set()
        for monitor in cfg.monitors:
            if monitor.name not in JsonConfigLoader._CHECKPOINT_MONITORS:
                raise ValueError(
                    "checkpointing.monitors.name must be one of "
                    f"{sorted(JsonConfigLoader._CHECKPOINT_MONITORS)}"
                )
            if monitor.name in seen:
                raise ValueError(
                    "checkpointing.monitors must not contain duplicate monitor names"
                )
            seen.add(monitor.name)
            allowed_modes = JsonConfigLoader._CHECKPOINT_MONITORS[monitor.name]
            if monitor.mode not in allowed_modes:
                raise ValueError(
                    f"checkpointing monitor {monitor.name} requires mode one of "
                    f"{sorted(allowed_modes)}"
                )
            if monitor.top_k <= 0:
                raise ValueError(
                    "checkpointing.monitors.top_k must be greater than zero"
                )
            if (
                monitor.filename_prefix is not None
                and not monitor.filename_prefix.strip()
            ):
                raise ValueError(
                    "checkpointing.monitors.filename_prefix must not be empty"
                )

    @staticmethod
    def _parse_experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
        kwargs = dict(raw)
        logging_raw = kwargs.get("logging", {})
        if logging_raw is None:
            logging_raw = {}
        if not isinstance(logging_raw, Mapping):
            raise TypeError("experiment.logging must be an object")
        kwargs["logging"] = ExperimentLoggingConfig(**dict(logging_raw))
        cfg = ExperimentConfig(**kwargs)
        JsonConfigLoader._validate_experiment(cfg)
        return cfg

    @staticmethod
    def _parse_data(raw: Mapping[str, Any]) -> DataConfig:
        kwargs = dict(raw)
        kwargs.setdefault("train_dirs", [])
        kwargs.setdefault("val_dirs", [])
        kwargs.setdefault("eval_dirs", [])
        kwargs["audio"] = AudioConfig(**dict(raw["audio"]))
        preprocessing = dict(raw["preprocessing"])
        if "ast_fbank" not in preprocessing:
            raise ValueError("data.preprocessing.ast_fbank is required")
        preprocessing["bandpass"] = BandPassConfig(**preprocessing.get("bandpass", {}))
        preprocessing["ast_fbank"] = AstFbankConfig(**dict(preprocessing["ast_fbank"]))
        kwargs["preprocessing"] = PreprocessingConfig(**preprocessing)
        augmentation = dict(raw.get("augmentation", {}))
        policy = dict(augmentation.get("policy", {}))
        policy_choices = policy.get("choices", ())
        if policy_choices:
            if not isinstance(policy_choices, Sequence) or isinstance(
                policy_choices, (str, bytes)
            ):
                raise TypeError("data.augmentation.policy.choices must be a list")
            policy["choices"] = tuple(
                AugmentationPolicyChoiceConfig(**dict(choice_raw))
                for choice_raw in policy_choices
            )
        else:
            policy["choices"] = ()
        augmentation["policy"] = AugmentationPolicyConfig(**policy)
        waveform = dict(augmentation.get("waveform", {}))
        waveform["gain"] = RandomGainConfig(**dict(waveform.get("gain", {})))
        waveform["noise"] = AdditiveNoiseConfig(**dict(waveform.get("noise", {})))
        waveform["time_shift"] = TimeShiftConfig(**dict(waveform.get("time_shift", {})))
        augmentation["waveform"] = WaveformAugmentationConfig(**waveform)
        fbank = dict(augmentation.get("fbank", {}))
        fbank["time_mask"] = MaskConfig(**dict(fbank.get("time_mask", {})))
        fbank["freq_mask"] = MaskConfig(**dict(fbank.get("freq_mask", {})))
        augmentation["fbank"] = FbankAugmentationConfig(**fbank)
        kwargs["augmentation"] = DataAugmentationConfig(**augmentation)
        cfg = DataConfig(**kwargs)
        JsonConfigLoader._validate_data(cfg)
        return cfg

    @staticmethod
    def _coerce_pair(
        values: Sequence[object],
        *,
        field_name: str,
    ) -> tuple[int, int]:
        if len(values) != 2:
            raise ValueError(f"{field_name} must contain exactly two integers")
        first, second = values
        if not isinstance(first, int) or not isinstance(second, int):
            raise TypeError(f"{field_name} must contain integers")
        return first, second

    @staticmethod
    def _coerce_int_tuple(values: object, *, field_name: str) -> tuple[int, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"{field_name} must be a list of integers")
        result: list[int] = []
        for value in values:
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must contain integers")
            result.append(value)
        return tuple(result)

    @staticmethod
    def _coerce_float_tuple(values: object, *, field_name: str) -> tuple[float, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"{field_name} must be a list of numbers")
        result: list[float] = []
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must contain numbers")
            result.append(float(value))
        return tuple(result)

    @staticmethod
    def _parse_patch_branch(raw: Mapping[str, Any]) -> PatchBranchConfig:
        patch_size_raw = raw.get("patch_size")
        stride_raw = raw.get("stride")
        if not isinstance(patch_size_raw, Sequence) or isinstance(
            patch_size_raw, (str, bytes)
        ):
            raise TypeError("patch_size must be a 2-item list")
        if not isinstance(stride_raw, Sequence) or isinstance(stride_raw, (str, bytes)):
            raise TypeError("stride must be a 2-item list")
        return PatchBranchConfig(
            patch_size=JsonConfigLoader._coerce_pair(
                patch_size_raw,
                field_name="patch_size",
            ),
            stride=JsonConfigLoader._coerce_pair(
                stride_raw,
                field_name="stride",
            ),
        )

    @staticmethod
    def _parse_evidence_pooling(
        raw: Mapping[str, Any],
        *,
        label_to_index: Mapping[str, int],
    ) -> EvidencePoolingConfig:
        kwargs = dict(raw)
        class_gate = dict(kwargs.get("class_gate", {}))
        global_residual = dict(class_gate.get("global_residual", {}))
        global_residual["warmup"] = ClassGateGlobalResidualWarmupConfig(
            **dict(global_residual.get("warmup", {}))
        )
        global_residual["bounding"] = ClassGateGlobalResidualBoundingConfig(
            **dict(global_residual.get("bounding", {}))
        )
        correction = dict(global_residual.get("correction", {}))
        correction["confidence_aware_gate"] = (
            ClassGateResidualConfidenceAwareGateConfig(
                **dict(correction.get("confidence_aware_gate", {}))
            )
        )
        global_residual["correction"] = ClassGateGlobalResidualCorrectionConfig(
            **correction
        )
        class_gate["global_residual"] = ClassGateGlobalResidualConfig(**global_residual)
        evidence_scorer = dict(class_gate.get("evidence_scorer", {}))
        score_decomposition = dict(evidence_scorer.get("score_decomposition", {}))
        score_decomposition["interaction_scale_schedule"] = (
            ClassGateInteractionScaleScheduleConfig(
                **dict(score_decomposition.get("interaction_scale_schedule", {}))
            )
        )
        score_bounding = dict(score_decomposition.get("score_bounding", {}))
        score_bounding["embedding"] = ClassGateScoreBoundComponentConfig(
            **dict(score_bounding.get("embedding", {}))
        )
        score_bounding["interaction"] = ClassGateScoreBoundComponentConfig(
            **dict(score_bounding.get("interaction", {}))
        )
        score_decomposition["score_bounding"] = ClassGateScoreBoundingConfig(
            **score_bounding
        )
        evidence_scorer["score_decomposition"] = ClassGateScoreDecompositionConfig(
            **score_decomposition
        )
        branch_direct_score = dict(evidence_scorer.get("branch_direct_score", {}))
        branch_direct_score["gate_reliability_mixture"] = (
            ClassGateReliabilityMixtureConfig(
                **dict(branch_direct_score.get("gate_reliability_mixture", {}))
            )
        )
        branch_direct_score["top_relative_correction"] = (
            ClassGateTopRelativeCorrectionConfig(
                **dict(branch_direct_score.get("top_relative_correction", {}))
            )
        )
        top_support_direct_path = dict(
            branch_direct_score.get("top_support_direct_path", {})
        )
        negative_relative_cap = dict(
            top_support_direct_path.get("negative_relative_cap", {})
        )
        negative_relative_cap["max_negative_fraction_by_label"] = dict(
            negative_relative_cap.get("max_negative_fraction_by_label", {})
        )
        negative_relative_cap["negative_cap_by_label"] = dict(
            negative_relative_cap.get("negative_cap_by_label", {})
        )
        negative_relative_cap["max_negative_fraction_by_class"] = (
            resolve_label_float_overrides(
                default=float(negative_relative_cap.get("max_negative_fraction", 0.75)),
                overrides=negative_relative_cap["max_negative_fraction_by_label"],
                label_to_index=label_to_index,
            )
        )
        negative_relative_cap["negative_cap_by_class"] = resolve_label_float_overrides(
            default=float(negative_relative_cap.get("negative_cap", 1.5)),
            overrides=negative_relative_cap["negative_cap_by_label"],
            label_to_index=label_to_index,
        )
        top_support_direct_path["negative_relative_cap"] = (
            ClassGateTopSupportNegativeRelativeCapConfig(**negative_relative_cap)
        )
        branch_direct_score["top_support_direct_path"] = (
            ClassGateTopSupportDirectPathConfig(**top_support_direct_path)
        )
        evidence_scorer["branch_direct_score"] = ClassGateBranchDirectScoreConfig(
            **branch_direct_score
        )
        evidence_scorer["branch_feature_transform"] = (
            ClassGateBranchFeatureTransformConfig(
                **dict(evidence_scorer.get("branch_feature_transform", {}))
            )
        )
        class_gate["evidence_scorer"] = ClassGateEvidenceScorerConfig(**evidence_scorer)
        class_gate["evidence_auxiliary"] = ClassGateEvidenceAuxiliaryConfig(
            **dict(class_gate.get("evidence_auxiliary", {}))
        )
        class_gate["branch_logit_feature"] = ClassGateBranchLogitFeatureConfig(
            **dict(class_gate.get("branch_logit_feature", {}))
        )
        class_gate["gate_mixing"] = ClassGateMixingConfig(
            **dict(class_gate.get("gate_mixing", {}))
        )
        kwargs["class_gate"] = ClassGateConfig(**class_gate)
        return EvidencePoolingConfig(**kwargs)

    @staticmethod
    def _parse_model(raw: Mapping[str, Any], data_cfg: DataConfig) -> ModelConfig:
        kwargs = dict(raw)
        encoder = dict(raw["encoder"])
        architecture = dict(encoder.get("architecture", {}))
        if "latent_query_count" in architecture:
            raise ValueError(
                "latent_query_count is deprecated in the event-MIL architecture. "
                "Use model.encoder.architecture.rdt instead."
            )
        if "summary_tokens_per_scale" in architecture:
            raise ValueError(
                "summary_tokens_per_scale is deprecated in the event-MIL architecture. "
                "Use model.encoder.architecture.rdt.top_tokens_per_branch instead."
            )
        if "rdt_steps" in architecture:
            raise ValueError(
                "Flat rdt_steps is deprecated in the event-MIL architecture. "
                "Use model.encoder.architecture.rdt.steps instead."
            )
        patch_branches_raw = architecture.get("patch_branches")
        if patch_branches_raw is not None:
            if not isinstance(patch_branches_raw, Sequence) or isinstance(
                patch_branches_raw, (str, bytes)
            ):
                raise TypeError(
                    "model.encoder.architecture.patch_branches must be a list"
                )
            architecture["patch_branches"] = tuple(
                JsonConfigLoader._parse_patch_branch(dict(branch_raw))
                for branch_raw in patch_branches_raw
            )
        rdt = dict(architecture.get("rdt", {}))
        if "exclude_branches_from_evidence" in rdt:
            rdt["exclude_branches_from_evidence"] = JsonConfigLoader._coerce_int_tuple(
                rdt["exclude_branches_from_evidence"],
                field_name=(
                    "model.encoder.architecture.rdt.exclude_branches_from_evidence"
                ),
            )
        architecture["rdt"] = RdtConfig(**rdt)
        architecture["mil"] = MilConfig(**dict(architecture.get("mil", {})))
        architecture["evidence_pooling"] = JsonConfigLoader._parse_evidence_pooling(
            dict(architecture.get("evidence_pooling", {})),
            label_to_index=data_cfg.label_to_index,
        )
        token_augmentation = dict(architecture.get("token_augmentation", {}))
        token_augmentation["branch_event_dropout"] = BranchEventDropoutConfig(
            **dict(token_augmentation.get("branch_event_dropout", {}))
        )
        token_augmentation["selected_evidence_dropout"] = SelectedEvidenceDropoutConfig(
            **dict(token_augmentation.get("selected_evidence_dropout", {}))
        )
        architecture["token_augmentation"] = TokenAugmentationConfig(
            **token_augmentation
        )
        encoder["adaptation"] = EncoderAdaptationConfig(
            **dict(encoder.get("adaptation", {}))
        )
        encoder["architecture"] = MultiScaleRdtArchitectureConfig(**architecture)
        kwargs["encoder"] = ModelEncoderConfig(**encoder)
        classifier = dict(raw["classifier"])
        classifier["fusion_projector"] = FusionProjectorConfig(
            **dict(classifier.get("fusion_projector", {}))
        )
        kwargs["classifier"] = ClassifierConfig(**classifier)
        cfg = ModelConfig(**kwargs)
        JsonConfigLoader._validate_model(cfg, data_cfg)
        return cfg

    @staticmethod
    def _parse_weak_positive_support_bands(
        raw: Mapping[str, Any] | None,
    ) -> dict[str, WeakPositiveSupportBandConfig]:
        if raw is None:
            return {}
        return {
            str(label_name): WeakPositiveSupportBandConfig(**dict(value))
            for label_name, value in dict(raw).items()
        }

    @staticmethod
    def _parse_weak_positive_support_weighting(
        raw: Mapping[str, Any],
    ) -> WeakPositiveSupportWeightingConfig:
        data = dict(raw)
        data["support_band_by_label"] = (
            JsonConfigLoader._parse_weak_positive_support_bands(
                data.get("support_band_by_label", {})
            )
        )
        return WeakPositiveSupportWeightingConfig(**data)

    @staticmethod
    def _parse_weak_positive_boost(
        raw: Mapping[str, Any],
        *,
        config_type: type[Any],
    ) -> WeakPositiveTargetBoostConfig | WeakPositiveMarginBoostConfig:
        data = dict(raw)
        data["boost_by_label"] = dict(data.get("boost_by_label", {}))
        data["support_band_by_label"] = (
            JsonConfigLoader._parse_weak_positive_support_bands(
                data.get("support_band_by_label", {})
            )
        )
        return config_type(**data)

    @staticmethod
    def _parse_margin_hardness_weighting_by_label(
        raw: Mapping[str, Any] | None,
        *,
        defaults: MarginHardnessWeightingConfig | None = None,
    ) -> dict[str, MarginHardnessWeightingConfig]:
        if raw is None:
            return {}
        base = defaults or MarginHardnessWeightingConfig()
        parsed: dict[str, MarginHardnessWeightingConfig] = {}
        for label_name, value in dict(raw).items():
            data: dict[str, Any] = {
                "enabled": base.enabled,
                "source": base.source,
                "mode": base.mode,
                "gain": base.gain,
                "cap": base.cap,
            }
            data.update(dict(value))
            parsed[str(label_name)] = MarginHardnessWeightingConfig(**data)
        return parsed

    @staticmethod
    def _parse_train(raw: Mapping[str, Any]) -> TrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = OptimizerConfig(**dict(raw["optimizer"]))
        kwargs["scheduler"] = SchedulerConfig(**dict(raw.get("scheduler", {})))
        loss = dict(raw.get("loss", {}))
        loss["class_weighting"] = ClassWeightingConfig(
            **dict(loss.get("class_weighting", {}))
        )
        loss["label_smoothing"] = LabelSmoothingConfig(
            **dict(loss.get("label_smoothing", {}))
        )
        branch_auxiliary = dict(loss.get("branch_auxiliary", {}))
        if "weights" in branch_auxiliary and branch_auxiliary["weights"] is not None:
            branch_auxiliary["weights"] = JsonConfigLoader._coerce_float_tuple(
                branch_auxiliary["weights"],
                field_name="train.loss.branch_auxiliary.weights",
            )
        loss["branch_auxiliary"] = BranchAuxiliaryLossConfig(**branch_auxiliary)
        branch_binary_auxiliary = dict(loss.get("branch_binary_auxiliary", {}))
        branch_binary_auxiliary["label_to_index"] = dict(
            branch_binary_auxiliary.get("label_to_index", {})
        )
        branch_binary_auxiliary["pos_weight"] = BranchBinaryPosWeightConfig(
            **dict(branch_binary_auxiliary.get("pos_weight", {}))
        )
        branch_binary_auxiliary["schedule"] = BranchBinaryAuxiliaryScheduleConfig(
            **dict(branch_binary_auxiliary.get("schedule", {}))
        )
        branch_binary_auxiliary["monitor"] = BranchBinaryAuxiliaryMonitorConfig(
            **dict(branch_binary_auxiliary.get("monitor", {}))
        )
        loss["branch_binary_auxiliary"] = BranchBinaryAuxiliaryLossConfig(
            **branch_binary_auxiliary
        )
        loss["attention_entropy"] = AttentionEntropyLossConfig(
            **dict(loss.get("attention_entropy", {}))
        )
        loss["gate_entropy_regularization"] = GateEntropyRegularizationConfig(
            **dict(loss.get("gate_entropy_regularization", {}))
        )
        loss["class_gate_diversity_regularization"] = (
            ClassGateDiversityRegularizationConfig(
                **dict(loss.get("class_gate_diversity_regularization", {}))
            )
        )
        class_evidence_margin = dict(loss.get("class_evidence_margin", {}))
        class_evidence_margin["support_weighting"] = MarginSupportWeightingConfig(
            **dict(class_evidence_margin.get("support_weighting", {}))
        )
        loss["class_evidence_margin"] = ClassEvidenceMarginConfig(
            **class_evidence_margin
        )
        class_evidence_gap_cap = dict(
            loss.get("class_evidence_gap_cap_regularization", {})
        )
        class_evidence_gap_cap["negative_gap_cap_by_label"] = dict(
            class_evidence_gap_cap.get("negative_gap_cap_by_label", {})
        )
        class_evidence_gap_cap["label_weight_by_label"] = dict(
            class_evidence_gap_cap.get("label_weight_by_label", {})
        )
        loss["class_evidence_gap_cap_regularization"] = (
            ClassEvidenceGapCapRegularizationConfig(**class_evidence_gap_cap)
        )
        loss["class_evidence_positive_gap_cap_regularization"] = (
            ClassEvidencePositiveGapCapRegularizationConfig(
                **dict(
                    loss.get(
                        "class_evidence_positive_gap_cap_regularization",
                        {},
                    )
                )
            )
        )
        loss["interaction_gap_cap_regularization"] = (
            InteractionGapCapRegularizationConfig(
                **dict(loss.get("interaction_gap_cap_regularization", {}))
            )
        )
        top_support_score_margin = dict(loss.get("top_support_score_margin", {}))
        top_support_score_margin["label_weight_by_label"] = dict(
            top_support_score_margin.get("label_weight_by_label", {})
        )
        top_support_score_margin["support_conditioned_multiplier"] = (
            MarginSupportWeightingConfig(
                **dict(
                    top_support_score_margin.get(
                        "support_conditioned_multiplier",
                        {},
                    )
                )
            )
        )
        top_support_score_margin["hardness_weighting"] = MarginHardnessWeightingConfig(
            **dict(top_support_score_margin.get("hardness_weighting", {}))
        )
        loss["top_support_score_margin"] = TopSupportScoreMarginConfig(
            **top_support_score_margin
        )
        top_support_gap_min_constraint = dict(
            loss.get("top_support_gap_min_constraint", {})
        )
        top_support_gap_min_constraint["base_min_gap_by_label"] = dict(
            top_support_gap_min_constraint.get("base_min_gap_by_label", {})
        )
        loss["top_support_gap_min_constraint"] = TopSupportGapMinConstraintConfig(
            **top_support_gap_min_constraint
        )
        class_top_branch_relative_margin = dict(
            loss.get("class_top_branch_relative_margin", {})
        )
        class_top_branch_relative_margin["margin_by_label"] = dict(
            class_top_branch_relative_margin.get("margin_by_label", {})
        )
        class_top_branch_relative_margin["support_weighting"] = (
            MarginSupportWeightingConfig(
                **dict(
                    class_top_branch_relative_margin.get(
                        "support_weighting",
                        {},
                    )
                )
            )
        )
        class_top_branch_relative_margin["hardness_weighting"] = (
            MarginHardnessWeightingConfig(
                **dict(
                    class_top_branch_relative_margin.get(
                        "hardness_weighting",
                        {},
                    )
                )
            )
        )
        class_top_branch_relative_margin["hardness_weighting_by_label"] = (
            JsonConfigLoader._parse_margin_hardness_weighting_by_label(
                class_top_branch_relative_margin.get(
                    "hardness_weighting_by_label",
                    {},
                ),
                defaults=class_top_branch_relative_margin["hardness_weighting"],
            )
        )
        class_top_branch_relative_margin["weak_positive_support_weighting"] = (
            JsonConfigLoader._parse_weak_positive_support_weighting(
                dict(
                    class_top_branch_relative_margin.get(
                        "weak_positive_support_weighting",
                        {},
                    )
                )
            )
        )
        class_top_branch_relative_margin["weak_positive_margin_boost"] = (
            JsonConfigLoader._parse_weak_positive_boost(
                dict(
                    class_top_branch_relative_margin.get(
                        "weak_positive_margin_boost",
                        {},
                    )
                ),
                config_type=WeakPositiveMarginBoostConfig,
            )
        )
        loss["class_top_branch_relative_margin"] = ClassTopBranchRelativeMarginConfig(
            **class_top_branch_relative_margin
        )
        top_teacher_gap_min_constraint = dict(
            loss.get("top_teacher_gap_min_constraint", {})
        )
        top_teacher_gap_min_constraint["base_min_gap_by_label"] = dict(
            top_teacher_gap_min_constraint.get("base_min_gap_by_label", {})
        )
        top_teacher_gap_min_constraint["support_gain_by_label"] = dict(
            top_teacher_gap_min_constraint.get("support_gain_by_label", {})
        )
        top_teacher_gap_min_constraint["support_cap_by_label"] = dict(
            top_teacher_gap_min_constraint.get("support_cap_by_label", {})
        )
        top_teacher_gap_min_constraint["weak_positive_support_weighting"] = (
            JsonConfigLoader._parse_weak_positive_support_weighting(
                dict(
                    top_teacher_gap_min_constraint.get(
                        "weak_positive_support_weighting",
                        {},
                    )
                )
            )
        )
        top_teacher_gap_min_constraint["weak_positive_target_boost"] = (
            JsonConfigLoader._parse_weak_positive_boost(
                dict(
                    top_teacher_gap_min_constraint.get(
                        "weak_positive_target_boost",
                        {},
                    )
                ),
                config_type=WeakPositiveTargetBoostConfig,
            )
        )
        loss["top_teacher_gap_min_constraint"] = TopTeacherGapMinConstraintConfig(
            **top_teacher_gap_min_constraint
        )
        loss["branch_support_score_margin"] = BranchSupportScoreMarginConfig(
            **dict(loss.get("branch_support_score_margin", {}))
        )
        loss["branch_direct_score_margin"] = BranchDirectScoreMarginConfig(
            **dict(loss.get("branch_direct_score_margin", {}))
        )
        branch_path_dominance = dict(loss.get("branch_path_dominance_constraint", {}))
        branch_path_dominance["allowed_drop_by_label"] = dict(
            branch_path_dominance.get("allowed_drop_by_label", {})
        )
        branch_path_dominance["label_weight_by_label"] = dict(
            branch_path_dominance.get("label_weight_by_label", {})
        )
        branch_path_dominance["support_weighting"] = MarginSupportWeightingConfig(
            **dict(branch_path_dominance.get("support_weighting", {}))
        )
        loss["branch_path_dominance_constraint"] = BranchPathDominanceConstraintConfig(
            **branch_path_dominance
        )
        loss["branch_support_disagreement_cap_regularization"] = (
            BranchSupportDisagreementCapRegularizationConfig(
                **dict(
                    loss.get(
                        "branch_support_disagreement_cap_regularization",
                        {},
                    )
                )
            )
        )
        loss["class_gated_branch_logit_margin"] = ClassGatedBranchLogitMarginConfig(
            **dict(loss.get("class_gated_branch_logit_margin", {}))
        )
        branch_to_evidence = dict(
            loss.get("branch_to_evidence_ranking_consistency", {})
        )
        branch_to_evidence["teacher_floor_by_label"] = dict(
            branch_to_evidence.get("teacher_floor_by_label", {})
        )
        branch_to_evidence["support_weighting"] = MarginSupportWeightingConfig(
            **dict(branch_to_evidence.get("support_weighting", {}))
        )
        branch_to_evidence["hardness_weighting"] = MarginHardnessWeightingConfig(
            **dict(branch_to_evidence.get("hardness_weighting", {}))
        )
        loss["branch_to_evidence_ranking_consistency"] = (
            BranchToEvidenceRankingConsistencyConfig(**branch_to_evidence)
        )
        loss["global_residual_anti_veto"] = GlobalResidualAntiVetoConfig(
            **dict(loss.get("global_residual_anti_veto", {}))
        )
        loss["residual_contradiction_regularization"] = (
            ResidualContradictionRegularizationConfig(
                **dict(loss.get("residual_contradiction_regularization", {}))
            )
        )
        if "gate_branch_alignment" in loss:
            raise ValueError(
                "train.loss.gate_branch_alignment is no longer supported; "
                "use train.loss.gate_weighted_branch_margin"
            )
        gate_weighted_branch_margin = dict(loss.get("gate_weighted_branch_margin", {}))
        if "top_k" in gate_weighted_branch_margin:
            raise ValueError(
                "train.loss.gate_weighted_branch_margin.top_k is no longer "
                "supported; use branch_selection='gate_weighted'"
            )
        loss["gate_weighted_branch_margin"] = GateWeightedBranchMarginConfig(
            **gate_weighted_branch_margin
        )
        gate_best_branch_alignment = dict(loss.get("gate_best_branch_alignment", {}))
        gate_best_branch_alignment["label_weight_by_label"] = dict(
            gate_best_branch_alignment.get("label_weight_by_label", {})
        )
        gate_best_branch_alignment["min_best_margin_by_label"] = dict(
            gate_best_branch_alignment.get("min_best_margin_by_label", {})
        )
        gate_best_branch_alignment["max_best_margin_by_label"] = dict(
            gate_best_branch_alignment.get("max_best_margin_by_label", {})
        )
        gate_best_branch_alignment["mismatch_margin_drop_by_label"] = dict(
            gate_best_branch_alignment.get("mismatch_margin_drop_by_label", {})
        )
        loss["gate_best_branch_alignment"] = GateBestBranchAlignmentConfig(
            **gate_best_branch_alignment
        )
        gate_branch_regret = dict(loss.get("gate_branch_regret", {}))
        gate_branch_regret["positive_threshold_by_label"] = dict(
            gate_branch_regret.get("positive_threshold_by_label", {})
        )
        gate_branch_regret["weight_schedule"] = GateBranchRegretWeightScheduleConfig(
            **dict(gate_branch_regret.get("weight_schedule", {}))
        )
        gate_branch_regret["auto_positive_threshold_by_train_stats"] = (
            AutoPositiveThresholdByTrainStatsConfig(
                **dict(
                    gate_branch_regret.get(
                        "auto_positive_threshold_by_train_stats",
                        {},
                    )
                )
            )
        )
        loss["gate_branch_regret"] = GateBranchRegretConfig(**gate_branch_regret)
        gate_bad_branch_suppression = dict(loss.get("gate_bad_branch_suppression", {}))
        gate_bad_branch_suppression["bad_margin_threshold_by_label"] = dict(
            gate_bad_branch_suppression.get("bad_margin_threshold_by_label", {})
        )
        gate_bad_branch_suppression["weight_schedule"] = (
            GateBranchRegretWeightScheduleConfig(
                **dict(gate_bad_branch_suppression.get("weight_schedule", {}))
            )
        )
        gate_bad_branch_suppression["auto_bad_margin_threshold_by_train_stats"] = (
            AutoBadBranchThresholdByTrainStatsConfig(
                **dict(
                    gate_bad_branch_suppression.get(
                        "auto_bad_margin_threshold_by_train_stats",
                        {},
                    )
                )
            )
        )
        loss["gate_bad_branch_suppression"] = GateBadBranchSuppressionConfig(
            **gate_bad_branch_suppression
        )
        top_branch_margin = dict(loss.get("top_branch_margin", {}))
        top_branch_margin["margin_by_label"] = dict(
            top_branch_margin.get("margin_by_label", {})
        )
        top_branch_margin["auto_margin_by_train_stats"] = AutoMarginByTrainStatsConfig(
            **dict(top_branch_margin.get("auto_margin_by_train_stats", {}))
        )
        phase_weight_schedule = dict(top_branch_margin.get("phase_weight_schedule", {}))
        phase_weight_schedule["start_multiplier_by_label"] = dict(
            phase_weight_schedule.get("start_multiplier_by_label", {})
        )
        phase_weight_schedule["label_multiplier_by_label"] = dict(
            phase_weight_schedule.get("label_multiplier_by_label", {})
        )
        top_branch_margin["phase_weight_schedule"] = (
            TopBranchMarginPhaseWeightScheduleConfig(**phase_weight_schedule)
        )
        top_branch_margin["hardness_weighting"] = (
            TopBranchMarginHardnessWeightingConfig(
                **dict(top_branch_margin.get("hardness_weighting", {}))
            )
        )
        loss["top_branch_margin"] = TopBranchMarginConfig(**top_branch_margin)
        kwargs["loss"] = LossConfig(**loss)
        kwargs["sampler"] = SamplerConfig(**dict(raw.get("sampler", {})))
        kwargs["early_stopping"] = EarlyStoppingConfig(
            **dict(raw.get("early_stopping", {}))
        )
        kwargs["initialization"] = TrainingInitializationConfig(
            **dict(raw.get("initialization", {}))
        )
        return TrainConfig(**kwargs)

    @staticmethod
    def _parse_val(raw: Mapping[str, Any] | None) -> ValidationConfig:
        if raw is None:
            return ValidationConfig()
        kwargs = dict(raw)
        kwargs["loss"] = ValidationLossConfig(**dict(raw.get("loss", {})))
        cfg = ValidationConfig(**kwargs)
        JsonConfigLoader._validate_val(cfg)
        return cfg

    @staticmethod
    def _parse_checkpointing(
        raw: Mapping[str, Any] | None,
    ) -> CheckpointingConfig | None:
        if raw is None:
            return None
        monitors_raw = raw.get("monitors", ())
        if not isinstance(monitors_raw, Sequence) or isinstance(
            monitors_raw, (str, bytes)
        ):
            raise TypeError("checkpointing.monitors must be a list")
        monitors: list[CheckpointMonitorConfig] = []
        for monitor_raw in monitors_raw:
            monitor_kwargs = dict(monitor_raw)
            keep_top_k = monitor_kwargs.pop("keep_top_k", None)
            if keep_top_k is not None:
                if "top_k" in monitor_kwargs and monitor_kwargs["top_k"] != keep_top_k:
                    raise ValueError(
                        "checkpointing.monitors.keep_top_k and top_k must match "
                        "when both are provided"
                    )
                monitor_kwargs["top_k"] = keep_top_k
            monitors.append(CheckpointMonitorConfig(**monitor_kwargs))
        cfg = CheckpointingConfig(monitors=tuple(monitors))
        JsonConfigLoader._validate_checkpointing(cfg)
        return cfg

    @staticmethod
    def _parse_analysis(raw: Mapping[str, Any] | None) -> AnalysisConfig:
        if raw is None:
            return AnalysisConfig()
        kwargs = dict(raw)
        kwargs["outputs"] = AnalysisOutputConfig(**dict(raw.get("outputs", {})))
        return AnalysisConfig(**kwargs)

    @staticmethod
    def _parse_threshold_optimization(
        raw: Mapping[str, Any] | None,
    ) -> ThresholdOptimizationConfig:
        cfg = ThresholdOptimizationConfig(**dict(raw or {}))
        if cfg.metric not in JsonConfigLoader._THRESHOLD_METRICS:
            raise ValueError(
                "threshold_optimization.metric must be one of "
                f"{sorted(JsonConfigLoader._THRESHOLD_METRICS)}"
            )
        return cfg

    @staticmethod
    def load_training(path: str | Path) -> TrainingRunConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_data(raw["data"])
        model_cfg = JsonConfigLoader._parse_model(raw["model"], data_cfg)
        cfg = TrainingRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=data_cfg,
            model=model_cfg,
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            val=JsonConfigLoader._parse_val(raw.get("val")),
            checkpointing=JsonConfigLoader._parse_checkpointing(
                raw.get("checkpointing")
            ),
        )
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
            num_branches=len(cfg.model.encoder.architecture.patch_branches),
            label_to_index=cfg.data.label_to_index,
            evidence_pooling_type=cfg.model.encoder.architecture.evidence_pooling.type,
            evidence_scorer_type=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.type
            ),
            score_decomposition_enabled=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.score_decomposition.enabled
            ),
            branch_direct_score_enabled=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_direct_score.enabled
            ),
        )
        JsonConfigLoader._validate_val(cfg.val)
        JsonConfigLoader._validate_checkpointing(cfg.checkpointing)
        return cfg

    @staticmethod
    def load_eval(path: str | Path) -> EvalConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_data(raw["data"])
        return EvalConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            data=data_cfg,
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            threshold_optimization=JsonConfigLoader._parse_threshold_optimization(
                raw.get("threshold_optimization")
            ),
        )

    @staticmethod
    def _parse_fold(raw: Mapping[str, Any]) -> CvFoldConfig:
        cfg = CvFoldConfig(**dict(raw))
        if not cfg.name:
            raise ValueError("fold.name must not be empty")
        return cfg

    @staticmethod
    def load_cv(path: str | Path) -> CvRunConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_data(raw["data"])
        model_cfg = JsonConfigLoader._parse_model(raw["model"], data_cfg)
        cfg = CvRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=data_cfg,
            model=model_cfg,
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            folds=[JsonConfigLoader._parse_fold(item) for item in raw["folds"]],
            val=JsonConfigLoader._parse_val(raw.get("val")),
            checkpointing=JsonConfigLoader._parse_checkpointing(
                raw.get("checkpointing")
            ),
        )
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
            num_branches=len(cfg.model.encoder.architecture.patch_branches),
            label_to_index=cfg.data.label_to_index,
            evidence_pooling_type=cfg.model.encoder.architecture.evidence_pooling.type,
            evidence_scorer_type=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.type
            ),
            score_decomposition_enabled=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.score_decomposition.enabled
            ),
            branch_direct_score_enabled=(
                cfg.model.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_direct_score.enabled
            ),
        )
        JsonConfigLoader._validate_val(cfg.val)
        JsonConfigLoader._validate_checkpointing(cfg.checkpointing)
        return cfg
