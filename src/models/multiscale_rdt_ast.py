from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.classifier import ClassifierDims, LinearClassifier, MlpClassifier

PATCH_INIT_STD = 0.02


def _resolve_hold_decay_schedule(
    *,
    enabled: bool,
    start_value: float,
    end_value: float,
    hold_epochs: int,
    decay_epochs: int,
    epoch: int | None,
) -> float:
    if not enabled:
        return end_value
    if epoch is None:
        return end_value
    current_epoch = int(epoch)
    if current_epoch <= int(hold_epochs):
        return start_value
    if int(decay_epochs) <= 0:
        return end_value
    if int(decay_epochs) == 1:
        return end_value
    decay_epoch = current_epoch - int(hold_epochs)
    if decay_epoch >= int(decay_epochs):
        return end_value
    progress = float(decay_epoch - 1) / float(int(decay_epochs) - 1)
    return start_value + ((end_value - start_value) * progress)


def _sigmoid_max_parameter_init(init_value: float, max_value: float) -> Tensor:
    ratio = float(init_value) / float(max_value)
    ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def _bounded_sigmoid_parameter_init(
    init_value: float,
    min_value: float,
    max_value: float,
) -> Tensor:
    ratio = (float(init_value) - float(min_value)) / (
        float(max_value) - float(min_value)
    )
    ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def _softplus_parameter_init(init_value: float) -> Tensor:
    value = max(float(init_value), 1e-6)
    return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32)))


@dataclass(frozen=True)
class AstFeatureDims:
    num_mel_bins: int
    max_length: int


@dataclass(frozen=True)
class PatchBranchConfig:
    patch_size: tuple[int, int]
    stride: tuple[int, int]


def default_patch_branches() -> tuple[PatchBranchConfig, ...]:
    return (
        PatchBranchConfig(patch_size=(16, 16), stride=(8, 16)),
        PatchBranchConfig(patch_size=(8, 32), stride=(4, 32)),
        PatchBranchConfig(patch_size=(4, 64), stride=(2, 64)),
        PatchBranchConfig(patch_size=(2, 128), stride=(1, 128)),
    )


@dataclass(frozen=True)
class EncoderAdaptationConfig:
    mode: Literal["frozen", "full", "partial"] = "full"
    num_layers: int = 0


@dataclass(frozen=True)
class RdtConfig:
    enabled: bool = False
    steps: int = 3
    top_tokens_per_branch: int = 2
    gated_residual: bool = True
    layerscale_init: float = 0.01
    evidence_score_source: Literal[
        "attention_weight", "attention_logit", "instance_logit"
    ] = "attention_weight"
    exclude_branches_from_evidence: tuple[int, ...] = ()


@dataclass(frozen=True)
class MilConfig:
    attention_temperature: float = 1.0


@dataclass(frozen=True)
class ClassGateGlobalResidualWarmupConfig:
    enabled: bool = False
    mode: Literal["zero_to_learned"] = "zero_to_learned"
    start_multiplier: float = 0.0
    end_multiplier: float = 1.0
    hold_epochs: int = 0
    decay_epochs: int = 0


@dataclass(frozen=True)
class ClassGateGlobalResidualBoundingConfig:
    enabled: bool = False
    bound: float = 1.0
    temperature: float = 1.0


@dataclass(frozen=True)
class ClassGateResidualConfidenceAwareGateConfig:
    enabled: bool = False
    source: Literal["evidence_gap"] = "evidence_gap"
    mode: Literal["damped_sigmoid"] = "damped_sigmoid"
    temperature: float = 1.0
    damping: float = 0.0


@dataclass(frozen=True)
class ClassGateGlobalResidualCorrectionConfig:
    mode: Literal["additive", "gated_zero_mean"] = "additive"
    gate_hidden_size: int = 128
    dropout: float = 0.05
    zero_mean: bool = True
    rebound: bool = True
    confidence_aware_gate: ClassGateResidualConfidenceAwareGateConfig = field(
        default_factory=ClassGateResidualConfidenceAwareGateConfig
    )


@dataclass(frozen=True)
class ClassGateGlobalResidualConfig:
    enabled: bool = True
    init_scale: float = 0.1
    learnable: bool = True
    warmup: ClassGateGlobalResidualWarmupConfig = field(
        default_factory=ClassGateGlobalResidualWarmupConfig
    )
    bounding: ClassGateGlobalResidualBoundingConfig = field(
        default_factory=ClassGateGlobalResidualBoundingConfig
    )
    correction: ClassGateGlobalResidualCorrectionConfig = field(
        default_factory=ClassGateGlobalResidualCorrectionConfig
    )


@dataclass(frozen=True)
class ClassGateEvidenceAuxiliaryConfig:
    enabled: bool = False
    weight: float = 0.1


@dataclass(frozen=True)
class ClassGateBranchLogitFeatureConfig:
    mode: Literal["raw", "hardest_negative_margin"] = "raw"


@dataclass(frozen=True)
class ClassGateBranchFeatureTransformConfig:
    mode: Literal["tanh"] = "tanh"
    temperature: float = 1.0


@dataclass(frozen=True)
class ClassGateInteractionScaleScheduleConfig:
    enabled: bool = False
    start_epoch: int = 1
    end_epoch: int = 1
    start_multiplier: float = 1.0
    end_multiplier: float = 1.0


@dataclass(frozen=True)
class ClassGateScoreBoundComponentConfig:
    enabled: bool = False
    mode: Literal["tanh_bound"] = "tanh_bound"
    bound: float = 1.0
    temperature: float = 1.0


@dataclass(frozen=True)
class ClassGateScoreBoundingConfig:
    enabled: bool = False
    embedding: ClassGateScoreBoundComponentConfig = field(
        default_factory=ClassGateScoreBoundComponentConfig
    )
    interaction: ClassGateScoreBoundComponentConfig = field(
        default_factory=ClassGateScoreBoundComponentConfig
    )


@dataclass(frozen=True)
class ClassGateScoreDecompositionConfig:
    enabled: bool = False
    branch_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "sigmoid_max"
    branch_scale_min: float = 0.0
    branch_scale_init: float = 0.3
    branch_scale_max: float = 1.0
    interaction_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "sigmoid_max"
    interaction_scale_min: float = 0.0
    interaction_scale_init: float = 1.0
    interaction_scale_max: float = 1.5
    interaction_scale_schedule: ClassGateInteractionScaleScheduleConfig = field(
        default_factory=ClassGateInteractionScaleScheduleConfig
    )
    score_bounding: ClassGateScoreBoundingConfig = field(
        default_factory=ClassGateScoreBoundingConfig
    )


@dataclass(frozen=True)
class ClassGateReliabilityMixtureConfig:
    enabled: bool = False
    source: Literal["top_vs_gated_margin_regret"] = "top_vs_gated_margin_regret"
    mode: Literal["exp_neg_regret"] = "exp_neg_regret"
    temperature: float = 1.0
    tolerance: float = 0.05
    detach: bool = True


@dataclass(frozen=True)
class ClassGateTopRelativeCorrectionConfig:
    enabled: bool = False
    positive_scale: float = 0.2
    negative_scale: float = 0.05
    negative_clip: float = 1.0


@dataclass(frozen=True)
class ClassGateTopSupportNegativeRelativeCapConfig:
    enabled: bool = False
    mode: Literal["raw_fraction_cap"] = "raw_fraction_cap"
    max_negative_fraction: float = 0.75
    negative_cap: float = 1.5
    max_negative_fraction_by_label: Mapping[str, float] = field(default_factory=dict)
    negative_cap_by_label: Mapping[str, float] = field(default_factory=dict)
    max_negative_fraction_by_class: tuple[float, ...] | None = None
    negative_cap_by_class: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ClassGateTopSupportDirectPathConfig:
    enabled: bool = False
    mode: Literal["monotonic_raw_relative"] = "monotonic_raw_relative"
    positive_transform: Literal["softplus"] = "softplus"
    raw_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "bounded_sigmoid"
    raw_scale_min: float = 0.7
    raw_scale_init: float = 1.0
    raw_scale_max: float = 2.0
    relative_positive_scale_mode: Literal[
        "sigmoid_max",
        "bounded_sigmoid",
    ] = "bounded_sigmoid"
    relative_positive_scale_min: float = 0.2
    relative_positive_scale_init: float = 0.5
    relative_positive_scale_max: float = 1.5
    relative_negative_scale_mode: Literal[
        "sigmoid_max",
        "bounded_sigmoid",
    ] = "sigmoid_max"
    relative_negative_scale_min: float = 0.0
    relative_negative_scale_init: float = 0.1
    relative_negative_scale_max: float = 0.5
    residual_hidden_size: int = 64
    residual_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "sigmoid_max"
    residual_scale_min: float = 0.0
    residual_scale_init: float = 0.05
    residual_scale_max: float = 0.3
    residual_bound: float = 1.0
    residual_temperature: float = 1.0
    negative_relative_cap: ClassGateTopSupportNegativeRelativeCapConfig = field(
        default_factory=ClassGateTopSupportNegativeRelativeCapConfig
    )


@dataclass(frozen=True)
class ClassGateBranchDirectScoreConfig:
    enabled: bool = False
    positive_weight_mode: Literal["softplus"] = "softplus"
    top_support_mode: Literal[
        "learned_weighted_sum",
        "raw_existential_plus_relative_correction",
    ] = "learned_weighted_sum"
    top_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "bounded_sigmoid"
    top_scale_min: float = 0.7
    top_scale_init: float = 1.0
    top_scale_max: float = 2.0
    gated_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "bounded_sigmoid"
    gated_scale_min: float = 0.0
    gated_scale_init: float = 0.5
    gated_scale_max: float = 1.5
    # Legacy aliases kept so older local configs/tests still deserialize. The
    # class-axis direct scorer uses top/gated scale fields.
    raw_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "bounded_sigmoid"
    raw_scale_min: float = 0.7
    raw_scale_init: float = 1.0
    raw_scale_max: float = 2.0
    relative_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "bounded_sigmoid"
    relative_scale_min: float = 0.0
    relative_scale_init: float = 0.3
    relative_scale_max: float = 1.0
    residual_hidden_size: int = 128
    residual_scale_mode: Literal["sigmoid_max", "bounded_sigmoid"] = "sigmoid_max"
    residual_scale_min: float = 0.0
    residual_scale_init: float = 0.2
    residual_scale_max: float = 0.5
    residual_bound: float = 1.0
    residual_temperature: float = 1.0
    gate_reliability_mixture: ClassGateReliabilityMixtureConfig = field(
        default_factory=ClassGateReliabilityMixtureConfig
    )
    top_relative_correction: ClassGateTopRelativeCorrectionConfig = field(
        default_factory=ClassGateTopRelativeCorrectionConfig
    )
    top_support_direct_path: ClassGateTopSupportDirectPathConfig = field(
        default_factory=ClassGateTopSupportDirectPathConfig
    )


@dataclass(frozen=True)
class ClassGateEvidenceScorerConfig:
    type: Literal[
        "embedding_mlp",
        "two_tower_mlp",
        "class_axis_attention",
    ] = "embedding_mlp"
    embedding_hidden_size: int = 512
    branch_hidden_size: int = 64
    fusion_hidden_size: int = 512
    num_attention_heads: int = 4
    num_attention_layers: int = 1
    dropout: float = 0.05
    use_class_embedding: bool = True
    logit_centering: bool = False
    score_decomposition: ClassGateScoreDecompositionConfig = field(
        default_factory=ClassGateScoreDecompositionConfig
    )
    branch_direct_score: ClassGateBranchDirectScoreConfig = field(
        default_factory=ClassGateBranchDirectScoreConfig
    )
    branch_feature_transform: ClassGateBranchFeatureTransformConfig = field(
        default_factory=ClassGateBranchFeatureTransformConfig
    )


@dataclass(frozen=True)
class ClassGateMixingConfig:
    enabled: bool = False
    mode: Literal["uniform_to_learned"] = "uniform_to_learned"
    start_alpha: float = 1.0
    end_alpha: float = 0.0
    hold_epochs: int = 0
    decay_epochs: int = 0


@dataclass(frozen=True)
class ClassGateConfig:
    mode: Literal["query"] = "query"
    scorer: Literal["diagonal", "normalized_mlp"] = "diagonal"
    scorer_hidden_size: int | None = None
    scorer_dropout: float = 0.0
    evidence_scorer: ClassGateEvidenceScorerConfig = field(
        default_factory=ClassGateEvidenceScorerConfig
    )
    gate_mixing: ClassGateMixingConfig = field(default_factory=ClassGateMixingConfig)
    global_residual: ClassGateGlobalResidualConfig = field(
        default_factory=ClassGateGlobalResidualConfig
    )
    evidence_auxiliary: ClassGateEvidenceAuxiliaryConfig = field(
        default_factory=ClassGateEvidenceAuxiliaryConfig
    )
    branch_logit_feature: ClassGateBranchLogitFeatureConfig = field(
        default_factory=ClassGateBranchLogitFeatureConfig
    )


@dataclass(frozen=True)
class EvidencePoolingConfig:
    type: Literal["mean", "branch_gated", "class_aware_branch_gated"] = "mean"
    gate_hidden_size: int | None = None
    dropout: float = 0.1
    temperature: float = 1.0
    class_gate: ClassGateConfig = field(default_factory=ClassGateConfig)


@dataclass(frozen=True)
class BranchEventDropoutConfig:
    enabled: bool = False
    probability: float = 0.0
    mode: Literal["zero_mask"] = "zero_mask"
    min_keep_tokens: int = 1


@dataclass(frozen=True)
class SelectedEvidenceDropoutConfig:
    enabled: bool = False
    probability: float = 0.0
    mode: Literal["zero"] = "zero"
    min_keep_per_branch: int = 1


@dataclass(frozen=True)
class TokenAugmentationConfig:
    branch_event_dropout: BranchEventDropoutConfig = field(
        default_factory=BranchEventDropoutConfig
    )
    selected_evidence_dropout: SelectedEvidenceDropoutConfig = field(
        default_factory=SelectedEvidenceDropoutConfig
    )


@dataclass(frozen=True)
class MultiScaleRdtArchitectureConfig:
    hidden_size: int = 192
    num_attention_heads: int = 4
    mlp_ratio: float = 2.0
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    shared_stem_depth: int = 2
    adapter_depth: int = 1
    patch_branches: tuple[PatchBranchConfig, ...] = field(
        default_factory=default_patch_branches
    )
    rdt: RdtConfig = field(default_factory=RdtConfig)
    mil: MilConfig = field(default_factory=MilConfig)
    evidence_pooling: EvidencePoolingConfig = field(
        default_factory=EvidencePoolingConfig
    )
    token_augmentation: TokenAugmentationConfig = field(
        default_factory=TokenAugmentationConfig
    )


@dataclass(frozen=True)
class MultiScaleRdtEncoderConfig:
    feature_dims: AstFeatureDims
    type: Literal["multiscale_rdt_ast"] = "multiscale_rdt_ast"
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: MultiScaleRdtArchitectureConfig = field(
        default_factory=MultiScaleRdtArchitectureConfig
    )


@dataclass(frozen=True)
class FusionProjectorConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 640
    layer_norm: bool = False


@dataclass(frozen=True)
class ClassifierConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0
    pooling: Literal["latent_mean"] = "latent_mean"
    fusion_projector: FusionProjectorConfig = field(
        default_factory=FusionProjectorConfig
    )


@dataclass(frozen=True)
class MultiScaleRdtAstModelConfig:
    encoder: MultiScaleRdtEncoderConfig
    classifier: ClassifierConfig
    num_classes: int


@dataclass(frozen=True)
class AstModelOutput:
    logits: Tensor
    pooled_embedding: Tensor
    branch_logits: Tensor | None = None
    branch_binary_logits: Tensor | None = None
    branch_attention_weights: tuple[Tensor, ...] | None = None
    selected_evidence_tokens: Tensor | None = None
    selected_evidence_indices: Tensor | None = None
    selected_evidence_scores: Tensor | None = None
    selected_evidence_branch_ids: Tensor | None = None
    evidence_score_source: str | None = None
    evidence_pooling_type: str | None = None
    evidence_gate_weights: Tensor | None = None
    evidence_gate_entropy: Tensor | None = None
    class_evidence_embeddings: Tensor | None = None
    class_evidence_logits: Tensor | None = None
    global_residual_logits: Tensor | None = None
    class_evidence_gate_weights: Tensor | None = None
    class_evidence_gate_entropy: Tensor | None = None
    class_evidence_learned_gate_weights: Tensor | None = None
    class_evidence_learned_gate_entropy: Tensor | None = None
    class_evidence_gate_mixing_alpha: Tensor | None = None
    class_gated_branch_logits: Tensor | None = None
    class_gated_branch_logit_features: Tensor | None = None
    class_gated_branch_logit_feature_mode: str | None = None
    class_top_branch_margin_features: Tensor | None = None
    class_gated_branch_logit_relative_features: Tensor | None = None
    class_top_branch_margin_relative_features: Tensor | None = None
    class_evidence_scorer_branch_raw_features: Tensor | None = None
    class_evidence_scorer_branch_features: Tensor | None = None
    class_evidence_attention_weights: Tensor | None = None
    class_evidence_embedding_scores: Tensor | None = None
    class_evidence_raw_embedding_scores: Tensor | None = None
    class_evidence_bounded_embedding_scores: Tensor | None = None
    class_evidence_top_support_scores: Tensor | None = None
    class_evidence_direct_top_scores: Tensor | None = None
    class_evidence_top_support_residual_scores: Tensor | None = None
    class_evidence_gated_support_scores: Tensor | None = None
    class_evidence_top_raw_existential_scores: Tensor | None = None
    class_evidence_top_relative_correction_scores: Tensor | None = None
    class_evidence_top_support_raw_positive_component: Tensor | None = None
    class_evidence_top_support_relative_positive_component: Tensor | None = None
    class_evidence_top_support_relative_negative_component: Tensor | None = None
    class_evidence_top_support_relative_negative_component_uncapped: Tensor | None = (
        None
    )
    class_evidence_top_support_relative_negative_component_capped: Tensor | None = None
    class_evidence_top_support_relative_negative_cap_value: Tensor | None = None
    class_evidence_top_support_relative_negative_cap_active: Tensor | None = None
    class_evidence_top_support_direct_raw_scale: Tensor | None = None
    class_evidence_top_support_direct_relative_positive_scale: Tensor | None = None
    class_evidence_top_support_direct_relative_negative_scale: Tensor | None = None
    class_evidence_top_support_direct_residual_scale: Tensor | None = None
    class_evidence_top_relative_positive: Tensor | None = None
    class_evidence_top_relative_negative: Tensor | None = None
    class_evidence_gate_reliability: Tensor | None = None
    class_evidence_gate_reliability_regret: Tensor | None = None
    class_evidence_top_margin: Tensor | None = None
    class_evidence_gated_margin: Tensor | None = None
    class_evidence_branch_existential_scores: Tensor | None = None
    class_evidence_branch_competitive_scores: Tensor | None = None
    class_evidence_branch_direct_scores: Tensor | None = None
    class_evidence_branch_residual_scores: Tensor | None = None
    class_evidence_branch_support_scores: Tensor | None = None
    class_evidence_interaction_scores: Tensor | None = None
    class_evidence_raw_interaction_scores: Tensor | None = None
    class_evidence_bounded_interaction_scores: Tensor | None = None
    class_evidence_branch_scale: Tensor | None = None
    class_evidence_branch_direct_top_scale: Tensor | None = None
    class_evidence_branch_direct_gated_scale: Tensor | None = None
    class_evidence_branch_direct_raw_scale: Tensor | None = None
    class_evidence_branch_direct_relative_scale: Tensor | None = None
    class_evidence_branch_direct_residual_scale: Tensor | None = None
    class_evidence_branch_direct_top_weights: Tensor | None = None
    class_evidence_branch_direct_gated_weights: Tensor | None = None
    class_evidence_branch_direct_existential_weights: Tensor | None = None
    class_evidence_branch_direct_competitive_weights: Tensor | None = None
    class_evidence_interaction_scale: Tensor | None = None
    class_evidence_interaction_scale_multiplier: Tensor | None = None
    class_evidence_interaction_effective_scale: Tensor | None = None
    class_evidence_embedding_score_bound: Tensor | None = None
    class_evidence_embedding_score_temperature: Tensor | None = None
    class_evidence_interaction_score_bound: Tensor | None = None
    class_evidence_interaction_score_temperature: Tensor | None = None
    class_evidence_scorer_type: str | None = None
    class_evidence_scorer_branch_feature_transform_mode: str | None = None
    class_evidence_scorer_branch_feature_transform_temperature: Tensor | None = None
    global_residual_scale: Tensor | None = None
    global_residual_schedule_multiplier: Tensor | None = None
    global_residual_effective_scale: Tensor | None = None
    bounded_global_residual_logits: Tensor | None = None
    centered_bounded_global_residual_logits: Tensor | None = None
    global_residual_learned_gate: Tensor | None = None
    global_residual_evidence_confidence: Tensor | None = None
    global_residual_confidence_factor: Tensor | None = None
    global_residual_gate: Tensor | None = None
    global_residual_contribution: Tensor | None = None
    global_residual_zero_mean_enabled: bool | None = None
    global_residual_rebound_enabled: bool | None = None
    global_residual_bound: Tensor | None = None
    global_residual_temperature: Tensor | None = None
    branch_evidence_norms: Tensor | None = None
    selected_evidence_dropout_mask: Tensor | None = None
    selected_evidence_keep_ratio: Tensor | None = None


@dataclass(frozen=True)
class MultiScaleEncoderOutput:
    branch_event_tokens: tuple[Tensor, ...]
    context_tokens: Tensor


@dataclass(frozen=True)
class BranchMilOutput:
    logits: Tensor
    attention_weights: Tensor
    attention_logits: Tensor
    instance_logits: Tensor | None
    embedding: Tensor


@dataclass(frozen=True)
class SelectedEvidence:
    tokens: Tensor
    indices: Tensor
    scores: Tensor


@dataclass(frozen=True)
class EvidencePoolingOutput:
    pooled_embedding: Tensor
    gate_weights: Tensor | None = None
    gate_entropy: Tensor | None = None
    class_evidence_embeddings: Tensor | None = None
    class_evidence_logits: Tensor | None = None
    class_gate_weights: Tensor | None = None
    class_gate_entropy: Tensor | None = None
    learned_class_gate_weights: Tensor | None = None
    learned_class_gate_entropy: Tensor | None = None
    class_gate_mixing_alpha: Tensor | None = None
    branch_evidence_summary: Tensor | None = None
    branch_evidence_norms: Tensor | None = None


@dataclass(frozen=True)
class ClassEvidenceScoreOutput:
    logits: Tensor
    embedding_scores: Tensor | None = None
    raw_embedding_scores: Tensor | None = None
    bounded_embedding_scores: Tensor | None = None
    top_support_scores: Tensor | None = None
    direct_top_scores: Tensor | None = None
    top_support_residual_scores: Tensor | None = None
    gated_support_scores: Tensor | None = None
    top_raw_existential_scores: Tensor | None = None
    top_relative_correction_scores: Tensor | None = None
    top_support_raw_positive_component: Tensor | None = None
    top_support_relative_positive_component: Tensor | None = None
    top_support_relative_negative_component: Tensor | None = None
    top_support_relative_negative_component_uncapped: Tensor | None = None
    top_support_relative_negative_component_capped: Tensor | None = None
    top_support_relative_negative_cap_value: Tensor | None = None
    top_support_relative_negative_cap_active: Tensor | None = None
    top_support_direct_raw_scale: Tensor | None = None
    top_support_direct_relative_positive_scale: Tensor | None = None
    top_support_direct_relative_negative_scale: Tensor | None = None
    top_support_direct_residual_scale: Tensor | None = None
    top_relative_positive: Tensor | None = None
    top_relative_negative: Tensor | None = None
    gate_reliability: Tensor | None = None
    gate_reliability_regret: Tensor | None = None
    top_margin: Tensor | None = None
    gated_margin: Tensor | None = None
    branch_existential_scores: Tensor | None = None
    branch_competitive_scores: Tensor | None = None
    branch_direct_scores: Tensor | None = None
    branch_residual_scores: Tensor | None = None
    branch_support_scores: Tensor | None = None
    interaction_scores: Tensor | None = None
    raw_interaction_scores: Tensor | None = None
    bounded_interaction_scores: Tensor | None = None
    branch_scale: Tensor | None = None
    branch_direct_top_scale: Tensor | None = None
    branch_direct_gated_scale: Tensor | None = None
    branch_direct_raw_scale: Tensor | None = None
    branch_direct_relative_scale: Tensor | None = None
    branch_direct_residual_scale: Tensor | None = None
    branch_direct_top_weights: Tensor | None = None
    branch_direct_gated_weights: Tensor | None = None
    branch_direct_existential_weights: Tensor | None = None
    branch_direct_competitive_weights: Tensor | None = None
    interaction_scale: Tensor | None = None
    interaction_scale_multiplier: Tensor | None = None
    interaction_effective_scale: Tensor | None = None
    embedding_score_bound: Tensor | None = None
    embedding_score_temperature: Tensor | None = None
    interaction_score_bound: Tensor | None = None
    interaction_score_temperature: Tensor | None = None


def compute_token_grid(
    *,
    feature_dims: AstFeatureDims,
    patch_branch: PatchBranchConfig,
) -> tuple[int, int]:
    patch_t, patch_f = patch_branch.patch_size
    stride_t, stride_f = patch_branch.stride
    n_t = ((feature_dims.max_length - patch_t) // stride_t) + 1
    n_f = ((feature_dims.num_mel_bins - patch_f) // stride_f) + 1
    return n_t, n_f


def compute_token_count(
    *,
    feature_dims: AstFeatureDims,
    patch_branch: PatchBranchConfig,
) -> int:
    n_t, n_f = compute_token_grid(
        feature_dims=feature_dims,
        patch_branch=patch_branch,
    )
    return n_t * n_f


def select_top_tokens(
    tokens: Tensor, scores: Tensor, *, top_k: int
) -> SelectedEvidence:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B, T, D), got {tuple(tokens.shape)}")
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B, T), got {tuple(scores.shape)}")
    if tokens.shape[:2] != scores.shape:
        raise ValueError(
            "scores must align with tokens on batch/time dimensions; "
            f"got tokens={tuple(tokens.shape)} scores={tuple(scores.shape)}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if top_k > int(tokens.shape[1]):
        raise ValueError(f"top_k={top_k} exceeds token length {int(tokens.shape[1])}")
    top = scores.topk(top_k, dim=1)
    top_indices = top.indices
    selected_tokens = tokens.gather(
        1,
        top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )
    return SelectedEvidence(
        tokens=selected_tokens,
        indices=top_indices,
        scores=top.values,
    )


def get_evidence_scores(
    mil_output: BranchMilOutput,
    *,
    source: str,
) -> Tensor:
    if source == "attention_weight":
        return mil_output.attention_weights
    if source == "attention_logit":
        return mil_output.attention_logits
    if source == "instance_logit":
        if mil_output.instance_logits is None:
            raise ValueError(
                "instance_logit evidence selection requires instance_logits"
            )
        if mil_output.instance_logits.ndim == 2:
            return mil_output.instance_logits
        if mil_output.instance_logits.ndim == 3:
            return mil_output.instance_logits.max(dim=-1).values
        raise ValueError(
            "instance_logits must have shape (B, T) or (B, T, C), "
            f"got {tuple(mil_output.instance_logits.shape)}"
        )
    raise ValueError(f"Unsupported evidence_score_source: {source}")


def summarize_evidence_by_branch(
    evidence_tokens: Tensor,
    branch_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    branch_values = torch.unique(branch_ids.detach().cpu()).sort().values.tolist()
    if not branch_values:
        raise ValueError("branch_ids must contain at least one branch")

    branch_summaries: list[Tensor] = []
    branch_present_masks: list[Tensor] = []
    for branch_value in branch_values:
        branch_mask = branch_ids == int(branch_value)
        branch_counts = branch_mask.sum(dim=1)
        branch_sum = (
            evidence_tokens * branch_mask.unsqueeze(-1).to(evidence_tokens.dtype)
        ).sum(dim=1)
        branch_summaries.append(branch_sum / branch_counts.clamp_min(1).unsqueeze(-1))
        branch_present_masks.append(branch_counts > 0)

    return (
        torch.stack(branch_summaries, dim=1),
        torch.stack(branch_present_masks, dim=1),
    )


class MeanEvidencePooling(nn.Module):
    def forward(
        self,
        evidence_tokens: Tensor,
        branch_ids: Tensor | None = None,
    ) -> EvidencePoolingOutput:
        del branch_ids
        if evidence_tokens.ndim != 3:
            raise ValueError(
                "evidence_tokens must have shape (B, K, D), "
                f"got {tuple(evidence_tokens.shape)}"
            )
        return EvidencePoolingOutput(pooled_embedding=evidence_tokens.mean(dim=1))


class BranchAwareGatedEvidencePooling(nn.Module):
    def __init__(self, *, hidden_size: int, cfg: EvidencePoolingConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        gate_hidden_size = cfg.gate_hidden_size or max(hidden_size // 2, 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(gate_hidden_size, 1),
        )

    def forward(
        self,
        evidence_tokens: Tensor,
        branch_ids: Tensor | None = None,
    ) -> EvidencePoolingOutput:
        if evidence_tokens.ndim != 3:
            raise ValueError(
                "evidence_tokens must have shape (B, K, D), "
                f"got {tuple(evidence_tokens.shape)}"
            )
        if branch_ids is None:
            raise ValueError("branch_gated evidence pooling requires branch_ids")
        if branch_ids.ndim != 2:
            raise ValueError(
                f"branch_ids must have shape (B, K), got {tuple(branch_ids.shape)}"
            )
        if branch_ids.shape != evidence_tokens.shape[:2]:
            raise ValueError(
                "branch_ids must align with evidence_tokens on batch/token "
                f"dimensions; got evidence_tokens={tuple(evidence_tokens.shape)} "
                f"branch_ids={tuple(branch_ids.shape)}"
            )

        branch_evidence_summary, branch_present_mask = summarize_evidence_by_branch(
            evidence_tokens,
            branch_ids,
        )
        if torch.any(branch_present_mask.sum(dim=1) == 0):
            raise ValueError("Every sample must contain at least one evidence branch")

        gate_logits = self.gate(branch_evidence_summary).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~branch_present_mask, float("-inf"))
        gate_weights = torch.softmax(gate_logits / self.temperature, dim=1)
        gate_entropy = -(
            gate_weights * (gate_weights + torch.finfo(gate_weights.dtype).eps).log()
        ).sum(dim=1)
        pooled_embedding = torch.sum(
            gate_weights.unsqueeze(-1) * branch_evidence_summary,
            dim=1,
        )
        branch_evidence_norms = branch_evidence_summary.norm(dim=-1)
        return EvidencePoolingOutput(
            pooled_embedding=pooled_embedding,
            gate_weights=gate_weights,
            gate_entropy=gate_entropy,
            branch_evidence_summary=branch_evidence_summary,
            branch_evidence_norms=branch_evidence_norms,
        )


class ClassAwareBranchGatedEvidencePooling(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_classes: int,
        cfg: EvidencePoolingConfig,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                "class_aware_branch_gated evidence pooling requires num_classes >= 2"
            )
        self.temperature = cfg.temperature
        self.num_classes = num_classes
        self.class_gate_scorer = cfg.class_gate.scorer
        self.gate_mixing = cfg.class_gate.gate_mixing
        self.evidence_scorer_cfg = cfg.class_gate.evidence_scorer
        self.evidence_scorer_type = self.evidence_scorer_cfg.type
        self.scorer_branch_feature_count = 0
        if self.evidence_scorer_type == "two_tower_mlp":
            self.scorer_branch_feature_count = 2
        elif self.evidence_scorer_type == "class_axis_attention":
            self.scorer_branch_feature_count = 4
        self._runtime_epoch: int | None = None
        gate_hidden_size = cfg.gate_hidden_size or max(hidden_size // 2, 1)
        self.branch_key = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(gate_hidden_size, hidden_size),
        )
        self.class_queries = nn.Parameter(torch.empty(num_classes, hidden_size))
        self.class_scorer_weight: nn.Parameter | None
        self.class_scorer_bias: nn.Parameter | None
        self.class_scorers: nn.ModuleList | None
        self.class_embedding_towers: nn.ModuleList | None
        self.class_branch_feature_towers: nn.ModuleList | None
        self.class_fusion_scorers: nn.ModuleList | None
        self.class_axis_embedding_tower: nn.Module | None
        self.class_axis_branch_feature_tower: nn.Module | None
        self.class_axis_token_projector: nn.Module | None
        self.class_axis_encoder: nn.TransformerEncoder | None
        self.class_axis_logit_norm: nn.Module | None
        self.class_axis_logit_head: nn.Module | None
        self.class_axis_class_embeddings: nn.Parameter | None
        self.class_axis_embedding_score_head: nn.Module | None
        self.class_axis_branch_score_head: nn.Module | None
        self.class_axis_branch_scale_param: nn.Parameter | None
        self.class_axis_branch_direct_existential_weight_param: nn.Parameter | None
        self.class_axis_branch_direct_competitive_weight_param: nn.Parameter | None
        self.class_axis_branch_direct_existential_bias: nn.Parameter | None
        self.class_axis_branch_direct_competitive_bias: nn.Parameter | None
        self.class_axis_branch_direct_raw_scale_param: nn.Parameter | None
        self.class_axis_branch_direct_relative_scale_param: nn.Parameter | None
        self.class_axis_branch_direct_residual_scale_param: nn.Parameter | None
        self.class_axis_top_support_direct_raw_scale_param: nn.Parameter | None
        self.class_axis_top_support_direct_relative_positive_scale_param: (
            nn.Parameter | None
        )
        self.class_axis_top_support_direct_relative_negative_scale_param: (
            nn.Parameter | None
        )
        self.class_axis_top_support_direct_residual_scale_param: nn.Parameter | None
        self.class_axis_top_support_direct_residual_mlp: nn.Module | None
        self.class_axis_branch_direct_residual_mlp: nn.Module | None
        self.class_axis_interaction_scale_param: nn.Parameter | None
        if self.evidence_scorer_type == "two_tower_mlp":
            self.class_scorer_weight = None
            self.class_scorer_bias = None
            self.class_scorers = None
            self.class_axis_embedding_tower = None
            self.class_axis_branch_feature_tower = None
            self.class_axis_token_projector = None
            self.class_axis_encoder = None
            self.class_axis_logit_norm = None
            self.class_axis_logit_head = None
            self.class_axis_class_embeddings = None
            self.class_axis_embedding_score_head = None
            self.class_axis_branch_score_head = None
            self.class_axis_branch_scale_param = None
            self.class_axis_branch_direct_existential_weight_param = None
            self.class_axis_branch_direct_competitive_weight_param = None
            self.class_axis_branch_direct_existential_bias = None
            self.class_axis_branch_direct_competitive_bias = None
            self.class_axis_branch_direct_raw_scale_param = None
            self.class_axis_branch_direct_relative_scale_param = None
            self.class_axis_branch_direct_residual_scale_param = None
            self.class_axis_top_support_direct_raw_scale_param = None
            self.class_axis_top_support_direct_relative_positive_scale_param = None
            self.class_axis_top_support_direct_relative_negative_scale_param = None
            self.class_axis_top_support_direct_residual_scale_param = None
            self.class_axis_top_support_direct_residual_mlp = None
            self.class_axis_branch_direct_residual_mlp = None
            self.class_axis_interaction_scale_param = None
            self.class_embedding_towers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(
                            hidden_size,
                            self.evidence_scorer_cfg.embedding_hidden_size,
                        ),
                        nn.GELU(),
                        nn.Dropout(self.evidence_scorer_cfg.dropout),
                    )
                    for _ in range(num_classes)
                ]
            )
            self.class_branch_feature_towers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.scorer_branch_feature_count,
                            self.evidence_scorer_cfg.branch_hidden_size,
                        ),
                        nn.GELU(),
                        nn.Dropout(self.evidence_scorer_cfg.dropout),
                    )
                    for _ in range(num_classes)
                ]
            )
            fusion_input_size = (
                self.evidence_scorer_cfg.embedding_hidden_size
                + self.evidence_scorer_cfg.branch_hidden_size
            )
            self.class_fusion_scorers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(fusion_input_size),
                        nn.Linear(
                            fusion_input_size,
                            self.evidence_scorer_cfg.fusion_hidden_size,
                        ),
                        nn.GELU(),
                        nn.Dropout(self.evidence_scorer_cfg.dropout),
                        nn.Linear(self.evidence_scorer_cfg.fusion_hidden_size, 1),
                    )
                    for _ in range(num_classes)
                ]
            )
        elif self.evidence_scorer_type == "class_axis_attention":
            self.class_scorer_weight = None
            self.class_scorer_bias = None
            self.class_scorers = None
            self.class_embedding_towers = None
            self.class_branch_feature_towers = None
            self.class_fusion_scorers = None
            self.class_axis_embedding_tower = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(
                    hidden_size,
                    self.evidence_scorer_cfg.embedding_hidden_size,
                ),
                nn.GELU(),
                nn.Dropout(self.evidence_scorer_cfg.dropout),
            )
            self.class_axis_branch_feature_tower = nn.Sequential(
                nn.Linear(
                    self.scorer_branch_feature_count,
                    self.evidence_scorer_cfg.branch_hidden_size,
                ),
                nn.GELU(),
                nn.Dropout(self.evidence_scorer_cfg.dropout),
            )
            token_input_size = (
                self.evidence_scorer_cfg.embedding_hidden_size
                + self.evidence_scorer_cfg.branch_hidden_size
            )
            self.class_axis_token_projector = nn.Sequential(
                nn.LayerNorm(token_input_size),
                nn.Linear(
                    token_input_size, self.evidence_scorer_cfg.fusion_hidden_size
                ),
                nn.GELU(),
                nn.Dropout(self.evidence_scorer_cfg.dropout),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.evidence_scorer_cfg.fusion_hidden_size,
                nhead=self.evidence_scorer_cfg.num_attention_heads,
                dim_feedforward=self.evidence_scorer_cfg.fusion_hidden_size * 2,
                dropout=self.evidence_scorer_cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.class_axis_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.evidence_scorer_cfg.num_attention_layers,
            )
            self.class_axis_logit_norm = nn.LayerNorm(
                self.evidence_scorer_cfg.fusion_hidden_size
            )
            self.class_axis_logit_head = nn.Linear(
                self.evidence_scorer_cfg.fusion_hidden_size,
                1,
            )
            if self.evidence_scorer_cfg.score_decomposition.enabled:
                self.class_axis_embedding_score_head = nn.Linear(
                    self.evidence_scorer_cfg.embedding_hidden_size,
                    1,
                )
                self.class_axis_branch_score_head = nn.Linear(
                    self.evidence_scorer_cfg.branch_hidden_size,
                    1,
                )
                self.class_axis_branch_scale_param = nn.Parameter(
                    self._score_scale_parameter_init(
                        mode=(
                            self.evidence_scorer_cfg.score_decomposition.branch_scale_mode
                        ),
                        init_value=(
                            self.evidence_scorer_cfg.score_decomposition.branch_scale_init
                        ),
                        min_value=(
                            self.evidence_scorer_cfg.score_decomposition.branch_scale_min
                        ),
                        max_value=(
                            self.evidence_scorer_cfg.score_decomposition.branch_scale_max
                        ),
                    )
                )
                self.class_axis_interaction_scale_param = nn.Parameter(
                    self._score_scale_parameter_init(
                        mode=(
                            self.evidence_scorer_cfg.score_decomposition.interaction_scale_mode
                        ),
                        init_value=(
                            self.evidence_scorer_cfg.score_decomposition.interaction_scale_init
                        ),
                        min_value=(
                            self.evidence_scorer_cfg.score_decomposition.interaction_scale_min
                        ),
                        max_value=(
                            self.evidence_scorer_cfg.score_decomposition.interaction_scale_max
                        ),
                    )
                )
                direct_cfg = self.evidence_scorer_cfg.branch_direct_score
                if direct_cfg.enabled:
                    self.class_axis_branch_direct_existential_weight_param = (
                        nn.Parameter(_softplus_parameter_init(1.0).repeat(2))
                    )
                    self.class_axis_branch_direct_competitive_weight_param = (
                        nn.Parameter(_softplus_parameter_init(1.0).repeat(2))
                    )
                    self.class_axis_branch_direct_existential_bias = nn.Parameter(
                        torch.zeros(num_classes)
                    )
                    self.class_axis_branch_direct_competitive_bias = nn.Parameter(
                        torch.zeros(num_classes)
                    )
                    self.class_axis_branch_direct_raw_scale_param = nn.Parameter(
                        self._score_scale_parameter_init(
                            mode=direct_cfg.top_scale_mode,
                            init_value=direct_cfg.top_scale_init,
                            min_value=direct_cfg.top_scale_min,
                            max_value=direct_cfg.top_scale_max,
                        )
                    )
                    self.class_axis_branch_direct_relative_scale_param = nn.Parameter(
                        self._score_scale_parameter_init(
                            mode=direct_cfg.gated_scale_mode,
                            init_value=direct_cfg.gated_scale_init,
                            min_value=direct_cfg.gated_scale_min,
                            max_value=direct_cfg.gated_scale_max,
                        )
                    )
                    self.class_axis_branch_direct_residual_scale_param = nn.Parameter(
                        self._score_scale_parameter_init(
                            mode=direct_cfg.residual_scale_mode,
                            init_value=direct_cfg.residual_scale_init,
                            min_value=direct_cfg.residual_scale_min,
                            max_value=direct_cfg.residual_scale_max,
                        )
                    )
                    top_direct_cfg = direct_cfg.top_support_direct_path
                    if top_direct_cfg.enabled:
                        self.class_axis_top_support_direct_raw_scale_param = (
                            nn.Parameter(
                                self._score_scale_parameter_init(
                                    mode=top_direct_cfg.raw_scale_mode,
                                    init_value=top_direct_cfg.raw_scale_init,
                                    min_value=top_direct_cfg.raw_scale_min,
                                    max_value=top_direct_cfg.raw_scale_max,
                                )
                            )
                        )
                        self.class_axis_top_support_direct_relative_positive_scale_param = nn.Parameter(
                            self._score_scale_parameter_init(
                                mode=(top_direct_cfg.relative_positive_scale_mode),
                                init_value=(
                                    top_direct_cfg.relative_positive_scale_init
                                ),
                                min_value=(top_direct_cfg.relative_positive_scale_min),
                                max_value=(top_direct_cfg.relative_positive_scale_max),
                            )
                        )
                        self.class_axis_top_support_direct_relative_negative_scale_param = nn.Parameter(
                            self._score_scale_parameter_init(
                                mode=(top_direct_cfg.relative_negative_scale_mode),
                                init_value=(
                                    top_direct_cfg.relative_negative_scale_init
                                ),
                                min_value=(top_direct_cfg.relative_negative_scale_min),
                                max_value=(top_direct_cfg.relative_negative_scale_max),
                            )
                        )
                        self.class_axis_top_support_direct_residual_scale_param = (
                            nn.Parameter(
                                self._score_scale_parameter_init(
                                    mode=top_direct_cfg.residual_scale_mode,
                                    init_value=top_direct_cfg.residual_scale_init,
                                    min_value=top_direct_cfg.residual_scale_min,
                                    max_value=top_direct_cfg.residual_scale_max,
                                )
                            )
                        )
                        self.class_axis_top_support_direct_residual_mlp = nn.Sequential(
                            nn.Linear(
                                self.scorer_branch_feature_count,
                                top_direct_cfg.residual_hidden_size,
                            ),
                            nn.GELU(),
                            nn.Dropout(self.evidence_scorer_cfg.dropout),
                            nn.Linear(top_direct_cfg.residual_hidden_size, 1),
                        )
                    else:
                        self.class_axis_top_support_direct_raw_scale_param = None
                        self.class_axis_top_support_direct_relative_positive_scale_param = None
                        self.class_axis_top_support_direct_relative_negative_scale_param = None
                        self.class_axis_top_support_direct_residual_scale_param = None
                        self.class_axis_top_support_direct_residual_mlp = None
                    self.class_axis_branch_direct_residual_mlp = nn.Sequential(
                        nn.Linear(
                            self.scorer_branch_feature_count,
                            direct_cfg.residual_hidden_size,
                        ),
                        nn.GELU(),
                        nn.Dropout(self.evidence_scorer_cfg.dropout),
                        nn.Linear(direct_cfg.residual_hidden_size, 1),
                    )
                else:
                    self.class_axis_branch_direct_existential_weight_param = None
                    self.class_axis_branch_direct_competitive_weight_param = None
                    self.class_axis_branch_direct_existential_bias = None
                    self.class_axis_branch_direct_competitive_bias = None
                    self.class_axis_branch_direct_raw_scale_param = None
                    self.class_axis_branch_direct_relative_scale_param = None
                    self.class_axis_branch_direct_residual_scale_param = None
                    self.class_axis_top_support_direct_raw_scale_param = None
                    self.class_axis_top_support_direct_relative_positive_scale_param = (
                        None
                    )
                    self.class_axis_top_support_direct_relative_negative_scale_param = (
                        None
                    )
                    self.class_axis_top_support_direct_residual_scale_param = None
                    self.class_axis_top_support_direct_residual_mlp = None
                    self.class_axis_branch_direct_residual_mlp = None
            else:
                self.class_axis_embedding_score_head = None
                self.class_axis_branch_score_head = None
                self.class_axis_branch_scale_param = None
                self.class_axis_branch_direct_existential_weight_param = None
                self.class_axis_branch_direct_competitive_weight_param = None
                self.class_axis_branch_direct_existential_bias = None
                self.class_axis_branch_direct_competitive_bias = None
                self.class_axis_branch_direct_raw_scale_param = None
                self.class_axis_branch_direct_relative_scale_param = None
                self.class_axis_branch_direct_residual_scale_param = None
                self.class_axis_top_support_direct_raw_scale_param = None
                self.class_axis_top_support_direct_relative_positive_scale_param = None
                self.class_axis_top_support_direct_relative_negative_scale_param = None
                self.class_axis_top_support_direct_residual_scale_param = None
                self.class_axis_top_support_direct_residual_mlp = None
                self.class_axis_branch_direct_residual_mlp = None
                self.class_axis_interaction_scale_param = None
            self.class_axis_class_embeddings = (
                nn.Parameter(
                    torch.empty(
                        num_classes,
                        self.evidence_scorer_cfg.fusion_hidden_size,
                    )
                )
                if self.evidence_scorer_cfg.use_class_embedding
                else None
            )
        elif self.class_gate_scorer == "diagonal":
            self.class_scorer_weight = nn.Parameter(
                torch.empty(num_classes, hidden_size)
            )
            self.class_scorer_bias = nn.Parameter(torch.zeros(num_classes))
            self.class_scorers = None
            self.class_embedding_towers = None
            self.class_branch_feature_towers = None
            self.class_fusion_scorers = None
            self.class_axis_embedding_tower = None
            self.class_axis_branch_feature_tower = None
            self.class_axis_token_projector = None
            self.class_axis_encoder = None
            self.class_axis_logit_norm = None
            self.class_axis_logit_head = None
            self.class_axis_class_embeddings = None
            self.class_axis_embedding_score_head = None
            self.class_axis_branch_score_head = None
            self.class_axis_branch_scale_param = None
            self.class_axis_branch_direct_existential_weight_param = None
            self.class_axis_branch_direct_competitive_weight_param = None
            self.class_axis_branch_direct_existential_bias = None
            self.class_axis_branch_direct_competitive_bias = None
            self.class_axis_branch_direct_raw_scale_param = None
            self.class_axis_branch_direct_relative_scale_param = None
            self.class_axis_branch_direct_residual_scale_param = None
            self.class_axis_top_support_direct_raw_scale_param = None
            self.class_axis_top_support_direct_relative_positive_scale_param = None
            self.class_axis_top_support_direct_relative_negative_scale_param = None
            self.class_axis_top_support_direct_residual_scale_param = None
            self.class_axis_top_support_direct_residual_mlp = None
            self.class_axis_branch_direct_residual_mlp = None
            self.class_axis_interaction_scale_param = None
        elif self.class_gate_scorer == "normalized_mlp":
            scorer_hidden_size = cfg.class_gate.scorer_hidden_size or max(
                hidden_size // 2,
                1,
            )
            scorer_input_size = hidden_size + self.scorer_branch_feature_count
            self.class_scorer_weight = None
            self.class_scorer_bias = None
            self.class_scorers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(scorer_input_size),
                        nn.Linear(scorer_input_size, scorer_hidden_size),
                        nn.GELU(),
                        nn.Dropout(cfg.class_gate.scorer_dropout),
                        nn.Linear(scorer_hidden_size, 1),
                    )
                    for _ in range(num_classes)
                ]
            )
            self.class_embedding_towers = None
            self.class_branch_feature_towers = None
            self.class_fusion_scorers = None
            self.class_axis_embedding_tower = None
            self.class_axis_branch_feature_tower = None
            self.class_axis_token_projector = None
            self.class_axis_encoder = None
            self.class_axis_logit_norm = None
            self.class_axis_logit_head = None
            self.class_axis_class_embeddings = None
            self.class_axis_embedding_score_head = None
            self.class_axis_branch_score_head = None
            self.class_axis_branch_scale_param = None
            self.class_axis_branch_direct_existential_weight_param = None
            self.class_axis_branch_direct_competitive_weight_param = None
            self.class_axis_branch_direct_existential_bias = None
            self.class_axis_branch_direct_competitive_bias = None
            self.class_axis_branch_direct_raw_scale_param = None
            self.class_axis_branch_direct_relative_scale_param = None
            self.class_axis_branch_direct_residual_scale_param = None
            self.class_axis_top_support_direct_raw_scale_param = None
            self.class_axis_top_support_direct_relative_positive_scale_param = None
            self.class_axis_top_support_direct_relative_negative_scale_param = None
            self.class_axis_top_support_direct_residual_scale_param = None
            self.class_axis_top_support_direct_residual_mlp = None
            self.class_axis_branch_direct_residual_mlp = None
            self.class_axis_interaction_scale_param = None
        else:
            raise ValueError(
                "class_aware_branch_gated class_gate.scorer must be 'diagonal' "
                "or 'normalized_mlp'"
            )
        nn.init.normal_(self.class_queries, std=PATCH_INIT_STD)
        if self.class_scorer_weight is not None:
            nn.init.normal_(self.class_scorer_weight, std=PATCH_INIT_STD)
        if self.class_axis_class_embeddings is not None:
            nn.init.normal_(self.class_axis_class_embeddings, std=PATCH_INIT_STD)

    def set_runtime_epoch(self, epoch: int | None) -> None:
        self._runtime_epoch = None if epoch is None else int(epoch)

    def _gate_mixing_alpha(self) -> float:
        return _resolve_hold_decay_schedule(
            enabled=self.gate_mixing.enabled,
            start_value=float(self.gate_mixing.start_alpha),
            end_value=float(self.gate_mixing.end_alpha),
            hold_epochs=int(self.gate_mixing.hold_epochs),
            decay_epochs=int(self.gate_mixing.decay_epochs),
            epoch=self._runtime_epoch,
        )

    @staticmethod
    def _center_class_scores(scores: Tensor) -> Tensor:
        return scores - scores.mean(dim=-1, keepdim=True)

    @staticmethod
    def _score_scale_parameter_init(
        *,
        mode: str,
        init_value: float,
        min_value: float,
        max_value: float,
    ) -> Tensor:
        if mode == "sigmoid_max":
            return _sigmoid_max_parameter_init(init_value, max_value)
        if mode == "bounded_sigmoid":
            return _bounded_sigmoid_parameter_init(init_value, min_value, max_value)
        raise ValueError("score decomposition scale mode is not supported")

    @staticmethod
    def _score_scale_from_parameter(
        parameter: Tensor,
        *,
        mode: str,
        min_value: float,
        max_value: float,
    ) -> Tensor:
        sigmoid_value = torch.sigmoid(parameter)
        if mode == "sigmoid_max":
            return sigmoid_value * float(max_value)
        if mode == "bounded_sigmoid":
            return float(min_value) + (
                sigmoid_value * (float(max_value) - float(min_value))
            )
        raise ValueError("score decomposition scale mode is not supported")

    def _interaction_scale_schedule_multiplier(self) -> float:
        cfg = self.evidence_scorer_cfg.score_decomposition.interaction_scale_schedule
        if not cfg.enabled:
            return 1.0
        if self._runtime_epoch is None:
            return 1.0
        current_epoch = int(self._runtime_epoch)
        if current_epoch < int(cfg.start_epoch):
            return float(cfg.start_multiplier)
        if current_epoch > int(cfg.end_epoch):
            return float(cfg.end_multiplier)
        if int(cfg.end_epoch) == int(cfg.start_epoch):
            return float(cfg.end_multiplier)
        progress = float(current_epoch - int(cfg.start_epoch)) / float(
            int(cfg.end_epoch) - int(cfg.start_epoch)
        )
        return float(cfg.start_multiplier) + (
            (float(cfg.end_multiplier) - float(cfg.start_multiplier)) * progress
        )

    def _score_decomposition_scales(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.evidence_scorer_cfg.score_decomposition
        if (
            self.class_axis_branch_scale_param is None
            or self.class_axis_interaction_scale_param is None
        ):
            raise RuntimeError(
                "class-axis score decomposition scales are not initialized"
            )
        branch_scale = self._score_scale_from_parameter(
            self.class_axis_branch_scale_param.to(device=device, dtype=dtype),
            mode=cfg.branch_scale_mode,
            min_value=float(cfg.branch_scale_min),
            max_value=float(cfg.branch_scale_max),
        )
        interaction_scale = self._score_scale_from_parameter(
            self.class_axis_interaction_scale_param.to(device=device, dtype=dtype),
            mode=cfg.interaction_scale_mode,
            min_value=float(cfg.interaction_scale_min),
            max_value=float(cfg.interaction_scale_max),
        )
        multiplier = torch.tensor(
            self._interaction_scale_schedule_multiplier(),
            device=device,
            dtype=dtype,
        )
        return (
            branch_scale,
            interaction_scale,
            multiplier,
            interaction_scale * multiplier,
        )

    @staticmethod
    def _bounded_score_component(
        scores: Tensor,
        cfg: ClassGateScoreBoundComponentConfig,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if not cfg.enabled:
            return scores, None, None
        if cfg.mode != "tanh_bound":
            raise ValueError("score bounding mode is unsupported")
        bound = torch.tensor(float(cfg.bound), device=scores.device, dtype=scores.dtype)
        temperature = torch.tensor(
            float(cfg.temperature),
            device=scores.device,
            dtype=scores.dtype,
        )
        return bound * torch.tanh(scores / temperature), bound, temperature

    def _branch_direct_score_scales(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.evidence_scorer_cfg.branch_direct_score
        if (
            self.class_axis_branch_direct_raw_scale_param is None
            or self.class_axis_branch_direct_relative_scale_param is None
            or self.class_axis_branch_direct_residual_scale_param is None
        ):
            raise RuntimeError("branch direct score scales are not initialized")
        top_scale = self._score_scale_from_parameter(
            self.class_axis_branch_direct_raw_scale_param.to(
                device=device, dtype=dtype
            ),
            mode=cfg.top_scale_mode,
            min_value=float(cfg.top_scale_min),
            max_value=float(cfg.top_scale_max),
        )
        gated_scale = self._score_scale_from_parameter(
            self.class_axis_branch_direct_relative_scale_param.to(
                device=device,
                dtype=dtype,
            ),
            mode=cfg.gated_scale_mode,
            min_value=float(cfg.gated_scale_min),
            max_value=float(cfg.gated_scale_max),
        )
        residual_scale = self._score_scale_from_parameter(
            self.class_axis_branch_direct_residual_scale_param.to(
                device=device,
                dtype=dtype,
            ),
            mode=cfg.residual_scale_mode,
            min_value=float(cfg.residual_scale_min),
            max_value=float(cfg.residual_scale_max),
        )
        return top_scale, gated_scale, residual_scale

    def _top_support_direct_scales(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.evidence_scorer_cfg.branch_direct_score.top_support_direct_path
        if (
            self.class_axis_top_support_direct_raw_scale_param is None
            or self.class_axis_top_support_direct_relative_positive_scale_param is None
            or self.class_axis_top_support_direct_relative_negative_scale_param is None
            or self.class_axis_top_support_direct_residual_scale_param is None
        ):
            raise RuntimeError("top support direct path scales are not initialized")
        raw_scale = self._score_scale_from_parameter(
            self.class_axis_top_support_direct_raw_scale_param.to(
                device=device,
                dtype=dtype,
            ),
            mode=cfg.raw_scale_mode,
            min_value=float(cfg.raw_scale_min),
            max_value=float(cfg.raw_scale_max),
        )
        relative_positive_scale = self._score_scale_from_parameter(
            self.class_axis_top_support_direct_relative_positive_scale_param.to(
                device=device,
                dtype=dtype,
            ),
            mode=cfg.relative_positive_scale_mode,
            min_value=float(cfg.relative_positive_scale_min),
            max_value=float(cfg.relative_positive_scale_max),
        )
        relative_negative_scale = self._score_scale_from_parameter(
            self.class_axis_top_support_direct_relative_negative_scale_param.to(
                device=device,
                dtype=dtype,
            ),
            mode=cfg.relative_negative_scale_mode,
            min_value=float(cfg.relative_negative_scale_min),
            max_value=float(cfg.relative_negative_scale_max),
        )
        residual_scale = self._score_scale_from_parameter(
            self.class_axis_top_support_direct_residual_scale_param.to(
                device=device,
                dtype=dtype,
            ),
            mode=cfg.residual_scale_mode,
            min_value=float(cfg.residual_scale_min),
            max_value=float(cfg.residual_scale_max),
        )
        return (
            raw_scale,
            relative_positive_scale,
            relative_negative_scale,
            residual_scale,
        )

    def _class_override_row(
        self,
        *,
        default: float,
        overrides_by_class: tuple[float, ...] | None,
        device: torch.device,
        dtype: torch.dtype,
        field_name: str,
    ) -> Tensor:
        if overrides_by_class is None:
            return torch.full(
                (1, self.num_classes),
                float(default),
                device=device,
                dtype=dtype,
            )
        if len(overrides_by_class) != self.num_classes:
            raise ValueError(
                f"{field_name} must have length {self.num_classes}, "
                f"got {len(overrides_by_class)}"
            )
        return torch.tensor(
            tuple(float(value) for value in overrides_by_class),
            device=device,
            dtype=dtype,
        ).view(1, -1)

    def _branch_direct_score_components(
        self,
        branch_feature_inputs: Tensor,
        branch_feature_raw_inputs: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        cfg = self.evidence_scorer_cfg.branch_direct_score
        if cfg.positive_weight_mode != "softplus":
            raise ValueError("branch direct score positive_weight_mode is unsupported")
        if (
            self.class_axis_branch_direct_existential_weight_param is None
            or self.class_axis_branch_direct_competitive_weight_param is None
            or self.class_axis_branch_direct_existential_bias is None
            or self.class_axis_branch_direct_competitive_bias is None
            or self.class_axis_branch_direct_residual_mlp is None
        ):
            raise RuntimeError("branch direct score modules are not initialized")
        correction_cfg = cfg.top_relative_correction
        top_direct_cfg = cfg.top_support_direct_path
        top_feature_inputs = branch_feature_inputs
        if top_direct_cfg.enabled and branch_feature_raw_inputs is not None:
            top_feature_inputs = branch_feature_raw_inputs.to(
                device=branch_feature_inputs.device,
                dtype=branch_feature_inputs.dtype,
            )
            if tuple(top_feature_inputs.shape) != tuple(branch_feature_inputs.shape):
                raise ValueError(
                    "branch_feature_raw_inputs must match branch_feature_inputs "
                    "shape for top support direct scoring"
                )
        top_features = top_feature_inputs[..., (2, 3)]
        gated_features = branch_feature_inputs[..., (0, 1)]
        top_weights = F.softplus(
            self.class_axis_branch_direct_existential_weight_param.to(
                device=branch_feature_inputs.device,
                dtype=branch_feature_inputs.dtype,
            )
        )
        gated_weights = F.softplus(
            self.class_axis_branch_direct_competitive_weight_param.to(
                device=branch_feature_inputs.device,
                dtype=branch_feature_inputs.dtype,
            )
        )
        top_bias = self.class_axis_branch_direct_existential_bias.to(
            device=branch_feature_inputs.device,
            dtype=branch_feature_inputs.dtype,
        ).view(1, -1)
        top_raw_feature = top_features[..., 0]
        top_relative_feature = top_features[..., 1]
        top_raw_existential_scores = (top_raw_feature * top_weights[0]) + top_bias
        top_relative_positive = torch.relu(top_relative_feature)
        top_relative_negative = torch.clamp(top_relative_feature, max=0.0)
        top_support_raw_positive_component: Tensor | None = None
        top_support_relative_positive_component: Tensor | None = None
        top_support_relative_negative_component: Tensor | None = None
        top_support_relative_negative_component_uncapped: Tensor | None = None
        top_support_relative_negative_component_capped: Tensor | None = None
        top_support_relative_negative_cap_value: Tensor | None = None
        top_support_relative_negative_cap_active: Tensor | None = None
        top_support_direct_raw_scale: Tensor | None = None
        top_support_direct_relative_positive_scale: Tensor | None = None
        top_support_direct_relative_negative_scale: Tensor | None = None
        top_support_direct_residual_scale: Tensor | None = None
        top_support_residual_scores = torch.zeros_like(top_raw_feature)
        if top_direct_cfg.enabled:
            if top_direct_cfg.mode != "monotonic_raw_relative":
                raise ValueError("top support direct path mode is unsupported")
            if top_direct_cfg.positive_transform != "softplus":
                raise ValueError(
                    "top support direct path positive_transform is unsupported"
                )
            if self.class_axis_top_support_direct_residual_mlp is None:
                raise RuntimeError(
                    "top support direct residual module is not initialized"
                )
            (
                top_support_direct_raw_scale,
                top_support_direct_relative_positive_scale,
                top_support_direct_relative_negative_scale,
                top_support_direct_residual_scale,
            ) = self._top_support_direct_scales(
                device=branch_feature_inputs.device,
                dtype=branch_feature_inputs.dtype,
            )
            top_support_raw_positive_component = (
                top_support_direct_raw_scale * top_raw_feature
            )
            top_support_relative_positive_component = (
                top_support_direct_relative_positive_scale
                * F.softplus(top_relative_feature)
            )
            top_support_relative_negative_component_uncapped = (
                top_support_direct_relative_negative_scale
                * F.softplus(-top_relative_feature)
            )
            top_support_relative_negative_component = (
                top_support_relative_negative_component_uncapped
            )
            cap_cfg = top_direct_cfg.negative_relative_cap
            if cap_cfg.enabled:
                if cap_cfg.mode != "raw_fraction_cap":
                    raise ValueError("negative relative cap mode is unsupported")
                max_negative_fraction = self._class_override_row(
                    default=float(cap_cfg.max_negative_fraction),
                    overrides_by_class=cap_cfg.max_negative_fraction_by_class,
                    device=branch_feature_inputs.device,
                    dtype=branch_feature_inputs.dtype,
                    field_name=(
                        "top_support_direct_path.negative_relative_cap."
                        "max_negative_fraction_by_class"
                    ),
                )
                negative_cap = self._class_override_row(
                    default=float(cap_cfg.negative_cap),
                    overrides_by_class=cap_cfg.negative_cap_by_class,
                    device=branch_feature_inputs.device,
                    dtype=branch_feature_inputs.dtype,
                    field_name=(
                        "top_support_direct_path.negative_relative_cap."
                        "negative_cap_by_class"
                    ),
                )
                top_support_relative_negative_cap_value = (
                    max_negative_fraction
                    * torch.relu(top_support_raw_positive_component)
                ) + negative_cap
                top_support_relative_negative_component = torch.minimum(
                    top_support_relative_negative_component_uncapped,
                    top_support_relative_negative_cap_value,
                )
                top_support_relative_negative_cap_active = (
                    top_support_relative_negative_component_uncapped
                    > top_support_relative_negative_cap_value
                ).to(branch_feature_inputs.dtype)
            top_support_relative_negative_component_capped = (
                top_support_relative_negative_component
            )
            top_raw_existential_scores = top_support_raw_positive_component + top_bias
            top_relative_correction_scores = (
                top_support_relative_positive_component
                - top_support_relative_negative_component
            )
            direct_top_scores = (
                top_raw_existential_scores + top_relative_correction_scores
            )
            top_support_residual_logits = (
                self.class_axis_top_support_direct_residual_mlp(
                    branch_feature_inputs
                ).squeeze(-1)
            )
            top_support_residual_scores = float(
                top_direct_cfg.residual_bound
            ) * torch.tanh(
                top_support_residual_logits / float(top_direct_cfg.residual_temperature)
            )
            top_support_scores = direct_top_scores + (
                top_support_direct_residual_scale * top_support_residual_scores
            )
        elif cfg.top_support_mode == "learned_weighted_sum":
            top_relative_correction_scores = top_relative_feature * top_weights[1]
            top_support_scores = (
                top_raw_existential_scores + top_relative_correction_scores
            )
            direct_top_scores = top_support_scores
        elif cfg.top_support_mode == "raw_existential_plus_relative_correction":
            if not correction_cfg.enabled:
                top_relative_correction_scores = torch.zeros_like(top_raw_feature)
            else:
                top_relative_negative = torch.clamp(
                    top_relative_feature,
                    min=-float(correction_cfg.negative_clip),
                    max=0.0,
                )
                top_relative_correction_scores = (
                    float(correction_cfg.positive_scale) * top_relative_positive
                ) + (float(correction_cfg.negative_scale) * top_relative_negative)
            top_support_scores = (
                top_raw_existential_scores + top_relative_correction_scores
            )
            direct_top_scores = top_support_scores
        else:
            raise ValueError(
                "branch direct score top_support_mode is unsupported: "
                f"{cfg.top_support_mode!r}"
            )
        gated_support_scores = (gated_features * gated_weights.view(1, 1, 2)).sum(
            dim=-1
        ) + self.class_axis_branch_direct_competitive_bias.to(
            device=branch_feature_inputs.device,
            dtype=branch_feature_inputs.dtype,
        ).view(1, -1)
        top_scale, gated_scale, residual_scale = self._branch_direct_score_scales(
            device=branch_feature_inputs.device,
            dtype=branch_feature_inputs.dtype,
        )
        if top_direct_cfg.enabled:
            residual_scale_cap = torch.tensor(
                float(top_direct_cfg.residual_scale_max),
                device=branch_feature_inputs.device,
                dtype=branch_feature_inputs.dtype,
            )
            residual_scale = torch.minimum(residual_scale, residual_scale_cap)
        if branch_feature_raw_inputs is None:
            margin_inputs = branch_feature_inputs
        else:
            margin_inputs = branch_feature_raw_inputs.to(
                device=branch_feature_inputs.device,
                dtype=branch_feature_inputs.dtype,
            )
        if tuple(margin_inputs.shape) != tuple(branch_feature_inputs.shape):
            raise ValueError(
                "branch_feature_raw_inputs must match branch_feature_inputs shape, "
                f"got {tuple(margin_inputs.shape)} vs "
                f"{tuple(branch_feature_inputs.shape)}"
            )
        top_margin = margin_inputs[..., 2]
        gated_margin = margin_inputs[..., 0]
        mixture_cfg = cfg.gate_reliability_mixture
        if mixture_cfg.enabled:
            if mixture_cfg.source != "top_vs_gated_margin_regret":
                raise ValueError("gate reliability mixture source is unsupported")
            if mixture_cfg.mode != "exp_neg_regret":
                raise ValueError("gate reliability mixture mode is unsupported")
            gate_reliability_regret = F.relu(
                top_margin - gated_margin - float(mixture_cfg.tolerance)
            )
            reliability_source = gate_reliability_regret
            if mixture_cfg.detach:
                reliability_source = reliability_source.detach()
            gate_reliability = torch.exp(
                -reliability_source / float(mixture_cfg.temperature)
            )
        else:
            gate_reliability_regret = torch.zeros_like(top_margin)
            gate_reliability = torch.ones_like(top_margin)
        direct_scores = (top_scale * top_support_scores) + (
            gated_scale * gate_reliability * gated_support_scores
        )
        residual_logits = self.class_axis_branch_direct_residual_mlp(
            branch_feature_inputs
        ).squeeze(-1)
        residual_scores = float(cfg.residual_bound) * torch.tanh(
            residual_logits / float(cfg.residual_temperature)
        )
        return (
            top_support_scores,
            direct_top_scores,
            top_support_residual_scores,
            gated_support_scores,
            direct_scores,
            residual_scores,
            top_scale,
            gated_scale,
            residual_scale,
            top_weights.detach(),
            gated_weights.detach(),
            gate_reliability,
            gate_reliability_regret,
            top_margin.detach(),
            gated_margin.detach(),
            top_raw_existential_scores,
            top_relative_correction_scores,
            top_relative_positive.detach(),
            top_relative_negative.detach(),
            top_support_raw_positive_component,
            top_support_relative_positive_component,
            top_support_relative_negative_component,
            top_support_relative_negative_component_uncapped,
            top_support_relative_negative_component_capped,
            top_support_relative_negative_cap_value,
            top_support_relative_negative_cap_active,
            top_support_direct_raw_scale.detach()
            if top_support_direct_raw_scale is not None
            else None,
            top_support_direct_relative_positive_scale.detach()
            if top_support_direct_relative_positive_scale is not None
            else None,
            top_support_direct_relative_negative_scale.detach()
            if top_support_direct_relative_negative_scale is not None
            else None,
            top_support_direct_residual_scale.detach()
            if top_support_direct_residual_scale is not None
            else None,
        )

    def score_class_evidence_with_components(
        self,
        class_evidence_embeddings: Tensor,
        branch_feature_inputs: Tensor | None = None,
        branch_feature_raw_inputs: Tensor | None = None,
    ) -> ClassEvidenceScoreOutput:
        if self.evidence_scorer_type == "class_axis_attention":
            if branch_feature_inputs is None:
                raise ValueError(
                    "class_axis_attention evidence scorer requires "
                    "branch_feature_inputs"
                )
            expected_shape = (
                class_evidence_embeddings.shape[0],
                class_evidence_embeddings.shape[1],
                self.scorer_branch_feature_count,
            )
            if tuple(branch_feature_inputs.shape) != expected_shape:
                raise ValueError(
                    "branch_feature_inputs must have shape "
                    f"{expected_shape}, got {tuple(branch_feature_inputs.shape)}"
                )
            if (
                branch_feature_raw_inputs is not None
                and tuple(branch_feature_raw_inputs.shape) != expected_shape
            ):
                raise ValueError(
                    "branch_feature_raw_inputs must have shape "
                    f"{expected_shape}, got {tuple(branch_feature_raw_inputs.shape)}"
                )
            if (
                self.class_axis_embedding_tower is None
                or self.class_axis_branch_feature_tower is None
                or self.class_axis_token_projector is None
                or self.class_axis_encoder is None
                or self.class_axis_logit_norm is None
                or self.class_axis_logit_head is None
            ):
                raise RuntimeError(
                    "class_axis_attention evidence scorer modules are not initialized"
                )
            embedding_hidden = self.class_axis_embedding_tower(
                class_evidence_embeddings
            )
            branch_hidden = self.class_axis_branch_feature_tower(branch_feature_inputs)
            class_tokens = self.class_axis_token_projector(
                torch.cat([embedding_hidden, branch_hidden], dim=-1)
            )
            if self.class_axis_class_embeddings is not None:
                class_tokens = (
                    class_tokens + self.class_axis_class_embeddings.unsqueeze(0)
                )
            attended_tokens = self.class_axis_encoder(class_tokens)
            interaction_scores = self.class_axis_logit_head(
                self.class_axis_logit_norm(attended_tokens)
            ).squeeze(-1)
            if self.evidence_scorer_cfg.score_decomposition.enabled:
                if (
                    self.class_axis_embedding_score_head is None
                    or self.class_axis_branch_score_head is None
                ):
                    raise RuntimeError(
                        "class-axis score decomposition heads are not initialized"
                    )
                raw_embedding_scores = self.class_axis_embedding_score_head(
                    embedding_hidden
                ).squeeze(-1)
                embedding_scores = raw_embedding_scores
                top_support_scores: Tensor | None = None
                direct_top_scores: Tensor | None = None
                top_support_residual_scores: Tensor | None = None
                gated_support_scores: Tensor | None = None
                gate_reliability: Tensor | None = None
                gate_reliability_regret: Tensor | None = None
                top_margin: Tensor | None = None
                gated_margin: Tensor | None = None
                branch_direct_scores: Tensor | None = None
                branch_residual_scores: Tensor | None = None
                top_raw_existential_scores: Tensor | None = None
                top_relative_correction_scores: Tensor | None = None
                top_support_raw_positive_component: Tensor | None = None
                top_support_relative_positive_component: Tensor | None = None
                top_support_relative_negative_component: Tensor | None = None
                top_support_relative_negative_component_uncapped: Tensor | None = None
                top_support_relative_negative_component_capped: Tensor | None = None
                top_support_relative_negative_cap_value: Tensor | None = None
                top_support_relative_negative_cap_active: Tensor | None = None
                top_support_direct_raw_scale: Tensor | None = None
                top_support_direct_relative_positive_scale: Tensor | None = None
                top_support_direct_relative_negative_scale: Tensor | None = None
                top_support_direct_residual_scale: Tensor | None = None
                top_relative_positive: Tensor | None = None
                top_relative_negative: Tensor | None = None
                branch_direct_top_scale: Tensor | None = None
                branch_direct_gated_scale: Tensor | None = None
                branch_direct_residual_scale: Tensor | None = None
                branch_direct_top_weights: Tensor | None = None
                branch_direct_gated_weights: Tensor | None = None
                if self.evidence_scorer_cfg.branch_direct_score.enabled:
                    (
                        top_support_scores,
                        direct_top_scores,
                        top_support_residual_scores,
                        gated_support_scores,
                        branch_direct_scores,
                        branch_residual_scores,
                        branch_direct_top_scale,
                        branch_direct_gated_scale,
                        branch_direct_residual_scale,
                        branch_direct_top_weights,
                        branch_direct_gated_weights,
                        gate_reliability,
                        gate_reliability_regret,
                        top_margin,
                        gated_margin,
                        top_raw_existential_scores,
                        top_relative_correction_scores,
                        top_relative_positive,
                        top_relative_negative,
                        top_support_raw_positive_component,
                        top_support_relative_positive_component,
                        top_support_relative_negative_component,
                        top_support_relative_negative_component_uncapped,
                        top_support_relative_negative_component_capped,
                        top_support_relative_negative_cap_value,
                        top_support_relative_negative_cap_active,
                        top_support_direct_raw_scale,
                        top_support_direct_relative_positive_scale,
                        top_support_direct_relative_negative_scale,
                        top_support_direct_residual_scale,
                    ) = self._branch_direct_score_components(
                        branch_feature_inputs,
                        branch_feature_raw_inputs=branch_feature_raw_inputs,
                    )
                    if self.evidence_scorer_cfg.logit_centering:
                        top_raw_existential_scores = self._center_class_scores(
                            top_raw_existential_scores
                        )
                        top_relative_correction_scores = self._center_class_scores(
                            top_relative_correction_scores
                        )
                        if top_support_raw_positive_component is not None:
                            top_support_raw_positive_component = (
                                self._center_class_scores(
                                    top_support_raw_positive_component
                                )
                            )
                        if top_support_relative_positive_component is not None:
                            top_support_relative_positive_component = (
                                self._center_class_scores(
                                    top_support_relative_positive_component
                                )
                            )
                        if top_support_relative_negative_component is not None:
                            top_support_relative_negative_component = (
                                self._center_class_scores(
                                    top_support_relative_negative_component
                                )
                            )
                        if top_support_relative_negative_component_uncapped is not None:
                            top_support_relative_negative_component_uncapped = (
                                self._center_class_scores(
                                    top_support_relative_negative_component_uncapped
                                )
                            )
                        if top_support_relative_negative_component_capped is not None:
                            top_support_relative_negative_component_capped = (
                                self._center_class_scores(
                                    top_support_relative_negative_component_capped
                                )
                            )
                        top_support_scores = self._center_class_scores(
                            top_support_scores
                        )
                        direct_top_scores = self._center_class_scores(direct_top_scores)
                        top_support_residual_scores = self._center_class_scores(
                            top_support_residual_scores
                        )
                        if top_support_direct_residual_scale is not None:
                            top_support_scores = direct_top_scores + (
                                top_support_direct_residual_scale
                                * top_support_residual_scores
                            )
                        gated_support_scores = self._center_class_scores(
                            gated_support_scores
                        )
                        branch_residual_scores = self._center_class_scores(
                            branch_residual_scores
                        )
                        branch_direct_scores = (
                            branch_direct_top_scale * top_support_scores
                        ) + (
                            branch_direct_gated_scale
                            * gate_reliability
                            * gated_support_scores
                        )
                    branch_support_scores = branch_direct_scores + (
                        branch_direct_residual_scale * branch_residual_scores
                    )
                else:
                    branch_support_scores = self.class_axis_branch_score_head(
                        branch_hidden
                    ).squeeze(-1)
                if self.evidence_scorer_cfg.logit_centering:
                    embedding_scores = self._center_class_scores(embedding_scores)
                    raw_embedding_scores = self._center_class_scores(
                        raw_embedding_scores
                    )
                    if not self.evidence_scorer_cfg.branch_direct_score.enabled:
                        branch_support_scores = self._center_class_scores(
                            branch_support_scores
                        )
                    interaction_scores = self._center_class_scores(interaction_scores)
                raw_interaction_scores = interaction_scores
                bounded_embedding_scores: Tensor | None = None
                bounded_interaction_scores: Tensor | None = None
                embedding_score_bound: Tensor | None = None
                embedding_score_temperature: Tensor | None = None
                interaction_score_bound: Tensor | None = None
                interaction_score_temperature: Tensor | None = None
                score_bounding = (
                    self.evidence_scorer_cfg.score_decomposition.score_bounding
                )
                if score_bounding.enabled:
                    (
                        embedding_scores,
                        embedding_score_bound,
                        embedding_score_temperature,
                    ) = self._bounded_score_component(
                        raw_embedding_scores,
                        score_bounding.embedding,
                    )
                    (
                        interaction_scores,
                        interaction_score_bound,
                        interaction_score_temperature,
                    ) = self._bounded_score_component(
                        raw_interaction_scores,
                        score_bounding.interaction,
                    )
                    bounded_embedding_scores = embedding_scores
                    bounded_interaction_scores = interaction_scores
                (
                    branch_scale,
                    interaction_scale,
                    interaction_scale_multiplier,
                    interaction_effective_scale,
                ) = self._score_decomposition_scales(
                    device=interaction_scores.device,
                    dtype=interaction_scores.dtype,
                )
                logits = (
                    embedding_scores
                    + (branch_scale * branch_support_scores)
                    + (interaction_effective_scale * interaction_scores)
                )
                if self.evidence_scorer_cfg.logit_centering:
                    logits = self._center_class_scores(logits)
                return ClassEvidenceScoreOutput(
                    logits=logits,
                    embedding_scores=embedding_scores,
                    raw_embedding_scores=raw_embedding_scores,
                    bounded_embedding_scores=bounded_embedding_scores,
                    top_support_scores=top_support_scores,
                    direct_top_scores=direct_top_scores,
                    top_support_residual_scores=top_support_residual_scores,
                    gated_support_scores=gated_support_scores,
                    top_raw_existential_scores=top_raw_existential_scores,
                    top_relative_correction_scores=top_relative_correction_scores,
                    top_support_raw_positive_component=(
                        top_support_raw_positive_component
                    ),
                    top_support_relative_positive_component=(
                        top_support_relative_positive_component
                    ),
                    top_support_relative_negative_component=(
                        top_support_relative_negative_component
                    ),
                    top_support_relative_negative_component_uncapped=(
                        top_support_relative_negative_component_uncapped
                    ),
                    top_support_relative_negative_component_capped=(
                        top_support_relative_negative_component_capped
                    ),
                    top_support_relative_negative_cap_value=(
                        top_support_relative_negative_cap_value
                    ),
                    top_support_relative_negative_cap_active=(
                        top_support_relative_negative_cap_active
                    ),
                    top_support_direct_raw_scale=top_support_direct_raw_scale,
                    top_support_direct_relative_positive_scale=(
                        top_support_direct_relative_positive_scale
                    ),
                    top_support_direct_relative_negative_scale=(
                        top_support_direct_relative_negative_scale
                    ),
                    top_support_direct_residual_scale=(
                        top_support_direct_residual_scale
                    ),
                    top_relative_positive=top_relative_positive,
                    top_relative_negative=top_relative_negative,
                    gate_reliability=gate_reliability,
                    gate_reliability_regret=gate_reliability_regret,
                    top_margin=top_margin,
                    gated_margin=gated_margin,
                    branch_existential_scores=top_support_scores,
                    branch_competitive_scores=gated_support_scores,
                    branch_direct_scores=branch_direct_scores,
                    branch_residual_scores=branch_residual_scores,
                    branch_support_scores=branch_support_scores,
                    interaction_scores=interaction_scores,
                    raw_interaction_scores=raw_interaction_scores,
                    bounded_interaction_scores=bounded_interaction_scores,
                    branch_scale=branch_scale.detach(),
                    branch_direct_top_scale=(
                        branch_direct_top_scale.detach()
                        if branch_direct_top_scale is not None
                        else None
                    ),
                    branch_direct_gated_scale=(
                        branch_direct_gated_scale.detach()
                        if branch_direct_gated_scale is not None
                        else None
                    ),
                    branch_direct_raw_scale=(
                        branch_direct_top_scale.detach()
                        if branch_direct_top_scale is not None
                        else None
                    ),
                    branch_direct_relative_scale=(
                        branch_direct_gated_scale.detach()
                        if branch_direct_gated_scale is not None
                        else None
                    ),
                    branch_direct_residual_scale=(
                        branch_direct_residual_scale.detach()
                        if branch_direct_residual_scale is not None
                        else None
                    ),
                    branch_direct_top_weights=branch_direct_top_weights,
                    branch_direct_gated_weights=branch_direct_gated_weights,
                    branch_direct_existential_weights=branch_direct_top_weights,
                    branch_direct_competitive_weights=branch_direct_gated_weights,
                    interaction_scale=interaction_scale.detach(),
                    interaction_scale_multiplier=(
                        interaction_scale_multiplier.detach()
                    ),
                    interaction_effective_scale=interaction_effective_scale.detach(),
                    embedding_score_bound=(
                        embedding_score_bound.detach()
                        if embedding_score_bound is not None
                        else None
                    ),
                    embedding_score_temperature=(
                        embedding_score_temperature.detach()
                        if embedding_score_temperature is not None
                        else None
                    ),
                    interaction_score_bound=(
                        interaction_score_bound.detach()
                        if interaction_score_bound is not None
                        else None
                    ),
                    interaction_score_temperature=(
                        interaction_score_temperature.detach()
                        if interaction_score_temperature is not None
                        else None
                    ),
                )
            logits = interaction_scores
            if self.evidence_scorer_cfg.logit_centering:
                logits = logits - logits.mean(dim=-1, keepdim=True)
            return ClassEvidenceScoreOutput(logits=logits)

        if self.evidence_scorer_type == "two_tower_mlp":
            if branch_feature_inputs is None:
                raise ValueError(
                    "two_tower_mlp evidence scorer requires branch_feature_inputs"
                )
            expected_shape = (
                class_evidence_embeddings.shape[0],
                class_evidence_embeddings.shape[1],
                self.scorer_branch_feature_count,
            )
            if tuple(branch_feature_inputs.shape) != expected_shape:
                raise ValueError(
                    "branch_feature_inputs must have shape "
                    f"{expected_shape}, got {tuple(branch_feature_inputs.shape)}"
                )
            if (
                self.class_embedding_towers is None
                or self.class_branch_feature_towers is None
                or self.class_fusion_scorers is None
            ):
                raise RuntimeError("two_tower_mlp class scorers are not initialized")
            logits = []
            for class_index in range(self.num_classes):
                embedding_hidden = self.class_embedding_towers[class_index](
                    class_evidence_embeddings[:, class_index, :]
                )
                branch_hidden = self.class_branch_feature_towers[class_index](
                    branch_feature_inputs[:, class_index, :]
                )
                fusion_input = torch.cat([embedding_hidden, branch_hidden], dim=-1)
                logits.append(
                    self.class_fusion_scorers[class_index](fusion_input).squeeze(-1)
                )
            return ClassEvidenceScoreOutput(logits=torch.stack(logits, dim=1))

        scorer_input = class_evidence_embeddings
        if self.scorer_branch_feature_count > 0:
            if branch_feature_inputs is None:
                raise ValueError(
                    "class evidence scorer branch feature concat is enabled but "
                    "branch_feature_inputs is None"
                )
            expected_shape = (
                class_evidence_embeddings.shape[0],
                class_evidence_embeddings.shape[1],
                self.scorer_branch_feature_count,
            )
            if tuple(branch_feature_inputs.shape) != expected_shape:
                raise ValueError(
                    "branch_feature_inputs must have shape "
                    f"{expected_shape}, got {tuple(branch_feature_inputs.shape)}"
                )
            scorer_input = torch.cat(
                [class_evidence_embeddings, branch_feature_inputs],
                dim=-1,
            )
        elif branch_feature_inputs is not None:
            raise ValueError(
                "branch_feature_inputs were provided but class evidence scorer "
                "branch feature concat is disabled"
            )
        if self.class_gate_scorer == "diagonal":
            if self.class_scorer_weight is None or self.class_scorer_bias is None:
                raise RuntimeError(
                    "diagonal class scorer parameters are not initialized"
                )
            logits = (scorer_input * self.class_scorer_weight.unsqueeze(0)).sum(
                dim=-1
            ) + self.class_scorer_bias.unsqueeze(0)
            return ClassEvidenceScoreOutput(logits=logits)
        if self.class_gate_scorer == "normalized_mlp":
            if self.class_scorers is None:
                raise RuntimeError("normalized_mlp class scorers are not initialized")
            logits = [
                scorer(scorer_input[:, class_index, :]).squeeze(-1)
                for class_index, scorer in enumerate(self.class_scorers)
            ]
            return ClassEvidenceScoreOutput(logits=torch.stack(logits, dim=1))
        raise RuntimeError(
            f"Unsupported class evidence scorer {self.class_gate_scorer}"
        )

    def score_class_evidence(
        self,
        class_evidence_embeddings: Tensor,
        branch_feature_inputs: Tensor | None = None,
        branch_feature_raw_inputs: Tensor | None = None,
    ) -> Tensor:
        return self.score_class_evidence_with_components(
            class_evidence_embeddings,
            branch_feature_inputs,
            branch_feature_raw_inputs=branch_feature_raw_inputs,
        ).logits

    def forward(
        self,
        evidence_tokens: Tensor,
        branch_ids: Tensor | None = None,
    ) -> EvidencePoolingOutput:
        if evidence_tokens.ndim != 3:
            raise ValueError(
                "evidence_tokens must have shape (B, K, D), "
                f"got {tuple(evidence_tokens.shape)}"
            )
        if branch_ids is None:
            raise ValueError(
                "class_aware_branch_gated evidence pooling requires branch_ids"
            )
        if branch_ids.ndim != 2:
            raise ValueError(
                f"branch_ids must have shape (B, K), got {tuple(branch_ids.shape)}"
            )
        if branch_ids.shape != evidence_tokens.shape[:2]:
            raise ValueError(
                "branch_ids must align with evidence_tokens on batch/token "
                f"dimensions; got evidence_tokens={tuple(evidence_tokens.shape)} "
                f"branch_ids={tuple(branch_ids.shape)}"
            )

        branch_evidence_summary, branch_present_mask = summarize_evidence_by_branch(
            evidence_tokens,
            branch_ids,
        )
        if torch.any(branch_present_mask.sum(dim=1) == 0):
            raise ValueError("Every sample must contain at least one evidence branch")

        branch_keys = self.branch_key(branch_evidence_summary)
        gate_logits = torch.einsum("brd,cd->bcr", branch_keys, self.class_queries)
        gate_logits = gate_logits / (branch_keys.shape[-1] ** 0.5)
        gate_logits = gate_logits.masked_fill(
            ~branch_present_mask.unsqueeze(1),
            float("-inf"),
        )
        learned_class_gate_weights = torch.softmax(
            gate_logits / self.temperature,
            dim=-1,
        )
        alpha = self._gate_mixing_alpha()
        if alpha == 0.0:
            class_gate_weights = learned_class_gate_weights
        else:
            branch_counts = branch_present_mask.sum(dim=1, keepdim=True).clamp_min(1)
            uniform_gate = branch_present_mask.to(
                dtype=learned_class_gate_weights.dtype
            ) / branch_counts.to(dtype=learned_class_gate_weights.dtype)
            uniform_gate = uniform_gate.unsqueeze(1).expand_as(
                learned_class_gate_weights
            )
            class_gate_weights = (float(alpha) * uniform_gate) + (
                (1.0 - float(alpha)) * learned_class_gate_weights
            )
        class_gate_entropy = -(
            class_gate_weights
            * (class_gate_weights + torch.finfo(class_gate_weights.dtype).eps).log()
        ).sum(dim=-1)
        learned_class_gate_entropy = -(
            learned_class_gate_weights
            * (
                learned_class_gate_weights
                + torch.finfo(learned_class_gate_weights.dtype).eps
            ).log()
        ).sum(dim=-1)
        class_gate_mixing_alpha = torch.tensor(
            float(alpha),
            device=class_gate_weights.device,
            dtype=class_gate_weights.dtype,
        )
        class_evidence_embeddings = torch.einsum(
            "bcr,brd->bcd",
            class_gate_weights,
            branch_evidence_summary,
        )
        class_evidence_logits = (
            None
            if self.scorer_branch_feature_count > 0
            else self.score_class_evidence(class_evidence_embeddings)
        )
        pooled_embedding = class_evidence_embeddings.mean(dim=1)
        aggregate_gate_weights = class_gate_weights.mean(dim=1)
        aggregate_gate_entropy = -(
            aggregate_gate_weights
            * (
                aggregate_gate_weights + torch.finfo(aggregate_gate_weights.dtype).eps
            ).log()
        ).sum(dim=1)
        branch_evidence_norms = branch_evidence_summary.norm(dim=-1)
        return EvidencePoolingOutput(
            pooled_embedding=pooled_embedding,
            gate_weights=aggregate_gate_weights,
            gate_entropy=aggregate_gate_entropy,
            class_evidence_embeddings=class_evidence_embeddings,
            class_evidence_logits=class_evidence_logits,
            class_gate_weights=class_gate_weights,
            class_gate_entropy=class_gate_entropy,
            learned_class_gate_weights=learned_class_gate_weights,
            learned_class_gate_entropy=learned_class_gate_entropy,
            class_gate_mixing_alpha=class_gate_mixing_alpha,
            branch_evidence_summary=branch_evidence_summary,
            branch_evidence_norms=branch_evidence_norms,
        )


class GlobalResidualLogitCombiner(nn.Module):
    def __init__(
        self,
        *,
        init_scale: float,
        learnable: bool,
        bounding: ClassGateGlobalResidualBoundingConfig,
        correction: ClassGateGlobalResidualCorrectionConfig | None = None,
    ) -> None:
        super().__init__()
        self.bounding = bounding
        self.correction = correction or ClassGateGlobalResidualCorrectionConfig()
        scale = torch.tensor(float(init_scale), dtype=torch.float32)
        self.scale: nn.Parameter | Tensor
        if learnable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def scale_tensor(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.scale.to(device=device, dtype=dtype)

    def bounded_residual_logits(self, residual_logits: Tensor) -> Tensor:
        if not self.bounding.enabled:
            return residual_logits
        return float(self.bounding.bound) * torch.tanh(
            residual_logits / float(self.bounding.temperature)
        )

    def centered_bounded_residual_logits(self, residual_logits: Tensor) -> Tensor:
        residual = self.bounded_residual_logits(residual_logits)
        if self.correction.mode != "gated_zero_mean":
            return residual
        if self.correction.zero_mean:
            residual = residual - residual.mean(dim=-1, keepdim=True)
        if self.correction.rebound and self.bounding.enabled:
            bound = float(self.bounding.bound)
            residual = torch.clamp(residual, min=-bound, max=bound)
        return residual

    def forward(
        self,
        evidence_logits: Tensor,
        residual_logits: Tensor,
        *,
        multiplier: float = 1.0,
        residual_gate: Tensor | None = None,
    ) -> Tensor:
        scale = self.scale.to(
            device=residual_logits.device, dtype=residual_logits.dtype
        )
        effective_scale = scale * float(multiplier)
        residual = self.centered_bounded_residual_logits(residual_logits)
        if self.correction.mode == "gated_zero_mean":
            if residual_gate is None:
                raise ValueError(
                    "gated_zero_mean residual correction requires residual_gate"
                )
            if tuple(residual_gate.shape) != tuple(residual.shape):
                raise ValueError(
                    "residual_gate must match residual logits shape; got "
                    f"{tuple(residual_gate.shape)} and {tuple(residual.shape)}"
                )
            residual = residual_gate * residual
        elif residual_gate is not None:
            raise ValueError(
                "residual_gate was provided but residual correction mode is not "
                "'gated_zero_mean'"
            )
        return evidence_logits + effective_scale * residual


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        mlp_ratio: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.norm(x))


class PatchTokenizer(nn.Module):
    def __init__(self, *, hidden_size: int, branch: PatchBranchConfig) -> None:
        super().__init__()
        self.branch = branch
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=branch.patch_size,
            stride=branch.stride,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FrequencyAttentionPooler(nn.Module):
    def __init__(self, *, hidden_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, branch_grid: Tensor) -> Tensor:
        norm_grid = self.norm(branch_grid)
        attention_logits = self.score(norm_grid).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=2)
        return torch.sum(branch_grid * attention_weights.unsqueeze(-1), dim=2)


class BranchEventTokenDropout(nn.Module):
    def __init__(self, cfg: BranchEventDropoutConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _ensure_min_keep(keep_mask: Tensor, min_keep_tokens: int) -> Tensor:
        for batch_index in range(keep_mask.shape[0]):
            keep_count = int(keep_mask[batch_index].sum().item())
            if keep_count >= min_keep_tokens:
                continue
            dropped_positions = torch.where(~keep_mask[batch_index])[0]
            if dropped_positions.numel() == 0:
                continue
            needed = min(min_keep_tokens - keep_count, int(dropped_positions.numel()))
            order = torch.randperm(
                int(dropped_positions.numel()),
                device=keep_mask.device,
            )
            keep_mask[batch_index, dropped_positions[order[:needed]]] = True
        return keep_mask

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape (B, T, D), got {tuple(tokens.shape)}"
            )
        batch_size, token_count, _ = tokens.shape
        all_true = torch.ones(
            batch_size,
            token_count,
            dtype=torch.bool,
            device=tokens.device,
        )
        if not self.training or not self.cfg.enabled or self.cfg.probability <= 0:
            return tokens, all_true

        keep_mask = torch.rand(batch_size, token_count, device=tokens.device) >= float(
            self.cfg.probability
        )
        min_keep_tokens = min(int(self.cfg.min_keep_tokens), token_count)
        keep_mask = self._ensure_min_keep(keep_mask, min_keep_tokens)
        dropped_tokens = tokens * keep_mask.unsqueeze(-1).to(tokens.dtype)
        return dropped_tokens, keep_mask


class SelectedEvidenceDropout(nn.Module):
    def __init__(self, cfg: SelectedEvidenceDropoutConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        selected_tokens: Tensor,
        selected_branch_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if selected_tokens.ndim != 3:
            raise ValueError(
                "selected_tokens must have shape (B, K, D), "
                f"got {tuple(selected_tokens.shape)}"
            )
        if selected_branch_ids.ndim != 2:
            raise ValueError(
                "selected_branch_ids must have shape (B, K), "
                f"got {tuple(selected_branch_ids.shape)}"
            )
        if selected_branch_ids.shape != selected_tokens.shape[:2]:
            raise ValueError(
                "selected_branch_ids must align with selected_tokens on batch/token "
                f"dimensions; got selected_tokens={tuple(selected_tokens.shape)} "
                f"selected_branch_ids={tuple(selected_branch_ids.shape)}"
            )

        batch_size, selected_count, _ = selected_tokens.shape
        keep_mask = torch.ones(
            batch_size,
            selected_count,
            dtype=torch.bool,
            device=selected_tokens.device,
        )
        if not self.training or not self.cfg.enabled or self.cfg.probability <= 0:
            return selected_tokens, keep_mask

        for batch_index in range(batch_size):
            branch_ids = torch.unique(selected_branch_ids[batch_index])
            for branch_id in branch_ids:
                positions = torch.where(selected_branch_ids[batch_index] == branch_id)[
                    0
                ]
                if positions.numel() == 0:
                    continue
                branch_keep = torch.rand(
                    int(positions.numel()), device=selected_tokens.device
                ) >= float(self.cfg.probability)
                min_keep = min(
                    int(self.cfg.min_keep_per_branch), int(positions.numel())
                )
                keep_count = int(branch_keep.sum().item())
                if keep_count < min_keep:
                    dropped_positions = torch.where(~branch_keep)[0]
                    needed = min(min_keep - keep_count, int(dropped_positions.numel()))
                    order = torch.randperm(
                        int(dropped_positions.numel()),
                        device=selected_tokens.device,
                    )
                    branch_keep[dropped_positions[order[:needed]]] = True
                keep_mask[batch_index, positions] = branch_keep

        dropped_tokens = selected_tokens * keep_mask.unsqueeze(-1).to(
            selected_tokens.dtype
        )
        return dropped_tokens, keep_mask


class BranchMilHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        output_dim: int,
        layer_norm_eps: float,
        attention_temperature: float,
    ) -> None:
        super().__init__()
        self.attention_temperature = attention_temperature
        self.token_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = nn.Linear(hidden_size, 1)
        self.instance_logit_proj = nn.Linear(hidden_size, output_dim)
        self.embedding_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.logit_proj = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        event_tokens: Tensor,
        token_mask: Tensor | None = None,
    ) -> BranchMilOutput:
        normalized_tokens = self.token_norm(event_tokens)
        attention_logits = self.attention(normalized_tokens).squeeze(-1)
        if token_mask is not None:
            if token_mask.shape != attention_logits.shape:
                raise ValueError(
                    "token_mask must align with event token batch/time dimensions; "
                    f"got token_mask={tuple(token_mask.shape)} "
                    f"event_tokens={tuple(event_tokens.shape)}"
                )
            attention_logits = attention_logits.masked_fill(~token_mask, -1e4)
        attention_weights = torch.softmax(
            attention_logits / self.attention_temperature,
            dim=1,
        )
        embedding = torch.sum(
            event_tokens * attention_weights.unsqueeze(-1),
            dim=1,
        )
        instance_logits = self.instance_logit_proj(normalized_tokens)
        if instance_logits.ndim == 3 and instance_logits.shape[-1] == 1:
            instance_logits = instance_logits.squeeze(-1)
        logits = self.logit_proj(self.embedding_norm(embedding))
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return BranchMilOutput(
            logits=logits,
            attention_weights=attention_weights,
            attention_logits=attention_logits,
            instance_logits=instance_logits,
            embedding=embedding,
        )


class MultiScalePatchStemAdapterEncoder(nn.Module):
    def __init__(self, cfg: MultiScaleRdtEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        architecture = cfg.architecture
        hidden_size = architecture.hidden_size
        self.branch_token_grids = tuple(
            compute_token_grid(feature_dims=cfg.feature_dims, patch_branch=branch)
            for branch in architecture.patch_branches
        )
        self.branch_time_lengths = tuple(
            time_steps for time_steps, _ in self.branch_token_grids
        )
        self.branch_frequency_lengths = tuple(
            freq_steps for _, freq_steps in self.branch_token_grids
        )
        self.branch_token_counts = tuple(
            time_steps * freq_steps
            for time_steps, freq_steps in self.branch_token_grids
        )
        self.total_token_count = sum(self.branch_token_counts)
        self.total_temporal_length = sum(self.branch_time_lengths)
        self.patch_tokenizers = nn.ModuleList(
            PatchTokenizer(hidden_size=hidden_size, branch=branch)
            for branch in architecture.patch_branches
        )
        self.position_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(1, token_count, hidden_size))
            for token_count in self.branch_token_counts
        )
        self.scale_embeddings = nn.Parameter(
            torch.empty(len(architecture.patch_branches), hidden_size)
        )
        self.shared_stem = nn.ModuleList(
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=architecture.num_attention_heads,
                mlp_ratio=architecture.mlp_ratio,
                dropout=architecture.hidden_dropout_prob,
                attention_dropout=architecture.attention_probs_dropout_prob,
                layer_norm_eps=architecture.layer_norm_eps,
            )
            for _ in range(architecture.shared_stem_depth)
        )
        self.adapters = nn.ModuleList(
            nn.ModuleList(
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=architecture.num_attention_heads,
                    mlp_ratio=architecture.mlp_ratio,
                    dropout=architecture.hidden_dropout_prob,
                    attention_dropout=architecture.attention_probs_dropout_prob,
                    layer_norm_eps=architecture.layer_norm_eps,
                )
                for _ in range(architecture.adapter_depth)
            )
            for _ in architecture.patch_branches
        )
        self.frequency_poolers = nn.ModuleList(
            FrequencyAttentionPooler(
                hidden_size=hidden_size,
                layer_norm_eps=architecture.layer_norm_eps,
            )
            for _ in architecture.patch_branches
        )
        self.output_norm = nn.LayerNorm(hidden_size, eps=architecture.layer_norm_eps)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for position_embedding in self.position_embeddings:
            nn.init.normal_(position_embedding, std=PATCH_INIT_STD)
        nn.init.normal_(self.scale_embeddings, std=PATCH_INIT_STD)

    def forward(self, x: Tensor) -> MultiScaleEncoderOutput:
        branch_event_outputs: list[Tensor] = []
        batch_size = x.shape[0]
        for branch_index, tokenizer in enumerate(self.patch_tokenizers):
            branch_tokens = tokenizer(x)
            branch_tokens = (
                branch_tokens
                + self.position_embeddings[branch_index]
                + self.scale_embeddings[branch_index].view(1, 1, -1)
            )
            for block in self.shared_stem:
                branch_tokens = block(branch_tokens)
            branch_adapter = cast(nn.ModuleList, self.adapters[branch_index])
            for block in branch_adapter:
                branch_tokens = block(branch_tokens)

            time_steps, freq_steps = self.branch_token_grids[branch_index]
            branch_grid = branch_tokens.reshape(
                batch_size,
                time_steps,
                freq_steps,
                branch_tokens.shape[-1],
            )
            frequency_pooler = cast(
                FrequencyAttentionPooler,
                self.frequency_poolers[branch_index],
            )
            branch_event_tokens = frequency_pooler(branch_grid)
            branch_event_outputs.append(self.output_norm(branch_event_tokens))

        context_tokens = torch.cat(branch_event_outputs, dim=1)
        return MultiScaleEncoderOutput(
            branch_event_tokens=tuple(branch_event_outputs),
            context_tokens=context_tokens,
        )


class RdtRefinementBlock(nn.Module):
    def __init__(self, cfg: MultiScaleRdtArchitectureConfig) -> None:
        super().__init__()
        self.gated_residual = cfg.rdt.gated_residual
        self.latent_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.context_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.feed_forward = FeedForwardBlock(
            hidden_size=cfg.hidden_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.hidden_dropout_prob,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.self_attn_scale: nn.Parameter | None
        self.cross_attn_scale: nn.Parameter | None
        self.ffn_scale: nn.Parameter | None
        if self.gated_residual:
            init = float(cfg.rdt.layerscale_init)
            self.self_attn_scale = nn.Parameter(torch.full((cfg.hidden_size,), init))
            self.cross_attn_scale = nn.Parameter(torch.full((cfg.hidden_size,), init))
            self.ffn_scale = nn.Parameter(torch.full((cfg.hidden_size,), init))
        else:
            self.self_attn_scale = None
            self.cross_attn_scale = None
            self.ffn_scale = None

    def _apply_residual(
        self,
        x: Tensor,
        delta: Tensor,
        scale: nn.Parameter | None,
    ) -> Tensor:
        if scale is None:
            return x + delta
        return x + scale.view(1, 1, -1) * delta

    def forward(self, latent_states: Tensor, evidence_memory: Tensor) -> Tensor:
        norm_latent = self.latent_norm(latent_states)
        self_attended, _ = self.latent_self_attn(
            norm_latent,
            norm_latent,
            norm_latent,
            need_weights=False,
        )
        latent_states = self._apply_residual(
            latent_states,
            self.dropout(self_attended),
            self.self_attn_scale,
        )

        norm_queries = self.query_norm(latent_states)
        norm_memory = self.context_norm(evidence_memory)
        cross_attended, _ = self.cross_attn(
            norm_queries,
            norm_memory,
            norm_memory,
            need_weights=False,
        )
        latent_states = self._apply_residual(
            latent_states,
            self.dropout(cross_attended),
            self.cross_attn_scale,
        )

        ff_delta = self.feed_forward(latent_states)
        return self._apply_residual(latent_states, ff_delta, self.ffn_scale)


class MultiScaleRdtAstModel(nn.Module):
    def __init__(self, cfg: MultiScaleRdtAstModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._runtime_epoch: int | None = None
        architecture = cfg.encoder.architecture
        self.encoder = MultiScalePatchStemAdapterEncoder(cfg.encoder)
        self.class_aware_evidence_pooling = (
            architecture.evidence_pooling.type == "class_aware_branch_gated"
        )
        output_dim = (
            cfg.num_classes
            if self.class_aware_evidence_pooling
            else (1 if cfg.num_classes == 2 else cfg.num_classes)
        )
        self.branch_mil_heads = nn.ModuleList(
            BranchMilHead(
                hidden_size=architecture.hidden_size,
                output_dim=output_dim,
                layer_norm_eps=architecture.layer_norm_eps,
                attention_temperature=architecture.mil.attention_temperature,
            )
            for _ in architecture.patch_branches
        )
        self.branch_binary_head = nn.Linear(architecture.hidden_size, 1)
        branch_event_dropout_cfg = architecture.token_augmentation.branch_event_dropout
        self.branch_event_dropout = (
            BranchEventTokenDropout(branch_event_dropout_cfg)
            if branch_event_dropout_cfg.enabled
            else None
        )
        selected_evidence_dropout_cfg = (
            architecture.token_augmentation.selected_evidence_dropout
        )
        self.selected_evidence_dropout = (
            SelectedEvidenceDropout(selected_evidence_dropout_cfg)
            if selected_evidence_dropout_cfg.enabled
            else None
        )
        self.rdt_block = (
            RdtRefinementBlock(architecture) if architecture.rdt.enabled else None
        )
        self.global_residual_combiner: GlobalResidualLogitCombiner | None = None
        self.global_residual_gate: nn.Module | None = None
        if architecture.evidence_pooling.type == "mean":
            self.evidence_pooler: nn.Module = MeanEvidencePooling()
        elif architecture.evidence_pooling.type == "branch_gated":
            self.evidence_pooler = BranchAwareGatedEvidencePooling(
                hidden_size=architecture.hidden_size,
                cfg=architecture.evidence_pooling,
            )
        elif architecture.evidence_pooling.type == "class_aware_branch_gated":
            if cfg.num_classes < 2:
                raise ValueError(
                    "class_aware_branch_gated evidence pooling requires num_classes >= 2"
                )
            self.evidence_pooler = ClassAwareBranchGatedEvidencePooling(
                hidden_size=architecture.hidden_size,
                num_classes=cfg.num_classes,
                cfg=architecture.evidence_pooling,
            )
            residual_cfg = architecture.evidence_pooling.class_gate.global_residual
            if residual_cfg.enabled:
                self.global_residual_combiner = GlobalResidualLogitCombiner(
                    init_scale=residual_cfg.init_scale,
                    learnable=residual_cfg.learnable,
                    bounding=residual_cfg.bounding,
                    correction=residual_cfg.correction,
                )
                if residual_cfg.correction.mode == "gated_zero_mean":
                    self.global_residual_gate = self._build_global_residual_gate(
                        input_dim=5,
                        hidden_size=residual_cfg.correction.gate_hidden_size,
                        dropout=residual_cfg.correction.dropout,
                    )
        else:
            raise ValueError(
                "Unsupported evidence pooling type: "
                f"{architecture.evidence_pooling.type}"
            )
        num_branches = len(architecture.patch_branches)
        if self.class_aware_evidence_pooling:
            fusion_input_dim = (cfg.num_classes * architecture.hidden_size) + output_dim
        else:
            fusion_input_dim = (2 * architecture.hidden_size) + (
                num_branches * output_dim
            )
        self.fusion_projector = self._build_fusion_projector(
            input_dim=fusion_input_dim,
            output_dim=architecture.hidden_size,
            cfg=cfg.classifier.fusion_projector,
            dropout=cfg.classifier.dropout,
        )
        classifier_dims = ClassifierDims(
            in_dim=architecture.hidden_size,
            num_classes=output_dim,
            hidden_dim=cfg.classifier.hidden_dim,
            dropout=cfg.classifier.dropout,
        )
        if cfg.classifier.type == "linear":
            self.classifier: nn.Module = LinearClassifier(classifier_dims)
        elif cfg.classifier.type == "mlp":
            self.classifier = MlpClassifier(classifier_dims)
        else:
            raise ValueError(f"Unsupported classifier type: {cfg.classifier.type}")

    def set_runtime_epoch(self, epoch: int | None) -> None:
        self._runtime_epoch = None if epoch is None else int(epoch)
        set_runtime_epoch = getattr(self.evidence_pooler, "set_runtime_epoch", None)
        if callable(set_runtime_epoch):
            set_runtime_epoch(self._runtime_epoch)

    @staticmethod
    def _build_fusion_projector(
        *,
        input_dim: int,
        output_dim: int,
        cfg: FusionProjectorConfig,
        dropout: float,
    ) -> nn.Sequential:
        if cfg.type == "linear":
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        if cfg.type == "mlp":
            layers: list[nn.Module] = []
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend(
                [
                    nn.Linear(input_dim, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(cfg.hidden_dim, output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            return nn.Sequential(*layers)
        raise ValueError(f"Unsupported fusion projector type: {cfg.type}")

    def encoder_side_modules(self) -> tuple[nn.Module, ...]:
        return (self.encoder, self.branch_mil_heads, self.branch_binary_head)

    def head_side_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [
            self.evidence_pooler,
            self.fusion_projector,
            self.classifier,
        ]
        if self.global_residual_combiner is not None:
            modules.append(self.global_residual_combiner)
        if self.global_residual_gate is not None:
            modules.append(self.global_residual_gate)
        if self.rdt_block is not None:
            modules.insert(0, self.rdt_block)
        return tuple(modules)

    def _stack_branch_logits(
        self,
        branch_logits: list[Tensor],
    ) -> Tensor:
        if self.cfg.num_classes == 2:
            return torch.stack(branch_logits, dim=1)
        return torch.stack(branch_logits, dim=1)

    @staticmethod
    def _class_gated_branch_logit_features(
        raw_scores: Tensor,
        *,
        mode: Literal["raw", "hardest_negative_margin"],
    ) -> Tensor:
        if mode == "raw":
            return raw_scores
        if mode != "hardest_negative_margin":
            raise ValueError(f"Unsupported branch logit feature mode: {mode}")
        if raw_scores.ndim != 2:
            raise ValueError(
                "class_gated_branch_logits must have shape (B, C), "
                f"got {tuple(raw_scores.shape)}"
            )
        num_classes = raw_scores.shape[1]
        if num_classes < 2:
            raise ValueError(
                "hardest_negative_margin branch logit features require "
                "at least two classes"
            )
        negative_scores = raw_scores.unsqueeze(1).expand(-1, num_classes, -1)
        self_mask = torch.eye(
            num_classes,
            dtype=torch.bool,
            device=raw_scores.device,
        ).unsqueeze(0)
        hardest_negative = (
            negative_scores.masked_fill(
                self_mask,
                float("-inf"),
            )
            .max(dim=-1)
            .values
        )
        return raw_scores - hardest_negative

    @staticmethod
    def _class_top_branch_margin_features(branch_logits: Tensor) -> Tensor:
        if branch_logits.ndim != 3:
            raise ValueError(
                "branch_logits must have shape (B, R, C), "
                f"got {tuple(branch_logits.shape)}"
            )
        num_classes = int(branch_logits.shape[2])
        if num_classes < 2:
            raise ValueError(
                "top_branch_margin_style features require at least two classes"
            )
        expanded = branch_logits.unsqueeze(2).expand(-1, -1, num_classes, -1)
        self_mask = torch.eye(
            num_classes,
            dtype=torch.bool,
            device=branch_logits.device,
        ).view(1, 1, num_classes, num_classes)
        hardest_negative = (
            expanded.masked_fill(self_mask, float("-inf")).max(dim=-1).values
        )
        branch_class_margins = branch_logits - hardest_negative
        return branch_class_margins.max(dim=1).values

    @staticmethod
    def _class_relative_features(class_values: Tensor) -> Tensor:
        if class_values.ndim != 2:
            raise ValueError(
                f"class_values must have shape (B, C), got {tuple(class_values.shape)}"
            )
        num_classes = int(class_values.shape[1])
        if num_classes < 2:
            raise ValueError("relative class features require at least two classes")
        expanded = class_values.unsqueeze(1).expand(-1, num_classes, -1)
        self_mask = torch.eye(
            num_classes,
            dtype=torch.bool,
            device=class_values.device,
        ).unsqueeze(0)
        hardest_other = (
            expanded.masked_fill(self_mask, float("-inf")).max(dim=-1).values
        )
        return class_values - hardest_other

    @staticmethod
    def _build_global_residual_gate(
        *,
        input_dim: int,
        hidden_size: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _global_residual_gate_values(
        self,
        *,
        class_evidence_logits: Tensor,
        class_gated_branch_logit_features: Tensor,
        class_gated_branch_logit_relative_features: Tensor,
        class_top_branch_margin_features: Tensor,
        class_top_branch_margin_relative_features: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if self.global_residual_gate is None:
            return None, None, None, None
        gate_input = torch.stack(
            [
                class_evidence_logits,
                class_gated_branch_logit_features,
                class_gated_branch_logit_relative_features,
                class_top_branch_margin_features,
                class_top_branch_margin_relative_features,
            ],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.global_residual_gate(gate_input).squeeze(-1))
        correction = self.cfg.encoder.architecture.evidence_pooling.class_gate.global_residual.correction
        confidence_cfg = correction.confidence_aware_gate
        if not confidence_cfg.enabled:
            return learned_gate, learned_gate, None, None
        if (
            confidence_cfg.source != "evidence_gap"
            or confidence_cfg.mode != "damped_sigmoid"
        ):
            raise ValueError(
                "confidence-aware residual gate supports only "
                "source='evidence_gap' and mode='damped_sigmoid'"
            )
        evidence_gap = self._class_relative_features(class_evidence_logits)
        evidence_confidence = torch.sigmoid(
            evidence_gap / float(confidence_cfg.temperature)
        )
        confidence_factor = 1.0 - (float(confidence_cfg.damping) * evidence_confidence)
        return (
            learned_gate * confidence_factor,
            learned_gate,
            evidence_confidence,
            confidence_factor,
        )

    def _class_evidence_scorer_branch_raw_features(
        self,
        *,
        class_gated_branch_logit_features: Tensor,
        class_gated_branch_logit_relative_features: Tensor,
        class_top_branch_margin_features: Tensor,
        class_top_branch_margin_relative_features: Tensor,
    ) -> Tensor | None:
        class_gate = self.cfg.encoder.architecture.evidence_pooling.class_gate
        evidence_scorer = class_gate.evidence_scorer
        if evidence_scorer.type == "two_tower_mlp":
            return torch.stack(
                [
                    class_gated_branch_logit_features,
                    class_top_branch_margin_features,
                ],
                dim=-1,
            )
        if evidence_scorer.type == "class_axis_attention":
            return torch.stack(
                [
                    class_gated_branch_logit_features,
                    class_gated_branch_logit_relative_features,
                    class_top_branch_margin_features,
                    class_top_branch_margin_relative_features,
                ],
                dim=-1,
            )
        return None

    def _class_evidence_scorer_branch_features(
        self,
        *,
        raw_features: Tensor | None,
    ) -> Tensor | None:
        if raw_features is None:
            return None
        class_gate = self.cfg.encoder.architecture.evidence_pooling.class_gate
        evidence_scorer = class_gate.evidence_scorer
        transform = evidence_scorer.branch_feature_transform
        if transform.mode != "tanh":
            raise ValueError(
                "Unsupported class evidence branch feature transform mode: "
                f"{transform.mode}"
            )
        return torch.tanh(raw_features / float(transform.temperature))

    def _global_residual_schedule_multiplier(self) -> float:
        residual_cfg = (
            self.cfg.encoder.architecture.evidence_pooling.class_gate.global_residual
        )
        warmup = residual_cfg.warmup
        return _resolve_hold_decay_schedule(
            enabled=warmup.enabled,
            start_value=float(warmup.start_multiplier),
            end_value=float(warmup.end_multiplier),
            hold_epochs=int(warmup.hold_epochs),
            decay_epochs=int(warmup.decay_epochs),
            epoch=self._runtime_epoch,
        )

    def forward(self, input_values: Tensor) -> AstModelOutput:
        if input_values.ndim != 3:
            raise ValueError(
                "input_values must have shape (B, max_length, num_mel_bins), "
                f"got {tuple(input_values.shape)}"
            )
        expected_shape = (
            self.cfg.encoder.feature_dims.max_length,
            self.cfg.encoder.feature_dims.num_mel_bins,
        )
        actual_shape = (int(input_values.shape[1]), int(input_values.shape[2]))
        if actual_shape != expected_shape:
            raise ValueError(
                "input_values feature dims do not match model feature dims. "
                f"expected {expected_shape}, got {actual_shape}"
            )

        encoder_output = self.encoder(input_values.unsqueeze(1))
        branch_logits: list[Tensor] = []
        branch_binary_logits: list[Tensor] = []
        branch_embeddings: list[Tensor] = []
        branch_attention_weights: list[Tensor] = []
        selected_tokens: list[Tensor] = []
        selected_indices: list[Tensor] = []
        selected_scores: list[Tensor] = []
        selected_branch_ids: list[Tensor] = []
        top_k = self.cfg.encoder.architecture.rdt.top_tokens_per_branch
        evidence_source = self.cfg.encoder.architecture.rdt.evidence_score_source
        excluded_branches = set(
            self.cfg.encoder.architecture.rdt.exclude_branches_from_evidence
        )
        for branch_index, (branch_tokens, branch_head) in enumerate(
            zip(
                encoder_output.branch_event_tokens,
                self.branch_mil_heads,
                strict=True,
            )
        ):
            branch_token_mask = None
            if self.branch_event_dropout is not None:
                branch_tokens, branch_token_mask = self.branch_event_dropout(
                    branch_tokens
                )
            mil_output = cast(
                BranchMilOutput,
                branch_head(branch_tokens, token_mask=branch_token_mask),
            )
            branch_logits.append(mil_output.logits)
            branch_binary_logits.append(
                self.branch_binary_head(mil_output.embedding).squeeze(-1)
            )
            branch_embeddings.append(mil_output.embedding)
            branch_attention_weights.append(mil_output.attention_weights)
            if branch_index in excluded_branches:
                continue
            evidence_scores = get_evidence_scores(
                mil_output,
                source=evidence_source,
            )
            selected = select_top_tokens(
                branch_tokens,
                evidence_scores,
                top_k=top_k,
            )
            selected_tokens.append(selected.tokens)
            selected_indices.append(selected.indices)
            selected_scores.append(selected.scores)
            selected_branch_ids.append(
                torch.full_like(selected.indices, fill_value=branch_index)
            )

        stacked_branch_logits = self._stack_branch_logits(branch_logits)
        stacked_branch_binary_logits = torch.stack(branch_binary_logits, dim=1)
        if not selected_tokens:
            raise ValueError("At least one branch must contribute selected evidence")
        selected_evidence_tokens = torch.cat(selected_tokens, dim=1)
        selected_evidence_indices = torch.cat(selected_indices, dim=1)
        selected_evidence_scores = torch.cat(selected_scores, dim=1)
        selected_evidence_branch_ids = torch.cat(selected_branch_ids, dim=1)
        evidence_tokens = selected_evidence_tokens
        selected_evidence_dropout_mask = None
        selected_evidence_keep_ratio = None
        if self.selected_evidence_dropout is not None:
            evidence_tokens, selected_evidence_dropout_mask = (
                self.selected_evidence_dropout(
                    evidence_tokens,
                    selected_evidence_branch_ids,
                )
            )
            selected_evidence_keep_ratio = selected_evidence_dropout_mask.to(
                evidence_tokens.dtype
            ).mean(dim=1)
        if self.rdt_block is not None:
            for _ in range(self.cfg.encoder.architecture.rdt.steps):
                evidence_tokens = self.rdt_block(
                    evidence_tokens,
                    encoder_output.context_tokens,
                )

        pooling_output = self.evidence_pooler(
            evidence_tokens,
            selected_evidence_branch_ids,
        )
        evidence_embedding = pooling_output.pooled_embedding
        class_evidence_logits = pooling_output.class_evidence_logits
        global_residual_logits: Tensor | None = None
        bounded_global_residual_logits: Tensor | None = None
        global_residual_scale: Tensor | None = None
        global_residual_schedule_multiplier: Tensor | None = None
        global_residual_effective_scale: Tensor | None = None
        global_residual_bound: Tensor | None = None
        global_residual_temperature: Tensor | None = None
        centered_bounded_global_residual_logits: Tensor | None = None
        global_residual_learned_gate: Tensor | None = None
        global_residual_evidence_confidence: Tensor | None = None
        global_residual_confidence_factor: Tensor | None = None
        global_residual_gate: Tensor | None = None
        global_residual_contribution: Tensor | None = None
        global_residual_zero_mean_enabled: bool | None = None
        global_residual_rebound_enabled: bool | None = None
        class_gated_branch_logits: Tensor | None = None
        class_gated_branch_logit_features: Tensor | None = None
        class_gated_branch_logit_feature_mode: str | None = None
        class_top_branch_margin_features: Tensor | None = None
        class_gated_branch_logit_relative_features: Tensor | None = None
        class_top_branch_margin_relative_features: Tensor | None = None
        class_evidence_scorer_branch_raw_features: Tensor | None = None
        class_evidence_scorer_branch_features: Tensor | None = None
        class_evidence_embedding_scores: Tensor | None = None
        class_evidence_raw_embedding_scores: Tensor | None = None
        class_evidence_bounded_embedding_scores: Tensor | None = None
        class_evidence_top_support_scores: Tensor | None = None
        class_evidence_direct_top_scores: Tensor | None = None
        class_evidence_top_support_residual_scores: Tensor | None = None
        class_evidence_gated_support_scores: Tensor | None = None
        class_evidence_top_raw_existential_scores: Tensor | None = None
        class_evidence_top_relative_correction_scores: Tensor | None = None
        class_evidence_top_support_raw_positive_component: Tensor | None = None
        class_evidence_top_support_relative_positive_component: Tensor | None = None
        class_evidence_top_support_relative_negative_component: Tensor | None = None
        class_evidence_top_support_relative_negative_component_uncapped: (
            Tensor | None
        ) = None
        class_evidence_top_support_relative_negative_component_capped: Tensor | None = (
            None
        )
        class_evidence_top_support_relative_negative_cap_value: Tensor | None = None
        class_evidence_top_support_relative_negative_cap_active: Tensor | None = None
        class_evidence_top_support_direct_raw_scale: Tensor | None = None
        class_evidence_top_support_direct_relative_positive_scale: Tensor | None = None
        class_evidence_top_support_direct_relative_negative_scale: Tensor | None = None
        class_evidence_top_support_direct_residual_scale: Tensor | None = None
        class_evidence_top_relative_positive: Tensor | None = None
        class_evidence_top_relative_negative: Tensor | None = None
        class_evidence_gate_reliability: Tensor | None = None
        class_evidence_gate_reliability_regret: Tensor | None = None
        class_evidence_top_margin: Tensor | None = None
        class_evidence_gated_margin: Tensor | None = None
        class_evidence_branch_existential_scores: Tensor | None = None
        class_evidence_branch_competitive_scores: Tensor | None = None
        class_evidence_branch_direct_scores: Tensor | None = None
        class_evidence_branch_residual_scores: Tensor | None = None
        class_evidence_branch_support_scores: Tensor | None = None
        class_evidence_interaction_scores: Tensor | None = None
        class_evidence_raw_interaction_scores: Tensor | None = None
        class_evidence_bounded_interaction_scores: Tensor | None = None
        class_evidence_branch_scale: Tensor | None = None
        class_evidence_branch_direct_top_scale: Tensor | None = None
        class_evidence_branch_direct_gated_scale: Tensor | None = None
        class_evidence_branch_direct_raw_scale: Tensor | None = None
        class_evidence_branch_direct_relative_scale: Tensor | None = None
        class_evidence_branch_direct_residual_scale: Tensor | None = None
        class_evidence_branch_direct_top_weights: Tensor | None = None
        class_evidence_branch_direct_gated_weights: Tensor | None = None
        class_evidence_branch_direct_existential_weights: Tensor | None = None
        class_evidence_branch_direct_competitive_weights: Tensor | None = None
        class_evidence_interaction_scale: Tensor | None = None
        class_evidence_interaction_scale_multiplier: Tensor | None = None
        class_evidence_interaction_effective_scale: Tensor | None = None
        class_evidence_embedding_score_bound: Tensor | None = None
        class_evidence_embedding_score_temperature: Tensor | None = None
        class_evidence_interaction_score_bound: Tensor | None = None
        class_evidence_interaction_score_temperature: Tensor | None = None
        if self.class_aware_evidence_pooling:
            class_evidence_gate_weights = pooling_output.class_gate_weights
            if class_evidence_gate_weights is None:
                raise ValueError(
                    "class_aware_branch_gated evidence pooling must return "
                    "class_gate_weights"
                )
            if class_evidence_gate_weights.ndim != 3:
                raise ValueError(
                    "class_gate_weights must have shape (B, C, R), "
                    f"got {tuple(class_evidence_gate_weights.shape)}"
                )
            if stacked_branch_logits.ndim != 3:
                raise ValueError(
                    "class_aware_branch_gated branch logits must have shape "
                    f"(B, R, C), got {tuple(stacked_branch_logits.shape)}"
                )
            if (
                class_evidence_gate_weights.shape[0] != stacked_branch_logits.shape[0]
                or class_evidence_gate_weights.shape[1]
                != stacked_branch_logits.shape[2]
                or class_evidence_gate_weights.shape[2]
                != stacked_branch_logits.shape[1]
            ):
                raise ValueError(
                    "class_gate_weights must align with branch logits as "
                    "(B, C, R) vs (B, R, C); got "
                    f"{tuple(class_evidence_gate_weights.shape)} and "
                    f"{tuple(stacked_branch_logits.shape)}"
                )
            class_evidence_embeddings = pooling_output.class_evidence_embeddings
            if class_evidence_embeddings is None:
                raise ValueError(
                    "class_aware_branch_gated evidence pooling must return "
                    "class_evidence_embeddings"
                )
            if class_evidence_embeddings.ndim != 3:
                raise ValueError(
                    "class_evidence_embeddings must have shape (B, C, D), "
                    f"got {tuple(class_evidence_embeddings.shape)}"
                )
            class_gated_branch_logits = torch.einsum(
                "bcr,brc->bc",
                class_evidence_gate_weights,
                stacked_branch_logits,
            )
            branch_feature_cfg = self.cfg.encoder.architecture.evidence_pooling.class_gate.branch_logit_feature
            class_gated_branch_logit_feature_mode = branch_feature_cfg.mode
            class_gated_branch_logit_features = self._class_gated_branch_logit_features(
                class_gated_branch_logits,
                mode=branch_feature_cfg.mode,
            )
            class_top_branch_margin_features = self._class_top_branch_margin_features(
                stacked_branch_logits
            )
            class_gated_branch_logit_relative_features = self._class_relative_features(
                class_gated_branch_logit_features
            )
            class_top_branch_margin_relative_features = self._class_relative_features(
                class_top_branch_margin_features
            )
            class_evidence_scorer_branch_raw_features = (
                self._class_evidence_scorer_branch_raw_features(
                    class_gated_branch_logit_features=(
                        class_gated_branch_logit_features
                    ),
                    class_gated_branch_logit_relative_features=(
                        class_gated_branch_logit_relative_features
                    ),
                    class_top_branch_margin_features=(class_top_branch_margin_features),
                    class_top_branch_margin_relative_features=(
                        class_top_branch_margin_relative_features
                    ),
                )
            )
            class_evidence_scorer_branch_features = (
                self._class_evidence_scorer_branch_features(
                    raw_features=class_evidence_scorer_branch_raw_features,
                )
            )
            if class_evidence_scorer_branch_features is not None:
                score_class_evidence = getattr(
                    self.evidence_pooler,
                    "score_class_evidence_with_components",
                    None,
                )
                if not callable(score_class_evidence):
                    raise RuntimeError(
                        "class-aware evidence pooler cannot score class evidence"
                    )
                class_evidence_score_output = score_class_evidence(
                    class_evidence_embeddings,
                    class_evidence_scorer_branch_features,
                    branch_feature_raw_inputs=(
                        class_evidence_scorer_branch_raw_features
                    ),
                )
                class_evidence_logits = class_evidence_score_output.logits
                class_evidence_embedding_scores = (
                    class_evidence_score_output.embedding_scores
                )
                class_evidence_raw_embedding_scores = (
                    class_evidence_score_output.raw_embedding_scores
                )
                class_evidence_bounded_embedding_scores = (
                    class_evidence_score_output.bounded_embedding_scores
                )
                class_evidence_top_support_scores = (
                    class_evidence_score_output.top_support_scores
                )
                class_evidence_direct_top_scores = (
                    class_evidence_score_output.direct_top_scores
                )
                class_evidence_top_support_residual_scores = (
                    class_evidence_score_output.top_support_residual_scores
                )
                class_evidence_gated_support_scores = (
                    class_evidence_score_output.gated_support_scores
                )
                class_evidence_top_raw_existential_scores = (
                    class_evidence_score_output.top_raw_existential_scores
                )
                class_evidence_top_relative_correction_scores = (
                    class_evidence_score_output.top_relative_correction_scores
                )
                class_evidence_top_support_raw_positive_component = (
                    class_evidence_score_output.top_support_raw_positive_component
                )
                class_evidence_top_support_relative_positive_component = (
                    class_evidence_score_output.top_support_relative_positive_component
                )
                class_evidence_top_support_relative_negative_component = (
                    class_evidence_score_output.top_support_relative_negative_component
                )
                class_evidence_top_support_relative_negative_component_uncapped = class_evidence_score_output.top_support_relative_negative_component_uncapped
                class_evidence_top_support_relative_negative_component_capped = class_evidence_score_output.top_support_relative_negative_component_capped
                class_evidence_top_support_relative_negative_cap_value = (
                    class_evidence_score_output.top_support_relative_negative_cap_value
                )
                class_evidence_top_support_relative_negative_cap_active = (
                    class_evidence_score_output.top_support_relative_negative_cap_active
                )
                class_evidence_top_support_direct_raw_scale = (
                    class_evidence_score_output.top_support_direct_raw_scale
                )
                class_evidence_top_support_direct_relative_positive_scale = class_evidence_score_output.top_support_direct_relative_positive_scale
                class_evidence_top_support_direct_relative_negative_scale = class_evidence_score_output.top_support_direct_relative_negative_scale
                class_evidence_top_support_direct_residual_scale = (
                    class_evidence_score_output.top_support_direct_residual_scale
                )
                class_evidence_top_relative_positive = (
                    class_evidence_score_output.top_relative_positive
                )
                class_evidence_top_relative_negative = (
                    class_evidence_score_output.top_relative_negative
                )
                class_evidence_gate_reliability = (
                    class_evidence_score_output.gate_reliability
                )
                class_evidence_gate_reliability_regret = (
                    class_evidence_score_output.gate_reliability_regret
                )
                class_evidence_top_margin = class_evidence_score_output.top_margin
                class_evidence_gated_margin = class_evidence_score_output.gated_margin
                class_evidence_branch_existential_scores = (
                    class_evidence_score_output.branch_existential_scores
                )
                class_evidence_branch_competitive_scores = (
                    class_evidence_score_output.branch_competitive_scores
                )
                class_evidence_branch_direct_scores = (
                    class_evidence_score_output.branch_direct_scores
                )
                class_evidence_branch_residual_scores = (
                    class_evidence_score_output.branch_residual_scores
                )
                class_evidence_branch_support_scores = (
                    class_evidence_score_output.branch_support_scores
                )
                class_evidence_interaction_scores = (
                    class_evidence_score_output.interaction_scores
                )
                class_evidence_raw_interaction_scores = (
                    class_evidence_score_output.raw_interaction_scores
                )
                class_evidence_bounded_interaction_scores = (
                    class_evidence_score_output.bounded_interaction_scores
                )
                class_evidence_branch_scale = class_evidence_score_output.branch_scale
                class_evidence_branch_direct_top_scale = (
                    class_evidence_score_output.branch_direct_top_scale
                )
                class_evidence_branch_direct_gated_scale = (
                    class_evidence_score_output.branch_direct_gated_scale
                )
                class_evidence_branch_direct_raw_scale = (
                    class_evidence_score_output.branch_direct_raw_scale
                )
                class_evidence_branch_direct_relative_scale = (
                    class_evidence_score_output.branch_direct_relative_scale
                )
                class_evidence_branch_direct_residual_scale = (
                    class_evidence_score_output.branch_direct_residual_scale
                )
                class_evidence_branch_direct_top_weights = (
                    class_evidence_score_output.branch_direct_top_weights
                )
                class_evidence_branch_direct_gated_weights = (
                    class_evidence_score_output.branch_direct_gated_weights
                )
                class_evidence_branch_direct_existential_weights = (
                    class_evidence_score_output.branch_direct_existential_weights
                )
                class_evidence_branch_direct_competitive_weights = (
                    class_evidence_score_output.branch_direct_competitive_weights
                )
                class_evidence_interaction_scale = (
                    class_evidence_score_output.interaction_scale
                )
                class_evidence_interaction_scale_multiplier = (
                    class_evidence_score_output.interaction_scale_multiplier
                )
                class_evidence_interaction_effective_scale = (
                    class_evidence_score_output.interaction_effective_scale
                )
                class_evidence_embedding_score_bound = (
                    class_evidence_score_output.embedding_score_bound
                )
                class_evidence_embedding_score_temperature = (
                    class_evidence_score_output.embedding_score_temperature
                )
                class_evidence_interaction_score_bound = (
                    class_evidence_score_output.interaction_score_bound
                )
                class_evidence_interaction_score_temperature = (
                    class_evidence_score_output.interaction_score_temperature
                )
            elif class_evidence_logits is None:
                score_class_evidence = getattr(
                    self.evidence_pooler,
                    "score_class_evidence_with_components",
                    None,
                )
                if not callable(score_class_evidence):
                    raise RuntimeError(
                        "class-aware evidence pooler cannot score class evidence"
                    )
                class_evidence_score_output = score_class_evidence(
                    class_evidence_embeddings
                )
                class_evidence_logits = class_evidence_score_output.logits
                class_evidence_embedding_scores = (
                    class_evidence_score_output.embedding_scores
                )
                class_evidence_raw_embedding_scores = (
                    class_evidence_score_output.raw_embedding_scores
                )
                class_evidence_bounded_embedding_scores = (
                    class_evidence_score_output.bounded_embedding_scores
                )
                class_evidence_top_support_scores = (
                    class_evidence_score_output.top_support_scores
                )
                class_evidence_direct_top_scores = (
                    class_evidence_score_output.direct_top_scores
                )
                class_evidence_top_support_residual_scores = (
                    class_evidence_score_output.top_support_residual_scores
                )
                class_evidence_gated_support_scores = (
                    class_evidence_score_output.gated_support_scores
                )
                class_evidence_top_raw_existential_scores = (
                    class_evidence_score_output.top_raw_existential_scores
                )
                class_evidence_top_relative_correction_scores = (
                    class_evidence_score_output.top_relative_correction_scores
                )
                class_evidence_top_support_raw_positive_component = (
                    class_evidence_score_output.top_support_raw_positive_component
                )
                class_evidence_top_support_relative_positive_component = (
                    class_evidence_score_output.top_support_relative_positive_component
                )
                class_evidence_top_support_relative_negative_component = (
                    class_evidence_score_output.top_support_relative_negative_component
                )
                class_evidence_top_support_relative_negative_component_uncapped = class_evidence_score_output.top_support_relative_negative_component_uncapped
                class_evidence_top_support_relative_negative_component_capped = class_evidence_score_output.top_support_relative_negative_component_capped
                class_evidence_top_support_relative_negative_cap_value = (
                    class_evidence_score_output.top_support_relative_negative_cap_value
                )
                class_evidence_top_support_relative_negative_cap_active = (
                    class_evidence_score_output.top_support_relative_negative_cap_active
                )
                class_evidence_top_support_direct_raw_scale = (
                    class_evidence_score_output.top_support_direct_raw_scale
                )
                class_evidence_top_support_direct_relative_positive_scale = class_evidence_score_output.top_support_direct_relative_positive_scale
                class_evidence_top_support_direct_relative_negative_scale = class_evidence_score_output.top_support_direct_relative_negative_scale
                class_evidence_top_support_direct_residual_scale = (
                    class_evidence_score_output.top_support_direct_residual_scale
                )
                class_evidence_top_relative_positive = (
                    class_evidence_score_output.top_relative_positive
                )
                class_evidence_top_relative_negative = (
                    class_evidence_score_output.top_relative_negative
                )
                class_evidence_gate_reliability = (
                    class_evidence_score_output.gate_reliability
                )
                class_evidence_gate_reliability_regret = (
                    class_evidence_score_output.gate_reliability_regret
                )
                class_evidence_top_margin = class_evidence_score_output.top_margin
                class_evidence_gated_margin = class_evidence_score_output.gated_margin
                class_evidence_branch_existential_scores = (
                    class_evidence_score_output.branch_existential_scores
                )
                class_evidence_branch_competitive_scores = (
                    class_evidence_score_output.branch_competitive_scores
                )
                class_evidence_branch_direct_scores = (
                    class_evidence_score_output.branch_direct_scores
                )
                class_evidence_branch_residual_scores = (
                    class_evidence_score_output.branch_residual_scores
                )
                class_evidence_branch_support_scores = (
                    class_evidence_score_output.branch_support_scores
                )
                class_evidence_interaction_scores = (
                    class_evidence_score_output.interaction_scores
                )
                class_evidence_raw_interaction_scores = (
                    class_evidence_score_output.raw_interaction_scores
                )
                class_evidence_bounded_interaction_scores = (
                    class_evidence_score_output.bounded_interaction_scores
                )
                class_evidence_branch_scale = class_evidence_score_output.branch_scale
                class_evidence_branch_direct_top_scale = (
                    class_evidence_score_output.branch_direct_top_scale
                )
                class_evidence_branch_direct_gated_scale = (
                    class_evidence_score_output.branch_direct_gated_scale
                )
                class_evidence_branch_direct_raw_scale = (
                    class_evidence_score_output.branch_direct_raw_scale
                )
                class_evidence_branch_direct_relative_scale = (
                    class_evidence_score_output.branch_direct_relative_scale
                )
                class_evidence_branch_direct_residual_scale = (
                    class_evidence_score_output.branch_direct_residual_scale
                )
                class_evidence_branch_direct_top_weights = (
                    class_evidence_score_output.branch_direct_top_weights
                )
                class_evidence_branch_direct_gated_weights = (
                    class_evidence_score_output.branch_direct_gated_weights
                )
                class_evidence_branch_direct_existential_weights = (
                    class_evidence_score_output.branch_direct_existential_weights
                )
                class_evidence_branch_direct_competitive_weights = (
                    class_evidence_score_output.branch_direct_competitive_weights
                )
                class_evidence_interaction_scale = (
                    class_evidence_score_output.interaction_scale
                )
                class_evidence_interaction_scale_multiplier = (
                    class_evidence_score_output.interaction_scale_multiplier
                )
                class_evidence_interaction_effective_scale = (
                    class_evidence_score_output.interaction_effective_scale
                )
                class_evidence_embedding_score_bound = (
                    class_evidence_score_output.embedding_score_bound
                )
                class_evidence_embedding_score_temperature = (
                    class_evidence_score_output.embedding_score_temperature
                )
                class_evidence_interaction_score_bound = (
                    class_evidence_score_output.interaction_score_bound
                )
                class_evidence_interaction_score_temperature = (
                    class_evidence_score_output.interaction_score_temperature
                )
            if class_evidence_logits is None:
                raise RuntimeError(
                    "class-aware evidence scorer did not return class evidence logits"
                )
            if self.global_residual_combiner is None:
                pooled_embedding = evidence_embedding
                logits = class_evidence_logits
            else:
                class_evidence_features = class_evidence_embeddings.reshape(
                    class_evidence_embeddings.shape[0],
                    -1,
                )
                fusion_input = torch.cat(
                    [class_evidence_features, class_gated_branch_logit_features],
                    dim=1,
                )
                pooled_embedding = self.fusion_projector(fusion_input)
                global_residual_logits = self.classifier(pooled_embedding)
                residual_multiplier = self._global_residual_schedule_multiplier()
                (
                    global_residual_gate,
                    global_residual_learned_gate,
                    global_residual_evidence_confidence,
                    global_residual_confidence_factor,
                ) = self._global_residual_gate_values(
                    class_evidence_logits=class_evidence_logits,
                    class_gated_branch_logit_features=(
                        class_gated_branch_logit_features
                    ),
                    class_gated_branch_logit_relative_features=(
                        class_gated_branch_logit_relative_features
                    ),
                    class_top_branch_margin_features=(class_top_branch_margin_features),
                    class_top_branch_margin_relative_features=(
                        class_top_branch_margin_relative_features
                    ),
                )
                bounded_global_residual_logits = (
                    self.global_residual_combiner.bounded_residual_logits(
                        global_residual_logits
                    )
                )
                centered_bounded_global_residual_logits = (
                    self.global_residual_combiner.centered_bounded_residual_logits(
                        global_residual_logits
                    )
                )
                logits = self.global_residual_combiner(
                    class_evidence_logits,
                    global_residual_logits,
                    multiplier=residual_multiplier,
                    residual_gate=global_residual_gate,
                )
                raw_scale = self.global_residual_combiner.scale_tensor(
                    device=global_residual_logits.device,
                    dtype=global_residual_logits.dtype,
                )
                global_residual_scale = raw_scale.detach()
                global_residual_schedule_multiplier = torch.tensor(
                    float(residual_multiplier),
                    device=global_residual_logits.device,
                    dtype=global_residual_logits.dtype,
                )
                global_residual_effective_scale = (
                    raw_scale * float(residual_multiplier)
                ).detach()
                residual_contribution_base = centered_bounded_global_residual_logits
                if global_residual_gate is not None:
                    residual_contribution_base = (
                        global_residual_gate * residual_contribution_base
                    )
                global_residual_contribution = (
                    raw_scale * float(residual_multiplier) * residual_contribution_base
                )
                residual_bounding = self.global_residual_combiner.bounding
                residual_correction = self.global_residual_combiner.correction
                global_residual_zero_mean_enabled = bool(
                    residual_correction.zero_mean
                    and residual_correction.mode == "gated_zero_mean"
                )
                global_residual_rebound_enabled = bool(
                    residual_correction.rebound
                    and residual_correction.mode == "gated_zero_mean"
                )
                if residual_bounding.enabled:
                    global_residual_bound = torch.tensor(
                        float(residual_bounding.bound),
                        device=global_residual_logits.device,
                        dtype=global_residual_logits.dtype,
                    )
                    global_residual_temperature = torch.tensor(
                        float(residual_bounding.temperature),
                        device=global_residual_logits.device,
                        dtype=global_residual_logits.dtype,
                    )
        else:
            branch_logit_features = stacked_branch_logits.reshape(
                stacked_branch_logits.shape[0],
                -1,
            )
            branch_embedding_mean = torch.stack(branch_embeddings, dim=1).mean(dim=1)
            fusion_input = torch.cat(
                [evidence_embedding, branch_embedding_mean, branch_logit_features],
                dim=1,
            )
            pooled_embedding = self.fusion_projector(fusion_input)
            logits = self.classifier(pooled_embedding)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
        return AstModelOutput(
            logits=logits,
            pooled_embedding=pooled_embedding,
            branch_logits=stacked_branch_logits,
            branch_binary_logits=stacked_branch_binary_logits,
            branch_attention_weights=tuple(branch_attention_weights),
            selected_evidence_tokens=selected_evidence_tokens,
            selected_evidence_indices=selected_evidence_indices,
            selected_evidence_scores=selected_evidence_scores,
            selected_evidence_branch_ids=selected_evidence_branch_ids,
            evidence_score_source=evidence_source,
            evidence_pooling_type=self.cfg.encoder.architecture.evidence_pooling.type,
            evidence_gate_weights=pooling_output.gate_weights,
            evidence_gate_entropy=pooling_output.gate_entropy,
            class_evidence_embeddings=pooling_output.class_evidence_embeddings,
            class_evidence_logits=class_evidence_logits,
            global_residual_logits=global_residual_logits,
            class_evidence_gate_weights=pooling_output.class_gate_weights,
            class_evidence_gate_entropy=pooling_output.class_gate_entropy,
            class_evidence_learned_gate_weights=(
                pooling_output.learned_class_gate_weights
            ),
            class_evidence_learned_gate_entropy=(
                pooling_output.learned_class_gate_entropy
            ),
            class_evidence_gate_mixing_alpha=pooling_output.class_gate_mixing_alpha,
            class_gated_branch_logits=class_gated_branch_logits,
            class_gated_branch_logit_features=class_gated_branch_logit_features,
            class_gated_branch_logit_feature_mode=(
                class_gated_branch_logit_feature_mode
            ),
            class_top_branch_margin_features=class_top_branch_margin_features,
            class_gated_branch_logit_relative_features=(
                class_gated_branch_logit_relative_features
            ),
            class_top_branch_margin_relative_features=(
                class_top_branch_margin_relative_features
            ),
            class_evidence_scorer_branch_raw_features=(
                class_evidence_scorer_branch_raw_features
            ),
            class_evidence_scorer_branch_features=(
                class_evidence_scorer_branch_features
            ),
            class_evidence_embedding_scores=class_evidence_embedding_scores,
            class_evidence_raw_embedding_scores=class_evidence_raw_embedding_scores,
            class_evidence_bounded_embedding_scores=(
                class_evidence_bounded_embedding_scores
            ),
            class_evidence_top_support_scores=class_evidence_top_support_scores,
            class_evidence_direct_top_scores=class_evidence_direct_top_scores,
            class_evidence_top_support_residual_scores=(
                class_evidence_top_support_residual_scores
            ),
            class_evidence_gated_support_scores=class_evidence_gated_support_scores,
            class_evidence_top_raw_existential_scores=(
                class_evidence_top_raw_existential_scores
            ),
            class_evidence_top_relative_correction_scores=(
                class_evidence_top_relative_correction_scores
            ),
            class_evidence_top_support_raw_positive_component=(
                class_evidence_top_support_raw_positive_component
            ),
            class_evidence_top_support_relative_positive_component=(
                class_evidence_top_support_relative_positive_component
            ),
            class_evidence_top_support_relative_negative_component=(
                class_evidence_top_support_relative_negative_component
            ),
            class_evidence_top_support_relative_negative_component_uncapped=(
                class_evidence_top_support_relative_negative_component_uncapped
            ),
            class_evidence_top_support_relative_negative_component_capped=(
                class_evidence_top_support_relative_negative_component_capped
            ),
            class_evidence_top_support_relative_negative_cap_value=(
                class_evidence_top_support_relative_negative_cap_value
            ),
            class_evidence_top_support_relative_negative_cap_active=(
                class_evidence_top_support_relative_negative_cap_active
            ),
            class_evidence_top_support_direct_raw_scale=(
                class_evidence_top_support_direct_raw_scale
            ),
            class_evidence_top_support_direct_relative_positive_scale=(
                class_evidence_top_support_direct_relative_positive_scale
            ),
            class_evidence_top_support_direct_relative_negative_scale=(
                class_evidence_top_support_direct_relative_negative_scale
            ),
            class_evidence_top_support_direct_residual_scale=(
                class_evidence_top_support_direct_residual_scale
            ),
            class_evidence_top_relative_positive=class_evidence_top_relative_positive,
            class_evidence_top_relative_negative=class_evidence_top_relative_negative,
            class_evidence_gate_reliability=class_evidence_gate_reliability,
            class_evidence_gate_reliability_regret=(
                class_evidence_gate_reliability_regret
            ),
            class_evidence_top_margin=class_evidence_top_margin,
            class_evidence_gated_margin=class_evidence_gated_margin,
            class_evidence_branch_existential_scores=(
                class_evidence_branch_existential_scores
            ),
            class_evidence_branch_competitive_scores=(
                class_evidence_branch_competitive_scores
            ),
            class_evidence_branch_direct_scores=class_evidence_branch_direct_scores,
            class_evidence_branch_residual_scores=class_evidence_branch_residual_scores,
            class_evidence_branch_support_scores=(class_evidence_branch_support_scores),
            class_evidence_interaction_scores=class_evidence_interaction_scores,
            class_evidence_raw_interaction_scores=(
                class_evidence_raw_interaction_scores
            ),
            class_evidence_bounded_interaction_scores=(
                class_evidence_bounded_interaction_scores
            ),
            class_evidence_branch_scale=class_evidence_branch_scale,
            class_evidence_branch_direct_top_scale=(
                class_evidence_branch_direct_top_scale
            ),
            class_evidence_branch_direct_gated_scale=(
                class_evidence_branch_direct_gated_scale
            ),
            class_evidence_branch_direct_raw_scale=(
                class_evidence_branch_direct_raw_scale
            ),
            class_evidence_branch_direct_relative_scale=(
                class_evidence_branch_direct_relative_scale
            ),
            class_evidence_branch_direct_residual_scale=(
                class_evidence_branch_direct_residual_scale
            ),
            class_evidence_branch_direct_top_weights=(
                class_evidence_branch_direct_top_weights
            ),
            class_evidence_branch_direct_gated_weights=(
                class_evidence_branch_direct_gated_weights
            ),
            class_evidence_branch_direct_existential_weights=(
                class_evidence_branch_direct_existential_weights
            ),
            class_evidence_branch_direct_competitive_weights=(
                class_evidence_branch_direct_competitive_weights
            ),
            class_evidence_interaction_scale=class_evidence_interaction_scale,
            class_evidence_interaction_scale_multiplier=(
                class_evidence_interaction_scale_multiplier
            ),
            class_evidence_interaction_effective_scale=(
                class_evidence_interaction_effective_scale
            ),
            class_evidence_embedding_score_bound=class_evidence_embedding_score_bound,
            class_evidence_embedding_score_temperature=(
                class_evidence_embedding_score_temperature
            ),
            class_evidence_interaction_score_bound=(
                class_evidence_interaction_score_bound
            ),
            class_evidence_interaction_score_temperature=(
                class_evidence_interaction_score_temperature
            ),
            class_evidence_scorer_type=(
                self.cfg.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.type
            ),
            class_evidence_scorer_branch_feature_transform_mode=(
                self.cfg.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_feature_transform.mode
            ),
            class_evidence_scorer_branch_feature_transform_temperature=torch.tensor(
                float(
                    self.cfg.encoder.architecture.evidence_pooling.class_gate.evidence_scorer.branch_feature_transform.temperature
                ),
                device=logits.device,
                dtype=logits.dtype,
            ),
            global_residual_scale=global_residual_scale,
            global_residual_schedule_multiplier=global_residual_schedule_multiplier,
            global_residual_effective_scale=global_residual_effective_scale,
            bounded_global_residual_logits=bounded_global_residual_logits,
            centered_bounded_global_residual_logits=(
                centered_bounded_global_residual_logits
            ),
            global_residual_learned_gate=global_residual_learned_gate,
            global_residual_evidence_confidence=global_residual_evidence_confidence,
            global_residual_confidence_factor=global_residual_confidence_factor,
            global_residual_gate=global_residual_gate,
            global_residual_contribution=global_residual_contribution,
            global_residual_zero_mean_enabled=global_residual_zero_mean_enabled,
            global_residual_rebound_enabled=global_residual_rebound_enabled,
            global_residual_bound=global_residual_bound,
            global_residual_temperature=global_residual_temperature,
            branch_evidence_norms=pooling_output.branch_evidence_norms,
            selected_evidence_dropout_mask=selected_evidence_dropout_mask,
            selected_evidence_keep_ratio=selected_evidence_keep_ratio,
        )
