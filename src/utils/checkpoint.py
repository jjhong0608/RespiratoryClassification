from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from src.models.model import (
    AstFeatureDims,
    BranchEventDropoutConfig,
    ClassGateBranchFeatureTransformConfig,
    ClassGateBranchLogitFeatureConfig,
    ClassGateConfig,
    ClassGateEvidenceAuxiliaryConfig,
    ClassGateEvidenceScorerConfig,
    ClassGateGlobalResidualBoundingConfig,
    ClassGateGlobalResidualConfig,
    ClassGateGlobalResidualWarmupConfig,
    ClassGateMixingConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    EvidencePoolingConfig,
    FusionProjectorConfig,
    MilConfig,
    MultiScaleRdtArchitectureConfig,
    MultiScaleRdtAstModelConfig,
    MultiScaleRdtEncoderConfig,
    PatchBranchConfig,
    RdtConfig,
    SelectedEvidenceDropoutConfig,
    TokenAugmentationConfig,
)


def torch_load_compat(path: str, *, device: torch.device, weights_only: bool) -> dict:
    if "weights_only" in torch.load.__code__.co_varnames:
        checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    else:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dict")
    return checkpoint


def load_checkpoint(path: str, *, device: torch.device, unsafe: bool = False) -> dict:
    try:
        return torch_load_compat(path, device=device, weights_only=True)
    except Exception:
        if not unsafe:
            raise
        return torch_load_compat(path, device=device, weights_only=False)


def _parse_pair(values: object, *, field_name: str) -> tuple[int, int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a 2-item list")
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly two integers")
    first, second = values
    if not isinstance(first, int) or not isinstance(second, int):
        raise TypeError(f"{field_name} must contain integers")
    return first, second


def _parse_patch_branch(raw: object) -> PatchBranchConfig:
    if not isinstance(raw, Mapping):
        raise TypeError("patch branch entries must be objects")
    patch_size = _parse_pair(raw.get("patch_size"), field_name="patch_size")
    stride = _parse_pair(raw.get("stride"), field_name="stride")
    return PatchBranchConfig(patch_size=patch_size, stride=stride)


def _parse_evidence_pooling(raw: object) -> EvidencePoolingConfig:
    if raw is None:
        return EvidencePoolingConfig()
    if not isinstance(raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling must be a dict"
        )
    kwargs = dict(raw)
    class_gate_raw = kwargs.get("class_gate", {})
    if not isinstance(class_gate_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate must be a dict"
        )
    class_gate_kwargs = dict(class_gate_raw)
    global_residual_raw = class_gate_kwargs.get("global_residual", {})
    if not isinstance(global_residual_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "global_residual must be a dict"
        )
    evidence_auxiliary_raw = class_gate_kwargs.get("evidence_auxiliary", {})
    if not isinstance(evidence_auxiliary_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "evidence_auxiliary must be a dict"
        )
    evidence_scorer_raw = class_gate_kwargs.get("evidence_scorer", {})
    if not isinstance(evidence_scorer_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "evidence_scorer must be a dict"
        )
    branch_logit_feature_raw = class_gate_kwargs.get("branch_logit_feature", {})
    if not isinstance(branch_logit_feature_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "branch_logit_feature must be a dict"
        )
    gate_mixing_raw = class_gate_kwargs.get("gate_mixing", {})
    if not isinstance(gate_mixing_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "gate_mixing must be a dict"
        )
    global_residual_kwargs = dict(global_residual_raw)
    warmup_raw = global_residual_kwargs.get("warmup", {})
    if not isinstance(warmup_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "global_residual.warmup must be a dict"
        )
    bounding_raw = global_residual_kwargs.get("bounding", {})
    if not isinstance(bounding_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "global_residual.bounding must be a dict"
        )
    global_residual_kwargs["warmup"] = ClassGateGlobalResidualWarmupConfig(
        **dict(warmup_raw)
    )
    global_residual_kwargs["bounding"] = ClassGateGlobalResidualBoundingConfig(
        **dict(bounding_raw)
    )
    class_gate_kwargs["global_residual"] = ClassGateGlobalResidualConfig(
        **global_residual_kwargs
    )
    evidence_scorer_kwargs = dict(evidence_scorer_raw)
    branch_feature_transform_raw = evidence_scorer_kwargs.get(
        "branch_feature_transform",
        {},
    )
    if not isinstance(branch_feature_transform_raw, Mapping):
        raise TypeError(
            "model_cfg.encoder.architecture.evidence_pooling.class_gate."
            "evidence_scorer.branch_feature_transform must be a dict"
        )
    evidence_scorer_kwargs["branch_feature_transform"] = (
        ClassGateBranchFeatureTransformConfig(**dict(branch_feature_transform_raw))
    )
    class_gate_kwargs["evidence_scorer"] = ClassGateEvidenceScorerConfig(
        **evidence_scorer_kwargs
    )
    class_gate_kwargs["evidence_auxiliary"] = ClassGateEvidenceAuxiliaryConfig(
        **dict(evidence_auxiliary_raw)
    )
    class_gate_kwargs["branch_logit_feature"] = ClassGateBranchLogitFeatureConfig(
        **dict(branch_logit_feature_raw)
    )
    class_gate_kwargs["gate_mixing"] = ClassGateMixingConfig(**dict(gate_mixing_raw))
    kwargs["class_gate"] = ClassGateConfig(**class_gate_kwargs)
    return EvidencePoolingConfig(**kwargs)


def parse_model_cfg(raw: object) -> MultiScaleRdtAstModelConfig:
    if isinstance(raw, MultiScaleRdtAstModelConfig):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("model_cfg must be a dict or MultiScaleRdtAstModelConfig")

    encoder_raw = raw.get("encoder")
    if not isinstance(encoder_raw, Mapping):
        raise TypeError("model_cfg.encoder must be a dict")
    feature_dims_raw = encoder_raw.get("feature_dims")
    if not isinstance(feature_dims_raw, Mapping):
        raise TypeError("model_cfg.encoder.feature_dims must be a dict")
    classifier_raw = raw.get("classifier")
    if not isinstance(classifier_raw, Mapping):
        raise TypeError("model_cfg.classifier must be a dict")
    num_classes_raw = raw.get("num_classes")
    if not isinstance(num_classes_raw, int):
        raise TypeError("model_cfg.num_classes must be an int")

    adaptation_raw = encoder_raw.get("adaptation", {})
    if not isinstance(adaptation_raw, Mapping):
        raise TypeError("model_cfg.encoder.adaptation must be a dict")
    architecture_raw = encoder_raw.get("architecture", {})
    if not isinstance(architecture_raw, Mapping):
        raise TypeError("model_cfg.encoder.architecture must be a dict")

    architecture_kwargs = dict(architecture_raw)
    if "latent_query_count" in architecture_kwargs:
        raise ValueError(
            "latent_query_count is deprecated in the event-MIL architecture. "
            "Use model.encoder.architecture.rdt instead."
        )
    if "summary_tokens_per_scale" in architecture_kwargs:
        raise ValueError(
            "summary_tokens_per_scale is deprecated in the event-MIL architecture. "
            "Use model.encoder.architecture.rdt.top_tokens_per_branch instead."
        )
    if "rdt_steps" in architecture_kwargs:
        raise ValueError(
            "Flat rdt_steps is deprecated in the event-MIL architecture. "
            "Use model.encoder.architecture.rdt.steps instead."
        )
    patch_branches_raw = architecture_kwargs.get("patch_branches")
    if patch_branches_raw is not None:
        if not isinstance(patch_branches_raw, Sequence) or isinstance(
            patch_branches_raw, (str, bytes)
        ):
            raise TypeError(
                "model_cfg.encoder.architecture.patch_branches must be a list"
            )
        architecture_kwargs["patch_branches"] = tuple(
            _parse_patch_branch(branch_raw) for branch_raw in patch_branches_raw
        )
    rdt_kwargs = dict(architecture_kwargs.get("rdt", {}))
    excluded = rdt_kwargs.get("exclude_branches_from_evidence")
    if excluded is not None:
        if not isinstance(excluded, Sequence) or isinstance(excluded, (str, bytes)):
            raise TypeError(
                "model_cfg.encoder.architecture.rdt.exclude_branches_from_evidence "
                "must be a list"
            )
        rdt_kwargs["exclude_branches_from_evidence"] = tuple(
            int(item) for item in excluded
        )
    architecture_kwargs["rdt"] = RdtConfig(**rdt_kwargs)
    architecture_kwargs["mil"] = MilConfig(**dict(architecture_kwargs.get("mil", {})))
    architecture_kwargs["evidence_pooling"] = _parse_evidence_pooling(
        architecture_kwargs.get("evidence_pooling")
    )
    token_augmentation = dict(architecture_kwargs.get("token_augmentation", {}))
    token_augmentation["branch_event_dropout"] = BranchEventDropoutConfig(
        **dict(token_augmentation.get("branch_event_dropout", {}))
    )
    token_augmentation["selected_evidence_dropout"] = SelectedEvidenceDropoutConfig(
        **dict(token_augmentation.get("selected_evidence_dropout", {}))
    )
    architecture_kwargs["token_augmentation"] = TokenAugmentationConfig(
        **token_augmentation
    )

    classifier_kwargs = dict(classifier_raw)
    fusion_projector_raw = classifier_kwargs.get("fusion_projector", {})
    if not isinstance(fusion_projector_raw, Mapping):
        raise TypeError("model_cfg.classifier.fusion_projector must be a dict")
    classifier_kwargs["fusion_projector"] = FusionProjectorConfig(
        **dict(fusion_projector_raw)
    )

    return MultiScaleRdtAstModelConfig(
        encoder=MultiScaleRdtEncoderConfig(
            feature_dims=AstFeatureDims(**dict(feature_dims_raw)),
            type=encoder_raw.get("type", "multiscale_rdt_ast"),
            adaptation=EncoderAdaptationConfig(**dict(adaptation_raw)),
            architecture=MultiScaleRdtArchitectureConfig(**architecture_kwargs),
        ),
        classifier=ClassifierConfig(**classifier_kwargs),
        num_classes=num_classes_raw,
    )
