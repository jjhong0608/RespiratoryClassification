from __future__ import annotations

from typing import Literal, cast

import torch

from src.models.model import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFeatureDims,
    AstMilModelConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    GatedAttentionMilConfig,
    InstanceHeadConfig,
    LinearSoftmaxMilConfig,
    MilConfig,
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


def parse_model_cfg(raw: object) -> AstMilModelConfig:
    if isinstance(raw, AstMilModelConfig):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("model_cfg must be a dict or AstMilModelConfig")

    encoder_raw = raw.get("encoder")
    if not isinstance(encoder_raw, dict):
        raise TypeError("model_cfg.encoder must be a dict")
    feature_dims_raw = encoder_raw.get("feature_dims")
    if not isinstance(feature_dims_raw, dict):
        raise TypeError("model_cfg.encoder.feature_dims must be a dict")
    adaptation_raw = encoder_raw.get("adaptation", {})
    architecture_raw = encoder_raw.get("architecture", {})
    instance_head_raw = raw.get("instance_head", {})
    mil_raw = raw.get("mil", {})
    classifier_raw = raw.get("classifier")
    if not isinstance(classifier_raw, dict):
        raise TypeError("model_cfg.classifier must be a dict")
    num_classes_raw = raw.get("num_classes")
    if not isinstance(num_classes_raw, int):
        raise TypeError("model_cfg.num_classes must be an int")

    return AstMilModelConfig(
        encoder=AstEncoderConfig(
            feature_dims=AstFeatureDims(**feature_dims_raw),
            type=encoder_raw.get("type", "ast"),
            pretrained_name_or_path=encoder_raw.get("pretrained_name_or_path"),
            cache_dir=encoder_raw.get("cache_dir"),
            pooling=encoder_raw.get("pooling", "cls"),
            adaptation=EncoderAdaptationConfig(**dict(adaptation_raw)),
            architecture=AstArchitectureConfig(**dict(architecture_raw)),
        ),
        instance_head=InstanceHeadConfig(**dict(instance_head_raw)),
        mil=MilConfig(
            type=cast(
                Literal["gated_attention", "linear_softmax"],
                mil_raw.get("type", "gated_attention"),
            ),
            gated_attention=GatedAttentionMilConfig(
                **dict(mil_raw.get("gated_attention", {}))
            ),
            linear_softmax=LinearSoftmaxMilConfig(
                **dict(mil_raw.get("linear_softmax", {}))
            ),
        ),
        classifier=ClassifierConfig(**dict(classifier_raw)),
        num_classes=num_classes_raw,
    )
