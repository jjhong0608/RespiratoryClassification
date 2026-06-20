from __future__ import annotations

from typing import Literal, cast

import torch

from src.models.model import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFeatureDims,
    AstModelConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
)
from src.models.resnet50_model import (
    ResNet50AdaptationConfig,
    ResNet50ClassifierConfig,
    ResNet50EncoderRuntimeConfig,
    ResNet50ModelConfig,
)
from src.models.whisper_encoder import WhisperEncoderDims
from src.models.whisper_model import WhisperModelConfig


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


RuntimeModelConfig = AstModelConfig | WhisperModelConfig | ResNet50ModelConfig


def parse_model_cfg(raw: object) -> RuntimeModelConfig:
    if isinstance(raw, (AstModelConfig, WhisperModelConfig, ResNet50ModelConfig)):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("model_cfg must be a dict or runtime model config")

    encoder_raw = raw.get("encoder")
    if not isinstance(encoder_raw, dict):
        raise TypeError("model_cfg.encoder must be a dict")
    encoder_type = encoder_raw.get("type")
    if encoder_type is None:
        encoder_type = "whisper" if "n_audio_ctx" in encoder_raw else "ast"
    if encoder_type == "whisper":
        return _parse_whisper_model_cfg(raw, encoder_raw)
    if encoder_type == "resnet50":
        return _parse_resnet50_model_cfg(raw, encoder_raw)
    if encoder_type != "ast":
        raise ValueError(f"Unsupported checkpoint model encoder type: {encoder_type}")
    return _parse_ast_model_cfg(raw, encoder_raw)


def _parse_ast_model_cfg(raw: dict, encoder_raw: dict) -> AstModelConfig:
    feature_dims_raw = encoder_raw.get("feature_dims")
    if not isinstance(feature_dims_raw, dict):
        raise TypeError("model_cfg.encoder.feature_dims must be a dict")
    adaptation_raw = encoder_raw.get("adaptation", {})
    architecture_raw = encoder_raw.get("architecture", {})
    classifier_raw = raw.get("classifier")
    if not isinstance(classifier_raw, dict):
        raise TypeError("model_cfg.classifier must be a dict")
    num_classes_raw = raw.get("num_classes")
    if not isinstance(num_classes_raw, int):
        raise TypeError("model_cfg.num_classes must be an int")

    return AstModelConfig(
        encoder=AstEncoderConfig(
            feature_dims=AstFeatureDims(**feature_dims_raw),
            type=encoder_raw.get("type", "ast"),
            pretrained_name_or_path=encoder_raw.get("pretrained_name_or_path"),
            cache_dir=encoder_raw.get("cache_dir"),
            adaptation=EncoderAdaptationConfig(**dict(adaptation_raw)),
            architecture=AstArchitectureConfig(**dict(architecture_raw)),
        ),
        classifier=ClassifierConfig(**dict(classifier_raw)),
        num_classes=num_classes_raw,
    )


def _parse_whisper_model_cfg(raw: dict, encoder_raw: dict) -> WhisperModelConfig:
    num_classes_raw = raw.get("num_classes")
    if not isinstance(num_classes_raw, int):
        raise TypeError("model_cfg.num_classes must be an int")
    adaptation_raw = raw.get(
        "adaptation",
        encoder_raw.get(
            "adaptation",
            {
                "mode": "frozen",
                "num_layers": 1,
            },
        ),
    )
    if not isinstance(adaptation_raw, dict):
        raise TypeError("model_cfg.adaptation must be a dict")
    head_type = raw.get("head_type", "hf")
    if head_type not in {"hf", "linear", "mlp"}:
        raise ValueError(f"Unsupported Whisper head_type: {head_type}")
    pooling = raw.get("pooling", "mean")
    if pooling not in {"mean", "cls"}:
        raise ValueError(f"Unsupported Whisper pooling: {pooling}")
    return WhisperModelConfig(
        encoder=WhisperEncoderDims(
            n_mels=int(encoder_raw["n_mels"]),
            n_audio_ctx=int(encoder_raw["n_audio_ctx"]),
            n_audio_state=int(encoder_raw["n_audio_state"]),
            n_audio_head=int(encoder_raw["n_audio_head"]),
            n_audio_layer=int(encoder_raw["n_audio_layer"]),
        ),
        num_classes=num_classes_raw,
        adaptation=EncoderAdaptationConfig(**dict(adaptation_raw)),
        head_type=head_type,
        pooling=pooling,
        use_weighted_layer_sum=bool(raw.get("use_weighted_layer_sum", False)),
        classifier_proj_size=int(raw.get("classifier_proj_size", 256)),
        hidden_dim=int(raw.get("hidden_dim", 256)),
        dropout=float(raw.get("dropout", 0.0)),
    )


def _parse_resnet50_model_cfg(raw: dict, encoder_raw: dict) -> ResNet50ModelConfig:
    num_classes_raw = raw.get("num_classes")
    if not isinstance(num_classes_raw, int):
        raise TypeError("model_cfg.num_classes must be an int")
    classifier_raw = raw.get("classifier")
    if not isinstance(classifier_raw, dict):
        raise TypeError("model_cfg.classifier must be a dict")
    adaptation_raw = encoder_raw.get("adaptation", {})
    if not isinstance(adaptation_raw, dict):
        raise TypeError("model_cfg.encoder.adaptation must be a dict")
    classifier_type = classifier_raw.get("type", "linear")
    if classifier_type not in {"linear", "mlp"}:
        raise ValueError(f"Unsupported ResNet50 classifier type: {classifier_type}")
    resnet_classifier_type = cast(Literal["linear", "mlp"], classifier_type)
    weights = encoder_raw.get("weights", "imagenet")
    if weights not in {"imagenet", "none"}:
        raise ValueError(f"Unsupported ResNet50 weights: {weights}")
    resnet_weights = cast(Literal["imagenet", "none"], weights)
    image_mean = cast(
        tuple[float, float, float],
        tuple(encoder_raw.get("image_mean", (0.485, 0.456, 0.406))),
    )
    image_std = cast(
        tuple[float, float, float],
        tuple(encoder_raw.get("image_std", (0.229, 0.224, 0.225))),
    )
    return ResNet50ModelConfig(
        encoder=ResNet50EncoderRuntimeConfig(
            weights=resnet_weights,
            adaptation=ResNet50AdaptationConfig(**dict(adaptation_raw)),
            input_channels=int(encoder_raw.get("input_channels", 3)),
            image_size=int(encoder_raw.get("image_size", 224)),
            image_mean=image_mean,
            image_std=image_std,
        ),
        classifier=ResNet50ClassifierConfig(
            type=resnet_classifier_type,
            hidden_dim=int(classifier_raw.get("hidden_dim", 256)),
            dropout=float(classifier_raw.get("dropout", 0.0)),
        ),
        num_classes=num_classes_raw,
    )
