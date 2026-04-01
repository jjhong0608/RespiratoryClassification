from __future__ import annotations

from typing import Literal, cast

import torch

from src.models.model import WhisperClassifierConfig
from src.models.whisper_encoder import WhisperEncoderDims
from src.utils.logging import logger


def torch_load_compat(path: str, *, device: torch.device, weights_only: bool) -> dict:
    if "weights_only" in torch.load.__code__.co_varnames:
        out = torch.load(path, map_location=device, weights_only=weights_only)
    else:
        out = torch.load(path, map_location=device)
    if not isinstance(out, dict):
        raise TypeError("Expected checkpoint dict")
    return out


def load_checkpoint(path: str, *, device: torch.device, unsafe: bool) -> dict:
    try:
        return torch_load_compat(path, device=device, weights_only=True)
    except Exception:
        if not unsafe:
            raise
        logger.info(
            "Retrying checkpoint load with weights_only=False (unsafe_pickle_load=true)."
        )
        return torch_load_compat(path, device=device, weights_only=False)


def parse_model_cfg(raw: object) -> WhisperClassifierConfig:
    if isinstance(raw, WhisperClassifierConfig):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(
            "model_cfg must be a dict (new checkpoints) or WhisperClassifierConfig (legacy)"
        )
    encoder_raw = raw.get("encoder")
    if not isinstance(encoder_raw, dict):
        raise TypeError("model_cfg['encoder'] must be a dict")
    encoder = WhisperEncoderDims(**encoder_raw)

    head_type_raw = raw.get("head_type", raw.get("classifier_type", "hf"))
    if head_type_raw not in ("hf", "linear", "mlp"):
        raise ValueError(f"Unsupported head_type: {head_type_raw}")
    head_type = cast(Literal["hf", "linear", "mlp"], head_type_raw)

    pooling_raw = raw.get("pooling", "mean")
    if pooling_raw not in ("mean", "cls"):
        raise ValueError(f"Unsupported pooling: {pooling_raw}")
    if head_type != "hf" and pooling_raw != "mean":
        raise ValueError(
            f"Unsupported pooling `{pooling_raw}` for legacy head_type `{head_type}`"
        )
    pooling = cast(Literal["mean", "cls"], pooling_raw)

    return WhisperClassifierConfig(
        encoder=encoder,
        num_classes=int(raw["num_classes"]),
        head_type=head_type,
        pooling=pooling,
        use_weighted_layer_sum=bool(raw.get("use_weighted_layer_sum", False)),
        classifier_proj_size=int(raw.get("classifier_proj_size", 256)),
        hidden_dim=int(raw.get("hidden_dim", 256)),
        dropout=float(raw.get("dropout", 0.0)),
    )
