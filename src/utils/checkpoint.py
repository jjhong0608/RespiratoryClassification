from __future__ import annotations

import torch

from src.models.model import MILModelConfig
from src.models.whisper_encoder import WhisperEncoderDims


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


def parse_model_cfg(raw: object) -> MILModelConfig:
    if isinstance(raw, MILModelConfig):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("model_cfg must be a dict or MILModelConfig")
    encoder_raw = raw.get("encoder")
    if not isinstance(encoder_raw, dict):
        raise TypeError("model_cfg.encoder must be a dict")
    encoder = WhisperEncoderDims(**encoder_raw)
    return MILModelConfig(
        encoder=encoder,
        instance_head_type=raw.get("instance_head_type", "linear"),
        instance_hidden_dim=int(raw.get("instance_hidden_dim", 256)),
        instance_dropout=float(raw.get("instance_dropout", 0.0)),
        aggregator=raw.get("aggregator", "max"),
        topk_k=int(raw.get("topk_k", 1)),
        attention_hidden_dim=int(raw.get("attention_hidden_dim", 128)),
        attention_dropout=float(raw.get("attention_dropout", 0.0)),
        attention_gated=bool(raw.get("attention_gated", True)),
        logsumexp_temperature=float(raw.get("logsumexp_temperature", 1.0)),
        softmax_weighted_temperature=float(
            raw.get("softmax_weighted_temperature", 1.0)
        ),
        noisy_or_clamp_eps=float(raw.get("noisy_or_clamp_eps", 1e-6)),
    )
