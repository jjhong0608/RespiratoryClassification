from __future__ import annotations

import torch

from src.models.model import (
    InterAttentionConfig,
    InstanceHeadConfig,
    MILConfig,
    MILModelConfig,
    SegmentEncoderAdaptationConfig,
    SegmentEncoderConfig,
    TopKConfig,
)
from src.models.segment_encoder import SegmentEncoderPoolingConfig
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

    segment_encoder_raw = raw.get("segment_encoder")
    if not isinstance(segment_encoder_raw, dict):
        raise TypeError("model_cfg.segment_encoder must be a dict")
    dims_raw = segment_encoder_raw.get("dims")
    if not isinstance(dims_raw, dict):
        raise TypeError("model_cfg.segment_encoder.dims must be a dict")
    pooling_raw = segment_encoder_raw.get("pooling", {})
    adaptation_raw = segment_encoder_raw.get("adaptation", {})
    instance_head_raw = raw.get("instance_head")
    if not isinstance(instance_head_raw, dict):
        raise TypeError("model_cfg.instance_head must be a dict")
    mil_raw = raw.get("mil")
    if not isinstance(mil_raw, dict):
        raise TypeError("model_cfg.mil must be a dict")

    return MILModelConfig(
        segment_encoder=SegmentEncoderConfig(
            dims=WhisperEncoderDims(**dims_raw),
            type=segment_encoder_raw.get("type", "whisper"),
            backbone=segment_encoder_raw.get("backbone", "custom"),
            pretrained_name_or_path=segment_encoder_raw.get(
                "pretrained_name_or_path"
            ),
            strict=bool(segment_encoder_raw.get("strict", True)),
            download_root=segment_encoder_raw.get("download_root"),
            pooling=SegmentEncoderPoolingConfig(**dict(pooling_raw)),
            adaptation=SegmentEncoderAdaptationConfig(**dict(adaptation_raw)),
        ),
        instance_head=InstanceHeadConfig(**dict(instance_head_raw)),
        mil=MILConfig(
            aggregator=mil_raw.get("aggregator", "attention"),
            attention=InterAttentionConfig(**dict(mil_raw.get("attention", {}))),
            topk=TopKConfig(**dict(mil_raw.get("topk", {}))),
        ),
    )
