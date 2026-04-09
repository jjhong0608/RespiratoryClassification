from __future__ import annotations

from dataclasses import dataclass

from torch.optim import AdamW

from src.models.model import (
    InterAttentionConfig,
    InstanceHeadConfig,
    MILConfig,
    MILModelConfig,
    RespiratoryMILModel,
    SegmentEncoderAdaptationConfig,
    SegmentEncoderConfig,
    TopKConfig,
)
from src.models.segment_encoder import SegmentEncoderPoolingConfig
from src.models.whisper_encoder import WhisperEncoderDims
from src.pretrained.whisper import LoadedPretrainedInfo, OpenAIWhisperCheckpointLoader
from src.utils.config import ModelConfig as RunModelConfig


@dataclass(frozen=True)
class EncoderAdaptationSummary:
    mode: str
    num_layers: int
    trainable_parameters: int
    frozen_parameters: int


@dataclass(frozen=True)
class OptimizerGroupSummary:
    encoder_lr: float
    head_lr: float
    encoder_trainable_parameters: int
    head_trainable_parameters: int
    param_group_count: int


def build_mil_model(
    cfg: RunModelConfig,
    *,
    segment_n_mels: int,
    segment_audio_ctx: int,
) -> RespiratoryMILModel:
    model_cfg = MILModelConfig(
        segment_encoder=SegmentEncoderConfig(
            type=cfg.segment_encoder.type,
            backbone=cfg.segment_encoder.backbone,
            pretrained_name_or_path=cfg.segment_encoder.pretrained_name_or_path,
            strict=cfg.segment_encoder.strict,
            download_root=cfg.segment_encoder.download_root,
            dims=WhisperEncoderDims(
                n_mels=segment_n_mels,
                n_audio_ctx=segment_audio_ctx,
                n_audio_state=cfg.segment_encoder.dims.n_audio_state,
                n_audio_head=cfg.segment_encoder.dims.n_audio_head,
                n_audio_layer=cfg.segment_encoder.dims.n_audio_layer,
            ),
            pooling=SegmentEncoderPoolingConfig(
                type=cfg.segment_encoder.pooling.type,
                hidden_dim=cfg.segment_encoder.pooling.hidden_dim,
                dropout=cfg.segment_encoder.pooling.dropout,
                gated=cfg.segment_encoder.pooling.gated,
            ),
            adaptation=SegmentEncoderAdaptationConfig(
                mode=cfg.segment_encoder.adaptation.mode,
                num_layers=cfg.segment_encoder.adaptation.num_layers,
            ),
        ),
        instance_head=InstanceHeadConfig(
            type=cfg.instance_head.type,
            hidden_dim=cfg.instance_head.hidden_dim,
            dropout=cfg.instance_head.dropout,
        ),
        mil=MILConfig(
            aggregator=cfg.mil.aggregator,
            attention=InterAttentionConfig(
                hidden_dim=cfg.mil.attention.hidden_dim,
                dropout=cfg.mil.attention.dropout,
                gated=cfg.mil.attention.gated,
            ),
            topk=TopKConfig(k=cfg.mil.topk.k),
        ),
    )
    return RespiratoryMILModel(model_cfg)


def maybe_initialize_encoder(
    model: RespiratoryMILModel,
    cfg: RunModelConfig,
) -> LoadedPretrainedInfo | None:
    pretrained_name_or_path = cfg.segment_encoder.pretrained_name_or_path
    if pretrained_name_or_path is None:
        return None
    loader = OpenAIWhisperCheckpointLoader()
    return loader.load_encoder_into(
        model,
        cfg.segment_encoder,
        target_dims=model.cfg.segment_encoder.dims,
    )


def apply_encoder_adaptation(
    model: RespiratoryMILModel,
    cfg: SegmentEncoderAdaptationConfig,
) -> EncoderAdaptationSummary:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    if cfg.mode == "full":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True
    elif cfg.mode == "partial":
        for block in model.encoder.blocks[-cfg.num_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.encoder.ln_post.parameters():
            parameter.requires_grad = True
    elif cfg.mode != "frozen":
        raise ValueError(f"Unsupported adaptation mode: {cfg.mode}")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    return EncoderAdaptationSummary(
        mode=cfg.mode,
        num_layers=cfg.num_layers,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
    )


def build_grouped_optimizer(
    model: RespiratoryMILModel,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[AdamW, OptimizerGroupSummary]:
    encoder_params = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    encoder_param_ids = {id(parameter) for parameter in model.encoder.parameters()}
    head_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_param_ids
    ]
    param_groups = []
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters available for optimizer creation")
    optimizer = AdamW(param_groups)
    summary = OptimizerGroupSummary(
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        encoder_trainable_parameters=sum(p.numel() for p in encoder_params),
        head_trainable_parameters=sum(p.numel() for p in head_params),
        param_group_count=len(param_groups),
    )
    return optimizer, summary
