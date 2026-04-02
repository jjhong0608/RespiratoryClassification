from __future__ import annotations

from src.models.model import MILModelConfig, RespiratoryMILModel
from src.models.whisper_encoder import WhisperEncoderDims
from src.pretrained.whisper import LoadedPretrainedInfo, OpenAIWhisperCheckpointLoader
from src.utils.config import ModelConfig as RunModelConfig


def build_mil_model(
    cfg: RunModelConfig,
    *,
    segment_n_mels: int,
    segment_audio_ctx: int,
) -> RespiratoryMILModel:
    model_cfg = MILModelConfig(
        encoder=WhisperEncoderDims(
            n_mels=segment_n_mels,
            n_audio_ctx=segment_audio_ctx,
            n_audio_state=cfg.encoder.n_audio_state,
            n_audio_head=cfg.encoder.n_audio_head,
            n_audio_layer=cfg.encoder.n_audio_layer,
        ),
        instance_head_type=cfg.instance_head.type,
        instance_hidden_dim=cfg.instance_head.hidden_dim,
        instance_dropout=cfg.instance_head.dropout,
        aggregator=cfg.mil.aggregator,
        topk_k=cfg.mil.topk.k,
        attention_hidden_dim=cfg.mil.attention.hidden_dim,
        attention_dropout=cfg.mil.attention.dropout,
        attention_gated=cfg.mil.attention.gated,
        logsumexp_temperature=cfg.mil.logsumexp.temperature,
        softmax_weighted_temperature=cfg.mil.softmax_weighted.temperature,
        noisy_or_clamp_eps=cfg.mil.noisy_or.clamp_eps,
    )
    return RespiratoryMILModel(model_cfg)


def maybe_initialize_encoder(
    model: RespiratoryMILModel,
    cfg: RunModelConfig,
) -> LoadedPretrainedInfo | None:
    if cfg.encoder.pretrained_name_or_path is None:
        if cfg.encoder.freeze:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
        return None
    loader = OpenAIWhisperCheckpointLoader()
    return loader.load_encoder_into(
        model,
        cfg.encoder,
        target_dims=model.cfg.encoder,
    )
