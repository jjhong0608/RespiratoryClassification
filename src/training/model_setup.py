from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from torch import nn
from torch.optim import AdamW

from src.data.dataset import RespiratoryClipDataset
from src.models.model import (
    AstModelConfig,
    EncoderAdaptationConfig,
    RespiratoryAstModel,
)
from src.models.resnet50_model import (
    ResNet50AdaptationConfig,
    ResNet50ClassifierConfig,
    ResNet50EncoderRuntimeConfig,
    ResNet50ModelConfig,
    ResNet50PretrainedInfo,
    RespiratoryResNet50Model,
)
from src.models.whisper_encoder import WhisperEncoderDims
from src.models.whisper_model import RespiratoryWhisperModel, WhisperModelConfig
from src.pretrained.whisper import (
    LoadedWhisperPretrainedInfo,
    OpenAIWhisperCheckpointLoader,
)
from src.training.ast_setup import (
    EncoderAdaptationSummary,
    OptimizerGroupSummary,
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
    inspect_pretrained_encoder,
)
from src.utils.config import (
    ClassifierConfig,
    ModelConfig,
    ResNet50EncoderConfig,
    TrainConfig,
    WhisperEncoderConfig,
)


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    optimizer: AdamW
    model_family: str
    model_cfg: Any
    pretrained_info: Any
    adaptation_summary: EncoderAdaptationSummary
    optimizer_summary: OptimizerGroupSummary
    feature_cfg: dict[str, Any]


def model_family_from_runtime_config(
    model_cfg: AstModelConfig | WhisperModelConfig | ResNet50ModelConfig,
) -> str:
    if isinstance(model_cfg, AstModelConfig):
        return "ast"
    if isinstance(model_cfg, WhisperModelConfig):
        return "whisper"
    if isinstance(model_cfg, ResNet50ModelConfig):
        return "resnet50"
    raise TypeError(f"Unsupported model config type: {type(model_cfg)!r}")


def build_model_from_runtime_config(
    model_cfg: AstModelConfig | WhisperModelConfig | ResNet50ModelConfig,
) -> nn.Module:
    if isinstance(model_cfg, AstModelConfig):
        return RespiratoryAstModel(model_cfg)
    if isinstance(model_cfg, WhisperModelConfig):
        model = RespiratoryWhisperModel(model_cfg)
        apply_whisper_encoder_adaptation(model, model_cfg.adaptation)
        return model
    if isinstance(model_cfg, ResNet50ModelConfig):
        return RespiratoryResNet50Model(model_cfg)
    raise TypeError(f"Unsupported model config type: {type(model_cfg)!r}")


def _feature_cfg(dataset: RespiratoryClipDataset) -> dict[str, Any]:
    payload = {
        "feature_type": dataset.feature_type,
        "num_mel_bins": dataset.num_mel_bins,
        "max_length": dataset.max_length,
        "n_audio_ctx": dataset.n_audio_ctx,
    }
    if dataset.feature_type == "resnet_spectrogram":
        resnet_cfg = dataset.cfg.preprocessing.resnet_spectrogram
        payload.update(
            {
                "input_channels": dataset.input_channels,
                "image_size": dataset.image_size,
                "use_hpss": resnet_cfg.use_hpss,
                "image_mean": tuple(resnet_cfg.image_mean),
                "image_std": tuple(resnet_cfg.image_std),
            }
        )
    return payload


def _count_encoder_parameters(model: nn.Module) -> EncoderAdaptationSummary:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise TypeError("Model must expose an nn.Module encoder")
    trainable = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in encoder.parameters())
    mode = "partial" if trainable and trainable < total else "full"
    if trainable == 0:
        mode = "frozen"
    return EncoderAdaptationSummary(
        mode=mode,
        num_layers=0,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
    )


def _build_generic_grouped_optimizer(
    model: nn.Module,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[AdamW, OptimizerGroupSummary]:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise TypeError("Model must expose an nn.Module encoder")
    encoder_params = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    encoder_param_ids = {id(parameter) for parameter in encoder.parameters()}
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
    return optimizer, OptimizerGroupSummary(
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        encoder_trainable_parameters=sum(p.numel() for p in encoder_params),
        head_trainable_parameters=sum(p.numel() for p in head_params),
        param_group_count=len(param_groups),
    )


def _build_whisper_model_cfg(
    encoder: WhisperEncoderConfig,
    classifier: ClassifierConfig,
    *,
    num_classes: int,
) -> WhisperModelConfig:
    head_type = classifier.type
    if head_type not in {"hf", "linear", "mlp"}:
        raise ValueError(f"Unsupported Whisper classifier type: {head_type}")
    return WhisperModelConfig(
        encoder=WhisperEncoderDims(
            n_mels=encoder.n_mels,
            n_audio_ctx=encoder.n_audio_ctx,
            n_audio_state=encoder.n_audio_state,
            n_audio_head=encoder.n_audio_head,
            n_audio_layer=encoder.n_audio_layer,
        ),
        num_classes=num_classes,
        adaptation=EncoderAdaptationConfig(
            mode=encoder.adaptation.mode,
            num_layers=encoder.adaptation.num_layers,
        ),
        head_type=head_type,
        pooling=classifier.pooling,
        use_weighted_layer_sum=classifier.use_weighted_layer_sum,
        classifier_proj_size=classifier.classifier_proj_size,
        hidden_dim=classifier.hidden_dim,
        dropout=classifier.dropout,
    )


def _build_resnet50_model_cfg(
    encoder: ResNet50EncoderConfig,
    classifier: ClassifierConfig,
    *,
    num_classes: int,
) -> ResNet50ModelConfig:
    if classifier.type not in {"linear", "mlp"}:
        raise ValueError("ResNet50 setup supports only linear or mlp classifier heads")
    classifier_type = cast(Literal["linear", "mlp"], classifier.type)
    image_mean = cast(tuple[float, float, float], tuple(encoder.image_mean))
    image_std = cast(tuple[float, float, float], tuple(encoder.image_std))
    return ResNet50ModelConfig(
        encoder=ResNet50EncoderRuntimeConfig(
            type=encoder.type,
            weights=encoder.weights,
            adaptation=ResNet50AdaptationConfig(
                mode=encoder.adaptation.mode,
                num_layers=encoder.adaptation.num_layers,
            ),
            input_channels=encoder.input_channels,
            image_size=encoder.image_size,
            image_mean=image_mean,
            image_std=image_std,
        ),
        classifier=ResNet50ClassifierConfig(
            type=classifier_type,
            hidden_dim=classifier.hidden_dim,
            dropout=classifier.dropout,
        ),
        num_classes=num_classes,
    )


def apply_whisper_encoder_adaptation(
    model: RespiratoryWhisperModel,
    cfg: EncoderAdaptationConfig,
) -> EncoderAdaptationSummary:
    encoder = model.encoder
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    if cfg.mode == "full":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    elif cfg.mode == "partial":
        if not 1 <= cfg.num_layers <= len(encoder.blocks):
            raise ValueError(
                "Whisper partial adaptation requires num_layers between 1 "
                "and the number of encoder blocks"
            )
        for block in encoder.blocks[-cfg.num_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    elif cfg.mode != "frozen":
        raise ValueError(f"Unsupported Whisper adaptation mode: {cfg.mode}")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    return EncoderAdaptationSummary(
        mode=cfg.mode,
        num_layers=cfg.num_layers,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
    )


def apply_resnet50_encoder_adaptation(
    model: RespiratoryResNet50Model,
    cfg: ResNet50AdaptationConfig,
) -> EncoderAdaptationSummary:
    encoder = model.encoder
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    if cfg.mode == "full":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    elif cfg.mode == "partial":
        if not 1 <= cfg.num_layers <= 4:
            raise ValueError("ResNet50 partial adaptation requires num_layers in 1..4")
        stage_names = ("layer1", "layer2", "layer3", "layer4")
        for stage_name in stage_names[-cfg.num_layers :]:
            stage = getattr(encoder, stage_name)
            if not isinstance(stage, nn.Module):
                raise TypeError(f"ResNet50 encoder missing module: {stage_name}")
            for parameter in stage.parameters():
                parameter.requires_grad = True
    elif cfg.mode != "frozen":
        raise ValueError(f"Unsupported ResNet50 adaptation mode: {cfg.mode}")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    return EncoderAdaptationSummary(
        mode=cfg.mode,
        num_layers=cfg.num_layers,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
    )


def build_model_bundle(
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    dataset: RespiratoryClipDataset,
    *,
    num_classes: int,
) -> ModelBundle:
    if cfg.encoder.type == "ast":
        ast_model = build_ast_model(
            cfg,
            num_mel_bins=dataset.num_mel_bins,
            max_length=dataset.max_length,
            num_classes=num_classes,
        )
        ast_pretrained_info = inspect_pretrained_encoder(cfg)
        adaptation_summary = apply_encoder_adaptation(
            ast_model,
            ast_model.cfg.encoder.adaptation,
        )
        optimizer, optimizer_summary = build_grouped_optimizer(
            ast_model,
            encoder_lr=train_cfg.optimizer.encoder_lr,
            head_lr=train_cfg.optimizer.head_lr,
            weight_decay=train_cfg.optimizer.weight_decay,
        )
        return ModelBundle(
            model=ast_model,
            optimizer=optimizer,
            model_family="ast",
            model_cfg=ast_model.cfg,
            pretrained_info=ast_pretrained_info,
            adaptation_summary=adaptation_summary,
            optimizer_summary=optimizer_summary,
            feature_cfg=_feature_cfg(dataset),
        )
    if cfg.encoder.type == "whisper":
        if dataset.feature_type != "log_mel":
            raise ValueError("Whisper model requires log_mel frontend")
        whisper_model_cfg = _build_whisper_model_cfg(
            cfg.encoder,
            cfg.classifier,
            num_classes=num_classes,
        )
        whisper_model = RespiratoryWhisperModel(whisper_model_cfg)
        whisper_pretrained_info: LoadedWhisperPretrainedInfo | None = None
        if cfg.encoder.pretrained is not None:
            whisper_pretrained_info = OpenAIWhisperCheckpointLoader().load_encoder_into(
                whisper_model,
                cfg.encoder.pretrained,
            )
        adaptation_summary = apply_whisper_encoder_adaptation(
            whisper_model,
            whisper_model.cfg.adaptation,
        )
        optimizer, optimizer_summary = _build_generic_grouped_optimizer(
            whisper_model,
            encoder_lr=train_cfg.optimizer.encoder_lr,
            head_lr=train_cfg.optimizer.head_lr,
            weight_decay=train_cfg.optimizer.weight_decay,
        )
        return ModelBundle(
            model=whisper_model,
            optimizer=optimizer,
            model_family="whisper",
            model_cfg=whisper_model.cfg,
            pretrained_info=whisper_pretrained_info,
            adaptation_summary=adaptation_summary,
            optimizer_summary=optimizer_summary,
            feature_cfg=_feature_cfg(dataset),
        )
    if cfg.encoder.type == "resnet50":
        if dataset.feature_type != "resnet_spectrogram":
            raise ValueError("ResNet50 model requires resnet_spectrogram frontend")
        if dataset.input_channels != cfg.encoder.input_channels:
            raise ValueError(
                "ResNet50 dataset input channels do not match model encoder "
                f"({dataset.input_channels} vs {cfg.encoder.input_channels})"
            )
        if dataset.image_size != cfg.encoder.image_size:
            raise ValueError(
                "ResNet50 dataset image_size does not match model encoder "
                f"({dataset.image_size} vs {cfg.encoder.image_size})"
            )
        resnet_model_cfg = _build_resnet50_model_cfg(
            cfg.encoder,
            cfg.classifier,
            num_classes=num_classes,
        )
        resnet_model = RespiratoryResNet50Model(resnet_model_cfg)
        adaptation_summary = apply_resnet50_encoder_adaptation(
            resnet_model,
            resnet_model.cfg.encoder.adaptation,
        )
        optimizer, optimizer_summary = _build_generic_grouped_optimizer(
            resnet_model,
            encoder_lr=train_cfg.optimizer.encoder_lr,
            head_lr=train_cfg.optimizer.head_lr,
            weight_decay=train_cfg.optimizer.weight_decay,
        )
        pretrained_info = ResNet50PretrainedInfo(
            source="torchvision",
            weights=resnet_model_cfg.encoder.weights,
            embedding_dim=RespiratoryResNet50Model.embedding_dim,
            input_channels=resnet_model_cfg.encoder.input_channels,
            image_size=resnet_model_cfg.encoder.image_size,
        )
        return ModelBundle(
            model=resnet_model,
            optimizer=optimizer,
            model_family="resnet50",
            model_cfg=resnet_model.cfg,
            pretrained_info=pretrained_info,
            adaptation_summary=adaptation_summary,
            optimizer_summary=optimizer_summary,
            feature_cfg=_feature_cfg(dataset),
        )
    raise ValueError(f"Unsupported model.encoder.type: {cfg.encoder.type}")
