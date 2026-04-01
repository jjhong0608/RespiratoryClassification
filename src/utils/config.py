from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig


@dataclass(frozen=True)
class ModelConfig:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    pooling: Literal["mean", "cls"] = "mean"
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256


@dataclass(frozen=True)
class BandPassConfig:
    enabled: bool = False
    low_freq: float | None = None
    high_freq: float | None = None
    q: float = 0.707


@dataclass(frozen=True)
class DataConfig:
    label_to_index: Mapping[str, int]
    sample_rate: int
    clip_seconds: float
    batch_size: int
    num_workers: int
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)
    train_dirs: Sequence[str] = ()
    val_dirs: Sequence[str] = ()
    eval_dirs: Sequence[str] = ()


@dataclass(frozen=True)
class TrainingHyperparams:
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float


@dataclass(frozen=True)
class EncoderConfig:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int


@dataclass(frozen=True)
class ContrastiveProjectionHeadConfig:
    hidden_dim: int = 384
    output_dim: int = 128


@dataclass(frozen=True)
class ContrastiveAugmentationConfig:
    time_mask_param: int = 40
    time_mask_count: int = 2
    freq_mask_param: int = 8
    freq_mask_count: int = 2
    gaussian_noise_std: float = 0.0


@dataclass(frozen=True)
class SupervisedContrastiveConfig:
    temperature: float = 0.07
    normalize: bool = True
    projection_head: ContrastiveProjectionHeadConfig = field(
        default_factory=ContrastiveProjectionHeadConfig
    )
    augmentation: ContrastiveAugmentationConfig = field(
        default_factory=ContrastiveAugmentationConfig
    )


@dataclass(frozen=True)
class ImbalanceConfig:
    auto_pos_weight: bool = False
    pos_weight: float | None = None
    sampler: Literal["none", "weighted_random"] = "none"


@dataclass(frozen=True)
class EvalThresholdConfig:
    manual: float | None = None


@dataclass(frozen=True)
class FineTuneSettings:
    checkpoint_path: str
    encoder_lr: float | None = None
    classifier_lr: float | None = None
    unsafe_pickle_load: bool = False


@dataclass(frozen=True)
class PretrainedConfig:
    name_or_path: str
    load_encoder_only: bool = True
    strict: bool = True
    freeze_encoder: bool = False
    download_root: str | None = None


@dataclass(frozen=True)
class TrainingRunConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str
    top_k: int
    pretrained: PretrainedConfig | None
    data: DataConfig
    model: ModelConfig
    training: TrainingHyperparams
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )


@dataclass(frozen=True)
class SupConPretrainRunConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str
    top_k: int
    pretrained: PretrainedConfig | None
    data: DataConfig
    encoder: EncoderConfig
    training: TrainingHyperparams
    supervised_contrastive: SupervisedContrastiveConfig


@dataclass(frozen=True)
class EvalConfig:
    device: str
    checkpoint_path: str
    data: DataConfig
    threshold: EvalThresholdConfig = field(default_factory=EvalThresholdConfig)
    unsafe_pickle_load: bool = False


@dataclass(frozen=True)
class CvFoldConfig:
    name: str
    train_dirs: Sequence[str]
    val_dirs: Sequence[str]


@dataclass(frozen=True)
class CvRunConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str
    top_k: int
    pretrained: PretrainedConfig | None
    folds: Sequence[CvFoldConfig]
    label_to_index: Mapping[str, int]
    sample_rate: int
    clip_seconds: float
    batch_size: int
    num_workers: int
    model: ModelConfig
    training: TrainingHyperparams
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )


class JsonConfigLoader:
    @staticmethod
    def _validate_bandpass(
        bandpass: BandPassConfig,
        *,
        sample_rate: int,
        field_name: str = "bandpass",
    ) -> None:
        if not bandpass.enabled:
            return
        if bandpass.low_freq is None or bandpass.high_freq is None:
            raise ValueError(
                f"{field_name} requires both `low_freq` and `high_freq` when enabled"
            )
        if bandpass.q <= 0:
            raise ValueError(f"{field_name}.q must be greater than zero")
        nyquist = float(sample_rate) / 2.0
        if not (0.0 < bandpass.low_freq < bandpass.high_freq < nyquist):
            raise ValueError(
                f"{field_name} must satisfy 0 < low_freq < high_freq < Nyquist ({nyquist})"
            )

    @staticmethod
    def _validate_label_to_index(label_to_index: Mapping[str, int]) -> None:
        indices = sorted(int(v) for v in label_to_index.values())
        if indices != list(range(len(indices))):
            raise ValueError(
                f"label_to_index must be contiguous 0..N-1; got indices={indices}"
            )

    @staticmethod
    def _validate_threshold_optimization(
        cfg: ThresholdOptimizationConfig,
        *,
        num_classes: int,
        field_name: str = "threshold_optimization",
    ) -> None:
        if cfg.metric not in {"f1", "balanced_accuracy", "youden_j"}:
            raise ValueError(
                f"{field_name}.metric must be one of "
                "`f1`, `balanced_accuracy`, `youden_j`"
            )
        if cfg.enabled and num_classes != 2:
            raise ValueError(
                f"{field_name} is only supported for binary classification; "
                f"got num_classes={num_classes}"
            )

    @staticmethod
    def _validate_encoder_config(
        encoder: EncoderConfig,
        *,
        field_name: str = "encoder",
    ) -> None:
        if encoder.n_mels <= 0:
            raise ValueError(f"{field_name}.n_mels must be greater than zero")
        if encoder.n_audio_ctx <= 0:
            raise ValueError(f"{field_name}.n_audio_ctx must be greater than zero")
        if encoder.n_audio_state <= 0:
            raise ValueError(f"{field_name}.n_audio_state must be greater than zero")
        if encoder.n_audio_head <= 0:
            raise ValueError(f"{field_name}.n_audio_head must be greater than zero")
        if encoder.n_audio_layer <= 0:
            raise ValueError(f"{field_name}.n_audio_layer must be greater than zero")

    @staticmethod
    def _parse_model_config(raw_model: Mapping[str, Any]) -> ModelConfig:
        model_kwargs = dict(raw_model)
        legacy_classifier = model_kwargs.pop("classifier", None)
        if legacy_classifier is not None:
            if not isinstance(legacy_classifier, Mapping):
                raise TypeError("model.classifier must be an object")
            raw_proj_size = model_kwargs.get(
                "classifier_proj_size",
                legacy_classifier.get("hidden_dim", 256),
            )
            if raw_proj_size is None:
                raw_proj_size = 256
            model_kwargs["classifier_proj_size"] = int(raw_proj_size)
            model_kwargs["use_weighted_layer_sum"] = bool(
                model_kwargs.get("use_weighted_layer_sum", False)
            )
        model = ModelConfig(**model_kwargs)
        if model.pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {model.pooling}")
        if model.classifier_proj_size <= 0:
            raise ValueError("model.classifier_proj_size must be greater than zero")
        return model

    @staticmethod
    def _parse_encoder_config(raw_encoder: Mapping[str, Any]) -> EncoderConfig:
        encoder = EncoderConfig(**dict(raw_encoder))
        JsonConfigLoader._validate_encoder_config(encoder)
        return encoder

    @staticmethod
    def _validate_eval_threshold(
        cfg: EvalThresholdConfig,
        *,
        field_name: str = "threshold",
    ) -> None:
        if cfg.manual is None:
            return
        if not (0.0 <= cfg.manual <= 1.0):
            raise ValueError(f"{field_name}.manual must be within [0.0, 1.0]")

    @staticmethod
    def _validate_supervised_contrastive(
        cfg: SupervisedContrastiveConfig,
        *,
        field_name: str = "supervised_contrastive",
    ) -> None:
        if cfg.temperature <= 0:
            raise ValueError(f"{field_name}.temperature must be greater than zero")
        if cfg.projection_head.hidden_dim <= 0:
            raise ValueError(
                f"{field_name}.projection_head.hidden_dim must be greater than zero"
            )
        if cfg.projection_head.output_dim <= 0:
            raise ValueError(
                f"{field_name}.projection_head.output_dim must be greater than zero"
            )
        aug = cfg.augmentation
        if aug.time_mask_param < 0 or aug.time_mask_count < 0:
            raise ValueError(
                f"{field_name}.augmentation time masking values must be non-negative"
            )
        if aug.freq_mask_param < 0 or aug.freq_mask_count < 0:
            raise ValueError(
                f"{field_name}.augmentation frequency masking values must be non-negative"
            )
        if aug.gaussian_noise_std < 0:
            raise ValueError(
                f"{field_name}.augmentation.gaussian_noise_std must be non-negative"
            )

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_supcon_pretrain(path: str | Path) -> SupConPretrainRunConfig:
        raw = JsonConfigLoader.load_json(path)
        raw_data = dict(raw["data"])
        raw_data["bandpass"] = BandPassConfig(**raw_data.get("bandpass", {}))
        data = DataConfig(**raw_data)
        JsonConfigLoader._validate_label_to_index(data.label_to_index)
        JsonConfigLoader._validate_bandpass(data.bandpass, sample_rate=data.sample_rate)
        encoder = JsonConfigLoader._parse_encoder_config(raw["encoder"])
        training = TrainingHyperparams(**raw["training"])
        raw_supcon = dict(raw["supervised_contrastive"])
        raw_supcon["projection_head"] = ContrastiveProjectionHeadConfig(
            **raw_supcon.get("projection_head", {})
        )
        raw_supcon["augmentation"] = ContrastiveAugmentationConfig(
            **raw_supcon.get("augmentation", {})
        )
        supervised_contrastive = SupervisedContrastiveConfig(**raw_supcon)
        JsonConfigLoader._validate_supervised_contrastive(supervised_contrastive)
        pretrained = (
            PretrainedConfig(**raw["pretrained"]) if "pretrained" in raw else None
        )
        if pretrained is not None and pretrained.freeze_encoder:
            raise ValueError(
                "pretrained.freeze_encoder must be false for supervised contrastive pretraining"
            )
        return SupConPretrainRunConfig(
            run_name=raw["run_name"],
            seed=raw["seed"],
            device=raw["device"],
            output_dir=raw["output_dir"],
            top_k=raw["top_k"],
            pretrained=pretrained,
            data=data,
            encoder=encoder,
            training=training,
            supervised_contrastive=supervised_contrastive,
        )

    @staticmethod
    def load_training(path: str | Path) -> TrainingRunConfig:
        raw = JsonConfigLoader.load_json(path)
        raw_data = dict(raw["data"])
        raw_data["bandpass"] = BandPassConfig(**raw_data.get("bandpass", {}))
        data = DataConfig(**raw_data)
        JsonConfigLoader._validate_label_to_index(data.label_to_index)
        JsonConfigLoader._validate_bandpass(data.bandpass, sample_rate=data.sample_rate)
        pretrained = (
            PretrainedConfig(**raw["pretrained"]) if "pretrained" in raw else None
        )
        model = JsonConfigLoader._parse_model_config(raw["model"])
        training = TrainingHyperparams(**raw["training"])
        imbalance = ImbalanceConfig(**raw.get("imbalance", {}))
        threshold_optimization = ThresholdOptimizationConfig(
            **raw.get("threshold_optimization", {})
        )
        JsonConfigLoader._validate_threshold_optimization(
            threshold_optimization,
            num_classes=len(data.label_to_index),
        )
        return TrainingRunConfig(
            run_name=raw["run_name"],
            seed=raw["seed"],
            device=raw["device"],
            output_dir=raw["output_dir"],
            top_k=raw["top_k"],
            pretrained=pretrained,
            data=data,
            model=model,
            training=training,
            imbalance=imbalance,
            threshold_optimization=threshold_optimization,
        )

    @staticmethod
    def load_eval(path: str | Path) -> EvalConfig:
        raw = JsonConfigLoader.load_json(path)
        raw_data = dict(raw["data"])
        raw_data["bandpass"] = BandPassConfig(**raw_data.get("bandpass", {}))
        data = DataConfig(**raw_data)
        JsonConfigLoader._validate_label_to_index(data.label_to_index)
        JsonConfigLoader._validate_bandpass(data.bandpass, sample_rate=data.sample_rate)
        threshold = EvalThresholdConfig(**raw.get("threshold", {}))
        JsonConfigLoader._validate_eval_threshold(threshold)
        return EvalConfig(
            device=raw["device"],
            checkpoint_path=raw["checkpoint_path"],
            data=data,
            threshold=threshold,
            unsafe_pickle_load=bool(raw.get("unsafe_pickle_load", False)),
        )

    @staticmethod
    def load_cv(path: str | Path) -> CvRunConfig:
        raw = JsonConfigLoader.load_json(path)
        folds = [CvFoldConfig(**fold) for fold in raw["folds"]]
        model = JsonConfigLoader._parse_model_config(raw["model"])
        training = TrainingHyperparams(**raw["training"])
        imbalance = ImbalanceConfig(**raw.get("imbalance", {}))
        threshold_optimization = ThresholdOptimizationConfig(
            **raw.get("threshold_optimization", {})
        )
        JsonConfigLoader._validate_label_to_index(raw["label_to_index"])
        JsonConfigLoader._validate_threshold_optimization(
            threshold_optimization,
            num_classes=len(raw["label_to_index"]),
        )
        bandpass = BandPassConfig(**raw.get("bandpass", {}))
        JsonConfigLoader._validate_bandpass(bandpass, sample_rate=raw["sample_rate"])
        pretrained = (
            PretrainedConfig(**raw["pretrained"]) if "pretrained" in raw else None
        )
        return CvRunConfig(
            run_name=raw["run_name"],
            seed=raw["seed"],
            device=raw["device"],
            output_dir=raw["output_dir"],
            top_k=raw["top_k"],
            pretrained=pretrained,
            folds=folds,
            label_to_index=raw["label_to_index"],
            sample_rate=raw["sample_rate"],
            clip_seconds=raw["clip_seconds"],
            source_type=raw.get("source_type", "original"),
            batch_size=raw["batch_size"],
            num_workers=raw["num_workers"],
            model=model,
            training=training,
            imbalance=imbalance,
            bandpass=bandpass,
            threshold_optimization=threshold_optimization,
        )

    @staticmethod
    def load_finetune(path: str | Path) -> FineTuneRunConfig:
        raw = JsonConfigLoader.load_json(path)
        raw_data = dict(raw["data"])
        raw_data["bandpass"] = BandPassConfig(**raw_data.get("bandpass", {}))
        data = DataConfig(**raw_data)
        JsonConfigLoader._validate_label_to_index(data.label_to_index)
        JsonConfigLoader._validate_bandpass(data.bandpass, sample_rate=data.sample_rate)
        training = TrainingHyperparams(**raw["training"])
        finetune = FineTuneSettings(**raw["finetune"])
        imbalance = ImbalanceConfig(**raw.get("imbalance", {}))
        threshold_optimization = ThresholdOptimizationConfig(
            **raw.get("threshold_optimization", {})
        )
        JsonConfigLoader._validate_threshold_optimization(
            threshold_optimization,
            num_classes=len(data.label_to_index),
        )
        return FineTuneRunConfig(
            run_name=raw["run_name"],
            seed=raw["seed"],
            device=raw["device"],
            output_dir=raw["output_dir"],
            top_k=raw["top_k"],
            finetune=finetune,
            data=data,
            training=training,
            imbalance=imbalance,
            threshold_optimization=threshold_optimization,
        )


@dataclass(frozen=True)
class FineTuneRunConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str
    top_k: int
    finetune: FineTuneSettings
    data: DataConfig
    training: TrainingHyperparams
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )
