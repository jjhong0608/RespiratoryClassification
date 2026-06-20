from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig
from src.pretrained.whisper import WhisperPretrainedConfig

DEFAULT_AST_PRETRAINED_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    mode: Literal["clip"]
    seed: int
    device: str
    output_dir: str


@dataclass(frozen=True)
class BandPassConfig:
    enabled: bool = False
    low_hz: float | None = None
    high_hz: float | None = None
    q: float = 0.707


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    clip_duration_sec: float

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_duration_sec))


@dataclass(frozen=True)
class AstFbankConfig:
    num_mel_bins: int = 128
    max_length: int = 1024
    do_normalize: bool = True
    mean: float = -4.2677393
    std: float = 4.5689974


@dataclass(frozen=True)
class LogMelConfig:
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80


@dataclass(frozen=True)
class ResNetSpectrogramConfig:
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    use_hpss: bool = True
    hpss_margin: float = 1.0
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingConfig:
    feature_type: Literal["ast_fbank", "log_mel", "resnet_spectrogram"] = "ast_fbank"
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)
    ast_fbank: AstFbankConfig = field(default_factory=AstFbankConfig)
    log_mel: LogMelConfig = field(default_factory=LogMelConfig)
    resnet_spectrogram: ResNetSpectrogramConfig = field(
        default_factory=ResNetSpectrogramConfig
    )


@dataclass(frozen=True)
class DataConfig:
    train_dirs: list[str]
    val_dirs: list[str]
    eval_dirs: list[str]
    label_to_index: Mapping[str, int]
    batch_size: int
    num_workers: int
    audio: AudioConfig
    preprocessing: PreprocessingConfig


@dataclass(frozen=True)
class EncoderAdaptationConfig:
    mode: Literal["frozen", "partial", "full"] = "partial"
    num_layers: int = 1


@dataclass(frozen=True)
class AstArchitectureConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    frequency_stride: int = 10
    time_stride: int = 10
    patch_size: int = 16
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


@dataclass(frozen=True)
class AstEncoderConfig:
    type: Literal["ast"] = "ast"
    pretrained_name_or_path: str | None = DEFAULT_AST_PRETRAINED_NAME
    cache_dir: str | None = None
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: AstArchitectureConfig = field(default_factory=AstArchitectureConfig)


@dataclass(frozen=True)
class WhisperEncoderConfig:
    type: Literal["whisper"] = "whisper"
    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4
    adaptation: EncoderAdaptationConfig = field(
        default_factory=lambda: EncoderAdaptationConfig(mode="frozen", num_layers=1)
    )
    pretrained: WhisperPretrainedConfig | None = field(
        default_factory=WhisperPretrainedConfig
    )


@dataclass(frozen=True)
class ResNet50EncoderConfig:
    type: Literal["resnet50"] = "resnet50"
    weights: Literal["imagenet", "none"] = "imagenet"
    adaptation: EncoderAdaptationConfig = field(
        default_factory=lambda: EncoderAdaptationConfig(mode="frozen", num_layers=1)
    )
    input_channels: int = 3
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


type EncoderConfig = AstEncoderConfig | WhisperEncoderConfig | ResNet50EncoderConfig


@dataclass(frozen=True)
class ClassifierConfig:
    type: Literal["linear", "mlp", "hf"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0
    pooling: Literal["cls", "mean"] = "cls"
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderConfig
    classifier: ClassifierConfig


@dataclass(frozen=True)
class OptimizerConfig:
    encoder_lr: float
    head_lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_ratio: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    type: Literal["bce", "focal", "cross_entropy"] = "bce"
    auto_pos_weight: bool = False
    pos_weight: float | None = None
    gamma: float = 2.0


@dataclass(frozen=True)
class SamplerConfig:
    weighted_random: bool = False


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: Literal["val_loss"] = "val_loss"
    patience: int = 15
    min_delta: float = 1e-4


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    top_k: int
    max_grad_norm: float
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class AnalysisOutputConfig:
    save_logits: bool = True
    save_probabilities: bool = True
    save_embeddings: bool = False
    save_clip_metadata: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    outputs: AnalysisOutputConfig = field(default_factory=AnalysisOutputConfig)


@dataclass(frozen=True)
class TrainingRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    analysis: AnalysisConfig


@dataclass(frozen=True)
class EvalConfig:
    experiment: ExperimentConfig
    checkpoint_path: str
    data: DataConfig
    analysis: AnalysisConfig
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )


@dataclass(frozen=True)
class CvFoldConfig:
    name: str
    train_dirs: list[str]
    val_dirs: list[str]


@dataclass(frozen=True)
class CvRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    analysis: AnalysisConfig
    folds: list[CvFoldConfig]


class JsonConfigLoader:
    _THRESHOLD_METRICS = {"f1", "balanced_accuracy", "youden_j"}
    _EARLY_STOPPING_MONITORS = {"val_loss"}

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Config root must be a JSON object")
        return payload

    @staticmethod
    def _validate_contiguous_labels(label_to_index: Mapping[str, int]) -> None:
        indices = sorted(int(value) for value in label_to_index.values())
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(f"label_to_index must be contiguous 0..N-1; got {indices}")
        if len(indices) < 2:
            raise ValueError("label_to_index must contain at least two classes")

    @staticmethod
    def _validate_experiment(cfg: ExperimentConfig) -> None:
        if cfg.mode != "clip":
            raise ValueError("experiment.mode must be 'clip' for this branch")
        if not cfg.name:
            raise ValueError("experiment.name must not be empty")
        if not cfg.task:
            raise ValueError("experiment.task must not be empty")

    @staticmethod
    def _validate_bandpass(cfg: BandPassConfig, sample_rate: int) -> None:
        if not cfg.enabled:
            return
        if cfg.low_hz is None or cfg.high_hz is None:
            raise ValueError(
                "data.preprocessing.bandpass requires low_hz and high_hz when enabled"
            )
        if cfg.q <= 0:
            raise ValueError("data.preprocessing.bandpass.q must be greater than zero")
        nyquist = float(sample_rate) / 2.0
        if not (0.0 < cfg.low_hz < cfg.high_hz < nyquist):
            raise ValueError(
                "data.preprocessing.bandpass must satisfy "
                "0 < low_hz < high_hz < sample_rate / 2"
            )

    @staticmethod
    def _validate_audio(cfg: AudioConfig) -> None:
        if cfg.sample_rate <= 0:
            raise ValueError("data.audio.sample_rate must be greater than zero")
        if cfg.clip_duration_sec <= 0:
            raise ValueError("data.audio.clip_duration_sec must be greater than zero")

    @staticmethod
    def _validate_preprocessing(cfg: PreprocessingConfig, sample_rate: int) -> None:
        if cfg.feature_type not in {"ast_fbank", "log_mel", "resnet_spectrogram"}:
            raise ValueError(
                "data.preprocessing.feature_type must be one of "
                "['ast_fbank', 'log_mel', 'resnet_spectrogram']"
            )
        if sample_rate != 16000:
            raise ValueError(
                "data.audio.sample_rate must be 16000 for clip classification"
            )
        if cfg.ast_fbank.num_mel_bins <= 0:
            raise ValueError(
                "data.preprocessing.ast_fbank.num_mel_bins must be greater than zero"
            )
        if cfg.ast_fbank.max_length <= 0:
            raise ValueError(
                "data.preprocessing.ast_fbank.max_length must be greater than zero"
            )
        if cfg.ast_fbank.do_normalize and cfg.ast_fbank.std <= 0:
            raise ValueError(
                "data.preprocessing.ast_fbank.std must be greater than zero when normalization is enabled"
            )
        if cfg.log_mel.n_fft <= 0:
            raise ValueError(
                "data.preprocessing.log_mel.n_fft must be greater than zero"
            )
        if cfg.log_mel.hop_length <= 0:
            raise ValueError(
                "data.preprocessing.log_mel.hop_length must be greater than zero"
            )
        if cfg.log_mel.win_length <= 0:
            raise ValueError(
                "data.preprocessing.log_mel.win_length must be greater than zero"
            )
        if cfg.log_mel.n_mels <= 0:
            raise ValueError(
                "data.preprocessing.log_mel.n_mels must be greater than zero"
            )
        if cfg.resnet_spectrogram.n_fft <= 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.n_fft must be greater than zero"
            )
        if cfg.resnet_spectrogram.hop_length <= 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.hop_length must be greater than zero"
            )
        if cfg.resnet_spectrogram.win_length <= 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.win_length must be greater than zero"
            )
        if cfg.resnet_spectrogram.n_mels <= 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.n_mels must be greater than zero"
            )
        if cfg.resnet_spectrogram.hpss_margin <= 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.hpss_margin must be greater than zero"
            )
        if cfg.resnet_spectrogram.image_size <= 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.image_size must be greater than zero"
            )
        if len(cfg.resnet_spectrogram.image_mean) != 3:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.image_mean must contain three values"
            )
        if len(cfg.resnet_spectrogram.image_std) != 3:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.image_std must contain three values"
            )
        if any(value <= 0 for value in cfg.resnet_spectrogram.image_std):
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.image_std values must be greater than zero"
            )
        nyquist = float(sample_rate) / 2.0
        if cfg.resnet_spectrogram.f_min < 0:
            raise ValueError(
                "data.preprocessing.resnet_spectrogram.f_min must be non-negative"
            )
        if cfg.resnet_spectrogram.f_max is not None and not (
            cfg.resnet_spectrogram.f_min < cfg.resnet_spectrogram.f_max <= nyquist
        ):
            raise ValueError(
                "data.preprocessing.resnet_spectrogram must satisfy "
                "f_min < f_max <= sample_rate / 2"
            )
        JsonConfigLoader._validate_bandpass(cfg.bandpass, sample_rate)

    @staticmethod
    def _validate_data(cfg: DataConfig) -> None:
        JsonConfigLoader._validate_contiguous_labels(cfg.label_to_index)
        JsonConfigLoader._validate_audio(cfg.audio)
        JsonConfigLoader._validate_preprocessing(
            cfg.preprocessing, cfg.audio.sample_rate
        )
        if cfg.batch_size <= 0:
            raise ValueError("data.batch_size must be greater than zero")
        if cfg.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")

    @staticmethod
    def _validate_ast_encoder(cfg: AstEncoderConfig) -> None:
        if cfg.adaptation.mode not in {"frozen", "partial", "full"}:
            raise ValueError(
                "model.encoder.adaptation.mode must be one of ['frozen', 'full', 'partial']"
            )
        if cfg.architecture.hidden_size <= 0:
            raise ValueError(
                "model.encoder.architecture.hidden_size must be greater than zero"
            )
        if cfg.architecture.num_hidden_layers <= 0:
            raise ValueError(
                "model.encoder.architecture.num_hidden_layers must be greater than zero"
            )
        if cfg.architecture.num_attention_heads <= 0:
            raise ValueError(
                "model.encoder.architecture.num_attention_heads must be greater than zero"
            )
        if cfg.architecture.intermediate_size <= 0:
            raise ValueError(
                "model.encoder.architecture.intermediate_size must be greater than zero"
            )
        if not (0.0 <= cfg.architecture.hidden_dropout_prob < 1.0):
            raise ValueError(
                "model.encoder.architecture.hidden_dropout_prob must be within [0, 1)"
            )
        if not (0.0 <= cfg.architecture.attention_probs_dropout_prob < 1.0):
            raise ValueError(
                "model.encoder.architecture.attention_probs_dropout_prob must be within [0, 1)"
            )
        if cfg.architecture.frequency_stride <= 0:
            raise ValueError(
                "model.encoder.architecture.frequency_stride must be greater than zero"
            )
        if cfg.architecture.time_stride <= 0:
            raise ValueError(
                "model.encoder.architecture.time_stride must be greater than zero"
            )
        if cfg.architecture.patch_size <= 0:
            raise ValueError(
                "model.encoder.architecture.patch_size must be greater than zero"
            )
        if cfg.adaptation.mode == "partial":
            if cfg.adaptation.num_layers <= 0:
                raise ValueError(
                    "model.encoder.adaptation.num_layers must be greater than zero"
                )
            if cfg.adaptation.num_layers > cfg.architecture.num_hidden_layers:
                raise ValueError(
                    "model.encoder.adaptation.num_layers must not exceed "
                    "model.encoder.architecture.num_hidden_layers"
                )

    @staticmethod
    def _validate_whisper_encoder(cfg: WhisperEncoderConfig) -> None:
        if cfg.adaptation.mode not in {"frozen", "partial", "full"}:
            raise ValueError(
                "model.encoder.adaptation.mode must be one of ['frozen', 'partial', 'full']"
            )
        if cfg.n_mels <= 0:
            raise ValueError("model.encoder.n_mels must be greater than zero")
        if cfg.n_audio_ctx <= 0:
            raise ValueError("model.encoder.n_audio_ctx must be greater than zero")
        if cfg.n_audio_state <= 0:
            raise ValueError("model.encoder.n_audio_state must be greater than zero")
        if cfg.n_audio_head <= 0:
            raise ValueError("model.encoder.n_audio_head must be greater than zero")
        if cfg.n_audio_layer <= 0:
            raise ValueError("model.encoder.n_audio_layer must be greater than zero")
        if cfg.n_audio_state % cfg.n_audio_head != 0:
            raise ValueError(
                "model.encoder.n_audio_state must be divisible by n_audio_head"
            )
        if cfg.adaptation.mode == "partial" and not (
            1 <= cfg.adaptation.num_layers <= cfg.n_audio_layer
        ):
            raise ValueError(
                "model.encoder.adaptation.num_layers must be between 1 and "
                "model.encoder.n_audio_layer for Whisper partial adaptation"
            )
        if cfg.pretrained is not None and not cfg.pretrained.load_encoder_only:
            raise ValueError("Whisper supports only pretrained.load_encoder_only=true")

    @staticmethod
    def _validate_resnet50_encoder(cfg: ResNet50EncoderConfig) -> None:
        if cfg.weights not in {"imagenet", "none"}:
            raise ValueError(
                "model.encoder.weights must be one of ['imagenet', 'none']"
            )
        if cfg.adaptation.mode not in {"frozen", "partial", "full"}:
            raise ValueError(
                "model.encoder.adaptation.mode must be one of ['frozen', 'partial', 'full']"
            )
        if cfg.adaptation.mode == "partial" and not (
            1 <= cfg.adaptation.num_layers <= 4
        ):
            raise ValueError(
                "model.encoder.adaptation.num_layers must be between 1 and 4 for ResNet50 partial adaptation"
            )
        if cfg.input_channels != 3:
            raise ValueError("ResNet50 disease frontend requires input_channels=3")
        if cfg.image_size <= 0:
            raise ValueError("model.encoder.image_size must be greater than zero")
        if len(cfg.image_mean) != 3:
            raise ValueError("model.encoder.image_mean must contain three values")
        if len(cfg.image_std) != 3:
            raise ValueError("model.encoder.image_std must contain three values")
        if any(value <= 0 for value in cfg.image_std):
            raise ValueError("model.encoder.image_std values must be greater than zero")

    @staticmethod
    def _validate_encoder(cfg: EncoderConfig) -> None:
        if cfg.type == "ast":
            JsonConfigLoader._validate_ast_encoder(cfg)
            return
        if cfg.type == "whisper":
            JsonConfigLoader._validate_whisper_encoder(cfg)
            return
        if cfg.type == "resnet50":
            JsonConfigLoader._validate_resnet50_encoder(cfg)
            return
        raise ValueError(f"Unsupported model.encoder.type: {cfg.type}")

    @staticmethod
    def _validate_encoder_against_data(
        data_cfg: DataConfig,
        encoder_cfg: EncoderConfig,
    ) -> None:
        if encoder_cfg.type == "whisper":
            if data_cfg.preprocessing.feature_type != "log_mel":
                raise ValueError(
                    "Whisper encoder requires data.preprocessing.feature_type='log_mel'"
                )
            log_mel = data_cfg.preprocessing.log_mel
            n_samples = int(
                round(data_cfg.audio.sample_rate * data_cfg.audio.clip_duration_sec)
            )
            n_frames = n_samples // log_mel.hop_length
            expected_ctx = (n_frames + 1) // 2
            if encoder_cfg.n_mels != log_mel.n_mels:
                raise ValueError(
                    "Whisper encoder n_mels must match data.preprocessing.log_mel.n_mels"
                )
            if encoder_cfg.n_audio_ctx != expected_ctx:
                raise ValueError(
                    "Whisper encoder n_audio_ctx must match log-mel frame count "
                    f"after stride-2 conv; expected {expected_ctx}, got "
                    f"{encoder_cfg.n_audio_ctx}"
                )
            return
        if encoder_cfg.type == "resnet50":
            if data_cfg.preprocessing.feature_type != "resnet_spectrogram":
                raise ValueError(
                    "ResNet50 encoder requires "
                    "data.preprocessing.feature_type='resnet_spectrogram'"
                )
            if data_cfg.preprocessing.source_type != "original":
                raise ValueError(
                    "ResNet50 spectrogram frontend requires "
                    "data.preprocessing.source_type='original'; HPSS channels are "
                    "computed inside the frontend"
                )
            if encoder_cfg.input_channels != 3:
                raise ValueError("ResNet50 encoder input_channels must be 3")
            resnet_spec = data_cfg.preprocessing.resnet_spectrogram
            if encoder_cfg.image_size != resnet_spec.image_size:
                raise ValueError(
                    "ResNet50 encoder image_size must match "
                    "data.preprocessing.resnet_spectrogram.image_size"
                )
            if tuple(encoder_cfg.image_mean) != tuple(resnet_spec.image_mean):
                raise ValueError(
                    "ResNet50 encoder image_mean must match "
                    "data.preprocessing.resnet_spectrogram.image_mean"
                )
            if tuple(encoder_cfg.image_std) != tuple(resnet_spec.image_std):
                raise ValueError(
                    "ResNet50 encoder image_std must match "
                    "data.preprocessing.resnet_spectrogram.image_std"
                )
            return
        if encoder_cfg.pretrained_name_or_path is None:
            return
        from transformers import ASTConfig

        pretrained_cfg = ASTConfig.from_pretrained(
            encoder_cfg.pretrained_name_or_path,
            cache_dir=encoder_cfg.cache_dir,
        )
        expected_bins = int(data_cfg.preprocessing.ast_fbank.num_mel_bins)
        expected_length = int(data_cfg.preprocessing.ast_fbank.max_length)
        actual_bins = int(pretrained_cfg.num_mel_bins)
        actual_length = int(pretrained_cfg.max_length)
        if actual_bins != expected_bins or actual_length != expected_length:
            raise ValueError(
                "Pretrained AST encoder input dims do not match "
                "data.preprocessing.ast_fbank.\n"
                f"- encoder: num_mel_bins={actual_bins}, max_length={actual_length}\n"
                f"- data:    num_mel_bins={expected_bins}, max_length={expected_length}\n"
                f"- name_or_path: {encoder_cfg.pretrained_name_or_path}"
            )

    @staticmethod
    def _validate_classifier(cfg: ClassifierConfig) -> None:
        if cfg.type not in {"linear", "mlp", "hf"}:
            raise ValueError(
                "model.classifier.type must be one of ['hf', 'linear', 'mlp']"
            )
        if cfg.hidden_dim <= 0:
            raise ValueError("model.classifier.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.classifier.dropout must be within [0, 1)")
        if cfg.pooling not in {"cls", "mean"}:
            raise ValueError("model.classifier.pooling must be one of ['cls', 'mean']")
        if cfg.classifier_proj_size <= 0:
            raise ValueError(
                "model.classifier.classifier_proj_size must be greater than zero"
            )

    @staticmethod
    def _validate_model(cfg: ModelConfig) -> None:
        JsonConfigLoader._validate_encoder(cfg.encoder)
        JsonConfigLoader._validate_classifier(cfg.classifier)
        if cfg.encoder.type == "ast" and cfg.classifier.type == "hf":
            raise ValueError("AST encoder supports only linear or mlp classifier heads")
        if (
            cfg.encoder.type == "whisper"
            and cfg.classifier.type != "hf"
            and cfg.classifier.pooling != "mean"
        ):
            raise ValueError(
                "Whisper linear/mlp classifier heads require pooling='mean'"
            )
        if cfg.encoder.type == "resnet50" and cfg.classifier.type == "hf":
            raise ValueError(
                "ResNet50 encoder supports only linear or mlp classifier heads"
            )

    @staticmethod
    def _validate_train(cfg: TrainConfig, *, num_classes: int) -> None:
        if cfg.epochs <= 0:
            raise ValueError("train.epochs must be greater than zero")
        if cfg.top_k <= 0:
            raise ValueError("train.top_k must be greater than zero")
        if cfg.max_grad_norm <= 0:
            raise ValueError("train.max_grad_norm must be greater than zero")
        if cfg.optimizer.encoder_lr <= 0:
            raise ValueError("train.optimizer.encoder_lr must be greater than zero")
        if cfg.optimizer.head_lr <= 0:
            raise ValueError("train.optimizer.head_lr must be greater than zero")
        if cfg.optimizer.weight_decay < 0:
            raise ValueError("train.optimizer.weight_decay must be non-negative")
        if not (0.0 <= cfg.scheduler.warmup_ratio <= 1.0):
            raise ValueError("train.scheduler.warmup_ratio must be within [0, 1]")
        if cfg.early_stopping.monitor not in JsonConfigLoader._EARLY_STOPPING_MONITORS:
            raise ValueError(
                "train.early_stopping.monitor must be one of "
                f"{sorted(JsonConfigLoader._EARLY_STOPPING_MONITORS)}"
            )
        if cfg.early_stopping.patience <= 0:
            raise ValueError("train.early_stopping.patience must be greater than zero")
        if cfg.early_stopping.min_delta < 0:
            raise ValueError("train.early_stopping.min_delta must be non-negative")
        if num_classes == 2:
            if cfg.loss.type not in {"bce", "focal"}:
                raise ValueError(
                    "Binary AST runs support only train.loss.type='bce' or 'focal'"
                )
            if cfg.loss.auto_pos_weight and cfg.loss.pos_weight is not None:
                raise ValueError(
                    "train.loss.auto_pos_weight and train.loss.pos_weight cannot both be set"
                )
            if cfg.loss.pos_weight is not None and cfg.loss.pos_weight <= 0:
                raise ValueError("train.loss.pos_weight must be greater than zero")
            return
        if cfg.loss.type != "cross_entropy":
            raise ValueError(
                "Multi-class AST runs require train.loss.type='cross_entropy'"
            )
        if cfg.loss.auto_pos_weight:
            raise ValueError(
                "train.loss.auto_pos_weight is only supported for binary classification"
            )
        if cfg.loss.pos_weight is not None:
            raise ValueError(
                "train.loss.pos_weight is only supported for binary classification"
            )

    @staticmethod
    def _parse_experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
        cfg = ExperimentConfig(**dict(raw))
        JsonConfigLoader._validate_experiment(cfg)
        return cfg

    @staticmethod
    def _parse_data(raw: Mapping[str, Any]) -> DataConfig:
        kwargs = dict(raw)
        kwargs.setdefault("train_dirs", [])
        kwargs.setdefault("val_dirs", [])
        kwargs.setdefault("eval_dirs", [])
        kwargs["audio"] = AudioConfig(**dict(raw["audio"]))
        preprocessing = dict(raw["preprocessing"])
        preprocessing.setdefault("feature_type", "ast_fbank")
        preprocessing["bandpass"] = BandPassConfig(**preprocessing.get("bandpass", {}))
        preprocessing["ast_fbank"] = AstFbankConfig(
            **dict(preprocessing.get("ast_fbank", {}))
        )
        preprocessing["log_mel"] = LogMelConfig(
            **dict(preprocessing.get("log_mel", {}))
        )
        preprocessing["resnet_spectrogram"] = ResNetSpectrogramConfig(
            **JsonConfigLoader._parse_resnet_spectrogram_dict(
                preprocessing.get("resnet_spectrogram", {})
            )
        )
        kwargs["preprocessing"] = PreprocessingConfig(**preprocessing)
        cfg = DataConfig(**kwargs)
        JsonConfigLoader._validate_data(cfg)
        return cfg

    @staticmethod
    def _parse_resnet_spectrogram_dict(raw: object) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError("data.preprocessing.resnet_spectrogram must be an object")
        parsed = dict(raw)
        if "image_mean" in parsed:
            parsed["image_mean"] = tuple(parsed["image_mean"])
        if "image_std" in parsed:
            parsed["image_std"] = tuple(parsed["image_std"])
        return parsed

    @staticmethod
    def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
        kwargs = dict(raw)
        encoder = dict(raw["encoder"])
        encoder_type = encoder.get("type", "ast")
        if encoder_type == "ast":
            encoder["adaptation"] = EncoderAdaptationConfig(
                **dict(encoder.get("adaptation", {}))
            )
            encoder["architecture"] = AstArchitectureConfig(
                **dict(encoder.get("architecture", {}))
            )
            kwargs["encoder"] = AstEncoderConfig(**encoder)
        elif encoder_type == "whisper":
            encoder["adaptation"] = EncoderAdaptationConfig(
                **dict(
                    encoder.get(
                        "adaptation",
                        {
                            "mode": "frozen",
                            "num_layers": 1,
                        },
                    )
                )
            )
            pretrained_raw = encoder.get("pretrained")
            if pretrained_raw is not None:
                encoder["pretrained"] = WhisperPretrainedConfig(**dict(pretrained_raw))
            kwargs["encoder"] = WhisperEncoderConfig(**encoder)
        elif encoder_type == "resnet50":
            encoder["adaptation"] = EncoderAdaptationConfig(
                **dict(encoder.get("adaptation", {}))
            )
            if "image_mean" in encoder:
                encoder["image_mean"] = tuple(encoder["image_mean"])
            if "image_std" in encoder:
                encoder["image_std"] = tuple(encoder["image_std"])
            kwargs["encoder"] = ResNet50EncoderConfig(**encoder)
        else:
            raise ValueError(f"Unsupported model.encoder.type: {encoder_type}")
        kwargs["classifier"] = ClassifierConfig(**dict(raw["classifier"]))
        cfg = ModelConfig(**kwargs)
        JsonConfigLoader._validate_model(cfg)
        return cfg

    @staticmethod
    def _parse_train(raw: Mapping[str, Any]) -> TrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = OptimizerConfig(**dict(raw["optimizer"]))
        kwargs["scheduler"] = SchedulerConfig(**dict(raw.get("scheduler", {})))
        kwargs["loss"] = LossConfig(**dict(raw.get("loss", {})))
        kwargs["sampler"] = SamplerConfig(**dict(raw.get("sampler", {})))
        kwargs["early_stopping"] = EarlyStoppingConfig(
            **dict(raw.get("early_stopping", {}))
        )
        return TrainConfig(**kwargs)

    @staticmethod
    def _parse_analysis(raw: Mapping[str, Any] | None) -> AnalysisConfig:
        if raw is None:
            return AnalysisConfig()
        kwargs = dict(raw)
        kwargs["outputs"] = AnalysisOutputConfig(**dict(raw.get("outputs", {})))
        return AnalysisConfig(**kwargs)

    @staticmethod
    def _parse_threshold_optimization(
        raw: Mapping[str, Any] | None,
    ) -> ThresholdOptimizationConfig:
        cfg = ThresholdOptimizationConfig(**dict(raw or {}))
        if cfg.metric not in JsonConfigLoader._THRESHOLD_METRICS:
            raise ValueError(
                "threshold_optimization.metric must be one of "
                f"{sorted(JsonConfigLoader._THRESHOLD_METRICS)}"
            )
        return cfg

    @staticmethod
    def load_training(path: str | Path) -> TrainingRunConfig:
        raw = JsonConfigLoader.load_json(path)
        cfg = TrainingRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            model=JsonConfigLoader._parse_model(raw["model"]),
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
        )
        JsonConfigLoader._validate_encoder_against_data(cfg.data, cfg.model.encoder)
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
        )
        return cfg

    @staticmethod
    def load_eval(path: str | Path) -> EvalConfig:
        raw = JsonConfigLoader.load_json(path)
        cfg = EvalConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            threshold_optimization=JsonConfigLoader._parse_threshold_optimization(
                raw.get("threshold_optimization")
            ),
        )
        return cfg

    @staticmethod
    def load_cv(path: str | Path) -> CvRunConfig:
        raw = JsonConfigLoader.load_json(path)
        folds = [CvFoldConfig(**dict(item)) for item in raw["folds"]]
        cfg = CvRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            model=JsonConfigLoader._parse_model(raw["model"]),
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            folds=folds,
        )
        JsonConfigLoader._validate_encoder_against_data(cfg.data, cfg.model.encoder)
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
        )
        return cfg
