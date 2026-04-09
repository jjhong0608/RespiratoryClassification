from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    mode: Literal["mil"]
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
class PreprocessingConfig:
    feature_type: Literal["log_mel"] = "log_mel"
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)


@dataclass(frozen=True)
class SegmentationConfig:
    mode: Literal["full_clip", "non_overlap", "sliding_window"] = "full_clip"
    length_sec: float | None = None
    stride_sec: float | None = None
    pad_last: bool = True
    drop_last: bool = False

    def effective_length_sec(self, clip_duration_sec: float) -> float:
        if self.mode == "full_clip":
            return clip_duration_sec
        if self.length_sec is None:
            raise ValueError("segment.length_sec is required for segmented MIL modes")
        return float(self.length_sec)

    def effective_stride_sec(self, clip_duration_sec: float) -> float:
        if self.mode == "full_clip":
            return clip_duration_sec
        if self.mode == "non_overlap":
            return self.effective_length_sec(clip_duration_sec)
        if self.stride_sec is None:
            raise ValueError("segment.stride_sec is required for sliding_window mode")
        return float(self.stride_sec)


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
    segment: SegmentationConfig


@dataclass(frozen=True)
class EncoderDimsConfig:
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4


@dataclass(frozen=True)
class SegmentEncoderPoolingConfig:
    type: Literal["attention"] = "attention"
    hidden_dim: int = 128
    dropout: float = 0.0
    gated: bool = True


@dataclass(frozen=True)
class EncoderAdaptationConfig:
    mode: Literal["frozen", "partial", "full"] = "partial"
    num_layers: int = 1


@dataclass(frozen=True)
class SegmentEncoderConfig:
    type: Literal["whisper"] = "whisper"
    backbone: str = "custom"
    pretrained_name_or_path: str | None = None
    strict: bool = True
    download_root: str | None = None
    dims: EncoderDimsConfig = field(default_factory=EncoderDimsConfig)
    pooling: SegmentEncoderPoolingConfig = field(
        default_factory=SegmentEncoderPoolingConfig
    )
    adaptation: EncoderAdaptationConfig = field(
        default_factory=EncoderAdaptationConfig
    )


@dataclass(frozen=True)
class InstanceHeadConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class InterAttentionConfig:
    hidden_dim: int = 128
    dropout: float = 0.0
    gated: bool = True


@dataclass(frozen=True)
class TopKConfig:
    k: int = 1


@dataclass(frozen=True)
class MILConfig:
    aggregator: Literal["attention", "max", "topk"] = "attention"
    attention: InterAttentionConfig = field(default_factory=InterAttentionConfig)
    topk: TopKConfig = field(default_factory=TopKConfig)


@dataclass(frozen=True)
class ModelConfig:
    segment_encoder: SegmentEncoderConfig
    instance_head: InstanceHeadConfig
    mil: MILConfig


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
    type: Literal["bce", "focal"] = "bce"
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
    save_segment_scores: bool = True
    save_instance_logits: bool = True
    save_intra_attention_weights: bool = True
    save_inter_attention_weights: bool = True
    save_topk_indices: bool = True
    save_bag_metadata: bool = True


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

    @staticmethod
    def _validate_binary_labels(label_to_index: Mapping[str, int]) -> None:
        JsonConfigLoader._validate_contiguous_labels(label_to_index)
        if len(label_to_index) != 2:
            raise ValueError(
                "The MIL branch currently supports bag-level binary tasks only"
            )

    @staticmethod
    def _validate_experiment(cfg: ExperimentConfig) -> None:
        if cfg.mode != "mil":
            raise ValueError("experiment.mode must be 'mil' for this branch")
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
        if cfg.feature_type != "log_mel":
            raise ValueError(
                "Only data.preprocessing.feature_type='log_mel' is supported"
            )
        if cfg.n_fft <= 0 or cfg.hop_length <= 0 or cfg.win_length <= 0:
            raise ValueError("STFT parameters must be greater than zero")
        if cfg.n_mels <= 0:
            raise ValueError("data.preprocessing.n_mels must be greater than zero")
        JsonConfigLoader._validate_bandpass(cfg.bandpass, sample_rate)

    @staticmethod
    def _validate_segment(cfg: SegmentationConfig) -> None:
        if cfg.pad_last and cfg.drop_last:
            raise ValueError("pad_last and drop_last cannot both be true")
        if cfg.mode == "full_clip":
            return
        if cfg.length_sec is None or cfg.length_sec <= 0:
            raise ValueError("data.segment.length_sec must be greater than zero")
        if cfg.mode == "sliding_window" and (
            cfg.stride_sec is None or cfg.stride_sec <= 0
        ):
            raise ValueError(
                "data.segment.stride_sec must be greater than zero for sliding_window"
            )

    @staticmethod
    def _validate_data(cfg: DataConfig) -> None:
        JsonConfigLoader._validate_binary_labels(cfg.label_to_index)
        JsonConfigLoader._validate_audio(cfg.audio)
        JsonConfigLoader._validate_preprocessing(
            cfg.preprocessing, cfg.audio.sample_rate
        )
        JsonConfigLoader._validate_segment(cfg.segment)
        if cfg.batch_size <= 0:
            raise ValueError("data.batch_size must be greater than zero")
        if cfg.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")

    @staticmethod
    def _validate_segment_encoder(cfg: SegmentEncoderConfig) -> None:
        if cfg.type != "whisper":
            raise ValueError("Only model.segment_encoder.type='whisper' is supported")
        if cfg.dims.n_audio_state <= 0:
            raise ValueError(
                "model.segment_encoder.dims.n_audio_state must be greater than zero"
            )
        if cfg.dims.n_audio_head <= 0:
            raise ValueError(
                "model.segment_encoder.dims.n_audio_head must be greater than zero"
            )
        if cfg.dims.n_audio_layer <= 0:
            raise ValueError(
                "model.segment_encoder.dims.n_audio_layer must be greater than zero"
            )
        if cfg.pooling.type != "attention":
            raise ValueError("model.segment_encoder.pooling.type must be 'attention'")
        if cfg.pooling.hidden_dim <= 0:
            raise ValueError(
                "model.segment_encoder.pooling.hidden_dim must be greater than zero"
            )
        if not (0.0 <= cfg.pooling.dropout < 1.0):
            raise ValueError(
                "model.segment_encoder.pooling.dropout must be within [0, 1)"
            )
        if cfg.adaptation.mode not in {"frozen", "partial", "full"}:
            raise ValueError(
                "model.segment_encoder.adaptation.mode must be one of "
                "['frozen', 'full', 'partial']"
            )
        if cfg.adaptation.mode == "partial":
            if cfg.adaptation.num_layers <= 0:
                raise ValueError(
                    "model.segment_encoder.adaptation.num_layers must be greater than zero"
                )
            if cfg.adaptation.num_layers > cfg.dims.n_audio_layer:
                raise ValueError(
                    "model.segment_encoder.adaptation.num_layers must not exceed "
                    "model.segment_encoder.dims.n_audio_layer"
                )

    @staticmethod
    def _validate_instance_head(cfg: InstanceHeadConfig) -> None:
        if cfg.hidden_dim <= 0:
            raise ValueError("model.instance_head.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.instance_head.dropout must be within [0, 1)")

    @staticmethod
    def _validate_mil(cfg: MILConfig) -> None:
        if cfg.aggregator not in {"attention", "max", "topk"}:
            raise ValueError("model.mil.aggregator must be one of ['attention', 'max', 'topk']")
        if cfg.topk.k <= 0:
            raise ValueError("model.mil.topk.k must be greater than zero")
        if cfg.attention.hidden_dim <= 0:
            raise ValueError("model.mil.attention.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.attention.dropout < 1.0):
            raise ValueError("model.mil.attention.dropout must be within [0, 1)")

    @staticmethod
    def _validate_model(cfg: ModelConfig) -> None:
        JsonConfigLoader._validate_segment_encoder(cfg.segment_encoder)
        JsonConfigLoader._validate_instance_head(cfg.instance_head)
        JsonConfigLoader._validate_mil(cfg.mil)

    @staticmethod
    def _validate_train(cfg: TrainConfig) -> None:
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
        if cfg.loss.type not in ["bce", "focal"]:
            raise ValueError("Only train.loss.type='bce' or 'focal' is supported")
        if cfg.loss.auto_pos_weight and cfg.loss.pos_weight is not None:
            raise ValueError(
                "train.loss.auto_pos_weight and train.loss.pos_weight cannot both be set"
            )
        if cfg.loss.pos_weight is not None and cfg.loss.pos_weight <= 0:
            raise ValueError("train.loss.pos_weight must be greater than zero")
        if cfg.early_stopping.monitor not in JsonConfigLoader._EARLY_STOPPING_MONITORS:
            raise ValueError(
                "train.early_stopping.monitor must be one of "
                f"{sorted(JsonConfigLoader._EARLY_STOPPING_MONITORS)}"
            )
        if cfg.early_stopping.patience <= 0:
            raise ValueError(
                "train.early_stopping.patience must be greater than zero"
            )
        if cfg.early_stopping.min_delta < 0:
            raise ValueError(
                "train.early_stopping.min_delta must be non-negative"
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
        preprocessing["bandpass"] = BandPassConfig(**preprocessing.get("bandpass", {}))
        kwargs["preprocessing"] = PreprocessingConfig(**preprocessing)
        kwargs["segment"] = SegmentationConfig(**dict(raw["segment"]))
        cfg = DataConfig(**kwargs)
        JsonConfigLoader._validate_data(cfg)
        return cfg

    @staticmethod
    def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
        kwargs = dict(raw)
        segment_encoder = dict(raw["segment_encoder"])
        segment_encoder["dims"] = EncoderDimsConfig(**segment_encoder.get("dims", {}))
        segment_encoder["pooling"] = SegmentEncoderPoolingConfig(
            **segment_encoder.get("pooling", {})
        )
        segment_encoder["adaptation"] = EncoderAdaptationConfig(
            **segment_encoder.get("adaptation", {})
        )
        kwargs["segment_encoder"] = SegmentEncoderConfig(**segment_encoder)
        kwargs["instance_head"] = InstanceHeadConfig(**dict(raw["instance_head"]))
        mil = dict(raw["mil"])
        mil["attention"] = InterAttentionConfig(**mil.get("attention", {}))
        mil["topk"] = TopKConfig(**mil.get("topk", {}))
        kwargs["mil"] = MILConfig(**mil)
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
        cfg = TrainConfig(**kwargs)
        JsonConfigLoader._validate_train(cfg)
        return cfg

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
        return TrainingRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            model=JsonConfigLoader._parse_model(raw["model"]),
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
        )

    @staticmethod
    def load_eval(path: str | Path) -> EvalConfig:
        raw = JsonConfigLoader.load_json(path)
        return EvalConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            threshold_optimization=JsonConfigLoader._parse_threshold_optimization(
                raw.get("threshold_optimization")
            ),
        )

    @staticmethod
    def load_cv(path: str | Path) -> CvRunConfig:
        raw = JsonConfigLoader.load_json(path)
        folds = [CvFoldConfig(**dict(item)) for item in raw["folds"]]
        base_data = JsonConfigLoader._parse_data(raw["data"])
        return CvRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=base_data,
            model=JsonConfigLoader._parse_model(raw["model"]),
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            folds=folds,
        )
