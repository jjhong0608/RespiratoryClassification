from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


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
class EncoderConfig:
    type: Literal["whisper"] = "whisper"
    backbone: str = "custom"
    pretrained_name_or_path: str | None = None
    freeze: bool = False
    strict: bool = True
    download_root: str | None = None
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4


@dataclass(frozen=True)
class InstanceHeadConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class TopKConfig:
    k: int = 1


@dataclass(frozen=True)
class AttentionConfig:
    hidden_dim: int = 128
    dropout: float = 0.0
    gated: bool = True


@dataclass(frozen=True)
class TemperatureConfig:
    temperature: float = 1.0


@dataclass(frozen=True)
class NoisyOrConfig:
    clamp_eps: float = 1e-6


@dataclass(frozen=True)
class MILConfig:
    aggregator: Literal[
        "max",
        "mean",
        "topk",
        "attention",
        "logsumexp",
        "softmax_weighted",
        "noisy_or",
    ]
    return_instance_scores: bool = True
    topk: TopKConfig = field(default_factory=TopKConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    logsumexp: TemperatureConfig = field(default_factory=TemperatureConfig)
    softmax_weighted: TemperatureConfig = field(default_factory=TemperatureConfig)
    noisy_or: NoisyOrConfig = field(default_factory=NoisyOrConfig)


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderConfig
    instance_head: InstanceHeadConfig
    mil: MILConfig


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    type: Literal["bce"] = "bce"
    pos_weight: float | None = None


@dataclass(frozen=True)
class SamplerConfig:
    weighted_random: bool = False


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    top_k: int
    warmup_ratio: float
    max_grad_norm: float
    optimizer: OptimizerConfig
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass(frozen=True)
class AnalysisOutputConfig:
    save_segment_scores: bool = True
    save_attention_weights: bool = True
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
    def _validate_encoder(cfg: EncoderConfig) -> None:
        if cfg.type != "whisper":
            raise ValueError("Only model.encoder.type='whisper' is supported")
        if cfg.n_audio_state <= 0:
            raise ValueError("model.encoder.n_audio_state must be greater than zero")
        if cfg.n_audio_head <= 0:
            raise ValueError("model.encoder.n_audio_head must be greater than zero")
        if cfg.n_audio_layer <= 0:
            raise ValueError("model.encoder.n_audio_layer must be greater than zero")

    @staticmethod
    def _validate_instance_head(cfg: InstanceHeadConfig) -> None:
        if cfg.hidden_dim <= 0:
            raise ValueError("model.instance_head.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.instance_head.dropout must be within [0, 1)")

    @staticmethod
    def _validate_mil(cfg: MILConfig) -> None:
        if cfg.aggregator == "topk" and cfg.topk.k <= 0:
            raise ValueError("model.mil.topk.k must be greater than zero")
        if cfg.attention.hidden_dim <= 0:
            raise ValueError("model.mil.attention.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.attention.dropout < 1.0):
            raise ValueError("model.mil.attention.dropout must be within [0, 1)")
        if cfg.logsumexp.temperature <= 0:
            raise ValueError(
                "model.mil.logsumexp.temperature must be greater than zero"
            )
        if cfg.softmax_weighted.temperature <= 0:
            raise ValueError(
                "model.mil.softmax_weighted.temperature must be greater than zero"
            )
        if not (0.0 < cfg.noisy_or.clamp_eps < 0.5):
            raise ValueError("model.mil.noisy_or.clamp_eps must be within (0, 0.5)")

    @staticmethod
    def _validate_model(cfg: ModelConfig) -> None:
        JsonConfigLoader._validate_encoder(cfg.encoder)
        JsonConfigLoader._validate_instance_head(cfg.instance_head)
        JsonConfigLoader._validate_mil(cfg.mil)

    @staticmethod
    def _validate_train(cfg: TrainConfig) -> None:
        if cfg.epochs <= 0:
            raise ValueError("train.epochs must be greater than zero")
        if cfg.top_k <= 0:
            raise ValueError("train.top_k must be greater than zero")
        if not (0.0 <= cfg.warmup_ratio <= 1.0):
            raise ValueError("train.warmup_ratio must be within [0, 1]")
        if cfg.max_grad_norm <= 0:
            raise ValueError("train.max_grad_norm must be greater than zero")
        if cfg.optimizer.lr <= 0:
            raise ValueError("train.optimizer.lr must be greater than zero")
        if cfg.optimizer.weight_decay < 0:
            raise ValueError("train.optimizer.weight_decay must be non-negative")
        if cfg.loss.type != "bce":
            raise ValueError("Only train.loss.type='bce' is supported")
        if cfg.loss.pos_weight is not None and cfg.loss.pos_weight <= 0:
            raise ValueError("train.loss.pos_weight must be greater than zero")

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
        kwargs["encoder"] = EncoderConfig(**dict(raw["encoder"]))
        kwargs["instance_head"] = InstanceHeadConfig(**dict(raw["instance_head"]))
        mil = dict(raw["mil"])
        mil["topk"] = TopKConfig(**mil.get("topk", {}))
        mil["attention"] = AttentionConfig(**mil.get("attention", {}))
        mil["logsumexp"] = TemperatureConfig(**mil.get("logsumexp", {}))
        mil["softmax_weighted"] = TemperatureConfig(**mil.get("softmax_weighted", {}))
        mil["noisy_or"] = NoisyOrConfig(**mil.get("noisy_or", {}))
        kwargs["mil"] = MILConfig(**mil)
        cfg = ModelConfig(**kwargs)
        JsonConfigLoader._validate_model(cfg)
        return cfg

    @staticmethod
    def _parse_train(raw: Mapping[str, Any]) -> TrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = OptimizerConfig(**dict(raw["optimizer"]))
        kwargs["loss"] = LossConfig(**dict(raw.get("loss", {})))
        kwargs["sampler"] = SamplerConfig(**dict(raw.get("sampler", {})))
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
