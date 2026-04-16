from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig

DEFAULT_AST_PRETRAINED_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    mode: Literal["recording_mil"]
    seed: int
    device: str
    output_dir: str


@dataclass(frozen=True)
class SplitConfig:
    roots: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataSplitsConfig:
    train: SplitConfig = field(default_factory=SplitConfig)
    val: SplitConfig = field(default_factory=SplitConfig)
    eval: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True)
class MetadataConfig:
    label_to_index: Mapping[str, int]
    splits: DataSplitsConfig = field(default_factory=DataSplitsConfig)


@dataclass(frozen=True)
class BandPassConfig:
    enabled: bool = False
    low_hz: float | None = None
    high_hz: float | None = None
    q: float = 0.707


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int


@dataclass(frozen=True)
class InstanceConfig:
    window_sec: float = 2.0
    hop_sec: float = 1.0
    tail_policy: Literal["cover_end"] = "cover_end"

    def window_samples(self, sample_rate: int) -> int:
        return max(1, int(round(self.window_sec * sample_rate)))

    def hop_samples(self, sample_rate: int) -> int:
        return max(1, int(round(self.hop_sec * sample_rate)))


@dataclass(frozen=True)
class AstFbankConfig:
    num_mel_bins: int = 128
    max_length: int = 1024
    do_normalize: bool = True
    mean: float = -4.2677393
    std: float = 4.5689974


@dataclass(frozen=True)
class FeatureConfig:
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)
    ast_fbank: AstFbankConfig = field(default_factory=AstFbankConfig)


@dataclass(frozen=True)
class DataLoaderConfig:
    num_workers: int = 0


@dataclass(frozen=True)
class DataConfig:
    metadata: MetadataConfig
    audio: AudioConfig
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    @property
    def label_to_index(self) -> Mapping[str, int]:
        return self.metadata.label_to_index

    @property
    def num_classes(self) -> int:
        return len(self.metadata.label_to_index)

    @property
    def num_workers(self) -> int:
        return self.loader.num_workers

    def roots_for_split(self, split: str) -> list[str]:
        split_map = {
            "train": self.metadata.splits.train.roots,
            "val": self.metadata.splits.val.roots,
            "eval": self.metadata.splits.eval.roots,
        }
        if split not in split_map:
            raise ValueError(f"Unsupported split: {split}")
        return list(split_map[split])


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
    pooling: Literal["cls", "mean"] = "cls"
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: AstArchitectureConfig = field(default_factory=AstArchitectureConfig)


@dataclass(frozen=True)
class InstanceHeadConfig:
    projection_dim: int | None = None
    dropout: float = 0.0
    normalize: bool = False


@dataclass(frozen=True)
class GatedAttentionConfig:
    attention_dim: int = 128
    dropout: float = 0.0


@dataclass(frozen=True)
class LinearSoftmaxConfig:
    eps: float = 1e-6


@dataclass(frozen=True)
class MilConfig:
    type: Literal["gated_attention", "linear_softmax"] = "gated_attention"
    gated_attention: GatedAttentionConfig = field(default_factory=GatedAttentionConfig)
    linear_softmax: LinearSoftmaxConfig = field(default_factory=LinearSoftmaxConfig)


@dataclass(frozen=True)
class ClassifierConfig:
    type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    encoder: AstEncoderConfig
    instance_head: InstanceHeadConfig = field(default_factory=InstanceHeadConfig)
    mil: MilConfig = field(default_factory=MilConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


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
    batch_size: int
    epochs: int
    top_k: int
    max_grad_norm: float
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class EvalSectionConfig:
    batch_size: int
    checkpoint_path: str | None = None
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )


@dataclass(frozen=True)
class DiagnosticsConfig:
    save_bag_logits: bool = True
    save_bag_probabilities: bool = True
    save_bag_embedding: bool = False
    save_instance_logits: bool = True
    save_instance_probabilities: bool = True
    save_instance_embeddings: bool = False
    save_attention_weights: bool = True
    save_instance_metadata: bool = True
    top_k_instances: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass(frozen=True)
class TrainingRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalSectionConfig
    logging: LoggingConfig


@dataclass(frozen=True)
class EvaluationRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    eval: EvalSectionConfig
    logging: LoggingConfig


@dataclass(frozen=True)
class CvFoldConfig:
    name: str
    train: SplitConfig
    val: SplitConfig
    eval: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True)
class CvConfig:
    folds: list[CvFoldConfig]


@dataclass(frozen=True)
class CvRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalSectionConfig
    logging: LoggingConfig
    cv: CvConfig


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
        if cfg.mode != "recording_mil":
            raise ValueError("experiment.mode must be 'recording_mil' for this branch")
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
                "data.features.bandpass requires low_hz and high_hz when enabled"
            )
        if cfg.q <= 0:
            raise ValueError("data.features.bandpass.q must be greater than zero")
        nyquist = float(sample_rate) / 2.0
        if not (0.0 < cfg.low_hz < cfg.high_hz < nyquist):
            raise ValueError(
                "data.features.bandpass must satisfy "
                "0 < low_hz < high_hz < sample_rate / 2"
            )

    @staticmethod
    def _validate_audio(cfg: AudioConfig) -> None:
        if cfg.sample_rate <= 0:
            raise ValueError("data.audio.sample_rate must be greater than zero")

    @staticmethod
    def _validate_instance(cfg: InstanceConfig) -> None:
        if cfg.window_sec <= 0:
            raise ValueError("data.instance.window_sec must be greater than zero")
        if cfg.hop_sec <= 0:
            raise ValueError("data.instance.hop_sec must be greater than zero")
        if cfg.tail_policy != "cover_end":
            raise ValueError("data.instance.tail_policy must be 'cover_end'")

    @staticmethod
    def _validate_features(cfg: FeatureConfig, sample_rate: int) -> None:
        if cfg.ast_fbank.num_mel_bins <= 0:
            raise ValueError(
                "data.features.ast_fbank.num_mel_bins must be greater than zero"
            )
        if cfg.ast_fbank.max_length <= 0:
            raise ValueError(
                "data.features.ast_fbank.max_length must be greater than zero"
            )
        if cfg.ast_fbank.do_normalize and cfg.ast_fbank.std <= 0:
            raise ValueError(
                "data.features.ast_fbank.std must be greater than zero when normalization is enabled"
            )
        if sample_rate != 16000:
            raise ValueError(
                "data.audio.sample_rate must be 16000 for AST MIL classification"
            )
        JsonConfigLoader._validate_bandpass(cfg.bandpass, sample_rate)

    @staticmethod
    def _validate_data(cfg: DataConfig) -> None:
        JsonConfigLoader._validate_contiguous_labels(cfg.label_to_index)
        JsonConfigLoader._validate_audio(cfg.audio)
        JsonConfigLoader._validate_instance(cfg.instance)
        JsonConfigLoader._validate_features(cfg.features, cfg.audio.sample_rate)
        if cfg.loader.num_workers < 0:
            raise ValueError("data.loader.num_workers must be non-negative")

    @staticmethod
    def _validate_encoder(cfg: AstEncoderConfig) -> None:
        if cfg.type != "ast":
            raise ValueError("Only model.encoder.type='ast' is supported")
        if cfg.pooling not in {"cls", "mean"}:
            raise ValueError("model.encoder.pooling must be one of ['cls', 'mean']")
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
    def _validate_encoder_against_data(
        data_cfg: DataConfig,
        encoder_cfg: AstEncoderConfig,
    ) -> None:
        if encoder_cfg.pretrained_name_or_path is None:
            return
        from transformers import ASTConfig

        pretrained_cfg = ASTConfig.from_pretrained(
            encoder_cfg.pretrained_name_or_path,
            cache_dir=encoder_cfg.cache_dir,
        )
        expected_bins = int(data_cfg.features.ast_fbank.num_mel_bins)
        expected_length = int(data_cfg.features.ast_fbank.max_length)
        actual_bins = int(pretrained_cfg.num_mel_bins)
        actual_length = int(pretrained_cfg.max_length)
        if actual_bins != expected_bins or actual_length != expected_length:
            raise ValueError(
                "Pretrained AST encoder input dims do not match "
                "data.features.ast_fbank.\n"
                f"- encoder: num_mel_bins={actual_bins}, max_length={actual_length}\n"
                f"- data:    num_mel_bins={expected_bins}, max_length={expected_length}\n"
                f"- name_or_path: {encoder_cfg.pretrained_name_or_path}"
            )

    @staticmethod
    def _validate_instance_head(cfg: InstanceHeadConfig) -> None:
        if cfg.projection_dim is not None and cfg.projection_dim <= 0:
            raise ValueError("model.instance_head.projection_dim must be positive")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.instance_head.dropout must be within [0, 1)")

    @staticmethod
    def _validate_mil(cfg: MilConfig) -> None:
        if cfg.type not in {"gated_attention", "linear_softmax"}:
            raise ValueError(
                "model.mil.type must be one of ['gated_attention', 'linear_softmax']"
            )
        if cfg.gated_attention.attention_dim <= 0:
            raise ValueError(
                "model.mil.gated_attention.attention_dim must be greater than zero"
            )
        if not (0.0 <= cfg.gated_attention.dropout < 1.0):
            raise ValueError("model.mil.gated_attention.dropout must be within [0, 1)")
        if cfg.linear_softmax.eps <= 0:
            raise ValueError("model.mil.linear_softmax.eps must be greater than zero")

    @staticmethod
    def _validate_classifier(cfg: ClassifierConfig) -> None:
        if cfg.hidden_dim <= 0:
            raise ValueError("model.classifier.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.classifier.dropout must be within [0, 1)")

    @staticmethod
    def _validate_model(cfg: ModelConfig) -> None:
        JsonConfigLoader._validate_encoder(cfg.encoder)
        JsonConfigLoader._validate_instance_head(cfg.instance_head)
        JsonConfigLoader._validate_mil(cfg.mil)
        JsonConfigLoader._validate_classifier(cfg.classifier)

    @staticmethod
    def _validate_train(cfg: TrainConfig, *, num_classes: int) -> None:
        if cfg.batch_size <= 0:
            raise ValueError("train.batch_size must be greater than zero")
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
                    "Binary AST+MIL runs support only train.loss.type='bce' or 'focal'"
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
                "Multi-class AST+MIL runs require train.loss.type='cross_entropy'"
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
    def _validate_eval(
        cfg: EvalSectionConfig,
        *,
        require_checkpoint: bool,
    ) -> None:
        if cfg.batch_size <= 0:
            raise ValueError("eval.batch_size must be greater than zero")
        if require_checkpoint and not cfg.checkpoint_path:
            raise ValueError("eval.checkpoint_path is required for evaluation")
        if cfg.threshold_optimization.metric not in JsonConfigLoader._THRESHOLD_METRICS:
            raise ValueError(
                "eval.threshold_optimization.metric must be one of "
                f"{sorted(JsonConfigLoader._THRESHOLD_METRICS)}"
            )

    @staticmethod
    def _validate_logging(cfg: LoggingConfig) -> None:
        if cfg.diagnostics.top_k_instances < 0:
            raise ValueError("logging.diagnostics.top_k_instances must be >= 0")

    @staticmethod
    def _parse_experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
        cfg = ExperimentConfig(**dict(raw))
        JsonConfigLoader._validate_experiment(cfg)
        return cfg

    @staticmethod
    def _parse_split(raw: Mapping[str, Any] | None) -> SplitConfig:
        return SplitConfig(**dict(raw or {}))

    @staticmethod
    def _parse_data(raw: Mapping[str, Any]) -> DataConfig:
        kwargs = dict(raw)
        metadata = dict(raw["metadata"])
        splits = dict(metadata.get("splits", {}))
        metadata["splits"] = DataSplitsConfig(
            train=JsonConfigLoader._parse_split(splits.get("train")),
            val=JsonConfigLoader._parse_split(splits.get("val")),
            eval=JsonConfigLoader._parse_split(splits.get("eval")),
        )
        kwargs["metadata"] = MetadataConfig(**metadata)
        kwargs["audio"] = AudioConfig(**dict(raw["audio"]))
        kwargs["instance"] = InstanceConfig(**dict(raw.get("instance", {})))
        features = dict(raw.get("features", {}))
        if "ast_fbank" not in features:
            raise ValueError("data.features.ast_fbank is required")
        features["bandpass"] = BandPassConfig(**features.get("bandpass", {}))
        features["ast_fbank"] = AstFbankConfig(**dict(features["ast_fbank"]))
        kwargs["features"] = FeatureConfig(**features)
        kwargs["loader"] = DataLoaderConfig(**dict(raw.get("loader", {})))
        cfg = DataConfig(**kwargs)
        JsonConfigLoader._validate_data(cfg)
        return cfg

    @staticmethod
    def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
        kwargs = dict(raw)
        encoder = dict(raw["encoder"])
        encoder["adaptation"] = EncoderAdaptationConfig(
            **dict(encoder.get("adaptation", {}))
        )
        encoder["architecture"] = AstArchitectureConfig(
            **dict(encoder.get("architecture", {}))
        )
        kwargs["encoder"] = AstEncoderConfig(**encoder)
        kwargs["instance_head"] = InstanceHeadConfig(
            **dict(raw.get("instance_head", {}))
        )
        mil = dict(raw.get("mil", {}))
        kwargs["mil"] = MilConfig(
            type=mil.get("type", "gated_attention"),
            gated_attention=GatedAttentionConfig(
                **dict(mil.get("gated_attention", {}))
            ),
            linear_softmax=LinearSoftmaxConfig(**dict(mil.get("linear_softmax", {}))),
        )
        kwargs["classifier"] = ClassifierConfig(**dict(raw.get("classifier", {})))
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
    def _parse_eval(raw: Mapping[str, Any]) -> EvalSectionConfig:
        kwargs = dict(raw)
        kwargs["threshold_optimization"] = ThresholdOptimizationConfig(
            **dict(raw.get("threshold_optimization", {}))
        )
        return EvalSectionConfig(**kwargs)

    @staticmethod
    def _parse_logging(raw: Mapping[str, Any] | None) -> LoggingConfig:
        if raw is None:
            return LoggingConfig()
        kwargs = dict(raw)
        kwargs["diagnostics"] = DiagnosticsConfig(**dict(raw.get("diagnostics", {})))
        cfg = LoggingConfig(**kwargs)
        JsonConfigLoader._validate_logging(cfg)
        return cfg

    @staticmethod
    def _parse_cv(raw: Mapping[str, Any]) -> CvConfig:
        folds: list[CvFoldConfig] = []
        for item in raw.get("folds", []):
            fold_raw = dict(item)
            folds.append(
                CvFoldConfig(
                    name=str(fold_raw["name"]),
                    train=JsonConfigLoader._parse_split(fold_raw.get("train")),
                    val=JsonConfigLoader._parse_split(fold_raw.get("val")),
                    eval=JsonConfigLoader._parse_split(fold_raw.get("eval")),
                )
            )
        if not folds:
            raise ValueError("cv.folds must contain at least one fold")
        return CvConfig(folds=folds)

    @staticmethod
    def load_training(path: str | Path) -> TrainingRunConfig:
        raw = JsonConfigLoader.load_json(path)
        cfg = TrainingRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            model=JsonConfigLoader._parse_model(raw["model"]),
            train=JsonConfigLoader._parse_train(raw["train"]),
            eval=JsonConfigLoader._parse_eval(raw["eval"]),
            logging=JsonConfigLoader._parse_logging(raw.get("logging")),
        )
        JsonConfigLoader._validate_encoder_against_data(cfg.data, cfg.model.encoder)
        JsonConfigLoader._validate_train(cfg.train, num_classes=cfg.data.num_classes)
        JsonConfigLoader._validate_eval(cfg.eval, require_checkpoint=False)
        return cfg

    @staticmethod
    def load_eval(path: str | Path) -> EvaluationRunConfig:
        raw = JsonConfigLoader.load_json(path)
        cfg = EvaluationRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            eval=JsonConfigLoader._parse_eval(raw["eval"]),
            logging=JsonConfigLoader._parse_logging(raw.get("logging")),
        )
        JsonConfigLoader._validate_eval(cfg.eval, require_checkpoint=True)
        return cfg

    @staticmethod
    def load_cv(path: str | Path) -> CvRunConfig:
        raw = JsonConfigLoader.load_json(path)
        cfg = CvRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=JsonConfigLoader._parse_data(raw["data"]),
            model=JsonConfigLoader._parse_model(raw["model"]),
            train=JsonConfigLoader._parse_train(raw["train"]),
            eval=JsonConfigLoader._parse_eval(raw["eval"]),
            logging=JsonConfigLoader._parse_logging(raw.get("logging")),
            cv=JsonConfigLoader._parse_cv(raw["cv"]),
        )
        JsonConfigLoader._validate_encoder_against_data(cfg.data, cfg.model.encoder)
        JsonConfigLoader._validate_train(cfg.train, num_classes=cfg.data.num_classes)
        JsonConfigLoader._validate_eval(cfg.eval, require_checkpoint=False)
        return cfg
