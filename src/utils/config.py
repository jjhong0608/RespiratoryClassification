from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig
from src.models.model import (
    AstFeatureDims,
    ClassifierConfig,
    EncoderAdaptationConfig,
    MultiScaleRdtArchitectureConfig,
    PatchBranchConfig,
    RdtConfig,
    compute_token_count,
    compute_token_grid,
)


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
class PreprocessingConfig:
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass: BandPassConfig = field(default_factory=BandPassConfig)
    ast_fbank: AstFbankConfig = field(default_factory=AstFbankConfig)


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
class ModelEncoderConfig:
    type: Literal["multiscale_rdt_ast"] = "multiscale_rdt_ast"
    adaptation: EncoderAdaptationConfig = field(default_factory=EncoderAdaptationConfig)
    architecture: MultiScaleRdtArchitectureConfig = field(
        default_factory=MultiScaleRdtArchitectureConfig
    )


@dataclass(frozen=True)
class ModelConfig:
    encoder: ModelEncoderConfig
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
class BranchAuxiliaryLossConfig:
    enabled: bool = False
    weight: float = 0.3
    aggregation: Literal["mean"] = "mean"


@dataclass(frozen=True)
class LossConfig:
    type: Literal["bce", "focal", "cross_entropy"] = "bce"
    auto_pos_weight: bool = False
    pos_weight: float | None = None
    gamma: float = 2.0
    branch_auxiliary: BranchAuxiliaryLossConfig = field(
        default_factory=BranchAuxiliaryLossConfig
    )


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
class TrainingInitializationConfig:
    checkpoint_path: str | None = None
    load_model_state: bool = True
    strict: bool = False
    load_optimizer_state: bool = False


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
    initialization: TrainingInitializationConfig = field(
        default_factory=TrainingInitializationConfig
    )


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
        if sample_rate != 16000:
            raise ValueError(
                "data.audio.sample_rate must be 16000 for AST clip classification"
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
    def _validate_encoder(cfg: ModelEncoderConfig, data_cfg: DataConfig) -> None:
        if cfg.type != "multiscale_rdt_ast":
            raise ValueError(
                "Only model.encoder.type='multiscale_rdt_ast' is supported"
            )
        if cfg.adaptation.mode == "partial":
            raise ValueError(
                "model.encoder.adaptation.mode='partial' is not supported; "
                "use 'full' or 'frozen'"
            )
        if cfg.adaptation.mode not in {"frozen", "full", "partial"}:
            raise ValueError(
                "model.encoder.adaptation.mode must be one of "
                "['frozen', 'full', 'partial']"
            )
        if cfg.adaptation.num_layers != 0:
            raise ValueError("model.encoder.adaptation.num_layers must be 0")

        architecture = cfg.architecture
        if architecture.hidden_size <= 0:
            raise ValueError(
                "model.encoder.architecture.hidden_size must be greater than zero"
            )
        if architecture.num_attention_heads <= 0:
            raise ValueError(
                "model.encoder.architecture.num_attention_heads must be greater than zero"
            )
        if architecture.hidden_size % architecture.num_attention_heads != 0:
            raise ValueError(
                "model.encoder.architecture.hidden_size must be divisible by "
                "model.encoder.architecture.num_attention_heads"
            )
        if architecture.mlp_ratio <= 0:
            raise ValueError(
                "model.encoder.architecture.mlp_ratio must be greater than zero"
            )
        if not (0.0 <= architecture.hidden_dropout_prob < 1.0):
            raise ValueError(
                "model.encoder.architecture.hidden_dropout_prob must be within [0, 1)"
            )
        if not (0.0 <= architecture.attention_probs_dropout_prob < 1.0):
            raise ValueError(
                "model.encoder.architecture.attention_probs_dropout_prob must be within [0, 1)"
            )
        if architecture.shared_stem_depth <= 0:
            raise ValueError(
                "model.encoder.architecture.shared_stem_depth must be greater than zero"
            )
        if architecture.adapter_depth <= 0:
            raise ValueError(
                "model.encoder.architecture.adapter_depth must be greater than zero"
            )
        if not architecture.patch_branches:
            raise ValueError(
                "model.encoder.architecture.patch_branches must not be empty"
            )
        if not isinstance(architecture.rdt.enabled, bool):
            raise ValueError("model.encoder.architecture.rdt.enabled must be a boolean")
        if architecture.rdt.enabled and architecture.rdt.steps <= 0:
            raise ValueError(
                "model.encoder.architecture.rdt.steps must be greater than zero when rdt.enabled is true"
            )
        if architecture.rdt.top_tokens_per_branch <= 0:
            raise ValueError(
                "model.encoder.architecture.rdt.top_tokens_per_branch must be greater than zero"
            )
        if architecture.rdt.layerscale_init <= 0:
            raise ValueError(
                "model.encoder.architecture.rdt.layerscale_init must be greater than zero"
            )

        feature_dims = AstFeatureDims(
            num_mel_bins=data_cfg.preprocessing.ast_fbank.num_mel_bins,
            max_length=data_cfg.preprocessing.ast_fbank.max_length,
        )
        for branch_index, branch in enumerate(architecture.patch_branches):
            patch_t, patch_f = branch.patch_size
            stride_t, stride_f = branch.stride
            if patch_t <= 0 or patch_f <= 0:
                raise ValueError(
                    "model.encoder.architecture.patch_branches.patch_size values "
                    "must be greater than zero"
                )
            if stride_t <= 0 or stride_f <= 0:
                raise ValueError(
                    "model.encoder.architecture.patch_branches.stride values must "
                    "be greater than zero"
                )
            if patch_t > feature_dims.max_length or patch_f > feature_dims.num_mel_bins:
                raise ValueError(
                    "model.encoder.architecture.patch_branches contains a patch "
                    "that does not fit within data.preprocessing.ast_fbank"
                )
            token_count = compute_token_count(
                feature_dims=feature_dims,
                patch_branch=branch,
            )
            if token_count <= 0:
                raise ValueError(
                    "model.encoder.architecture.patch_branches must yield a positive "
                    f"token count; branch_index={branch_index}"
                )
            time_steps, _ = compute_token_grid(
                feature_dims=feature_dims,
                patch_branch=branch,
            )
            if architecture.rdt.top_tokens_per_branch > time_steps:
                raise ValueError(
                    "model.encoder.architecture.rdt.top_tokens_per_branch exceeds "
                    f"branch time length for branch_index={branch_index}"
                )

    @staticmethod
    def _validate_classifier(cfg: ClassifierConfig) -> None:
        if cfg.hidden_dim <= 0:
            raise ValueError("model.classifier.hidden_dim must be greater than zero")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("model.classifier.dropout must be within [0, 1)")
        if cfg.pooling != "latent_mean":
            raise ValueError(
                "model.classifier.pooling must be 'latent_mean' for this branch"
            )

    @staticmethod
    def _validate_model(cfg: ModelConfig, data_cfg: DataConfig) -> None:
        JsonConfigLoader._validate_encoder(cfg.encoder, data_cfg)
        JsonConfigLoader._validate_classifier(cfg.classifier)

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
        if cfg.initialization.checkpoint_path is not None and not isinstance(
            cfg.initialization.checkpoint_path, str
        ):
            raise TypeError(
                "train.initialization.checkpoint_path must be a string or null"
            )
        if not isinstance(cfg.initialization.load_model_state, bool):
            raise TypeError("train.initialization.load_model_state must be a boolean")
        if not isinstance(cfg.initialization.strict, bool):
            raise TypeError("train.initialization.strict must be a boolean")
        if not isinstance(cfg.initialization.load_optimizer_state, bool):
            raise TypeError(
                "train.initialization.load_optimizer_state must be a boolean"
            )
        if not isinstance(cfg.loss.branch_auxiliary.enabled, bool):
            raise ValueError("train.loss.branch_auxiliary.enabled must be a boolean")
        if cfg.loss.branch_auxiliary.aggregation != "mean":
            raise ValueError("train.loss.branch_auxiliary.aggregation must be 'mean'")
        if cfg.loss.branch_auxiliary.enabled and cfg.loss.branch_auxiliary.weight <= 0:
            raise ValueError(
                "train.loss.branch_auxiliary.weight must be greater than zero when branch auxiliary loss is enabled"
            )
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
        if "ast_fbank" not in preprocessing:
            raise ValueError("data.preprocessing.ast_fbank is required")
        preprocessing["bandpass"] = BandPassConfig(**preprocessing.get("bandpass", {}))
        preprocessing["ast_fbank"] = AstFbankConfig(**dict(preprocessing["ast_fbank"]))
        kwargs["preprocessing"] = PreprocessingConfig(**preprocessing)
        cfg = DataConfig(**kwargs)
        JsonConfigLoader._validate_data(cfg)
        return cfg

    @staticmethod
    def _coerce_pair(
        values: Sequence[object],
        *,
        field_name: str,
    ) -> tuple[int, int]:
        if len(values) != 2:
            raise ValueError(f"{field_name} must contain exactly two integers")
        first, second = values
        if not isinstance(first, int) or not isinstance(second, int):
            raise TypeError(f"{field_name} must contain integers")
        return first, second

    @staticmethod
    def _parse_patch_branch(raw: Mapping[str, Any]) -> PatchBranchConfig:
        patch_size_raw = raw.get("patch_size")
        stride_raw = raw.get("stride")
        if not isinstance(patch_size_raw, Sequence) or isinstance(
            patch_size_raw, (str, bytes)
        ):
            raise TypeError("patch_size must be a 2-item list")
        if not isinstance(stride_raw, Sequence) or isinstance(stride_raw, (str, bytes)):
            raise TypeError("stride must be a 2-item list")
        return PatchBranchConfig(
            patch_size=JsonConfigLoader._coerce_pair(
                patch_size_raw,
                field_name="patch_size",
            ),
            stride=JsonConfigLoader._coerce_pair(
                stride_raw,
                field_name="stride",
            ),
        )

    @staticmethod
    def _parse_model(raw: Mapping[str, Any], data_cfg: DataConfig) -> ModelConfig:
        kwargs = dict(raw)
        encoder = dict(raw["encoder"])
        architecture = dict(encoder.get("architecture", {}))
        if "latent_query_count" in architecture:
            raise ValueError(
                "latent_query_count is deprecated in the event-MIL architecture. "
                "Use model.encoder.architecture.rdt instead."
            )
        if "summary_tokens_per_scale" in architecture:
            raise ValueError(
                "summary_tokens_per_scale is deprecated in the event-MIL architecture. "
                "Use model.encoder.architecture.rdt.top_tokens_per_branch instead."
            )
        if "rdt_steps" in architecture:
            raise ValueError(
                "Flat rdt_steps is deprecated in the event-MIL architecture. "
                "Use model.encoder.architecture.rdt.steps instead."
            )
        patch_branches_raw = architecture.get("patch_branches")
        if patch_branches_raw is not None:
            if not isinstance(patch_branches_raw, Sequence) or isinstance(
                patch_branches_raw, (str, bytes)
            ):
                raise TypeError(
                    "model.encoder.architecture.patch_branches must be a list"
                )
            architecture["patch_branches"] = tuple(
                JsonConfigLoader._parse_patch_branch(dict(branch_raw))
                for branch_raw in patch_branches_raw
            )
        architecture["rdt"] = RdtConfig(**dict(architecture.get("rdt", {})))
        encoder["adaptation"] = EncoderAdaptationConfig(
            **dict(encoder.get("adaptation", {}))
        )
        encoder["architecture"] = MultiScaleRdtArchitectureConfig(**architecture)
        kwargs["encoder"] = ModelEncoderConfig(**encoder)
        kwargs["classifier"] = ClassifierConfig(**dict(raw["classifier"]))
        cfg = ModelConfig(**kwargs)
        JsonConfigLoader._validate_model(cfg, data_cfg)
        return cfg

    @staticmethod
    def _parse_train(raw: Mapping[str, Any]) -> TrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = OptimizerConfig(**dict(raw["optimizer"]))
        kwargs["scheduler"] = SchedulerConfig(**dict(raw.get("scheduler", {})))
        loss = dict(raw.get("loss", {}))
        loss["branch_auxiliary"] = BranchAuxiliaryLossConfig(
            **dict(loss.get("branch_auxiliary", {}))
        )
        kwargs["loss"] = LossConfig(**loss)
        kwargs["sampler"] = SamplerConfig(**dict(raw.get("sampler", {})))
        kwargs["early_stopping"] = EarlyStoppingConfig(
            **dict(raw.get("early_stopping", {}))
        )
        kwargs["initialization"] = TrainingInitializationConfig(
            **dict(raw.get("initialization", {}))
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
        data_cfg = JsonConfigLoader._parse_data(raw["data"])
        model_cfg = JsonConfigLoader._parse_model(raw["model"], data_cfg)
        cfg = TrainingRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=data_cfg,
            model=model_cfg,
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
        )
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
        )
        return cfg

    @staticmethod
    def load_eval(path: str | Path) -> EvalConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_data(raw["data"])
        return EvalConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            data=data_cfg,
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            threshold_optimization=JsonConfigLoader._parse_threshold_optimization(
                raw.get("threshold_optimization")
            ),
        )

    @staticmethod
    def _parse_fold(raw: Mapping[str, Any]) -> CvFoldConfig:
        cfg = CvFoldConfig(**dict(raw))
        if not cfg.name:
            raise ValueError("fold.name must not be empty")
        return cfg

    @staticmethod
    def load_cv(path: str | Path) -> CvRunConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_data(raw["data"])
        model_cfg = JsonConfigLoader._parse_model(raw["model"], data_cfg)
        cfg = CvRunConfig(
            experiment=JsonConfigLoader._parse_experiment(raw["experiment"]),
            data=data_cfg,
            model=model_cfg,
            train=JsonConfigLoader._parse_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            folds=[JsonConfigLoader._parse_fold(item) for item in raw["folds"]],
        )
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
        )
        return cfg
