from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.evaluation.thresholds import ThresholdOptimizationConfig
from src.models.model import (
    AstFeatureDims,
    BranchEventDropoutConfig,
    ClassifierBiasInitConfig,
    ClassifierConfig,
    EncoderAdaptationConfig,
    EvidencePoolingConfig,
    MilConfig,
    MultiScaleRdtArchitectureConfig,
    PatchBranchConfig,
    RdtConfig,
    SelectedEvidenceDropoutConfig,
    SslDecoderConfig,
    SslLossConfig,
    SslMaskingConfig,
    SslVisualizationConfig,
    TokenAugmentationConfig,
    compute_token_count,
    compute_token_grid,
)
from src.models.ssl import MaskedFbankSSLConfig


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
class RandomGainConfig:
    enabled: bool = False
    probability: float = 0.5
    min_db: float = -3.0
    max_db: float = 3.0


@dataclass(frozen=True)
class AdditiveNoiseConfig:
    enabled: bool = False
    probability: float = 0.3
    snr_db_min: float = 15.0
    snr_db_max: float = 30.0


@dataclass(frozen=True)
class TimeShiftConfig:
    enabled: bool = False
    probability: float = 0.5
    max_shift_fraction: float = 0.05
    mode: Literal["zero_pad", "roll"] = "zero_pad"


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    enabled: bool = False
    probability: float = 1.0
    gain: RandomGainConfig = field(default_factory=RandomGainConfig)
    noise: AdditiveNoiseConfig = field(default_factory=AdditiveNoiseConfig)
    time_shift: TimeShiftConfig = field(default_factory=TimeShiftConfig)


@dataclass(frozen=True)
class MaskConfig:
    enabled: bool = False
    num_masks: int = 1
    max_width: int = 32


@dataclass(frozen=True)
class FbankAugmentationConfig:
    enabled: bool = False
    probability: float = 0.5
    time_mask: MaskConfig = field(default_factory=MaskConfig)
    freq_mask: MaskConfig = field(default_factory=MaskConfig)
    mask_value: float = 0.0


@dataclass(frozen=True)
class AugmentationPolicyChoiceConfig:
    name: Literal["none", "waveform", "fbank", "both_light"]
    probability: float


@dataclass(frozen=True)
class AugmentationPolicyConfig:
    type: Literal["independent", "one_of"] = "independent"
    choices: tuple[AugmentationPolicyChoiceConfig, ...] = ()


@dataclass(frozen=True)
class DataAugmentationConfig:
    enabled: bool = False
    policy: AugmentationPolicyConfig = field(default_factory=AugmentationPolicyConfig)
    waveform: WaveformAugmentationConfig = field(
        default_factory=WaveformAugmentationConfig
    )
    fbank: FbankAugmentationConfig = field(default_factory=FbankAugmentationConfig)


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
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)


@dataclass(frozen=True)
class Fsd50kDataConfig:
    dataset: Literal["fsd50k"]
    mode: Literal["ssl", "supervised"]
    root: str
    dev_audio_dir: str
    eval_audio_dir: str
    ground_truth_dir: str
    vocabulary_csv: str
    dev_csv: str
    eval_csv: str
    val_ratio: float = 0.1
    split_seed: int = 42
    num_workers: int = 0
    audio: AudioConfig = field(
        default_factory=lambda: AudioConfig(sample_rate=16000, clip_duration_sec=30.0)
    )
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)


@dataclass(frozen=True)
class ModelEncoderConfig:
    type: Literal["multiscale_rdt_ast"] = "multiscale_rdt_ast"
    init_from: str | None = None
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
    weights: tuple[float, ...] | None = None
    aggregation: Literal["mean"] = "mean"


@dataclass(frozen=True)
class AttentionEntropyLossConfig:
    enabled: bool = False
    weight: float = 0.0


@dataclass(frozen=True)
class ClassWeightingConfig:
    enabled: bool = False
    type: Literal["sqrt_inverse_frequency"] = "sqrt_inverse_frequency"
    normalize: Literal["mean_one"] = "mean_one"
    source: Literal["train"] = "train"


@dataclass(frozen=True)
class LabelSmoothingConfig:
    enabled: bool = False
    value: float = 0.0


@dataclass(frozen=True)
class BranchBinaryPosWeightConfig:
    enabled: bool = False
    type: Literal["sqrt_normal_over_abnormal"] = "sqrt_normal_over_abnormal"
    source: Literal["train"] = "train"


@dataclass(frozen=True)
class BranchBinaryAuxiliaryScheduleConfig:
    enabled: bool = False
    type: Literal["none", "cosine_floor"] = "none"
    max_weight: float = 0.4
    min_weight: float = 0.1
    total_epochs: int | None = None


@dataclass(frozen=True)
class BranchBinaryAuxiliaryMonitorConfig:
    loss_weight: float = 0.3


@dataclass(frozen=True)
class BranchBinaryAuxiliaryLossConfig:
    enabled: bool = False
    weight: float = 0.3
    label_to_index: Mapping[str, int] = field(default_factory=dict)
    pos_weight: BranchBinaryPosWeightConfig = field(
        default_factory=BranchBinaryPosWeightConfig
    )
    schedule: BranchBinaryAuxiliaryScheduleConfig = field(
        default_factory=BranchBinaryAuxiliaryScheduleConfig
    )
    monitor: BranchBinaryAuxiliaryMonitorConfig = field(
        default_factory=BranchBinaryAuxiliaryMonitorConfig
    )
    aggregation: Literal["mean"] = "mean"


@dataclass(frozen=True)
class LossConfig:
    type: Literal["bce", "focal", "cross_entropy"] = "bce"
    auto_pos_weight: bool = False
    pos_weight: float | None = None
    gamma: float = 2.0
    class_weighting: ClassWeightingConfig = field(default_factory=ClassWeightingConfig)
    label_smoothing: LabelSmoothingConfig = field(default_factory=LabelSmoothingConfig)
    branch_auxiliary: BranchAuxiliaryLossConfig = field(
        default_factory=BranchAuxiliaryLossConfig
    )
    branch_binary_auxiliary: BranchBinaryAuxiliaryLossConfig = field(
        default_factory=BranchBinaryAuxiliaryLossConfig
    )
    attention_entropy: AttentionEntropyLossConfig = field(
        default_factory=AttentionEntropyLossConfig
    )


@dataclass(frozen=True)
class SamplerConfig:
    weighted_random: bool = False
    enabled: bool = False
    type: Literal["none", "sqrt_inverse_class"] = "none"
    replacement: bool = True
    num_samples: Literal["dataset_size"] | int = "dataset_size"
    source: Literal["train"] = "train"


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
class ValidationLossConfig:
    use_train_loss_config: bool = True
    class_weight_source: Literal["train"] = "train"
    binary_pos_weight_source: Literal["train"] = "train"


@dataclass(frozen=True)
class ValidationConfig:
    loss: ValidationLossConfig = field(default_factory=ValidationLossConfig)


@dataclass(frozen=True)
class CheckpointMonitorConfig:
    name: Literal[
        "val_macro_f1",
        "val_macro_recall",
        "val_loss",
        "val_loss_total_monitor",
        "val_ssl_loss",
        "val_macro_AP",
        "val_micro_AP",
        "last",
    ]
    mode: Literal["max", "min", "latest", "last"]
    top_k: int = 3
    filename_prefix: str | None = None


@dataclass(frozen=True)
class CheckpointingConfig:
    monitors: tuple[CheckpointMonitorConfig, ...] = ()


@dataclass(frozen=True)
class Fsd50kSchedulerConfig:
    type: Literal["linear_warmup_cosine"] = "linear_warmup_cosine"
    warmup_ratio: float = 0.05


@dataclass(frozen=True)
class Fsd50kSslOptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True)
class Fsd50kSslTrainConfig:
    epochs: int = 200
    batch_size: int = 32
    optimizer: Fsd50kSslOptimizerConfig = field(
        default_factory=Fsd50kSslOptimizerConfig
    )
    scheduler: Fsd50kSchedulerConfig = field(default_factory=Fsd50kSchedulerConfig)
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class MultiLabelPosWeightConfig:
    type: Literal["sqrt_negative_over_positive"] = "sqrt_negative_over_positive"
    cap: float = 10.0
    source: Literal["train"] = "train"
    save_to_checkpoint_metadata: bool = True


@dataclass(frozen=True)
class Fsd50kSupervisedLossConfig:
    type: Literal["bce_with_logits"] = "bce_with_logits"
    pos_weight: MultiLabelPosWeightConfig = field(
        default_factory=MultiLabelPosWeightConfig
    )
    branch_auxiliary: BranchAuxiliaryLossConfig = field(
        default_factory=BranchAuxiliaryLossConfig
    )
    branch_binary_auxiliary: BranchBinaryAuxiliaryLossConfig = field(
        default_factory=BranchBinaryAuxiliaryLossConfig
    )


@dataclass(frozen=True)
class Fsd50kSupervisedOptimizerConfig:
    encoder_lr: float = 1e-5
    body_lr: float = 3e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True)
class GradientClippingConfig:
    enabled: bool = False
    max_norm: float = 1.0


@dataclass(frozen=True)
class Fsd50kSupervisedTrainConfig:
    epochs: int = 120
    batch_size: int = 32
    optimizer: Fsd50kSupervisedOptimizerConfig = field(
        default_factory=Fsd50kSupervisedOptimizerConfig
    )
    scheduler: Fsd50kSchedulerConfig = field(default_factory=Fsd50kSchedulerConfig)
    gradient_clipping: GradientClippingConfig = field(
        default_factory=GradientClippingConfig
    )
    loss: Fsd50kSupervisedLossConfig = field(default_factory=Fsd50kSupervisedLossConfig)


@dataclass(frozen=True)
class Fsd50kTopKMetricsConfig:
    enabled: bool = True
    ks: tuple[int, ...] = (5, 10)
    save_label_frequency: bool = True
    top_n_frequency: int = 20


@dataclass(frozen=True)
class Fsd50kMetricsConfig:
    primary: Literal["macro_AP"] = "macro_AP"
    secondary: Literal["micro_AP"] = "micro_AP"
    f1_threshold: float = 0.5
    topk: Fsd50kTopKMetricsConfig = field(default_factory=Fsd50kTopKMetricsConfig)
    save_per_class_AP: bool = True
    zero_positive_class_policy: Literal["exclude_with_warning"] = "exclude_with_warning"


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
class TerminalConfig:
    width: int | None = None


@dataclass(frozen=True)
class TrainingRunConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    analysis: AnalysisConfig
    val: ValidationConfig = field(default_factory=ValidationConfig)
    checkpointing: CheckpointingConfig | None = None
    terminal: TerminalConfig = field(default_factory=TerminalConfig)


@dataclass(frozen=True)
class Fsd50kSslRunConfig:
    task: str
    experiment: ExperimentConfig
    data: Fsd50kDataConfig
    model: ModelConfig
    train: Fsd50kSslTrainConfig
    ssl: MaskedFbankSSLConfig
    checkpointing: CheckpointingConfig | None = None
    terminal: TerminalConfig = field(default_factory=TerminalConfig)


@dataclass(frozen=True)
class Fsd50kSupervisedRunConfig:
    task: str
    experiment: ExperimentConfig
    data: Fsd50kDataConfig
    model: ModelConfig
    train: Fsd50kSupervisedTrainConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    metrics: Fsd50kMetricsConfig = field(default_factory=Fsd50kMetricsConfig)
    checkpointing: CheckpointingConfig | None = None
    terminal: TerminalConfig = field(default_factory=TerminalConfig)


@dataclass(frozen=True)
class CnuhTransferConfig:
    checkpoint_path: str
    reset_classifier: bool = True
    freeze_encoder: bool = True
    strict: bool = False
    freeze_modules: tuple[str, ...] = (
        "patch_tokenizers",
        "position_embeddings",
        "scale_embeddings",
        "shared_stem",
        "scale_specific_adapters",
        "frequency_attention_poolers",
    )


@dataclass(frozen=True)
class CnuhTransferOptimizerConfig:
    frozen_encoder_lr: float = 0.0
    body_lr: float = 1e-5
    head_lr: float = 3e-4


@dataclass(frozen=True)
class CnuhTransferTemplateConfig:
    task: Literal["cnuh_4class"]
    transfer: CnuhTransferConfig
    optimizer: CnuhTransferOptimizerConfig


@dataclass(frozen=True)
class EvalConfig:
    experiment: ExperimentConfig
    checkpoint_path: str
    data: DataConfig
    analysis: AnalysisConfig
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )
    terminal: TerminalConfig = field(default_factory=TerminalConfig)


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
    val: ValidationConfig = field(default_factory=ValidationConfig)
    checkpointing: CheckpointingConfig | None = None
    terminal: TerminalConfig = field(default_factory=TerminalConfig)


class JsonConfigLoader:
    _THRESHOLD_METRICS = {"f1", "balanced_accuracy", "youden_j"}
    _EARLY_STOPPING_MONITORS = {"val_loss"}
    _EVIDENCE_SCORE_SOURCES = {
        "attention_weight",
        "attention_logit",
        "instance_logit",
    }
    _EVIDENCE_POOLING_TYPES = {"mean", "branch_gated"}
    _TIME_SHIFT_MODES = {"zero_pad", "roll"}
    _AUGMENTATION_POLICY_TYPES = {"independent", "one_of"}
    _AUGMENTATION_POLICY_CHOICES = {"none", "waveform", "fbank", "both_light"}
    _BRANCH_EVENT_DROPOUT_MODES = {"zero_mask"}
    _SELECTED_EVIDENCE_DROPOUT_MODES = {"zero"}
    _CLASS_WEIGHTING_TYPES = {"sqrt_inverse_frequency"}
    _CLASS_WEIGHTING_NORMALIZERS = {"mean_one"}
    _LOSS_WEIGHT_SOURCES = {"train"}
    _BRANCH_BINARY_POS_WEIGHT_TYPES = {"sqrt_normal_over_abnormal"}
    _BRANCH_BINARY_AUXILIARY_SCHEDULE_TYPES = {"none", "cosine_floor"}
    _SAMPLER_TYPES = {"none", "sqrt_inverse_class"}
    _CLASSIFIER_BIAS_INIT_TYPES = {"none", "prior", "weighted_prior"}
    _CHECKPOINT_MONITORS = {
        "val_macro_f1": {"max"},
        "val_macro_recall": {"max"},
        "val_loss": {"min"},
        "val_loss_total_monitor": {"min"},
        "val_ssl_loss": {"min"},
        "val_macro_AP": {"max"},
        "val_micro_AP": {"max"},
        "last": {"latest", "last"},
    }

    @staticmethod
    def _validate_probability(value: float, *, field_name: str) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{field_name} must be within [0, 1]")

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
    def _validate_mask(cfg: MaskConfig, *, field_name: str) -> None:
        if cfg.num_masks < 0:
            raise ValueError(f"{field_name}.num_masks must be non-negative")
        if cfg.max_width < 0:
            raise ValueError(f"{field_name}.max_width must be non-negative")

    @staticmethod
    def _validate_augmentation_policy(cfg: AugmentationPolicyConfig) -> None:
        if cfg.type not in JsonConfigLoader._AUGMENTATION_POLICY_TYPES:
            raise ValueError(
                "data.augmentation.policy.type must be one of "
                f"{sorted(JsonConfigLoader._AUGMENTATION_POLICY_TYPES)}"
            )
        if cfg.type == "independent":
            return
        if not cfg.choices:
            raise ValueError(
                "data.augmentation.policy.choices must not be empty for one_of policy"
            )
        total_probability = 0.0
        positive_count = 0
        seen_names: set[str] = set()
        for choice in cfg.choices:
            if choice.name not in JsonConfigLoader._AUGMENTATION_POLICY_CHOICES:
                raise ValueError(
                    "data.augmentation.policy.choices.name must be one of "
                    f"{sorted(JsonConfigLoader._AUGMENTATION_POLICY_CHOICES)}"
                )
            if choice.name in seen_names:
                raise ValueError(
                    "data.augmentation.policy.choices must not contain duplicate names"
                )
            seen_names.add(choice.name)
            if choice.probability < 0:
                raise ValueError(
                    "data.augmentation.policy.choices.probability must be non-negative"
                )
            total_probability += float(choice.probability)
            if choice.probability > 0:
                positive_count += 1
        if positive_count == 0:
            raise ValueError(
                "data.augmentation.policy.choices must contain at least one positive probability"
            )
        if abs(total_probability - 1.0) > 1e-6:
            raise ValueError(
                "data.augmentation.policy.choices probabilities must sum to 1.0"
            )

    @staticmethod
    def _validate_augmentation(cfg: DataAugmentationConfig) -> None:
        if not isinstance(cfg.enabled, bool):
            raise ValueError("data.augmentation.enabled must be a boolean")
        if not isinstance(cfg.waveform.enabled, bool):
            raise ValueError("data.augmentation.waveform.enabled must be a boolean")
        if not isinstance(cfg.fbank.enabled, bool):
            raise ValueError("data.augmentation.fbank.enabled must be a boolean")
        JsonConfigLoader._validate_augmentation_policy(cfg.policy)
        JsonConfigLoader._validate_probability(
            cfg.waveform.probability,
            field_name="data.augmentation.waveform.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.waveform.gain.probability,
            field_name="data.augmentation.waveform.gain.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.waveform.noise.probability,
            field_name="data.augmentation.waveform.noise.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.waveform.time_shift.probability,
            field_name="data.augmentation.waveform.time_shift.probability",
        )
        JsonConfigLoader._validate_probability(
            cfg.fbank.probability,
            field_name="data.augmentation.fbank.probability",
        )
        if cfg.waveform.gain.min_db > cfg.waveform.gain.max_db:
            raise ValueError(
                "data.augmentation.waveform.gain.min_db must be less than or equal to max_db"
            )
        if cfg.waveform.noise.snr_db_min <= 0:
            raise ValueError(
                "data.augmentation.waveform.noise.snr_db_min must be greater than zero"
            )
        if cfg.waveform.noise.snr_db_min > cfg.waveform.noise.snr_db_max:
            raise ValueError(
                "data.augmentation.waveform.noise.snr_db_min must be less than or equal to snr_db_max"
            )
        if not (0.0 <= cfg.waveform.time_shift.max_shift_fraction < 1.0):
            raise ValueError(
                "data.augmentation.waveform.time_shift.max_shift_fraction must be within [0, 1)"
            )
        if cfg.waveform.time_shift.mode not in JsonConfigLoader._TIME_SHIFT_MODES:
            raise ValueError(
                "data.augmentation.waveform.time_shift.mode must be one of "
                f"{sorted(JsonConfigLoader._TIME_SHIFT_MODES)}"
            )
        JsonConfigLoader._validate_mask(
            cfg.fbank.time_mask,
            field_name="data.augmentation.fbank.time_mask",
        )
        JsonConfigLoader._validate_mask(
            cfg.fbank.freq_mask,
            field_name="data.augmentation.fbank.freq_mask",
        )

    @staticmethod
    def _validate_data(cfg: DataConfig) -> None:
        JsonConfigLoader._validate_contiguous_labels(cfg.label_to_index)
        JsonConfigLoader._validate_audio(cfg.audio)
        JsonConfigLoader._validate_preprocessing(
            cfg.preprocessing, cfg.audio.sample_rate
        )
        JsonConfigLoader._validate_augmentation(cfg.augmentation)
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
        if (
            architecture.rdt.evidence_score_source
            not in JsonConfigLoader._EVIDENCE_SCORE_SOURCES
        ):
            raise ValueError(
                "model.encoder.architecture.rdt.evidence_score_source must be one of "
                f"{sorted(JsonConfigLoader._EVIDENCE_SCORE_SOURCES)}"
            )
        if architecture.mil.attention_temperature <= 0:
            raise ValueError(
                "model.encoder.architecture.mil.attention_temperature must be greater than zero"
            )
        if architecture.evidence_pooling.type not in (
            JsonConfigLoader._EVIDENCE_POOLING_TYPES
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.type must be one of "
                f"{sorted(JsonConfigLoader._EVIDENCE_POOLING_TYPES)}"
            )
        if architecture.evidence_pooling.temperature <= 0:
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.temperature must be greater than zero"
            )
        if not (0.0 <= architecture.evidence_pooling.dropout < 1.0):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.dropout must be within [0, 1)"
            )
        if (
            architecture.evidence_pooling.gate_hidden_size is not None
            and architecture.evidence_pooling.gate_hidden_size <= 0
        ):
            raise ValueError(
                "model.encoder.architecture.evidence_pooling.gate_hidden_size must be null or greater than zero"
            )
        token_augmentation = architecture.token_augmentation
        if not isinstance(token_augmentation.branch_event_dropout.enabled, bool):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.enabled must be a boolean"
            )
        if not (0.0 <= token_augmentation.branch_event_dropout.probability < 1.0):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.probability must be within [0, 1)"
            )
        if (
            token_augmentation.branch_event_dropout.mode
            not in JsonConfigLoader._BRANCH_EVENT_DROPOUT_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.mode must be 'zero_mask'"
            )
        if token_augmentation.branch_event_dropout.min_keep_tokens < 1:
            raise ValueError(
                "model.encoder.architecture.token_augmentation.branch_event_dropout.min_keep_tokens must be at least 1"
            )
        if not isinstance(token_augmentation.selected_evidence_dropout.enabled, bool):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.enabled must be a boolean"
            )
        if not (0.0 <= token_augmentation.selected_evidence_dropout.probability < 1.0):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.probability must be within [0, 1)"
            )
        if (
            token_augmentation.selected_evidence_dropout.mode
            not in JsonConfigLoader._SELECTED_EVIDENCE_DROPOUT_MODES
        ):
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.mode must be 'zero'"
            )
        if token_augmentation.selected_evidence_dropout.min_keep_per_branch < 1:
            raise ValueError(
                "model.encoder.architecture.token_augmentation.selected_evidence_dropout.min_keep_per_branch must be at least 1"
            )

        feature_dims = AstFeatureDims(
            num_mel_bins=data_cfg.preprocessing.ast_fbank.num_mel_bins,
            max_length=data_cfg.preprocessing.ast_fbank.max_length,
        )
        branch_count = len(architecture.patch_branches)
        excluded_branches = architecture.rdt.exclude_branches_from_evidence
        if len(set(excluded_branches)) != len(excluded_branches):
            raise ValueError(
                "model.encoder.architecture.rdt.exclude_branches_from_evidence "
                "must not contain duplicate branch indices"
            )
        for excluded_branch in excluded_branches:
            if excluded_branch < 0 or excluded_branch >= branch_count:
                raise ValueError(
                    "model.encoder.architecture.rdt.exclude_branches_from_evidence "
                    f"contains invalid branch index {excluded_branch}"
                )
        if len(excluded_branches) >= branch_count:
            raise ValueError(
                "model.encoder.architecture.rdt.exclude_branches_from_evidence "
                "must leave at least one evidence branch"
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
            if (
                branch_index not in excluded_branches
                and architecture.rdt.top_tokens_per_branch > time_steps
            ):
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
        if not isinstance(cfg.bias_init.enabled, bool):
            raise ValueError("model.classifier.bias_init.enabled must be a boolean")
        if cfg.bias_init.type not in JsonConfigLoader._CLASSIFIER_BIAS_INIT_TYPES:
            raise ValueError(
                "model.classifier.bias_init.type must be one of "
                f"{sorted(JsonConfigLoader._CLASSIFIER_BIAS_INIT_TYPES)}"
            )
        if cfg.bias_init.source != "train":
            raise ValueError("model.classifier.bias_init.source must be 'train'")
        if cfg.bias_init.eps <= 0:
            raise ValueError("model.classifier.bias_init.eps must be greater than zero")
        if cfg.bias_init.clamp_min >= cfg.bias_init.clamp_max:
            raise ValueError(
                "model.classifier.bias_init.clamp_min must be less than clamp_max"
            )
        if cfg.bias_init.enabled and cfg.bias_init.type == "none":
            raise ValueError(
                "model.classifier.bias_init.type must be 'prior' or "
                "'weighted_prior' when enabled"
            )

    @staticmethod
    def _validate_model(cfg: ModelConfig, data_cfg: DataConfig) -> None:
        JsonConfigLoader._validate_encoder(cfg.encoder, data_cfg)
        JsonConfigLoader._validate_classifier(cfg.classifier)

    @staticmethod
    def _validate_train(
        cfg: TrainConfig,
        *,
        num_classes: int,
        num_branches: int,
        label_to_index: Mapping[str, int],
    ) -> None:
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
        JsonConfigLoader._validate_sampler(cfg.sampler)
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
        if cfg.loss.branch_auxiliary.weights is not None:
            if len(cfg.loss.branch_auxiliary.weights) != num_branches:
                raise ValueError(
                    "train.loss.branch_auxiliary.weights length must match the number "
                    "of model.encoder.architecture.patch_branches"
                )
            if any(weight <= 0 for weight in cfg.loss.branch_auxiliary.weights):
                raise ValueError(
                    "train.loss.branch_auxiliary.weights values must be greater than zero"
                )
        if (
            cfg.loss.branch_auxiliary.enabled
            and cfg.loss.branch_auxiliary.weights is None
            and cfg.loss.branch_auxiliary.weight <= 0
        ):
            raise ValueError(
                "train.loss.branch_auxiliary.weight must be greater than zero when branch auxiliary loss is enabled"
            )
        if not isinstance(cfg.loss.attention_entropy.enabled, bool):
            raise ValueError("train.loss.attention_entropy.enabled must be a boolean")
        if (
            cfg.loss.attention_entropy.enabled
            and cfg.loss.attention_entropy.weight <= 0
        ):
            raise ValueError(
                "train.loss.attention_entropy.weight must be greater than zero when enabled"
            )
        if not isinstance(cfg.loss.class_weighting.enabled, bool):
            raise ValueError("train.loss.class_weighting.enabled must be a boolean")
        if cfg.loss.class_weighting.type not in JsonConfigLoader._CLASS_WEIGHTING_TYPES:
            raise ValueError(
                "train.loss.class_weighting.type must be 'sqrt_inverse_frequency'"
            )
        if (
            cfg.loss.class_weighting.normalize
            not in JsonConfigLoader._CLASS_WEIGHTING_NORMALIZERS
        ):
            raise ValueError("train.loss.class_weighting.normalize must be 'mean_one'")
        if cfg.loss.class_weighting.source not in JsonConfigLoader._LOSS_WEIGHT_SOURCES:
            raise ValueError("train.loss.class_weighting.source must be 'train'")
        if cfg.loss.class_weighting.enabled and (
            num_classes <= 2 or cfg.loss.type != "cross_entropy"
        ):
            raise ValueError(
                "train.loss.class_weighting is supported only for multiclass "
                "cross_entropy runs"
            )
        if not isinstance(cfg.loss.label_smoothing.enabled, bool):
            raise ValueError("train.loss.label_smoothing.enabled must be a boolean")
        if not (0.0 <= cfg.loss.label_smoothing.value < 1.0):
            raise ValueError("train.loss.label_smoothing.value must be within [0, 1)")
        JsonConfigLoader._validate_branch_binary_auxiliary(
            cfg.loss.branch_binary_auxiliary,
            label_to_index=label_to_index,
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
    def _validate_sampler(cfg: SamplerConfig) -> None:
        if not isinstance(cfg.weighted_random, bool):
            raise ValueError("train.sampler.weighted_random must be a boolean")
        if not isinstance(cfg.enabled, bool):
            raise ValueError("train.sampler.enabled must be a boolean")
        if cfg.type not in JsonConfigLoader._SAMPLER_TYPES:
            raise ValueError(
                "train.sampler.type must be one of "
                f"{sorted(JsonConfigLoader._SAMPLER_TYPES)}"
            )
        if not isinstance(cfg.replacement, bool):
            raise ValueError("train.sampler.replacement must be a boolean")
        if cfg.source != "train":
            raise ValueError("train.sampler.source must be 'train'")
        if isinstance(cfg.num_samples, str):
            if cfg.num_samples != "dataset_size":
                raise ValueError(
                    "train.sampler.num_samples must be 'dataset_size' or a positive integer"
                )
        elif isinstance(cfg.num_samples, bool) or not isinstance(cfg.num_samples, int):
            raise TypeError(
                "train.sampler.num_samples must be 'dataset_size' or a positive integer"
            )
        elif cfg.num_samples <= 0:
            raise ValueError(
                "train.sampler.num_samples must be 'dataset_size' or a positive integer"
            )
        if not cfg.enabled:
            return
        if cfg.type != "sqrt_inverse_class":
            raise ValueError(
                "train.sampler.type must be 'sqrt_inverse_class' when sampler is enabled"
            )
        if not cfg.replacement:
            raise ValueError(
                "train.sampler.replacement must be true for sqrt_inverse_class"
            )

    @staticmethod
    def _validate_branch_binary_auxiliary(
        cfg: BranchBinaryAuxiliaryLossConfig,
        *,
        label_to_index: Mapping[str, int],
    ) -> None:
        if not isinstance(cfg.enabled, bool):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.enabled must be a boolean"
            )
        if cfg.aggregation != "mean":
            raise ValueError(
                "train.loss.branch_binary_auxiliary.aggregation must be 'mean'"
            )
        if cfg.weight <= 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.weight must be greater than zero"
            )
        if not isinstance(cfg.schedule.enabled, bool):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.enabled must be a boolean"
            )
        if (
            cfg.schedule.type
            not in JsonConfigLoader._BRANCH_BINARY_AUXILIARY_SCHEDULE_TYPES
        ):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.type must be "
                "'none' or 'cosine_floor'"
            )
        if cfg.schedule.enabled and cfg.schedule.type != "cosine_floor":
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.type must be "
                "'cosine_floor' when schedule is enabled"
            )
        if cfg.schedule.max_weight <= 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.max_weight must be "
                "greater than zero"
            )
        if cfg.schedule.min_weight < 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.min_weight must be "
                "greater than or equal to zero"
            )
        if cfg.schedule.max_weight < cfg.schedule.min_weight:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.max_weight must be "
                "greater than or equal to min_weight"
            )
        if cfg.schedule.total_epochs is not None and cfg.schedule.total_epochs <= 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.schedule.total_epochs must be "
                "greater than zero when provided"
            )
        if cfg.monitor.loss_weight < 0:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.monitor.loss_weight must be "
                "greater than or equal to zero"
            )
        if not isinstance(cfg.pos_weight.enabled, bool):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.pos_weight.enabled must be a boolean"
            )
        if cfg.pos_weight.type not in JsonConfigLoader._BRANCH_BINARY_POS_WEIGHT_TYPES:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.pos_weight.type must be "
                "'sqrt_normal_over_abnormal'"
            )
        if cfg.pos_weight.source not in JsonConfigLoader._LOSS_WEIGHT_SOURCES:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.pos_weight.source must be 'train'"
            )
        if not cfg.enabled:
            return
        configured_labels = set(cfg.label_to_index.keys())
        expected_labels = set(label_to_index.keys())
        missing = sorted(expected_labels - configured_labels)
        extra = sorted(configured_labels - expected_labels)
        if missing or extra:
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index must contain "
                "exactly the same labels as data.label_to_index; "
                f"missing={missing} extra={extra}"
            )
        values = [int(value) for value in cfg.label_to_index.values()]
        if any(value not in {0, 1} for value in values):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index values must be 0 or 1"
            )
        if all(value == 0 for value in values):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index must include at least one abnormal label"
            )
        if all(value == 1 for value in values):
            raise ValueError(
                "train.loss.branch_binary_auxiliary.label_to_index must include at least one normal label"
            )

    @staticmethod
    def _validate_val(cfg: ValidationConfig) -> None:
        if not isinstance(cfg.loss.use_train_loss_config, bool):
            raise ValueError("val.loss.use_train_loss_config must be a boolean")
        if not cfg.loss.use_train_loss_config:
            raise ValueError(
                "val.loss.use_train_loss_config=false is not supported; validation "
                "loss uses train-derived weighting semantics"
            )
        if cfg.loss.class_weight_source != "train":
            raise ValueError("val.loss.class_weight_source must be 'train'")
        if cfg.loss.binary_pos_weight_source != "train":
            raise ValueError("val.loss.binary_pos_weight_source must be 'train'")

    @staticmethod
    def _validate_checkpointing(cfg: CheckpointingConfig | None) -> None:
        if cfg is None:
            return
        seen: set[str] = set()
        for monitor in cfg.monitors:
            if monitor.name not in JsonConfigLoader._CHECKPOINT_MONITORS:
                raise ValueError(
                    "checkpointing.monitors.name must be one of "
                    f"{sorted(JsonConfigLoader._CHECKPOINT_MONITORS)}"
                )
            if monitor.name in seen:
                raise ValueError(
                    "checkpointing.monitors must not contain duplicate monitor names"
                )
            seen.add(monitor.name)
            allowed_modes = JsonConfigLoader._CHECKPOINT_MONITORS[monitor.name]
            if monitor.mode not in allowed_modes:
                raise ValueError(
                    f"checkpointing monitor {monitor.name} requires mode one of "
                    f"{sorted(allowed_modes)}"
                )
            if monitor.top_k <= 0:
                raise ValueError(
                    "checkpointing.monitors.top_k must be greater than zero"
                )
            if (
                monitor.filename_prefix is not None
                and not monitor.filename_prefix.strip()
            ):
                raise ValueError(
                    "checkpointing.monitors.filename_prefix must not be empty"
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
        augmentation = dict(raw.get("augmentation", {}))
        policy = dict(augmentation.get("policy", {}))
        policy_choices = policy.get("choices", ())
        if policy_choices:
            if not isinstance(policy_choices, Sequence) or isinstance(
                policy_choices, (str, bytes)
            ):
                raise TypeError("data.augmentation.policy.choices must be a list")
            policy["choices"] = tuple(
                AugmentationPolicyChoiceConfig(**dict(choice_raw))
                for choice_raw in policy_choices
            )
        else:
            policy["choices"] = ()
        augmentation["policy"] = AugmentationPolicyConfig(**policy)
        waveform = dict(augmentation.get("waveform", {}))
        waveform["gain"] = RandomGainConfig(**dict(waveform.get("gain", {})))
        waveform["noise"] = AdditiveNoiseConfig(**dict(waveform.get("noise", {})))
        waveform["time_shift"] = TimeShiftConfig(**dict(waveform.get("time_shift", {})))
        augmentation["waveform"] = WaveformAugmentationConfig(**waveform)
        fbank = dict(augmentation.get("fbank", {}))
        fbank["time_mask"] = MaskConfig(**dict(fbank.get("time_mask", {})))
        fbank["freq_mask"] = MaskConfig(**dict(fbank.get("freq_mask", {})))
        augmentation["fbank"] = FbankAugmentationConfig(**fbank)
        kwargs["augmentation"] = DataAugmentationConfig(**augmentation)
        cfg = DataConfig(**kwargs)
        JsonConfigLoader._validate_data(cfg)
        return cfg

    @staticmethod
    def _parse_fsd50k_data(
        raw: Mapping[str, Any],
        *,
        expected_mode: Literal["ssl", "supervised"],
    ) -> Fsd50kDataConfig:
        kwargs = dict(raw)
        if kwargs.get("dataset") != "fsd50k":
            raise ValueError("data.dataset must be 'fsd50k'")
        if kwargs.get("mode") != expected_mode:
            raise ValueError(f"data.mode must be '{expected_mode}'")
        kwargs["audio"] = AudioConfig(**dict(raw.get("audio", {})))
        preprocessing = dict(raw.get("preprocessing", {}))
        preprocessing["bandpass"] = BandPassConfig(**preprocessing.get("bandpass", {}))
        preprocessing["ast_fbank"] = AstFbankConfig(
            **dict(preprocessing.get("ast_fbank", {}))
        )
        kwargs["preprocessing"] = PreprocessingConfig(**preprocessing)
        augmentation = dict(raw.get("augmentation", {}))
        policy = dict(augmentation.get("policy", {}))
        policy_choices = policy.get("choices", ())
        if policy_choices:
            if not isinstance(policy_choices, Sequence) or isinstance(
                policy_choices, (str, bytes)
            ):
                raise TypeError("data.augmentation.policy.choices must be a list")
            policy["choices"] = tuple(
                AugmentationPolicyChoiceConfig(**dict(choice_raw))
                for choice_raw in policy_choices
            )
        else:
            policy["choices"] = ()
        augmentation["policy"] = AugmentationPolicyConfig(**policy)
        waveform = dict(augmentation.get("waveform", {}))
        waveform["gain"] = RandomGainConfig(**dict(waveform.get("gain", {})))
        waveform["noise"] = AdditiveNoiseConfig(**dict(waveform.get("noise", {})))
        waveform["time_shift"] = TimeShiftConfig(**dict(waveform.get("time_shift", {})))
        augmentation["waveform"] = WaveformAugmentationConfig(**waveform)
        fbank = dict(augmentation.get("fbank", {}))
        fbank["time_mask"] = MaskConfig(**dict(fbank.get("time_mask", {})))
        fbank["freq_mask"] = MaskConfig(**dict(fbank.get("freq_mask", {})))
        augmentation["fbank"] = FbankAugmentationConfig(**fbank)
        kwargs["augmentation"] = DataAugmentationConfig(**augmentation)
        cfg = Fsd50kDataConfig(**kwargs)
        if not (0.0 < cfg.val_ratio < 1.0):
            raise ValueError("data.val_ratio must be within (0, 1)")
        if cfg.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        JsonConfigLoader._validate_audio(cfg.audio)
        JsonConfigLoader._validate_preprocessing(
            cfg.preprocessing,
            cfg.audio.sample_rate,
        )
        JsonConfigLoader._validate_augmentation(cfg.augmentation)
        return cfg

    @staticmethod
    def _compat_data_config_for_fsd50k(cfg: Fsd50kDataConfig) -> DataConfig:
        return DataConfig(
            train_dirs=[],
            val_dirs=[],
            eval_dirs=[],
            label_to_index={"negative": 0, "positive": 1},
            batch_size=1,
            num_workers=cfg.num_workers,
            audio=cfg.audio,
            preprocessing=cfg.preprocessing,
            augmentation=cfg.augmentation,
        )

    @staticmethod
    def _parse_ssl_config(raw: Mapping[str, Any] | None) -> MaskedFbankSSLConfig:
        kwargs = dict(raw or {})
        kwargs["masking"] = SslMaskingConfig(**dict(kwargs.get("masking", {})))
        kwargs["decoder"] = SslDecoderConfig(**dict(kwargs.get("decoder", {})))
        kwargs["loss"] = SslLossConfig(**dict(kwargs.get("loss", {})))
        kwargs["visualization"] = SslVisualizationConfig(
            **dict(kwargs.get("visualization", {}))
        )
        cfg = MaskedFbankSSLConfig(**kwargs)
        if cfg.target != "normalized_fbank":
            raise ValueError("ssl.target must be 'normalized_fbank'")
        if cfg.masking.type != "branch_specific_token_grid_input_patch":
            raise ValueError(
                "ssl.masking.type must be 'branch_specific_token_grid_input_patch'"
            )
        if cfg.masking.sampling != "random":
            raise ValueError("ssl.masking.sampling must be 'random'")
        if not (0.0 <= cfg.masking.token_mask_ratio <= 1.0):
            raise ValueError("ssl.masking.token_mask_ratio must be within [0, 1]")
        if cfg.masking.actual_mask_ratio_warning_threshold <= 0:
            raise ValueError(
                "ssl.masking.actual_mask_ratio_warning_threshold must be positive"
            )
        if cfg.decoder.type != "branch_conv1d":
            raise ValueError("ssl.decoder.type must be 'branch_conv1d'")
        if cfg.decoder.sharing != "separate":
            raise ValueError("ssl.decoder.sharing must be 'separate'")
        if cfg.decoder.channels <= 0:
            raise ValueError("ssl.decoder.channels must be greater than zero")
        if cfg.decoder.kernel_size <= 0:
            raise ValueError("ssl.decoder.kernel_size must be greater than zero")
        if cfg.decoder.num_layers <= 0:
            raise ValueError("ssl.decoder.num_layers must be greater than zero")
        if not (0.0 <= cfg.decoder.dropout < 1.0):
            raise ValueError("ssl.decoder.dropout must be within [0, 1)")
        if cfg.loss.type != "mean_branch_masked_mse":
            raise ValueError("ssl.loss.type must be 'mean_branch_masked_mse'")
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
    def _coerce_int_tuple(values: object, *, field_name: str) -> tuple[int, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"{field_name} must be a list of integers")
        result: list[int] = []
        for value in values:
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must contain integers")
            result.append(value)
        return tuple(result)

    @staticmethod
    def _coerce_float_tuple(values: object, *, field_name: str) -> tuple[float, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"{field_name} must be a list of numbers")
        result: list[float] = []
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must contain numbers")
            result.append(float(value))
        return tuple(result)

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
        rdt = dict(architecture.get("rdt", {}))
        if "exclude_branches_from_evidence" in rdt:
            rdt["exclude_branches_from_evidence"] = JsonConfigLoader._coerce_int_tuple(
                rdt["exclude_branches_from_evidence"],
                field_name=(
                    "model.encoder.architecture.rdt.exclude_branches_from_evidence"
                ),
            )
        architecture["rdt"] = RdtConfig(**rdt)
        architecture["mil"] = MilConfig(**dict(architecture.get("mil", {})))
        architecture["evidence_pooling"] = EvidencePoolingConfig(
            **dict(architecture.get("evidence_pooling", {}))
        )
        token_augmentation = dict(architecture.get("token_augmentation", {}))
        token_augmentation["branch_event_dropout"] = BranchEventDropoutConfig(
            **dict(token_augmentation.get("branch_event_dropout", {}))
        )
        token_augmentation["selected_evidence_dropout"] = SelectedEvidenceDropoutConfig(
            **dict(token_augmentation.get("selected_evidence_dropout", {}))
        )
        architecture["token_augmentation"] = TokenAugmentationConfig(
            **token_augmentation
        )
        encoder["adaptation"] = EncoderAdaptationConfig(
            **dict(encoder.get("adaptation", {}))
        )
        encoder["architecture"] = MultiScaleRdtArchitectureConfig(**architecture)
        kwargs["encoder"] = ModelEncoderConfig(**encoder)
        classifier = dict(raw["classifier"])
        classifier["bias_init"] = ClassifierBiasInitConfig(
            **dict(classifier.get("bias_init", {}))
        )
        kwargs["classifier"] = ClassifierConfig(**classifier)
        cfg = ModelConfig(**kwargs)
        JsonConfigLoader._validate_model(cfg, data_cfg)
        return cfg

    @staticmethod
    def _parse_train(raw: Mapping[str, Any]) -> TrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = OptimizerConfig(**dict(raw["optimizer"]))
        kwargs["scheduler"] = SchedulerConfig(**dict(raw.get("scheduler", {})))
        loss = dict(raw.get("loss", {}))
        loss["class_weighting"] = ClassWeightingConfig(
            **dict(loss.get("class_weighting", {}))
        )
        loss["label_smoothing"] = LabelSmoothingConfig(
            **dict(loss.get("label_smoothing", {}))
        )
        branch_auxiliary = dict(loss.get("branch_auxiliary", {}))
        if "weights" in branch_auxiliary and branch_auxiliary["weights"] is not None:
            branch_auxiliary["weights"] = JsonConfigLoader._coerce_float_tuple(
                branch_auxiliary["weights"],
                field_name="train.loss.branch_auxiliary.weights",
            )
        loss["branch_auxiliary"] = BranchAuxiliaryLossConfig(**branch_auxiliary)
        branch_binary_auxiliary = dict(loss.get("branch_binary_auxiliary", {}))
        branch_binary_auxiliary["label_to_index"] = dict(
            branch_binary_auxiliary.get("label_to_index", {})
        )
        branch_binary_auxiliary["pos_weight"] = BranchBinaryPosWeightConfig(
            **dict(branch_binary_auxiliary.get("pos_weight", {}))
        )
        branch_binary_auxiliary["schedule"] = BranchBinaryAuxiliaryScheduleConfig(
            **dict(branch_binary_auxiliary.get("schedule", {}))
        )
        branch_binary_auxiliary["monitor"] = BranchBinaryAuxiliaryMonitorConfig(
            **dict(branch_binary_auxiliary.get("monitor", {}))
        )
        loss["branch_binary_auxiliary"] = BranchBinaryAuxiliaryLossConfig(
            **branch_binary_auxiliary
        )
        loss["attention_entropy"] = AttentionEntropyLossConfig(
            **dict(loss.get("attention_entropy", {}))
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
    def _parse_fsd50k_ssl_train(raw: Mapping[str, Any]) -> Fsd50kSslTrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = Fsd50kSslOptimizerConfig(**dict(raw.get("optimizer", {})))
        kwargs["scheduler"] = Fsd50kSchedulerConfig(**dict(raw.get("scheduler", {})))
        cfg = Fsd50kSslTrainConfig(**kwargs)
        if cfg.epochs <= 0:
            raise ValueError("train.epochs must be greater than zero")
        if cfg.batch_size <= 0:
            raise ValueError("train.batch_size must be greater than zero")
        if cfg.optimizer.lr <= 0:
            raise ValueError("train.optimizer.lr must be greater than zero")
        if cfg.optimizer.weight_decay < 0:
            raise ValueError("train.optimizer.weight_decay must be non-negative")
        if cfg.scheduler.type != "linear_warmup_cosine":
            raise ValueError("train.scheduler.type must be 'linear_warmup_cosine'")
        if not (0.0 <= cfg.scheduler.warmup_ratio <= 1.0):
            raise ValueError("train.scheduler.warmup_ratio must be within [0, 1]")
        if cfg.max_grad_norm <= 0:
            raise ValueError("train.max_grad_norm must be greater than zero")
        return cfg

    @staticmethod
    def _parse_fsd50k_supervised_train(
        raw: Mapping[str, Any],
    ) -> Fsd50kSupervisedTrainConfig:
        kwargs = dict(raw)
        kwargs["optimizer"] = Fsd50kSupervisedOptimizerConfig(
            **dict(raw.get("optimizer", {}))
        )
        kwargs["scheduler"] = Fsd50kSchedulerConfig(**dict(raw.get("scheduler", {})))
        kwargs["gradient_clipping"] = GradientClippingConfig(
            **dict(raw.get("gradient_clipping", {}))
        )
        loss = dict(raw.get("loss", {}))
        loss["pos_weight"] = MultiLabelPosWeightConfig(
            **dict(loss.get("pos_weight", {}))
        )
        loss["branch_auxiliary"] = BranchAuxiliaryLossConfig(
            **dict(loss.get("branch_auxiliary", {}))
        )
        branch_binary = dict(loss.get("branch_binary_auxiliary", {}))
        branch_binary["label_to_index"] = dict(branch_binary.get("label_to_index", {}))
        branch_binary["pos_weight"] = BranchBinaryPosWeightConfig(
            **dict(branch_binary.get("pos_weight", {}))
        )
        branch_binary["schedule"] = BranchBinaryAuxiliaryScheduleConfig(
            **dict(branch_binary.get("schedule", {}))
        )
        branch_binary["monitor"] = BranchBinaryAuxiliaryMonitorConfig(
            **dict(branch_binary.get("monitor", {}))
        )
        loss["branch_binary_auxiliary"] = BranchBinaryAuxiliaryLossConfig(
            **branch_binary
        )
        kwargs["loss"] = Fsd50kSupervisedLossConfig(**loss)
        cfg = Fsd50kSupervisedTrainConfig(**kwargs)
        if cfg.epochs <= 0:
            raise ValueError("train.epochs must be greater than zero")
        if cfg.batch_size <= 0:
            raise ValueError("train.batch_size must be greater than zero")
        if cfg.optimizer.encoder_lr <= 0:
            raise ValueError("train.optimizer.encoder_lr must be greater than zero")
        if cfg.optimizer.body_lr <= 0:
            raise ValueError("train.optimizer.body_lr must be greater than zero")
        if cfg.optimizer.head_lr <= 0:
            raise ValueError("train.optimizer.head_lr must be greater than zero")
        if cfg.optimizer.weight_decay < 0:
            raise ValueError("train.optimizer.weight_decay must be non-negative")
        if cfg.scheduler.type != "linear_warmup_cosine":
            raise ValueError("train.scheduler.type must be 'linear_warmup_cosine'")
        if not (0.0 <= cfg.scheduler.warmup_ratio <= 1.0):
            raise ValueError("train.scheduler.warmup_ratio must be within [0, 1]")
        if not isinstance(cfg.gradient_clipping.enabled, bool):
            raise ValueError("train.gradient_clipping.enabled must be a boolean")
        if cfg.gradient_clipping.max_norm <= 0:
            raise ValueError(
                "train.gradient_clipping.max_norm must be greater than zero"
            )
        if cfg.loss.type != "bce_with_logits":
            raise ValueError("train.loss.type must be 'bce_with_logits'")
        if cfg.loss.pos_weight.type != "sqrt_negative_over_positive":
            raise ValueError(
                "train.loss.pos_weight.type must be 'sqrt_negative_over_positive'"
            )
        if cfg.loss.pos_weight.cap <= 0:
            raise ValueError("train.loss.pos_weight.cap must be greater than zero")
        if cfg.loss.pos_weight.source != "train":
            raise ValueError("train.loss.pos_weight.source must be 'train'")
        if cfg.loss.branch_auxiliary.enabled:
            raise ValueError(
                "FSD50K supervised training requires branch_auxiliary.enabled=false"
            )
        if cfg.loss.branch_binary_auxiliary.enabled:
            raise ValueError(
                "FSD50K supervised training requires branch_binary_auxiliary.enabled=false"
            )
        return cfg

    @staticmethod
    def _parse_fsd50k_metrics(raw: Mapping[str, Any] | None) -> Fsd50kMetricsConfig:
        kwargs = dict(raw or {})
        topk_raw = kwargs.get("topk", {})
        if not isinstance(topk_raw, Mapping):
            raise ValueError("metrics.topk must be an object")
        topk_kwargs = dict(topk_raw)
        if "ks" in topk_kwargs:
            raw_ks = topk_kwargs["ks"]
            if not isinstance(raw_ks, (list, tuple)):
                raise ValueError("metrics.topk.ks must be a list of positive integers")
            topk_kwargs["ks"] = tuple(raw_ks)
        kwargs["topk"] = Fsd50kTopKMetricsConfig(**topk_kwargs)
        cfg = Fsd50kMetricsConfig(**kwargs)
        if cfg.primary != "macro_AP":
            raise ValueError("metrics.primary must be 'macro_AP'")
        if cfg.secondary != "micro_AP":
            raise ValueError("metrics.secondary must be 'micro_AP'")
        if not (0.0 < cfg.f1_threshold < 1.0):
            raise ValueError("metrics.f1_threshold must be within (0, 1)")
        if type(cfg.topk.enabled) is not bool:
            raise ValueError("metrics.topk.enabled must be a boolean")
        if type(cfg.topk.save_label_frequency) is not bool:
            raise ValueError("metrics.topk.save_label_frequency must be a boolean")
        if not cfg.topk.ks:
            raise ValueError("metrics.topk.ks must contain at least one value")
        for k in cfg.topk.ks:
            if type(k) is not int or k <= 0:
                raise ValueError("metrics.topk.ks must contain positive integers")
        if type(cfg.topk.top_n_frequency) is not int or cfg.topk.top_n_frequency <= 0:
            raise ValueError("metrics.topk.top_n_frequency must be a positive integer")
        if cfg.zero_positive_class_policy != "exclude_with_warning":
            raise ValueError(
                "metrics.zero_positive_class_policy must be 'exclude_with_warning'"
            )
        return cfg

    @staticmethod
    def _parse_val(raw: Mapping[str, Any] | None) -> ValidationConfig:
        if raw is None:
            return ValidationConfig()
        kwargs = dict(raw)
        kwargs["loss"] = ValidationLossConfig(**dict(raw.get("loss", {})))
        cfg = ValidationConfig(**kwargs)
        JsonConfigLoader._validate_val(cfg)
        return cfg

    @staticmethod
    def _parse_checkpointing(
        raw: Mapping[str, Any] | None,
    ) -> CheckpointingConfig | None:
        if raw is None:
            return None
        monitors_raw = raw.get("monitors", ())
        if not isinstance(monitors_raw, Sequence) or isinstance(
            monitors_raw, (str, bytes)
        ):
            raise TypeError("checkpointing.monitors must be a list")
        monitors: list[CheckpointMonitorConfig] = []
        for monitor_raw in monitors_raw:
            monitor_kwargs = dict(monitor_raw)
            keep_top_k = monitor_kwargs.pop("keep_top_k", None)
            if keep_top_k is not None:
                if "top_k" in monitor_kwargs and monitor_kwargs["top_k"] != keep_top_k:
                    raise ValueError(
                        "checkpointing.monitors.keep_top_k and top_k must match "
                        "when both are provided"
                    )
                monitor_kwargs["top_k"] = keep_top_k
            monitors.append(CheckpointMonitorConfig(**monitor_kwargs))
        cfg = CheckpointingConfig(monitors=tuple(monitors))
        JsonConfigLoader._validate_checkpointing(cfg)
        return cfg

    @staticmethod
    def _parse_analysis(raw: Mapping[str, Any] | None) -> AnalysisConfig:
        if raw is None:
            return AnalysisConfig()
        kwargs = dict(raw)
        kwargs["outputs"] = AnalysisOutputConfig(**dict(raw.get("outputs", {})))
        return AnalysisConfig(**kwargs)

    @staticmethod
    def _parse_terminal(raw: Mapping[str, Any] | None) -> TerminalConfig:
        if raw is None:
            return TerminalConfig()
        if not isinstance(raw, Mapping):
            raise ValueError("terminal must be a JSON object")
        cfg = TerminalConfig(**dict(raw))
        if cfg.width is None:
            return cfg
        if isinstance(cfg.width, bool) or not isinstance(cfg.width, int):
            raise ValueError("terminal.width must be a positive integer or null")
        if cfg.width <= 0:
            raise ValueError("terminal.width must be a positive integer or null")
        return cfg

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
            val=JsonConfigLoader._parse_val(raw.get("val")),
            checkpointing=JsonConfigLoader._parse_checkpointing(
                raw.get("checkpointing")
            ),
            terminal=JsonConfigLoader._parse_terminal(raw.get("terminal")),
        )
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
            num_branches=len(cfg.model.encoder.architecture.patch_branches),
            label_to_index=cfg.data.label_to_index,
        )
        JsonConfigLoader._validate_val(cfg.val)
        JsonConfigLoader._validate_checkpointing(cfg.checkpointing)
        return cfg

    @staticmethod
    def _parse_experiment_or_default(
        raw: Mapping[str, Any],
        *,
        task: str,
        seed: int,
    ) -> ExperimentConfig:
        experiment_raw = raw.get("experiment")
        if experiment_raw is not None:
            return JsonConfigLoader._parse_experiment(dict(experiment_raw))
        return ExperimentConfig(
            name=task,
            task=task,
            mode="clip",
            seed=seed,
            device="cpu",
            output_dir="checkpoints",
        )

    @staticmethod
    def load_fsd50k_ssl(path: str | Path) -> Fsd50kSslRunConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_fsd50k_data(
            raw["data"],
            expected_mode="ssl",
        )
        model_cfg = JsonConfigLoader._parse_model(
            raw["model"],
            JsonConfigLoader._compat_data_config_for_fsd50k(data_cfg),
        )
        task = str(raw.get("task", "fsd50k_masked_fbank_ssl"))
        cfg = Fsd50kSslRunConfig(
            task=task,
            experiment=JsonConfigLoader._parse_experiment_or_default(
                raw,
                task=task,
                seed=data_cfg.split_seed,
            ),
            data=data_cfg,
            model=model_cfg,
            train=JsonConfigLoader._parse_fsd50k_ssl_train(raw["train"]),
            ssl=JsonConfigLoader._parse_ssl_config(raw.get("ssl")),
            checkpointing=JsonConfigLoader._parse_checkpointing(
                raw.get("checkpointing")
            ),
            terminal=JsonConfigLoader._parse_terminal(raw.get("terminal")),
        )
        JsonConfigLoader._validate_checkpointing(cfg.checkpointing)
        return cfg

    @staticmethod
    def load_fsd50k_supervised(path: str | Path) -> Fsd50kSupervisedRunConfig:
        raw = JsonConfigLoader.load_json(path)
        data_cfg = JsonConfigLoader._parse_fsd50k_data(
            raw["data"],
            expected_mode="supervised",
        )
        model_cfg = JsonConfigLoader._parse_model(
            raw["model"],
            JsonConfigLoader._compat_data_config_for_fsd50k(data_cfg),
        )
        task = str(raw.get("task", "fsd50k_multilabel"))
        cfg = Fsd50kSupervisedRunConfig(
            task=task,
            experiment=JsonConfigLoader._parse_experiment_or_default(
                raw,
                task=task,
                seed=data_cfg.split_seed,
            ),
            data=data_cfg,
            model=model_cfg,
            train=JsonConfigLoader._parse_fsd50k_supervised_train(raw["train"]),
            analysis=JsonConfigLoader._parse_analysis(raw.get("analysis")),
            metrics=JsonConfigLoader._parse_fsd50k_metrics(raw.get("metrics")),
            checkpointing=JsonConfigLoader._parse_checkpointing(
                raw.get("checkpointing")
            ),
            terminal=JsonConfigLoader._parse_terminal(raw.get("terminal")),
        )
        JsonConfigLoader._validate_checkpointing(cfg.checkpointing)
        return cfg

    @staticmethod
    def load_cnuh_transfer_template(path: str | Path) -> CnuhTransferTemplateConfig:
        raw = JsonConfigLoader.load_json(path)
        transfer = dict(raw["transfer"])
        freeze_modules_raw = transfer.get("freeze_modules")
        if freeze_modules_raw is not None:
            if not isinstance(freeze_modules_raw, Sequence) or isinstance(
                freeze_modules_raw,
                (str, bytes),
            ):
                raise TypeError("transfer.freeze_modules must be a list of strings")
            transfer["freeze_modules"] = tuple(str(item) for item in freeze_modules_raw)
        cfg = CnuhTransferTemplateConfig(
            task=raw.get("task", "cnuh_4class"),
            transfer=CnuhTransferConfig(**transfer),
            optimizer=CnuhTransferOptimizerConfig(**dict(raw.get("optimizer", {}))),
        )
        if not cfg.transfer.checkpoint_path:
            raise ValueError("transfer.checkpoint_path must not be empty")
        if not isinstance(cfg.transfer.reset_classifier, bool):
            raise ValueError("transfer.reset_classifier must be a boolean")
        if not isinstance(cfg.transfer.freeze_encoder, bool):
            raise ValueError("transfer.freeze_encoder must be a boolean")
        if not isinstance(cfg.transfer.strict, bool):
            raise ValueError("transfer.strict must be a boolean")
        if cfg.optimizer.frozen_encoder_lr < 0:
            raise ValueError("optimizer.frozen_encoder_lr must be non-negative")
        if cfg.optimizer.body_lr <= 0:
            raise ValueError("optimizer.body_lr must be greater than zero")
        if cfg.optimizer.head_lr <= 0:
            raise ValueError("optimizer.head_lr must be greater than zero")
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
            terminal=JsonConfigLoader._parse_terminal(raw.get("terminal")),
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
            val=JsonConfigLoader._parse_val(raw.get("val")),
            checkpointing=JsonConfigLoader._parse_checkpointing(
                raw.get("checkpointing")
            ),
            terminal=JsonConfigLoader._parse_terminal(raw.get("terminal")),
        )
        JsonConfigLoader._validate_train(
            cfg.train,
            num_classes=len(cfg.data.label_to_index),
            num_branches=len(cfg.model.encoder.architecture.patch_branches),
            label_to_index=cfg.data.label_to_index,
        )
        JsonConfigLoader._validate_val(cfg.val)
        JsonConfigLoader._validate_checkpointing(cfg.checkpointing)
        return cfg
