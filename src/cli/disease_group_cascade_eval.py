from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)

from src.data.audio import (
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    ResNetSpectrogramFeatureConfig,
    ResNetSpectrogramImage,
    SegmentFeatureExtractor,
    WaveformPreprocessor,
    WhisperLikeLogMel,
)
from src.data.io import WaveformLoader
from src.evaluation.metrics import MetricsComputer
from src.evaluation.prediction import class_probabilities_and_predictions
from src.evaluation.thresholds import (
    ThresholdOptimizationResult,
    load_checkpoint_threshold_optimization,
)
from src.models.model import AstModelConfig
from src.models.resnet50_model import ResNet50ModelConfig
from src.models.whisper_model import WhisperModelConfig
from src.plots.export import PlotlyExportMixin
from src.training.model_setup import build_model_from_runtime_config
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import CvRunConfig, DataConfig, JsonConfigLoader
from src.utils.logging import LoggingMixin, enable_file_logging, logger

FINAL_LABELS = ("Normal", "Airway", "Lung_Parenchymal")
FINAL_LABEL_TO_INDEX = {label: index for index, label in enumerate(FINAL_LABELS)}
DIRECT_TASK = "Normal_vs_Airway_vs_LungParenchymal"
STAGE1_TASK = "Normal_vs_Abnormal"
STAGE2_TASK = "Airway_vs_LungParenchymal"
FOLD_NAMES = tuple(f"fold_{index}" for index in range(5))
DEFAULT_TEST_ROOT = (
    "/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/"
    "DATA/DISEASE_CNUH_DATA/5_Folds/test"
)
METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


@dataclass(frozen=True)
class ManifestItem:
    audio_path: Path
    true_label: str

    @property
    def true_index(self) -> int:
        return FINAL_LABEL_TO_INDEX[self.true_label]


@dataclass(frozen=True)
class SelectedCheckpoint:
    task: str
    fold_name: str
    checkpoint_path: Path
    metrics_path: Path
    optimized_f1: float
    balanced_accuracy: float | None
    pr_auc: float | None
    brier_score: float | None
    filename_score: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["metrics_path"] = str(self.metrics_path)
        return payload


@dataclass(frozen=True)
class ModelPrediction:
    audio_path: Path
    predicted_label: str
    predicted_index: int
    probabilities: tuple[float, ...]
    threshold: float | None
    threshold_source: str | None


@dataclass(frozen=True)
class InferenceSpec:
    checkpoint: SelectedCheckpoint
    config: CvRunConfig
    label_order: tuple[str, ...]
    device: str


class CheckpointSelector(LoggingMixin):
    def __init__(self, results_root: str | Path):
        self.results_root = Path(results_root)

    def load_config(self, task: str) -> CvRunConfig:
        config_path = self._find_config_path(task)
        return JsonConfigLoader.load_cv(config_path)

    def select(self, task: str, fold_name: str) -> SelectedCheckpoint:
        fold_dir = self.results_root / task / fold_name
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        candidates = sorted(fold_dir.glob("eval_metrics__best_f1_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No eval_metrics__best_f1_*.json files found under {fold_dir}"
            )

        selected_path = max(candidates, key=self._metric_sort_key)
        metrics = self._load_metrics(selected_path)
        checkpoint_stem = selected_path.stem.removeprefix("eval_metrics__")
        checkpoint_path = selected_path.with_name(f"{checkpoint_stem}.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint matching {selected_path} not found: {checkpoint_path}"
            )

        optimized = self._optimized_metrics(metrics)
        optimized_f1 = self._require_float(optimized.get("f1_score"), "f1_score")
        selected = SelectedCheckpoint(
            task=task,
            fold_name=fold_name,
            checkpoint_path=checkpoint_path,
            metrics_path=selected_path,
            optimized_f1=optimized_f1,
            balanced_accuracy=self._optional_float(optimized.get("balanced_accuracy")),
            pr_auc=self._optional_float(optimized.get("pr_auc")),
            brier_score=self._optional_float(optimized.get("brier_score")),
            filename_score=self._filename_score(selected_path),
        )
        self.logger.info(
            "[%s/%s] selected %s with optimized_f1=%.6f",
            task,
            fold_name,
            checkpoint_path.name,
            optimized_f1,
        )
        return selected

    def _find_config_path(self, task: str) -> Path:
        task_dir = self.results_root / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        candidates = sorted(task_dir.glob("cv_run*.json"))
        if not candidates:
            raise FileNotFoundError(f"No cv_run*.json config found under {task_dir}")
        return candidates[0]

    @classmethod
    def _metric_sort_key(cls, path: Path) -> tuple[float, float, float, float, float]:
        metrics = cls._load_metrics(path)
        optimized = cls._optimized_metrics(metrics)
        f1 = cls._numeric_or_low(optimized.get("f1_score"))
        balanced = cls._numeric_or_low(optimized.get("balanced_accuracy"))
        pr_auc = cls._numeric_or_low(optimized.get("pr_auc"))
        brier = cls._optional_float(optimized.get("brier_score"))
        brier_key = -brier if brier is not None else -math.inf
        filename_score = cls._filename_score(path)
        filename_key = filename_score if filename_score is not None else -math.inf
        return (f1, balanced, pr_auc, brier_key, filename_key)

    @staticmethod
    def _load_metrics(path: Path) -> dict[str, Any]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Metrics file must be a JSON object: {path}")
        return raw

    @staticmethod
    def _optimized_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
        raw = metrics.get("optimized_metrics")
        if isinstance(raw, Mapping):
            return raw
        return metrics

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric

    @classmethod
    def _numeric_or_low(cls, value: object) -> float:
        numeric = cls._optional_float(value)
        return numeric if numeric is not None else -math.inf

    @classmethod
    def _require_float(cls, value: object, name: str) -> float:
        numeric = cls._optional_float(value)
        if numeric is None:
            raise ValueError(f"Missing finite numeric metric: {name}")
        return numeric

    @staticmethod
    def _filename_score(path: Path) -> float | None:
        try:
            return float(path.stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            return None


class InferenceRunner(LoggingMixin):
    def __init__(self, spec: InferenceSpec):
        self.spec = spec
        self.device = torch.device(spec.device)
        self.checkpoint = load_checkpoint(
            str(spec.checkpoint.checkpoint_path),
            device=self.device,
        )
        model_cfg_raw = self.checkpoint.get("model_cfg")
        if model_cfg_raw is None:
            raise RuntimeError(
                f"Checkpoint missing model_cfg: {spec.checkpoint.checkpoint_path}"
            )
        self.model_cfg = parse_model_cfg(model_cfg_raw)
        self.num_classes = int(
            self.checkpoint.get("num_classes", self.model_cfg.num_classes)
        )
        self._validate_labels()
        self.model = build_model_from_runtime_config(self.model_cfg)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.threshold_optimization = self._load_threshold()
        self._validate_frontend_dims(spec.config.data)
        self.waveform_loader = WaveformLoader(spec.config.data.audio.sample_rate)
        self.preprocessor = WaveformPreprocessor(
            self._build_audio_preprocess_config(spec.config.data)
        )
        self.feature_extractor = self._build_feature_extractor(spec.config.data)

    def predict(self, files: Sequence[Path]) -> dict[str, ModelPrediction]:
        predictions: dict[str, ModelPrediction] = {}
        batch_size = int(self.spec.config.data.batch_size)
        with torch.no_grad():
            for start in range(0, len(files), batch_size):
                batch_files = list(files[start : start + batch_size])
                inputs = torch.stack(
                    [self._load_features(path) for path in batch_files],
                    dim=0,
                ).to(self.device)
                output = self.model(inputs)
                batch_probabilities, batch_indices = self._probabilities_and_indices(
                    output.logits
                )
                for index, path in enumerate(batch_files):
                    probs = tuple(
                        float(item)
                        for item in batch_probabilities[index].cpu().tolist()
                    )
                    pred_index = int(batch_indices[index].item())
                    predictions[str(path)] = ModelPrediction(
                        audio_path=path,
                        predicted_label=self.spec.label_order[pred_index],
                        predicted_index=pred_index,
                        probabilities=probs,
                        threshold=(
                            self.threshold_optimization.selected_threshold
                            if self.num_classes == 2
                            else None
                        ),
                        threshold_source=(
                            self.threshold_optimization.threshold_source
                            if self.num_classes == 2
                            else None
                        ),
                    )
        return predictions

    @staticmethod
    def _build_audio_preprocess_config(cfg: DataConfig) -> AudioPreprocessConfig:
        return AudioPreprocessConfig(
            sample_rate=cfg.audio.sample_rate,
            clip_seconds=cfg.audio.clip_duration_sec,
            source_type=cfg.preprocessing.source_type,
            bandpass_enabled=cfg.preprocessing.bandpass.enabled,
            bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
            bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
            bandpass_q=cfg.preprocessing.bandpass.q,
        )

    @staticmethod
    def _build_feature_extractor(cfg: DataConfig) -> SegmentFeatureExtractor:
        if cfg.preprocessing.feature_type == "ast_fbank":
            return AstLikeFbank(
                AstFbankFeatureConfig(
                    sample_rate=cfg.audio.sample_rate,
                    clip_seconds=cfg.audio.clip_duration_sec,
                    num_mel_bins=cfg.preprocessing.ast_fbank.num_mel_bins,
                    max_length=cfg.preprocessing.ast_fbank.max_length,
                    do_normalize=cfg.preprocessing.ast_fbank.do_normalize,
                    mean=cfg.preprocessing.ast_fbank.mean,
                    std=cfg.preprocessing.ast_fbank.std,
                )
            )
        if cfg.preprocessing.feature_type == "log_mel":
            return WhisperLikeLogMel(
                AudioPreprocessConfig(
                    sample_rate=cfg.audio.sample_rate,
                    n_fft=cfg.preprocessing.log_mel.n_fft,
                    hop_length=cfg.preprocessing.log_mel.hop_length,
                    win_length=cfg.preprocessing.log_mel.win_length,
                    n_mels=cfg.preprocessing.log_mel.n_mels,
                    clip_seconds=cfg.audio.clip_duration_sec,
                    source_type=cfg.preprocessing.source_type,
                    bandpass_enabled=cfg.preprocessing.bandpass.enabled,
                    bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
                    bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
                    bandpass_q=cfg.preprocessing.bandpass.q,
                )
            )
        if cfg.preprocessing.feature_type == "resnet_spectrogram":
            return ResNetSpectrogramImage(
                ResNetSpectrogramFeatureConfig(
                    sample_rate=cfg.audio.sample_rate,
                    clip_seconds=cfg.audio.clip_duration_sec,
                    n_fft=cfg.preprocessing.resnet_spectrogram.n_fft,
                    hop_length=cfg.preprocessing.resnet_spectrogram.hop_length,
                    win_length=cfg.preprocessing.resnet_spectrogram.win_length,
                    n_mels=cfg.preprocessing.resnet_spectrogram.n_mels,
                    f_min=cfg.preprocessing.resnet_spectrogram.f_min,
                    f_max=cfg.preprocessing.resnet_spectrogram.f_max,
                    use_hpss=cfg.preprocessing.resnet_spectrogram.use_hpss,
                    hpss_margin=cfg.preprocessing.resnet_spectrogram.hpss_margin,
                    bandpass_enabled=cfg.preprocessing.bandpass.enabled,
                    bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
                    bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
                    bandpass_q=cfg.preprocessing.bandpass.q,
                    image_size=cfg.preprocessing.resnet_spectrogram.image_size,
                    image_mean=cfg.preprocessing.resnet_spectrogram.image_mean,
                    image_std=cfg.preprocessing.resnet_spectrogram.image_std,
                )
            )
        raise ValueError(f"Unsupported feature_type: {cfg.preprocessing.feature_type}")

    def _load_features(self, path: Path) -> torch.Tensor:
        waveform = self.waveform_loader.load(path)
        if self.spec.config.data.preprocessing.feature_type == "ast_fbank":
            clip_waveform = self.preprocessor.prepare(waveform)
            feature_map = self.feature_extractor(clip_waveform).transpose(0, 1)
        else:
            feature_map = self.feature_extractor(waveform)
        return feature_map.contiguous()

    def _probabilities_and_indices(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return class_probabilities_and_predictions(
            logits,
            num_classes=self.num_classes,
            threshold=float(self.threshold_optimization.selected_threshold),
        )

    def _load_threshold(self) -> ThresholdOptimizationResult:
        if self.num_classes != 2:
            return ThresholdOptimizationResult.disabled(
                "f1",
                reason="threshold optimization is only supported for binary classification",
            )
        return load_checkpoint_threshold_optimization(
            self.checkpoint.get("val_threshold_optimization"),
            "f1",
        )

    def _validate_frontend_dims(self, data_cfg: DataConfig) -> None:
        if isinstance(self.model_cfg, AstModelConfig):
            checkpoint_bins = self.model_cfg.encoder.feature_dims.num_mel_bins
            checkpoint_length = self.model_cfg.encoder.feature_dims.max_length
            data_bins = data_cfg.preprocessing.ast_fbank.num_mel_bins
            data_length = data_cfg.preprocessing.ast_fbank.max_length
            if checkpoint_bins == data_bins and checkpoint_length == data_length:
                return
            raise ValueError(
                "Evaluation frontend dims do not match checkpoint encoder dims. "
                f"checkpoint=({checkpoint_bins}, {checkpoint_length}), "
                f"data=({data_bins}, {data_length}), "
                f"checkpoint_path={self.spec.checkpoint.checkpoint_path}"
            )
        if isinstance(self.model_cfg, WhisperModelConfig):
            if data_cfg.preprocessing.feature_type != "log_mel":
                raise ValueError(
                    "Whisper checkpoint evaluation requires log_mel frontend"
                )
            log_mel = data_cfg.preprocessing.log_mel
            n_samples = int(
                round(data_cfg.audio.sample_rate * data_cfg.audio.clip_duration_sec)
            )
            expected_ctx = ((n_samples // log_mel.hop_length) + 1) // 2
            if (
                self.model_cfg.encoder.n_mels == log_mel.n_mels
                and self.model_cfg.encoder.n_audio_ctx == expected_ctx
            ):
                return
            raise ValueError(
                "Evaluation frontend dims do not match checkpoint encoder dims. "
                f"checkpoint=(n_mels={self.model_cfg.encoder.n_mels}, "
                f"n_audio_ctx={self.model_cfg.encoder.n_audio_ctx}), "
                f"data=(n_mels={log_mel.n_mels}, n_audio_ctx={expected_ctx}), "
                f"checkpoint_path={self.spec.checkpoint.checkpoint_path}"
            )
        if isinstance(self.model_cfg, ResNet50ModelConfig):
            if data_cfg.preprocessing.feature_type != "resnet_spectrogram":
                raise ValueError(
                    "ResNet50 checkpoint evaluation requires resnet_spectrogram frontend"
                )
            resnet_spec = data_cfg.preprocessing.resnet_spectrogram
            if self.model_cfg.encoder.image_size != resnet_spec.image_size:
                raise ValueError(
                    "Evaluation frontend image_size does not match checkpoint encoder. "
                    f"checkpoint={self.model_cfg.encoder.image_size}, "
                    f"data={resnet_spec.image_size}, "
                    f"checkpoint_path={self.spec.checkpoint.checkpoint_path}"
                )
            if self.model_cfg.encoder.input_channels != 3:
                raise ValueError("ResNet50 checkpoint must use input_channels=3")
            feature_cfg = self.checkpoint.get("feature_cfg")
            if isinstance(feature_cfg, Mapping):
                expected = {
                    "feature_type": "resnet_spectrogram",
                    "num_mel_bins": resnet_spec.n_mels,
                    "max_length": int(
                        round(
                            data_cfg.audio.sample_rate
                            * data_cfg.audio.clip_duration_sec
                        )
                    )
                    // resnet_spec.hop_length,
                    "input_channels": self.model_cfg.encoder.input_channels,
                    "image_size": self.model_cfg.encoder.image_size,
                }
                mismatches = {
                    key: (feature_cfg.get(key), value)
                    for key, value in expected.items()
                    if feature_cfg.get(key) != value
                }
                if mismatches:
                    raise ValueError(
                        "Evaluation frontend metadata does not match checkpoint "
                        f"feature_cfg. mismatches={mismatches}, "
                        f"checkpoint_path={self.spec.checkpoint.checkpoint_path}"
                    )
            return
        raise ValueError(f"Unsupported model config type: {type(self.model_cfg)!r}")

    def _validate_labels(self) -> None:
        if len(self.spec.label_order) != self.num_classes:
            raise ValueError(
                "Label order length does not match checkpoint class count: "
                f"{self.spec.label_order} vs {self.num_classes}"
            )


class CascadeComparisonEvaluator(LoggingMixin):
    def __init__(
        self,
        *,
        results_root: str | Path,
        test_root: str | Path,
        reports_dir: str | Path,
        device: str | None,
        formats: set[str],
    ):
        self.results_root = Path(results_root)
        self.test_root = Path(test_root)
        self.reports_dir = Path(reports_dir)
        self.device_override = device
        self.formats = formats
        self.selector = CheckpointSelector(self.results_root)

    def run(self) -> dict[str, Any]:
        manifest = self.build_test_manifest()
        config_by_task = {
            DIRECT_TASK: self.selector.load_config(DIRECT_TASK),
            STAGE1_TASK: self.selector.load_config(STAGE1_TASK),
            STAGE2_TASK: self.selector.load_config(STAGE2_TASK),
        }
        self._validate_config_labels(config_by_task)

        all_rows: list[dict[str, Any]] = []
        fold_metric_rows: list[dict[str, Any]] = []
        selected_by_fold: dict[str, dict[str, Any]] = {}

        files = [item.audio_path for item in manifest]
        true_indices = np.asarray([item.true_index for item in manifest], dtype=int)

        for fold_name in FOLD_NAMES:
            selected = {
                DIRECT_TASK: self.selector.select(DIRECT_TASK, fold_name),
                STAGE1_TASK: self.selector.select(STAGE1_TASK, fold_name),
                STAGE2_TASK: self.selector.select(STAGE2_TASK, fold_name),
            }
            selected_by_fold[fold_name] = {
                task: checkpoint.to_dict() for task, checkpoint in selected.items()
            }
            direct_predictions = self._run_inference(
                selected[DIRECT_TASK],
                config_by_task[DIRECT_TASK],
                FINAL_LABELS,
                files,
            )
            stage1_predictions = self._run_inference(
                selected[STAGE1_TASK],
                config_by_task[STAGE1_TASK],
                ("Normal", "Abnormal"),
                files,
            )
            stage2_predictions = self._run_inference(
                selected[STAGE2_TASK],
                config_by_task[STAGE2_TASK],
                ("Airway", "Lung_Parenchymal"),
                files,
            )

            fold_rows = self._build_fold_rows(
                fold_name=fold_name,
                manifest=manifest,
                selected=selected,
                direct_predictions=direct_predictions,
                stage1_predictions=stage1_predictions,
                stage2_predictions=stage2_predictions,
            )
            all_rows.extend(fold_rows)
            fold_metric_rows.append(
                self._build_fold_metric_row(fold_name, true_indices, fold_rows)
            )

        summary = self._build_summary(
            manifest=manifest,
            prediction_rows=all_rows,
            fold_metric_rows=fold_metric_rows,
            selected_by_fold=selected_by_fold,
        )
        ResultWriter(self.reports_dir, self.formats).write(
            prediction_rows=all_rows,
            fold_metric_rows=fold_metric_rows,
            summary=summary,
        )
        return summary

    def build_test_manifest(self) -> list[ManifestItem]:
        if not self.test_root.exists():
            raise FileNotFoundError(f"Test root not found: {self.test_root}")

        manifest: list[ManifestItem] = []
        for label in FINAL_LABELS:
            label_dir = self.test_root / label
            if not label_dir.exists():
                raise FileNotFoundError(
                    f"Required test label directory not found: {label_dir}"
                )
            for path in sorted(label_dir.glob("*.wav")):
                manifest.append(ManifestItem(audio_path=path, true_label=label))

        if not manifest:
            raise ValueError(f"No test .wav files found under {self.test_root}")
        self._validate_abnormal_overlay()
        self.logger.info(
            "Loaded test manifest with counts: %s",
            dict(Counter(item.true_label for item in manifest)),
        )
        return manifest

    def _validate_abnormal_overlay(self) -> None:
        abnormal_dir = self.test_root / "Abnormal"
        if not abnormal_dir.exists():
            raise FileNotFoundError(
                f"Required split-local Abnormal overlay not found: {abnormal_dir}"
            )
        subtype_names = {
            path.name
            for label in ("Airway", "Lung_Parenchymal")
            for path in (self.test_root / label).glob("*.wav")
        }
        abnormal_names = {path.name for path in abnormal_dir.glob("*.wav")}
        if subtype_names == abnormal_names:
            return
        missing = sorted(subtype_names - abnormal_names)
        extra = sorted(abnormal_names - subtype_names)
        raise ValueError(
            "Abnormal overlay does not match Airway + Lung_Parenchymal basenames. "
            f"missing={missing[:10]} extra={extra[:10]} "
            f"missing_count={len(missing)} extra_count={len(extra)}"
        )

    def _validate_config_labels(self, configs: Mapping[str, CvRunConfig]) -> None:
        expected = {
            DIRECT_TASK: FINAL_LABEL_TO_INDEX,
            STAGE1_TASK: {"Normal": 0, "Abnormal": 1},
            STAGE2_TASK: {"Airway": 0, "Lung_Parenchymal": 1},
        }
        for task, mapping in expected.items():
            actual = dict(configs[task].data.label_to_index)
            if actual != mapping:
                raise ValueError(
                    f"Unexpected label_to_index for {task}: {actual}; expected {mapping}"
                )

    def _run_inference(
        self,
        checkpoint: SelectedCheckpoint,
        config: CvRunConfig,
        label_order: tuple[str, ...],
        files: Sequence[Path],
    ) -> dict[str, ModelPrediction]:
        device = self.device_override or config.experiment.device
        runner = InferenceRunner(
            InferenceSpec(
                checkpoint=checkpoint,
                config=config,
                label_order=label_order,
                device=device,
            )
        )
        return runner.predict(files)

    def _build_fold_rows(
        self,
        *,
        fold_name: str,
        manifest: Sequence[ManifestItem],
        selected: Mapping[str, SelectedCheckpoint],
        direct_predictions: Mapping[str, ModelPrediction],
        stage1_predictions: Mapping[str, ModelPrediction],
        stage2_predictions: Mapping[str, ModelPrediction],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in manifest:
            key = str(item.audio_path)
            direct = direct_predictions[key]
            stage1 = stage1_predictions[key]
            stage2 = stage2_predictions[key]
            stage2_executed = stage1.predicted_label == "Abnormal"
            cascade_final = "Normal" if not stage2_executed else stage2.predicted_label
            cascade_error_type = self.resolve_cascade_error_type(
                item.true_label,
                stage1.predicted_label,
                stage2.predicted_label if stage2_executed else None,
            )
            stage1_normal = stage1.probabilities[0]
            stage1_abnormal = stage1.probabilities[1]
            stage2_airway = stage2.probabilities[0]
            stage2_lung = stage2.probabilities[1]
            cascade_probs = (
                stage1_normal,
                stage1_abnormal * stage2_airway,
                stage1_abnormal * stage2_lung,
            )
            rows.append(
                {
                    "fold": fold_name,
                    "audio_path": str(item.audio_path),
                    "true_label": item.true_label,
                    "direct_pred_label": direct.predicted_label,
                    "direct_correct": direct.predicted_label == item.true_label,
                    "direct_prob_normal": direct.probabilities[0],
                    "direct_prob_airway": direct.probabilities[1],
                    "direct_prob_lung_parenchymal": direct.probabilities[2],
                    "stage1_pred_label": stage1.predicted_label,
                    "stage1_prob_normal": stage1_normal,
                    "stage1_prob_abnormal": stage1_abnormal,
                    "stage1_threshold": stage1.threshold,
                    "stage1_threshold_source": stage1.threshold_source,
                    "stage2_executed": stage2_executed,
                    "stage2_pred_label": stage2.predicted_label
                    if stage2_executed
                    else "",
                    "stage2_prob_airway": stage2_airway if stage2_executed else "",
                    "stage2_prob_lung_parenchymal": stage2_lung
                    if stage2_executed
                    else "",
                    "stage2_threshold": stage2.threshold if stage2_executed else "",
                    "stage2_threshold_source": stage2.threshold_source
                    if stage2_executed
                    else "",
                    "cascade_prob_normal": cascade_probs[0],
                    "cascade_prob_airway": cascade_probs[1],
                    "cascade_prob_lung_parenchymal": cascade_probs[2],
                    "cascade_final_pred_label": cascade_final,
                    "cascade_correct": cascade_final == item.true_label,
                    "cascade_error_type": cascade_error_type,
                    "direct_checkpoint": str(selected[DIRECT_TASK].checkpoint_path),
                    "stage1_checkpoint": str(selected[STAGE1_TASK].checkpoint_path),
                    "stage2_checkpoint": str(selected[STAGE2_TASK].checkpoint_path),
                }
            )
        return rows

    @staticmethod
    def resolve_cascade_error_type(
        true_label: str,
        stage1_label: str,
        stage2_label: str | None,
    ) -> str:
        if true_label == "Normal":
            if stage1_label == "Normal":
                return "correct_normal"
            if stage2_label == "Airway":
                return "normal_false_abnormal_to_airway"
            if stage2_label == "Lung_Parenchymal":
                return "normal_false_abnormal_to_lung_parenchymal"
        if true_label == "Airway":
            if stage1_label == "Normal":
                return "airway_blocked_as_normal"
            if stage2_label == "Airway":
                return "airway_correct"
            if stage2_label == "Lung_Parenchymal":
                return "airway_to_lung_parenchymal"
        if true_label == "Lung_Parenchymal":
            if stage1_label == "Normal":
                return "lung_parenchymal_blocked_as_normal"
            if stage2_label == "Airway":
                return "lung_parenchymal_to_airway"
            if stage2_label == "Lung_Parenchymal":
                return "lung_parenchymal_correct"
        raise ValueError(
            "Unsupported cascade state: "
            f"true={true_label}, stage1={stage1_label}, stage2={stage2_label}"
        )

    def _build_fold_metric_row(
        self,
        fold_name: str,
        true_indices: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        direct_pred = np.asarray(
            [FINAL_LABEL_TO_INDEX[str(row["direct_pred_label"])] for row in rows],
            dtype=int,
        )
        cascade_pred = np.asarray(
            [
                FINAL_LABEL_TO_INDEX[str(row["cascade_final_pred_label"])]
                for row in rows
            ],
            dtype=int,
        )
        direct_prob = np.asarray(
            [
                [
                    float(row["direct_prob_normal"]),
                    float(row["direct_prob_airway"]),
                    float(row["direct_prob_lung_parenchymal"]),
                ]
                for row in rows
            ],
            dtype=float,
        )
        cascade_prob = np.asarray(
            [
                [
                    float(row["cascade_prob_normal"]),
                    float(row["cascade_prob_airway"]),
                    float(row["cascade_prob_lung_parenchymal"]),
                ]
                for row in rows
            ],
            dtype=float,
        )
        direct_metrics = compute_final_metrics(true_indices, direct_pred, direct_prob)
        cascade_metrics = compute_final_metrics(
            true_indices,
            cascade_pred,
            cascade_prob,
        )
        metric_row: dict[str, Any] = {"fold": fold_name}
        for metric in METRIC_NAMES:
            metric_row[f"direct_{metric}"] = direct_metrics["metrics"][metric]
            metric_row[f"cascade_{metric}"] = cascade_metrics["metrics"][metric]
        return metric_row

    def _build_summary(
        self,
        *,
        manifest: Sequence[ManifestItem],
        prediction_rows: Sequence[Mapping[str, Any]],
        fold_metric_rows: Sequence[Mapping[str, Any]],
        selected_by_fold: Mapping[str, Any],
    ) -> dict[str, Any]:
        true_indices = np.asarray(
            [FINAL_LABEL_TO_INDEX[str(row["true_label"])] for row in prediction_rows],
            dtype=int,
        )
        direct_pred = np.asarray(
            [
                FINAL_LABEL_TO_INDEX[str(row["direct_pred_label"])]
                for row in prediction_rows
            ],
            dtype=int,
        )
        cascade_pred = np.asarray(
            [
                FINAL_LABEL_TO_INDEX[str(row["cascade_final_pred_label"])]
                for row in prediction_rows
            ],
            dtype=int,
        )
        direct_prob = np.asarray(
            [
                [
                    float(row["direct_prob_normal"]),
                    float(row["direct_prob_airway"]),
                    float(row["direct_prob_lung_parenchymal"]),
                ]
                for row in prediction_rows
            ],
            dtype=float,
        )
        cascade_prob = np.asarray(
            [
                [
                    float(row["cascade_prob_normal"]),
                    float(row["cascade_prob_airway"]),
                    float(row["cascade_prob_lung_parenchymal"]),
                ]
                for row in prediction_rows
            ],
            dtype=float,
        )
        return {
            "results_root": str(self.results_root.resolve()),
            "test_root": str(self.test_root.resolve()),
            "reports_dir": str(self.reports_dir.resolve()),
            "labels": list(FINAL_LABELS),
            "folds": list(FOLD_NAMES),
            "test_label_counts": dict(Counter(item.true_label for item in manifest)),
            "prediction_rows": len(prediction_rows),
            "selected_checkpoints": selected_by_fold,
            "fold_metrics": list(fold_metric_rows),
            "metric_summary": summarize_fold_metrics(fold_metric_rows),
            "direct": compute_final_metrics(true_indices, direct_pred, direct_prob),
            "cascade": compute_final_metrics(
                true_indices,
                cascade_pred,
                cascade_prob,
            ),
            "cascade_stage_flow": build_stage_flow(prediction_rows),
            "cascade_error_counts": dict(
                Counter(str(row["cascade_error_type"]) for row in prediction_rows)
            ),
        }


class ResultWriter(PlotlyExportMixin):
    def __init__(self, reports_dir: str | Path, formats: set[str]):
        self.reports_dir = Path(reports_dir)
        self.formats = formats

    def write(
        self,
        *,
        prediction_rows: Sequence[Mapping[str, Any]],
        fold_metric_rows: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(
            prediction_rows,
            self.reports_dir / "cascade_test_predictions.csv",
        )
        self._write_csv(
            fold_metric_rows,
            self.reports_dir / "cascade_test_fold_metrics.csv",
        )
        summary_path = self.reports_dir / "cascade_test_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", summary_path)
        markdown_path = self.reports_dir / "cascade_test_summary.md"
        markdown_path.write_text(self._build_markdown(summary), encoding="utf-8")
        self.logger.info("Wrote %s", markdown_path)
        self._write_plots(summary)

    def _write_csv(self, rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
        if not rows:
            raise ValueError(f"Cannot write empty CSV: {out_path}")
        fieldnames = list(rows[0].keys())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.logger.info("Wrote %s", out_path)

    def _write_plots(self, summary: Mapping[str, Any]) -> None:
        direct = summary["direct"]
        cascade = summary["cascade"]
        if not isinstance(direct, Mapping) or not isinstance(cascade, Mapping):
            raise TypeError("summary direct/cascade blocks must be mappings")
        self.write_outputs(
            build_confusion_figure(
                direct["confusion_matrix"],
                direct["row_normalized_confusion_matrix"],
                "Direct 3-class row-normalized confusion matrix",
            ),
            self.reports_dir / "direct_confusion_matrix",
            formats=self.formats,
        )
        self.write_outputs(
            build_confusion_figure(
                cascade["confusion_matrix"],
                cascade["row_normalized_confusion_matrix"],
                "Cascade row-normalized confusion matrix",
            ),
            self.reports_dir / "cascade_confusion_matrix",
            formats=self.formats,
        )
        self.write_outputs(
            build_stage_flow_figure(summary["cascade_stage_flow"]),
            self.reports_dir / "cascade_stage_flow",
            formats=self.formats,
        )

    @staticmethod
    def _build_markdown(summary: Mapping[str, Any]) -> str:
        metric_summary = summary["metric_summary"]
        if not isinstance(metric_summary, Mapping):
            raise TypeError("metric_summary must be a mapping")
        direct = metric_summary["direct"]
        cascade = metric_summary["cascade"]
        if not isinstance(direct, Mapping) or not isinstance(cascade, Mapping):
            raise TypeError("direct/cascade metric summaries must be mappings")
        lines = [
            "# Disease-Group Direct vs Cascade Test Summary",
            "",
            f"- test root: `{summary['test_root']}`",
            f"- prediction rows: `{summary['prediction_rows']}`",
            f"- labels: `{', '.join(FINAL_LABELS)}`",
            "",
            "## Mean Fold Metrics",
            "",
            "| metric | direct mean +- std | cascade mean +- std |",
            "|---|---:|---:|",
        ]
        for metric in METRIC_NAMES:
            direct_metric = direct[metric]
            cascade_metric = cascade[metric]
            lines.append(
                "| {metric} | {dm:.6f} +- {ds:.6f} | {cm:.6f} +- {cs:.6f} |".format(
                    metric=metric,
                    dm=direct_metric["mean"],
                    ds=direct_metric["std"],
                    cm=cascade_metric["mean"],
                    cs=cascade_metric["std"],
                )
            )
        lines.extend(
            [
                "",
                "## Outputs",
                "",
                "- `cascade_test_predictions.csv`",
                "- `cascade_test_fold_metrics.csv`",
                "- `cascade_test_summary.json`",
                "- `direct_confusion_matrix.*`",
                "- `cascade_confusion_matrix.*`",
                "- `cascade_stage_flow.*`",
            ]
        )
        return "\n".join(lines) + "\n"


def compute_final_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    labels = list(range(len(FINAL_LABELS)))
    base_metrics = MetricsComputer.compute(y_true, y_pred, y_prob)
    cm = np.asarray(base_metrics.confusion_matrix, dtype=int)
    row_totals = cm.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals > 0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    metrics = {
        "accuracy": base_metrics.accuracy,
        "macro_precision": base_metrics.precision,
        "macro_recall": base_metrics.recall,
        "macro_f1": base_metrics.f1_score,
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": base_metrics.balanced_accuracy,
        "specificity": base_metrics.specificity,
        "roc_auc": base_metrics.roc_auc,
        "pr_auc": base_metrics.pr_auc,
        "brier_score": base_metrics.brier_score,
    }
    class_metrics = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(FINAL_LABELS)
    }
    return {
        "metrics": metrics,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "row_normalized_confusion_matrix": row_normalized.tolist(),
    }


def summarize_fold_metrics(
    fold_metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for approach in ("direct", "cascade"):
        summary[approach] = {}
        for metric in METRIC_NAMES:
            values = [
                float(row[f"{approach}_{metric}"])
                for row in fold_metric_rows
                if row[f"{approach}_{metric}"] is not None
            ]
            arr = np.asarray(values, dtype=float)
            summary[approach][metric] = {
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
                "n": float(arr.size),
            }
    return summary


def build_stage_flow(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    flow: dict[str, Counter[str]] = {label: Counter() for label in FINAL_LABELS}
    for row in rows:
        true_label = str(row["true_label"])
        stage1 = str(row["stage1_pred_label"])
        final = str(row["cascade_final_pred_label"])
        flow[true_label][f"stage1_{stage1}"] += 1
        if bool(row["stage2_executed"]):
            stage2 = str(row["stage2_pred_label"])
            flow[true_label]["stage2_executed"] += 1
            flow[true_label][f"stage2_{stage2}"] += 1
        else:
            flow[true_label]["stage2_not_executed"] += 1
        flow[true_label][f"final_{final}"] += 1
    return {label: dict(counts) for label, counts in flow.items()}


def build_confusion_figure(
    raw_matrix: object,
    normalized_matrix: object,
    title: str,
) -> go.Figure:
    raw = np.asarray(raw_matrix, dtype=int)
    normalized = np.asarray(normalized_matrix, dtype=float)
    hover = [
        [
            f"true={FINAL_LABELS[row]}<br>pred={FINAL_LABELS[col]}<br>"
            f"ratio={float(normalized[row, col]):.4f}<br>"
            f"count={int(raw[row, col])}"
            for col in range(raw.shape[1])
        ]
        for row in range(raw.shape[0])
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=normalized,
            x=list(FINAL_LABELS),
            y=list(FINAL_LABELS),
            text=[[f"{value:.2f}" for value in row] for row in normalized],
            texttemplate="%{text}",
            hovertext=hover,
            hoverinfo="text",
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "Row ratio"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        width=760,
        height=620,
    )
    return fig


def build_stage_flow_figure(stage_flow: object) -> go.Figure:
    if not isinstance(stage_flow, Mapping):
        raise TypeError("stage_flow must be a mapping")
    categories = [
        "stage1_Normal",
        "stage1_Abnormal",
        "stage2_executed",
        "stage2_not_executed",
        "stage2_Airway",
        "stage2_Lung_Parenchymal",
        "final_Normal",
        "final_Airway",
        "final_Lung_Parenchymal",
    ]
    fig = go.Figure()
    for label in FINAL_LABELS:
        counts_raw = stage_flow.get(label, {})
        if not isinstance(counts_raw, Mapping):
            raise TypeError(f"stage_flow[{label}] must be a mapping")
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[int(counts_raw.get(category, 0)) for category in categories],
                name=label,
            )
        )
    fig.update_layout(
        title="Cascade stage-flow counts by true label",
        xaxis_title="Flow bucket",
        yaxis_title="Count across folds",
        barmode="group",
        width=1200,
        height=620,
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def parse_formats(raw: str) -> set[str]:
    formats = {item.strip() for item in raw.split(",") if item.strip()}
    unsupported = formats - {"html", "png", "pdf"}
    if unsupported:
        raise ValueError(f"Unsupported output formats: {sorted(unsupported)}")
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="Disease_Group_Results")
    parser.add_argument(
        "--model-family",
        default=None,
        help=(
            "Optional model family subdirectory under results-root, "
            "for example AST or Whisper."
        ),
    )
    parser.add_argument("--test-root", default=DEFAULT_TEST_ROOT)
    parser.add_argument(
        "--reports-dir",
        default="Disease_Group_Results/reports/cascade_test_comparison",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override the device from each cv_run*.json config, e.g. cpu or mps.",
    )
    parser.add_argument(
        "--formats",
        default="html,png,pdf",
        help="Comma-separated output formats for Plotly figures: html,png,pdf",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if args.model_family:
        results_root = results_root / str(args.model_family)
    reports_dir = Path(args.reports_dir)
    enable_file_logging(reports_dir / "cascade_test_eval.log", mode="w")
    evaluator = CascadeComparisonEvaluator(
        results_root=results_root,
        test_root=args.test_root,
        reports_dir=reports_dir,
        device=args.device,
        formats=parse_formats(args.formats),
    )
    summary = evaluator.run()
    metric_summary = summary["metric_summary"]
    logger.info("Completed cascade comparison. Metric summary: %s", metric_summary)


if __name__ == "__main__":
    main()
