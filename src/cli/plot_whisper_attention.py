from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor

from src.data.audio import (
    AudioPreprocessConfig,
    WaveformPreprocessor,
    WhisperLikeLogMel,
)
from src.data.io import WaveformLoader
from src.models.classifier import HuggingFaceClassifier
from src.models.whisper_model import RespiratoryWhisperModel, WhisperModelConfig
from src.plots.export import PlotlyExportMixin
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import DataConfig, JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger

AttentionMethod = Literal[
    "last_time_attention",
    "last_time_attention_head_mean",
    "time_attention_rollout",
    "class_gradient_time_attention",
]

ATTENTION_METHODS: tuple[AttentionMethod, ...] = (
    "last_time_attention",
    "last_time_attention_head_mean",
    "time_attention_rollout",
    "class_gradient_time_attention",
)
SUPPORTED_FORMATS = {"html", "png", "pdf", "json"}
DEFAULT_OUT_DIR = "Disease_Group_Results/reports/whisper_attention_visualization"


@dataclass(frozen=True)
class AttentionCliConfig:
    checkpoint: Path
    wav: Path
    out_dir: Path
    attention_method: AttentionMethod
    target_class: str
    head: str
    top_k: int
    formats: set[str]
    device_override: str | None
    install_chrome: bool


@dataclass(frozen=True)
class LoadedCheckpoint:
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    data_cfg: DataConfig
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    device: torch.device


@dataclass(frozen=True)
class AttentionInputs:
    wav_path: Path
    log_mel: Tensor
    input_values: Tensor


@dataclass(frozen=True)
class PredictionSummary:
    logits: list[float]
    probabilities: dict[str, float]
    predicted_label: str
    predicted_index: int
    predicted_probability: float
    target_label: str
    target_index: int


@dataclass(frozen=True)
class AttentionRunResult:
    prediction: PredictionSummary
    attentions: tuple[Tensor, ...]
    logits: Tensor
    target_attention: Tensor | None


@dataclass(frozen=True)
class TimeTokenGeometry:
    num_mel_bins: int
    num_frames: int
    n_audio_ctx: int
    sample_rate: int
    hop_length: int
    prefix_tokens: int = 0

    @property
    def token_count(self) -> int:
        return self.prefix_tokens + self.n_audio_ctx


@dataclass(frozen=True)
class TimeTokenRecord:
    rank: int
    token_index: int
    attention_token_index: int
    frame_start: int
    frame_end: int
    time_start_sec: float
    time_end_sec: float
    attention_score: float
    normalized_attention_score: float


@dataclass(frozen=True)
class AttentionScoreResult:
    method: AttentionMethod
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    top_time_spans: list[TimeTokenRecord]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FigureBundle:
    log_mel: go.Figure
    attention_time_curve: go.Figure
    temporal_overlay: go.Figure
    top_time_spans_overlay: go.Figure


class WhisperAttentionInputLoader(LoggingMixin):
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg
        self.waveform_loader = WaveformLoader(data_cfg.audio.sample_rate)
        preprocess_cfg = self.build_audio_preprocess_config(data_cfg)
        self.preprocessor = WaveformPreprocessor(preprocess_cfg)
        self.feature_extractor = WhisperLikeLogMel(preprocess_cfg)

    def load(self, wav_path: Path) -> AttentionInputs:
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file does not exist: {wav_path}")
        if wav_path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a .wav file, got: {wav_path}")
        waveform = self.waveform_loader.load(wav_path)
        prepared = self.preprocessor.prepare(waveform)
        log_mel = self.feature_extractor(prepared).contiguous()
        input_values = log_mel.contiguous().unsqueeze(0)
        self.logger.info("Loaded %s | log_mel_shape=%s", wav_path, tuple(log_mel.shape))
        return AttentionInputs(
            wav_path=wav_path,
            log_mel=log_mel,
            input_values=input_values,
        )

    @staticmethod
    def build_audio_preprocess_config(cfg: DataConfig) -> AudioPreprocessConfig:
        return AudioPreprocessConfig(
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


class WhisperAttentionModelRunner(LoggingMixin):
    def __init__(self, loaded: LoadedCheckpoint):
        self.loaded = loaded
        model_cfg_raw = loaded.checkpoint.get("model_cfg")
        if model_cfg_raw is None:
            raise RuntimeError(
                f"Checkpoint missing model_cfg: {loaded.checkpoint_path}"
            )
        model_cfg = parse_model_cfg(model_cfg_raw)
        if not isinstance(model_cfg, WhisperModelConfig):
            raise ValueError(
                "Whisper attention visualization supports only Whisper checkpoints"
            )
        self.model_cfg = model_cfg
        self.model = RespiratoryWhisperModel(self.model_cfg)
        self.model.load_state_dict(loaded.checkpoint["model_state_dict"])
        self.model.to(loaded.device)
        self.model.eval()
        self._validate_frontend_dims(loaded.data_cfg)

    @property
    def prefix_token_count(self) -> int:
        if self.model_cfg.head_type == "hf" and self.model_cfg.pooling == "cls":
            return 1
        return 0

    def run(
        self,
        input_values: Tensor,
        *,
        attention_method: AttentionMethod,
        target_class: str,
    ) -> AttentionRunResult:
        model_input = input_values.to(self.loaded.device)
        target_attention: Tensor | None = None

        if attention_method == "class_gradient_time_attention":
            self.model.zero_grad(set_to_none=True)
            model_input = model_input.detach().requires_grad_(True)
            attentions, logits = self._forward_encoder(model_input)
            target_index = self._resolve_target_index(target_class, logits)
            target_attention = attentions[-1]
            if not target_attention.requires_grad:
                raise RuntimeError(
                    "Whisper attention tensor does not require gradients. "
                    "Class-gradient attention requires a differentiable forward pass."
                )
            target_attention.retain_grad()
            target_score = self._target_score(logits, target_index)
            target_score.backward()
        else:
            with torch.no_grad():
                attentions, logits = self._forward_encoder(model_input)
            target_index = self._resolve_target_index(target_class, logits)

        prediction = self._prediction_summary(
            logits.detach(), target_class, target_index
        )
        return AttentionRunResult(
            prediction=prediction,
            attentions=tuple(attention.detach() for attention in attentions),
            logits=logits.detach(),
            target_attention=target_attention,
        )

    def _forward_encoder(
        self, input_values: Tensor
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        if self.model_cfg.head_type == "hf":
            if not isinstance(self.model.classifier, HuggingFaceClassifier):
                raise RuntimeError("Expected HuggingFaceClassifier for head_type='hf'")
            prefix_tokens = self.model.classifier.build_prefix_tokens(
                input_values.shape[0],
                device=input_values.device,
                dtype=input_values.dtype,
            )
            features = self.model.encoder(
                input_values,
                output_hidden_states=self.model_cfg.use_weighted_layer_sum,
                output_attentions=True,
                prefix_tokens=prefix_tokens,
            )
            projected = self.model.classifier.project_sequence(
                features.last_hidden_state,
                features.hidden_states,
            )
            pooled_embedding = self.model.classifier.pool_projected(projected)
            logits = self.model.classifier.classifier(pooled_embedding)
        else:
            features = self.model.encoder(
                input_values,
                output_hidden_states=False,
                output_attentions=True,
            )
            pooled_embedding = features.last_hidden_state.mean(dim=1)
            logits = self.model.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        attentions = self._require_attentions(features.attentions)
        return attentions, logits

    def _require_attentions(
        self, attentions: tuple[Tensor, ...] | None
    ) -> tuple[Tensor, ...]:
        if not attentions:
            raise RuntimeError("Whisper encoder did not return attention tensors")
        return tuple(attentions)

    def _prediction_summary(
        self,
        logits: Tensor,
        target_class: str,
        target_index: int,
    ) -> PredictionSummary:
        probabilities = self._probability_tensor(logits)
        pred_index = int(probabilities.argmax(dim=-1).reshape(-1)[0].item())
        predicted_label = self.loaded.index_to_label[pred_index]
        target_label = self.loaded.index_to_label[target_index]
        label_probabilities = {
            self.loaded.index_to_label[index]: float(
                probabilities.reshape(1, -1)[0, index].item()
            )
            for index in sorted(self.loaded.index_to_label)
        }
        del target_class
        return PredictionSummary(
            logits=[float(item) for item in logits.detach().cpu().reshape(-1).tolist()],
            probabilities=label_probabilities,
            predicted_label=predicted_label,
            predicted_index=pred_index,
            predicted_probability=label_probabilities[predicted_label],
            target_label=target_label,
            target_index=target_index,
        )

    def _probability_tensor(self, logits: Tensor) -> Tensor:
        if self.model_cfg.num_classes == 2:
            positive = torch.sigmoid(logits.reshape(-1, 1))
            return torch.cat([1.0 - positive, positive], dim=1)
        return torch.softmax(logits.reshape(1, -1), dim=-1)

    def _target_score(self, logits: Tensor, target_index: int) -> Tensor:
        if self.model_cfg.num_classes == 2:
            raw_logit = logits.reshape(-1)[0]
            return raw_logit if target_index == 1 else -raw_logit
        return logits.reshape(1, -1)[0, target_index]

    def _resolve_target_index(self, target_class: str, logits: Tensor) -> int:
        if target_class == "predicted":
            probabilities = self._probability_tensor(logits.detach())
            return int(probabilities.argmax(dim=-1).reshape(-1)[0].item())
        if target_class in self.loaded.label_to_index:
            return int(self.loaded.label_to_index[target_class])
        try:
            target_index = int(target_class)
        except ValueError as exc:
            raise ValueError(
                f"Unknown --target-class={target_class!r}; expected 'predicted', "
                f"a label name from {sorted(self.loaded.label_to_index)}, or an integer index"
            ) from exc
        if target_index not in self.loaded.index_to_label:
            raise ValueError(
                f"Invalid --target-class index={target_index}; "
                f"known indices are {sorted(self.loaded.index_to_label)}"
            )
        return target_index

    def _validate_frontend_dims(self, data_cfg: DataConfig) -> None:
        if data_cfg.preprocessing.feature_type != "log_mel":
            raise ValueError(
                "Whisper attention visualization requires log_mel frontend"
            )
        preprocess_cfg = WhisperAttentionInputLoader.build_audio_preprocess_config(
            data_cfg
        )
        if self.model_cfg.encoder.n_mels != preprocess_cfg.n_mels:
            raise ValueError(
                "Checkpoint Whisper n_mels does not match run_config.data preprocessing dims.\n"
                f"- checkpoint: n_mels={self.model_cfg.encoder.n_mels}\n"
                f"- run_config: n_mels={preprocess_cfg.n_mels}\n"
                f"- checkpoint_path: {self.loaded.checkpoint_path}"
            )
        if self.model_cfg.encoder.n_audio_ctx != preprocess_cfg.n_audio_ctx:
            raise ValueError(
                "Checkpoint Whisper n_audio_ctx does not match log-mel frame count.\n"
                f"- checkpoint: n_audio_ctx={self.model_cfg.encoder.n_audio_ctx}\n"
                f"- run_config: n_audio_ctx={preprocess_cfg.n_audio_ctx}, "
                f"n_frames={preprocess_cfg.n_frames}\n"
                f"- checkpoint_path: {self.loaded.checkpoint_path}"
            )


class WhisperTimeTokenMapper:
    def __init__(self, geometry: TimeTokenGeometry):
        self.geometry = geometry

    @classmethod
    def from_runner(cls, runner: WhisperAttentionModelRunner) -> WhisperTimeTokenMapper:
        preprocess_cfg = WhisperAttentionInputLoader.build_audio_preprocess_config(
            runner.loaded.data_cfg
        )
        return cls(
            TimeTokenGeometry(
                num_mel_bins=preprocess_cfg.n_mels,
                num_frames=preprocess_cfg.n_frames,
                n_audio_ctx=runner.model_cfg.encoder.n_audio_ctx,
                sample_rate=preprocess_cfg.sample_rate,
                hop_length=preprocess_cfg.hop_length,
                prefix_tokens=runner.prefix_token_count,
            )
        )

    @property
    def audio_slice(self) -> slice:
        start = self.geometry.prefix_tokens
        return slice(start, start + self.geometry.n_audio_ctx)

    def validate_token_count(self, attention: Tensor) -> None:
        token_count = int(attention.shape[-1])
        query_count = int(attention.shape[-2])
        expected = self.geometry.token_count
        if token_count != expected or query_count != expected:
            raise ValueError(
                "Attention token count does not match Whisper time-token geometry. "
                f"attention_shape={tuple(attention.shape)}, expected_tokens={expected}, "
                f"geometry={asdict(self.geometry)}"
            )

    def token_record(
        self,
        *,
        rank: int,
        token_index: int,
        attention_score: float,
        normalized_attention_score: float,
    ) -> TimeTokenRecord:
        frame_start, frame_end = self.frame_range(token_index)
        return TimeTokenRecord(
            rank=rank,
            token_index=token_index,
            attention_token_index=token_index + self.geometry.prefix_tokens,
            frame_start=frame_start,
            frame_end=frame_end,
            time_start_sec=self.frame_to_seconds(frame_start),
            time_end_sec=self.frame_to_seconds(frame_end),
            attention_score=attention_score,
            normalized_attention_score=normalized_attention_score,
        )

    def frame_range(self, token_index: int) -> tuple[int, int]:
        if token_index < 0 or token_index >= self.geometry.n_audio_ctx:
            raise ValueError(
                f"token_index out of range 0..{self.geometry.n_audio_ctx - 1}: "
                f"{token_index}"
            )
        frames_per_token = self.geometry.num_frames / float(self.geometry.n_audio_ctx)
        start = int(math.floor(token_index * frames_per_token))
        end = int(math.ceil((token_index + 1) * frames_per_token))
        start = max(0, min(start, self.geometry.num_frames - 1))
        end = max(start + 1, min(end, self.geometry.num_frames))
        return start, end

    def frame_to_seconds(self, frame: int) -> float:
        return float(frame * self.geometry.hop_length) / float(
            self.geometry.sample_rate
        )

    def time_axis(self) -> np.ndarray:
        frames = np.arange(self.geometry.num_frames, dtype=np.float64)
        return frames * (self.geometry.hop_length / float(self.geometry.sample_rate))

    def token_center_seconds(self) -> np.ndarray:
        return np.asarray(
            [
                (self.frame_to_seconds(start) + self.frame_to_seconds(end)) / 2.0
                for start, end in (
                    self.frame_range(index)
                    for index in range(self.geometry.n_audio_ctx)
                )
            ],
            dtype=np.float64,
        )

    def top_time_spans(
        self,
        raw_scores: np.ndarray,
        normalized_scores: np.ndarray,
        *,
        top_k: int,
    ) -> list[TimeTokenRecord]:
        limit = min(top_k, raw_scores.size)
        order = np.argsort(raw_scores)[::-1][:limit]
        return [
            self.token_record(
                rank=rank,
                token_index=int(index),
                attention_score=float(raw_scores[index]),
                normalized_attention_score=float(normalized_scores[index]),
            )
            for rank, index in enumerate(order, start=1)
        ]

    def upsample_scores(self, normalized_scores: np.ndarray) -> np.ndarray:
        if normalized_scores.shape != (self.geometry.n_audio_ctx,):
            raise ValueError(
                "Expected one attention score per Whisper audio token, got "
                f"shape={normalized_scores.shape}"
            )
        frame_scores = np.zeros(self.geometry.num_frames, dtype=np.float64)
        for token_index, score in enumerate(normalized_scores):
            start, end = self.frame_range(token_index)
            frame_scores[start:end] = float(score)
        return np.tile(frame_scores[None, :], (self.geometry.num_mel_bins, 1))


class AttentionScoreExtractor:
    def __init__(self, mapper: WhisperTimeTokenMapper):
        self.mapper = mapper

    def extract(
        self,
        run_result: AttentionRunResult,
        *,
        method: AttentionMethod,
        head: str,
        top_k: int,
    ) -> AttentionScoreResult:
        if method == "last_time_attention":
            raw_scores, metadata = self._last_time_attention(
                run_result.attentions,
                head=head,
            )
        elif method == "last_time_attention_head_mean":
            raw_scores, metadata = self._last_time_attention_head_mean(
                run_result.attentions
            )
        elif method == "time_attention_rollout":
            raw_scores, metadata = self._time_attention_rollout(run_result.attentions)
        elif method == "class_gradient_time_attention":
            raw_scores, metadata = self._class_gradient_time_attention(run_result)
        else:
            raise ValueError(f"Unsupported attention method: {method}")

        normalized_scores = self._normalize(raw_scores)
        top_time_spans = self.mapper.top_time_spans(
            raw_scores,
            normalized_scores,
            top_k=top_k,
        )
        return AttentionScoreResult(
            method=method,
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            top_time_spans=top_time_spans,
            metadata=metadata,
        )

    def _last_time_attention(
        self,
        attentions: tuple[Tensor, ...],
        *,
        head: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        last_attention = attentions[-1]
        self.mapper.validate_token_count(last_attention)
        head_index = self._parse_head(head, num_heads=int(last_attention.shape[1]))
        scores = last_attention[0, head_index, self.mapper.audio_slice, :][
            :,
            self.mapper.audio_slice,
        ].mean(dim=0)
        return (
            self._to_numpy_scores(scores),
            {
                "layer": len(attentions) - 1,
                "head": head_index,
                "head_aggregation": "single",
                "query_aggregation": "audio_time_mean",
            },
        )

    def _last_time_attention_head_mean(
        self,
        attentions: tuple[Tensor, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        last_attention = attentions[-1]
        self.mapper.validate_token_count(last_attention)
        scores = (
            last_attention[0, :, self.mapper.audio_slice, :][
                :,
                :,
                self.mapper.audio_slice,
            ]
            .mean(dim=0)
            .mean(dim=0)
        )
        return (
            self._to_numpy_scores(scores),
            {
                "layer": len(attentions) - 1,
                "head": "mean",
                "head_aggregation": "mean",
                "query_aggregation": "audio_time_mean",
            },
        )

    def _time_attention_rollout(
        self,
        attentions: tuple[Tensor, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rollout: Tensor | None = None
        for attention in attentions:
            self.mapper.validate_token_count(attention)
            matrix = attention[0].mean(dim=0)
            matrix = matrix + torch.eye(
                matrix.shape[0],
                device=matrix.device,
                dtype=matrix.dtype,
            )
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(matrix.dtype).eps
            )
            rollout = matrix if rollout is None else matrix @ rollout
        if rollout is None:
            raise RuntimeError(
                "Cannot compute attention rollout without attention tensors"
            )
        scores = rollout[self.mapper.audio_slice, :][
            :,
            self.mapper.audio_slice,
        ].mean(dim=0)
        return (
            self._to_numpy_scores(scores),
            {
                "layers": len(attentions),
                "head": "mean",
                "residual": True,
                "row_normalized": True,
                "query_aggregation": "audio_time_mean",
            },
        )

    def _class_gradient_time_attention(
        self,
        run_result: AttentionRunResult,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if run_result.target_attention is None:
            raise RuntimeError(
                "class_gradient_time_attention requires target attention from a gradient run"
            )
        target_attention = run_result.target_attention
        self.mapper.validate_token_count(target_attention)
        gradient = target_attention.grad
        if gradient is None:
            raise RuntimeError("Attention gradient is missing after backward pass")
        weighted = torch.relu(target_attention * gradient)
        scores = (
            weighted[0, :, self.mapper.audio_slice, :][
                :,
                :,
                self.mapper.audio_slice,
            ]
            .mean(dim=0)
            .mean(dim=0)
        )
        return (
            self._to_numpy_scores(scores),
            {
                "layer": len(run_result.attentions) - 1,
                "head": "mean",
                "gradient_weighting": "relu(attention * gradient)",
                "query_aggregation": "audio_time_mean",
            },
        )

    @staticmethod
    def _parse_head(head: str, *, num_heads: int) -> int:
        if head == "mean":
            raise ValueError(
                "--head=mean is not valid with --method last_time_attention; "
                "use --method last_time_attention_head_mean instead"
            )
        try:
            head_index = int(head)
        except ValueError as exc:
            raise ValueError(
                f"--head must be an integer head index, got {head!r}"
            ) from exc
        if head_index < 0 or head_index >= num_heads:
            raise ValueError(
                f"--head index {head_index} out of range 0..{num_heads - 1}"
            )
        return head_index

    @staticmethod
    def _to_numpy_scores(scores: Tensor) -> np.ndarray:
        values = scores.detach().cpu().to(torch.float64).numpy()
        if values.ndim != 1:
            raise ValueError(
                f"Expected flat Whisper time-token scores, got shape={values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Attention scores contain non-finite values")
        return values

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            raise ValueError("Cannot normalize empty attention scores")
        minimum = float(scores.min())
        maximum = float(scores.max())
        if math.isclose(maximum, minimum):
            return np.zeros_like(scores, dtype=np.float64)
        return ((scores - minimum) / (maximum - minimum)).astype(np.float64)


class AttentionFigureBuilder:
    def __init__(self, mapper: WhisperTimeTokenMapper):
        self.mapper = mapper

    def build(
        self,
        *,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> FigureBundle:
        return FigureBundle(
            log_mel=self.log_mel_figure(inputs, score_result, prediction),
            attention_time_curve=self.attention_time_curve(
                inputs,
                score_result,
                prediction,
            ),
            temporal_overlay=self.temporal_overlay(inputs, score_result, prediction),
            top_time_spans_overlay=self.top_time_spans_overlay(
                inputs,
                score_result,
                prediction,
            ),
        )

    def log_mel_figure(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = self._base_log_mel_figure(inputs)
        fig.update_layout(
            title=self._title("Whisper log-mel", inputs, score_result, prediction)
        )
        return fig

    def attention_time_curve(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.mapper.token_center_seconds(),
                y=score_result.normalized_scores,
                mode="lines",
                line={"color": "rgb(230,92,0)", "width": 2},
                name="normalized attention",
                hovertemplate=(
                    "time=%{x:.3f}s<br>normalized_attention=%{y:.6f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=self._title(
                "Whisper time-token attention curve",
                inputs,
                score_result,
                prediction,
            ),
            xaxis_title="Time (s)",
            yaxis_title="Normalized attention",
            width=1300,
            height=420,
        )
        return fig

    def temporal_overlay(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = self._base_log_mel_figure(inputs)
        attention_map = self.mapper.upsample_scores(score_result.normalized_scores)
        fig.add_trace(
            go.Heatmap(
                z=attention_map,
                x=self.mapper.time_axis(),
                y=self._mel_axis(attention_map.shape[0]),
                colorscale="Inferno",
                opacity=0.48,
                colorbar={"title": "attention"},
                hovertemplate=(
                    "time=%{x:.3f}s<br>mel_bin=%{y}<br>"
                    "normalized_attention=%{z:.6f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=self._title(
                "Whisper temporal attention overlay",
                inputs,
                score_result,
                prediction,
            )
        )
        return fig

    def top_time_spans_overlay(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = self._base_log_mel_figure(inputs)
        for span in score_result.top_time_spans:
            color = self._rank_color(span.normalized_attention_score)
            fig.add_shape(
                type="rect",
                x0=span.time_start_sec,
                x1=span.time_end_sec,
                y0=0,
                y1=self.mapper.geometry.num_mel_bins - 1,
                line={"color": color, "width": 2},
                fillcolor=color.replace("1.0)", "0.22)"),
            )
            fig.add_annotation(
                x=(span.time_start_sec + span.time_end_sec) / 2.0,
                y=self.mapper.geometry.num_mel_bins - 1,
                text=f"#{span.rank}<br>{span.normalized_attention_score:.2f}",
                showarrow=False,
                font={"size": 10, "color": "white"},
                bgcolor="rgba(0,0,0,0.55)",
                borderpad=2,
            )
        fig.update_layout(
            title=self._title(
                "Whisper top time-span overlay",
                inputs,
                score_result,
                prediction,
            )
        )
        return fig

    def _base_log_mel_figure(self, inputs: AttentionInputs) -> go.Figure:
        log_mel = inputs.log_mel.detach().cpu().numpy()
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=log_mel,
                    x=self.mapper.time_axis(),
                    y=self._mel_axis(log_mel.shape[0]),
                    colorscale="Viridis",
                    colorbar={"title": "log-mel"},
                    hovertemplate=(
                        "time=%{x:.3f}s<br>mel_bin=%{y}<br>"
                        "value=%{z:.6f}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Mel bin",
            width=1300,
            height=720,
        )
        return fig

    def _title(
        self,
        prefix: str,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> str:
        return (
            f"{prefix} | file={inputs.wav_path.name} | method={score_result.method} | "
            f"pred={prediction.predicted_label} ({prediction.predicted_probability:.4f}) | "
            f"target={prediction.target_label}"
        )

    @staticmethod
    def _mel_axis(length: int) -> np.ndarray:
        return np.arange(length, dtype=np.int32)

    @staticmethod
    def _rank_color(score: float) -> str:
        clipped = max(0.0, min(1.0, score))
        red = 255
        green = int(210 - 150 * clipped)
        blue = int(40 - 40 * clipped)
        return f"rgba({red},{green},{blue},1.0)"


class AttentionResultWriter(PlotlyExportMixin):
    def __init__(self, out_dir: Path, formats: set[str], install_chrome: bool = False):
        self.out_dir = out_dir
        self.formats = formats
        self.install_chrome = install_chrome

    def write(
        self,
        *,
        inputs: AttentionInputs,
        loaded: LoadedCheckpoint,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
        figures: FigureBundle,
        cli_config: AttentionCliConfig,
        geometry: TimeTokenGeometry,
    ) -> dict[str, Any]:
        Fs.ensure_dir(self.out_dir)
        stem = self._safe_stem(inputs.wav_path)
        outputs: dict[str, list[str] | str] = {}
        outputs["log_mel"] = self._write_figure(
            figures.log_mel,
            self.out_dir / f"{stem}_log_mel",
        )
        outputs["attention_time_curve"] = self._write_figure(
            figures.attention_time_curve,
            self.out_dir / f"{stem}_attention_time_curve",
        )
        outputs["temporal_overlay"] = self._write_figure(
            figures.temporal_overlay,
            self.out_dir / f"{stem}_attention_temporal_overlay",
        )
        outputs["top_time_spans_overlay"] = self._write_figure(
            figures.top_time_spans_overlay,
            self.out_dir / f"{stem}_top_time_spans_overlay",
        )
        top_spans_path = self.out_dir / f"{stem}_top_time_spans.csv"
        metadata_path = self.out_dir / f"{stem}_attention_metadata.json"
        self._write_top_time_spans(top_spans_path, score_result.top_time_spans)
        metadata = self._metadata(
            inputs=inputs,
            loaded=loaded,
            score_result=score_result,
            prediction=prediction,
            cli_config=cli_config,
            geometry=geometry,
            outputs=outputs,
            top_spans_path=top_spans_path,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.logger.info("Wrote %s", top_spans_path)
        self.logger.info("Wrote %s", metadata_path)
        return metadata

    def _write_figure(self, fig: go.Figure, out_base: Path) -> list[str]:
        return [
            str(path)
            for path in self.write_outputs(
                fig,
                out_base,
                formats=self.formats,
                install_chrome=self.install_chrome,
            )
        ]

    @staticmethod
    def _safe_stem(path: Path) -> str:
        return "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem
        )

    @staticmethod
    def _write_top_time_spans(
        path: Path,
        spans: Sequence[TimeTokenRecord],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "token_index",
                    "attention_token_index",
                    "frame_start",
                    "frame_end",
                    "time_start_sec",
                    "time_end_sec",
                    "attention_score",
                    "normalized_attention_score",
                ],
            )
            writer.writeheader()
            for span in spans:
                writer.writerow(asdict(span))

    @staticmethod
    def _metadata(
        *,
        inputs: AttentionInputs,
        loaded: LoadedCheckpoint,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
        cli_config: AttentionCliConfig,
        geometry: TimeTokenGeometry,
        outputs: Mapping[str, list[str] | str],
        top_spans_path: Path,
    ) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "wav": str(inputs.wav_path),
            "checkpoint": str(loaded.checkpoint_path),
            "device": str(loaded.device),
            "attention_method": score_result.method,
            "attention_metadata": score_result.metadata,
            "target_class": cli_config.target_class,
            "prediction": asdict(prediction),
            "time_token_geometry": asdict(geometry),
            "top_k": cli_config.top_k,
            "top_time_spans_csv": str(top_spans_path),
            "outputs": outputs,
        }


class WhisperAttentionVisualizer(LoggingMixin):
    def __init__(self, config: AttentionCliConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        loaded = load_attention_checkpoint(
            self.config.checkpoint,
            self.config.device_override,
        )
        inputs = WhisperAttentionInputLoader(loaded.data_cfg).load(self.config.wav)
        runner = WhisperAttentionModelRunner(loaded)
        run_result = runner.run(
            inputs.input_values,
            attention_method=self.config.attention_method,
            target_class=self.config.target_class,
        )
        mapper = WhisperTimeTokenMapper.from_runner(runner)
        extractor = AttentionScoreExtractor(mapper)
        score_result = extractor.extract(
            run_result,
            method=self.config.attention_method,
            head=self.config.head,
            top_k=self.config.top_k,
        )
        figures = AttentionFigureBuilder(mapper).build(
            inputs=inputs,
            score_result=score_result,
            prediction=run_result.prediction,
        )
        metadata = AttentionResultWriter(
            self.config.out_dir,
            self.config.formats,
            install_chrome=self.config.install_chrome,
        ).write(
            inputs=inputs,
            loaded=loaded,
            score_result=score_result,
            prediction=run_result.prediction,
            figures=figures,
            cli_config=self.config,
            geometry=mapper.geometry,
        )
        self.logger.info(
            "Generated Whisper attention visualization | wav=%s | method=%s | out_dir=%s",
            self.config.wav,
            self.config.attention_method,
            self.config.out_dir,
        )
        return metadata


def load_attention_checkpoint(
    checkpoint_path: Path,
    device_override: str | None,
) -> LoadedCheckpoint:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    device = _resolve_device(checkpoint_path, device_override)
    checkpoint = load_checkpoint(str(checkpoint_path), device=device)
    data_cfg = _load_data_config(checkpoint, checkpoint_path)
    label_to_index = _load_label_to_index(checkpoint, checkpoint_path)
    model_cfg_raw = checkpoint.get("model_cfg")
    if model_cfg_raw is None:
        raise RuntimeError(f"Checkpoint missing model_cfg: {checkpoint_path}")
    model_cfg = parse_model_cfg(model_cfg_raw)
    if not isinstance(model_cfg, WhisperModelConfig):
        raise ValueError(
            "Whisper attention visualization supports only Whisper checkpoints"
        )
    if len(label_to_index) != model_cfg.num_classes:
        raise ValueError(
            "checkpoint label_to_index size does not match model_cfg.num_classes. "
            f"labels={label_to_index}, num_classes={model_cfg.num_classes}"
        )
    index_to_label = {index: label for label, index in label_to_index.items()}
    expected_indices = list(range(model_cfg.num_classes))
    if sorted(index_to_label) != expected_indices:
        raise ValueError(
            f"checkpoint label_to_index must be contiguous {expected_indices}; got {label_to_index}"
        )
    return LoadedCheckpoint(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        data_cfg=data_cfg,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        device=device,
    )


def _resolve_device(checkpoint_path: Path, device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    run_config = _read_checkpoint_run_config_for_device(checkpoint_path)
    device_raw = run_config.get("experiment", {}).get("device") if run_config else None
    if isinstance(device_raw, str) and _device_available(device_raw):
        return torch.device(device_raw)
    return torch.device("cpu")


def _read_checkpoint_run_config_for_device(
    checkpoint_path: Path,
) -> Mapping[str, Any] | None:
    try:
        checkpoint = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    except Exception:
        return None
    run_config = checkpoint.get("run_config")
    return run_config if isinstance(run_config, Mapping) else None


def _device_available(device_name: str) -> bool:
    if device_name == "cpu":
        return True
    if device_name == "cuda":
        return torch.cuda.is_available()
    if device_name == "mps":
        return bool(torch.backends.mps.is_available())
    if device_name.startswith("cuda:"):
        return torch.cuda.is_available()
    return False


def _load_data_config(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> DataConfig:
    run_config = checkpoint.get("run_config")
    if not isinstance(run_config, Mapping):
        raise RuntimeError(f"Checkpoint missing run_config mapping: {checkpoint_path}")
    data_raw = run_config.get("data")
    if isinstance(data_raw, DataConfig):
        return data_raw
    if not isinstance(data_raw, Mapping):
        raise RuntimeError(
            f"Checkpoint missing run_config.data mapping: {checkpoint_path}"
        )
    return JsonConfigLoader._parse_data(data_raw)


def _load_label_to_index(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, int]:
    raw = checkpoint.get("label_to_index")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"Checkpoint missing label_to_index mapping: {checkpoint_path}"
        )
    label_to_index = {str(label): int(index) for label, index in raw.items()}
    if len(label_to_index) < 2:
        raise ValueError(
            f"checkpoint label_to_index must contain at least two classes: {label_to_index}"
        )
    return label_to_index


def parse_formats(formats_arg: str | Sequence[str]) -> set[str]:
    raw_items = [formats_arg] if isinstance(formats_arg, str) else list(formats_arg)
    formats = {
        part.strip()
        for item in raw_items
        for part in str(item).split(",")
        if part.strip()
    }
    unsupported = formats - SUPPORTED_FORMATS
    if unsupported:
        raise ValueError(f"Unsupported format(s): {sorted(unsupported)}")
    if not formats:
        raise ValueError("--format/--formats must include at least one output format")
    return formats


def parse_args(argv: Sequence[str] | None = None) -> AttentionCliConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav", "--audio", dest="wav", required=True)
    parser.add_argument(
        "--out-dir", "--output-dir", dest="out_dir", default=DEFAULT_OUT_DIR
    )
    parser.add_argument(
        "--method",
        "--attention-method",
        dest="attention_method",
        default="last_time_attention_head_mean",
        choices=ATTENTION_METHODS,
    )
    parser.add_argument("--target-class", default="predicted")
    parser.add_argument("--head", default="0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--format",
        "--formats",
        dest="formats",
        action="append",
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.attention_method != "last_time_attention" and args.head != "0":
        raise ValueError("--head is only supported with --method last_time_attention")

    return AttentionCliConfig(
        checkpoint=Path(args.checkpoint),
        wav=Path(args.wav),
        out_dir=Path(args.out_dir),
        attention_method=args.attention_method,
        target_class=str(args.target_class),
        head=str(args.head),
        top_k=int(args.top_k),
        formats=parse_formats(args.formats or ["html,json"]),
        device_override=args.device,
        install_chrome=bool(args.install_chrome),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    Fs.ensure_dir(config.out_dir)
    enable_file_logging(config.out_dir / "plot_whisper_attention.log", mode="w")
    metadata = WhisperAttentionVisualizer(config).run()
    logger.info("Wrote Whisper attention metadata for %s", metadata["wav"])


if __name__ == "__main__":
    main()
