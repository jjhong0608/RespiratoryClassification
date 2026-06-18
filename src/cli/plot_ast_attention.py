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
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    WaveformPreprocessor,
)
from src.data.io import WaveformLoader
from src.models.model import RespiratoryAstModel
from src.plots.export import PlotlyExportMixin
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import DataConfig, JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger

AttentionMethod = Literal[
    "last_cls_patch",
    "last_cls_patch_head_mean",
    "attention_rollout",
    "class_gradient_attention",
]
VisualizationMode = Literal["rectangle", "heatmap", "both"]

ATTENTION_METHODS: tuple[AttentionMethod, ...] = (
    "last_cls_patch",
    "last_cls_patch_head_mean",
    "attention_rollout",
    "class_gradient_attention",
)
VISUALIZATION_MODES: tuple[VisualizationMode, ...] = ("rectangle", "heatmap", "both")
SUPPORTED_FORMATS = {"html", "png", "pdf"}
CLS_TOKEN_INDEX = 0
PATCH_TOKEN_OFFSET = 2
AST_FRAME_SHIFT_SECONDS = 0.01
DEFAULT_OUT_DIR = "Disease_Group_Results/reports/ast_attention_visualization"


@dataclass(frozen=True)
class AttentionCliConfig:
    checkpoint: Path
    wav: Path
    out_dir: Path
    attention_method: AttentionMethod
    visualization: VisualizationMode
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
    fbank: Tensor
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
class PatchGeometry:
    num_mel_bins: int
    max_length: int
    patch_size: int
    frequency_stride: int
    time_stride: int
    frequency_patches: int
    time_patches: int

    @property
    def num_patches(self) -> int:
        return self.frequency_patches * self.time_patches


@dataclass(frozen=True)
class PatchRecord:
    rank: int
    patch_index: int
    freq_index: int
    time_index: int
    freq_start_bin: int
    freq_end_bin: int
    time_start_frame: int
    time_end_frame: int
    time_start_sec: float
    time_end_sec: float
    attention_score: float
    normalized_attention_score: float


@dataclass(frozen=True)
class AttentionScoreResult:
    method: AttentionMethod
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    top_patches: list[PatchRecord]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FigureBundle:
    fbank: go.Figure
    heatmap_overlay: go.Figure | None
    rectangle_overlay: go.Figure | None


class AstAttentionInputLoader(LoggingMixin):
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg
        self.waveform_loader = WaveformLoader(data_cfg.audio.sample_rate)
        self.preprocessor = WaveformPreprocessor(
            self._build_audio_preprocess_config(data_cfg)
        )
        self.feature_extractor = AstLikeFbank(
            AstFbankFeatureConfig(
                sample_rate=data_cfg.audio.sample_rate,
                clip_seconds=data_cfg.audio.clip_duration_sec,
                num_mel_bins=data_cfg.preprocessing.ast_fbank.num_mel_bins,
                max_length=data_cfg.preprocessing.ast_fbank.max_length,
                do_normalize=data_cfg.preprocessing.ast_fbank.do_normalize,
                mean=data_cfg.preprocessing.ast_fbank.mean,
                std=data_cfg.preprocessing.ast_fbank.std,
            )
        )

    def load(self, wav_path: Path) -> AttentionInputs:
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file does not exist: {wav_path}")
        if wav_path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a .wav file, got: {wav_path}")
        waveform = self.waveform_loader.load(wav_path)
        prepared = self.preprocessor.prepare(waveform)
        fbank = self.feature_extractor(prepared).contiguous()
        input_values = fbank.transpose(0, 1).contiguous().unsqueeze(0)
        self.logger.info("Loaded %s | fbank_shape=%s", wav_path, tuple(fbank.shape))
        return AttentionInputs(
            wav_path=wav_path, fbank=fbank, input_values=input_values
        )

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


class AstAttentionModelRunner(LoggingMixin):
    def __init__(self, loaded: LoadedCheckpoint):
        self.loaded = loaded
        model_cfg_raw = loaded.checkpoint.get("model_cfg")
        if model_cfg_raw is None:
            raise RuntimeError(
                f"Checkpoint missing model_cfg: {loaded.checkpoint_path}"
            )
        self.model_cfg = parse_model_cfg(model_cfg_raw)
        self.model = RespiratoryAstModel(self.model_cfg)
        self.model.load_state_dict(loaded.checkpoint["model_state_dict"])
        self.model.to(loaded.device)
        self.model.eval()
        self._force_eager_attention()
        self._validate_frontend_dims(loaded.data_cfg)

    def run(
        self,
        input_values: Tensor,
        *,
        attention_method: AttentionMethod,
        target_class: str,
    ) -> AttentionRunResult:
        model_input = input_values.to(self.loaded.device)
        target_attention: Tensor | None = None

        if attention_method == "class_gradient_attention":
            self.model.zero_grad(set_to_none=True)
            model_input = model_input.detach()
            outputs, logits = self._forward_encoder(model_input)
            attentions = self._require_attentions(outputs)
            target_index = self._resolve_target_index(target_class, logits)
            target_attention = attentions[-1]
            target_attention.retain_grad()
            target_score = self._target_score(logits, target_index)
            target_score.backward()
        else:
            with torch.no_grad():
                outputs, logits = self._forward_encoder(model_input)
                attentions = self._require_attentions(outputs)
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

    def _forward_encoder(self, input_values: Tensor) -> tuple[Any, Tensor]:
        outputs = self.model.encoder(
            input_values=input_values,
            output_attentions=True,
            return_dict=True,
        )
        pooled_embedding = self._pool(outputs.last_hidden_state)
        logits = self.model.classifier(pooled_embedding)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return outputs, logits

    def _pool(self, hidden_states: Tensor) -> Tensor:
        if self.model.cfg.classifier.pooling == "cls":
            return hidden_states[:, 0, :]
        return hidden_states.mean(dim=1)

    def _require_attentions(self, outputs: Any) -> tuple[Tensor, ...]:
        attentions = tuple(outputs.attentions or ())
        if not attentions:
            raise RuntimeError(
                "AST encoder did not return attention tensors. "
                "The attention backend must support output_attentions; eager attention was requested."
            )
        return attentions

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

    def _force_eager_attention(self) -> None:
        if hasattr(self.model.encoder, "set_attn_implementation"):
            self.model.encoder.set_attn_implementation("eager")
            self.logger.info("Set AST encoder attention backend to eager")
            return
        self.model.encoder.config._attn_implementation = "eager"
        self.logger.info("Set AST encoder config _attn_implementation to eager")

    def _validate_frontend_dims(self, data_cfg: DataConfig) -> None:
        checkpoint_bins = self.model_cfg.encoder.feature_dims.num_mel_bins
        checkpoint_length = self.model_cfg.encoder.feature_dims.max_length
        data_bins = data_cfg.preprocessing.ast_fbank.num_mel_bins
        data_length = data_cfg.preprocessing.ast_fbank.max_length
        if checkpoint_bins == data_bins and checkpoint_length == data_length:
            return
        raise ValueError(
            "Checkpoint frontend dims do not match run_config.data preprocessing dims.\n"
            f"- checkpoint: num_mel_bins={checkpoint_bins}, max_length={checkpoint_length}\n"
            f"- run_config: num_mel_bins={data_bins}, max_length={data_length}\n"
            f"- checkpoint_path: {self.loaded.checkpoint_path}"
        )


class PatchGridMapper:
    def __init__(self, geometry: PatchGeometry):
        self.geometry = geometry

    @classmethod
    def from_model(cls, runner: AstAttentionModelRunner) -> PatchGridMapper:
        cfg = runner.model.encoder.config
        patch_size = _require_int_config_value(cfg.patch_size, "patch_size")
        frequency_stride = _require_int_config_value(
            cfg.frequency_stride,
            "frequency_stride",
        )
        time_stride = _require_int_config_value(cfg.time_stride, "time_stride")
        num_mel_bins = _require_int_config_value(cfg.num_mel_bins, "num_mel_bins")
        max_length = _require_int_config_value(cfg.max_length, "max_length")
        frequency_patches = (num_mel_bins - patch_size) // frequency_stride + 1
        time_patches = (max_length - patch_size) // time_stride + 1
        if frequency_patches <= 0 or time_patches <= 0:
            raise ValueError(
                "Invalid AST patch geometry: "
                f"num_mel_bins={num_mel_bins}, max_length={max_length}, "
                f"patch_size={patch_size}, frequency_stride={frequency_stride}, "
                f"time_stride={time_stride}"
            )
        return cls(
            PatchGeometry(
                num_mel_bins=num_mel_bins,
                max_length=max_length,
                patch_size=patch_size,
                frequency_stride=frequency_stride,
                time_stride=time_stride,
                frequency_patches=frequency_patches,
                time_patches=time_patches,
            )
        )

    def validate_token_count(self, attention: Tensor) -> None:
        token_count = int(attention.shape[-1])
        expected = self.geometry.num_patches + PATCH_TOKEN_OFFSET
        if token_count != expected:
            raise ValueError(
                "Attention token count does not match AST patch geometry. "
                f"attention_tokens={token_count}, expected={expected}, "
                f"geometry={asdict(self.geometry)}"
            )

    def patch_record(
        self,
        *,
        rank: int,
        patch_index: int,
        attention_score: float,
        normalized_attention_score: float,
    ) -> PatchRecord:
        freq_index = patch_index // self.geometry.time_patches
        time_index = patch_index % self.geometry.time_patches
        freq_start = freq_index * self.geometry.frequency_stride
        time_start = time_index * self.geometry.time_stride
        freq_end = min(
            freq_start + self.geometry.patch_size, self.geometry.num_mel_bins
        )
        time_end = min(time_start + self.geometry.patch_size, self.geometry.max_length)
        return PatchRecord(
            rank=rank,
            patch_index=patch_index,
            freq_index=freq_index,
            time_index=time_index,
            freq_start_bin=freq_start,
            freq_end_bin=freq_end,
            time_start_frame=time_start,
            time_end_frame=time_end,
            time_start_sec=time_start * AST_FRAME_SHIFT_SECONDS,
            time_end_sec=time_end * AST_FRAME_SHIFT_SECONDS,
            attention_score=attention_score,
            normalized_attention_score=normalized_attention_score,
        )

    def top_patches(
        self,
        raw_scores: np.ndarray,
        normalized_scores: np.ndarray,
        *,
        top_k: int,
    ) -> list[PatchRecord]:
        limit = min(top_k, raw_scores.size)
        order = np.argsort(raw_scores)[::-1][:limit]
        return [
            self.patch_record(
                rank=rank,
                patch_index=int(index),
                attention_score=float(raw_scores[index]),
                normalized_attention_score=float(normalized_scores[index]),
            )
            for rank, index in enumerate(order, start=1)
        ]

    def upsample_scores(self, normalized_scores: np.ndarray) -> np.ndarray:
        attention_map = np.zeros(
            (self.geometry.num_mel_bins, self.geometry.max_length),
            dtype=np.float64,
        )
        counts = np.zeros_like(attention_map)
        for patch_index, score in enumerate(normalized_scores):
            record = self.patch_record(
                rank=0,
                patch_index=patch_index,
                attention_score=float(score),
                normalized_attention_score=float(score),
            )
            attention_map[
                record.freq_start_bin : record.freq_end_bin,
                record.time_start_frame : record.time_end_frame,
            ] += float(score)
            counts[
                record.freq_start_bin : record.freq_end_bin,
                record.time_start_frame : record.time_end_frame,
            ] += 1.0
        np.divide(attention_map, counts, out=attention_map, where=counts > 0)
        return attention_map


class AttentionScoreExtractor:
    def __init__(self, mapper: PatchGridMapper):
        self.mapper = mapper

    def extract(
        self,
        run_result: AttentionRunResult,
        *,
        method: AttentionMethod,
        head: str,
        top_k: int,
    ) -> AttentionScoreResult:
        if method == "last_cls_patch":
            raw_scores, metadata = self._last_cls_patch(
                run_result.attentions, head=head
            )
        elif method == "last_cls_patch_head_mean":
            raw_scores, metadata = self._last_cls_patch_head_mean(run_result.attentions)
        elif method == "attention_rollout":
            raw_scores, metadata = self._attention_rollout(run_result.attentions)
        elif method == "class_gradient_attention":
            raw_scores, metadata = self._class_gradient_attention(run_result)
        else:
            raise ValueError(f"Unsupported attention method: {method}")

        normalized_scores = self._normalize(raw_scores)
        top_patches = self.mapper.top_patches(
            raw_scores, normalized_scores, top_k=top_k
        )
        return AttentionScoreResult(
            method=method,
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            top_patches=top_patches,
            metadata=metadata,
        )

    def _last_cls_patch(
        self,
        attentions: tuple[Tensor, ...],
        *,
        head: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        last_attention = attentions[-1]
        self.mapper.validate_token_count(last_attention)
        head_index = self._parse_head(head, num_heads=int(last_attention.shape[1]))
        scores = last_attention[0, head_index, CLS_TOKEN_INDEX, PATCH_TOKEN_OFFSET:]
        return (
            self._to_numpy_scores(scores),
            {
                "layer": len(attentions) - 1,
                "head": head_index,
                "head_aggregation": "single",
            },
        )

    def _last_cls_patch_head_mean(
        self,
        attentions: tuple[Tensor, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        last_attention = attentions[-1]
        self.mapper.validate_token_count(last_attention)
        scores = last_attention[0, :, CLS_TOKEN_INDEX, PATCH_TOKEN_OFFSET:].mean(dim=0)
        return (
            self._to_numpy_scores(scores),
            {"layer": len(attentions) - 1, "head": "mean", "head_aggregation": "mean"},
        )

    def _attention_rollout(
        self,
        attentions: tuple[Tensor, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rollout: Tensor | None = None
        for attention in attentions:
            self.mapper.validate_token_count(attention)
            matrix = attention[0].mean(dim=0)
            matrix = matrix + torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype
            )
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(matrix.dtype).eps
            )
            rollout = matrix if rollout is None else matrix @ rollout
        if rollout is None:
            raise RuntimeError(
                "Cannot compute attention rollout without attention tensors"
            )
        scores = rollout[CLS_TOKEN_INDEX, PATCH_TOKEN_OFFSET:]
        return (
            self._to_numpy_scores(scores),
            {
                "layers": len(attentions),
                "head": "mean",
                "residual": True,
                "row_normalized": True,
            },
        )

    def _class_gradient_attention(
        self,
        run_result: AttentionRunResult,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if run_result.target_attention is None:
            raise RuntimeError(
                "class_gradient_attention requires target attention from a gradient run"
            )
        target_attention = run_result.target_attention
        self.mapper.validate_token_count(target_attention)
        gradient = target_attention.grad
        if gradient is None:
            raise RuntimeError("Attention gradient is missing after backward pass")
        weighted = torch.relu(target_attention * gradient)
        scores = weighted[0, :, CLS_TOKEN_INDEX, PATCH_TOKEN_OFFSET:].mean(dim=0)
        return (
            self._to_numpy_scores(scores),
            {
                "layer": len(run_result.attentions) - 1,
                "head": "mean",
                "gradient_weighting": "relu(attention * gradient)",
            },
        )

    @staticmethod
    def _parse_head(head: str, *, num_heads: int) -> int:
        if head == "mean":
            raise ValueError(
                "--head=mean is not valid with --attention-method last_cls_patch; "
                "use --attention-method last_cls_patch_head_mean instead"
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
            raise ValueError(f"Expected flat patch scores, got shape={values.shape}")
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
    def __init__(self, mapper: PatchGridMapper):
        self.mapper = mapper

    def build(
        self,
        *,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
        visualization: VisualizationMode,
    ) -> FigureBundle:
        fbank = self.fbank_figure(inputs, score_result, prediction)
        heatmap_overlay = None
        rectangle_overlay = None
        if visualization in {"heatmap", "both"}:
            heatmap_overlay = self.heatmap_overlay(inputs, score_result, prediction)
        if visualization in {"rectangle", "both"}:
            rectangle_overlay = self.rectangle_overlay(inputs, score_result, prediction)
        return FigureBundle(
            fbank=fbank,
            heatmap_overlay=heatmap_overlay,
            rectangle_overlay=rectangle_overlay,
        )

    def fbank_figure(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = self._base_fbank_figure(inputs)
        fig.update_layout(
            title=self._title("AST fbank", inputs, score_result, prediction)
        )
        return fig

    def heatmap_overlay(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = self._base_fbank_figure(inputs)
        attention_map = self.mapper.upsample_scores(score_result.normalized_scores)
        fig.add_trace(
            go.Heatmap(
                z=attention_map,
                x=self._time_axis(attention_map.shape[1]),
                y=self._freq_axis(attention_map.shape[0]),
                colorscale="Inferno",
                opacity=0.48,
                colorbar={"title": "attention"},
                hovertemplate=(
                    "time=%{x:.3f}s<br>fbank_bin=%{y}<br>"
                    "normalized_attention=%{z:.6f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=self._title(
                "Attention heatmap overlay", inputs, score_result, prediction
            )
        )
        return fig

    def rectangle_overlay(
        self,
        inputs: AttentionInputs,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
    ) -> go.Figure:
        fig = self._base_fbank_figure(inputs)
        for patch in score_result.top_patches:
            color = self._rank_color(patch.normalized_attention_score)
            fig.add_shape(
                type="rect",
                x0=patch.time_start_sec,
                x1=patch.time_end_sec,
                y0=patch.freq_start_bin,
                y1=patch.freq_end_bin,
                line={"color": color, "width": 2},
                fillcolor=color.replace("1.0)", "0.22)"),
            )
            fig.add_annotation(
                x=(patch.time_start_sec + patch.time_end_sec) / 2.0,
                y=patch.freq_end_bin,
                text=f"#{patch.rank}<br>{patch.normalized_attention_score:.2f}",
                showarrow=False,
                font={"size": 10, "color": "white"},
                bgcolor="rgba(0,0,0,0.55)",
                borderpad=2,
            )
        fig.update_layout(
            title=self._title(
                "Top patch rectangle overlay", inputs, score_result, prediction
            )
        )
        return fig

    def _base_fbank_figure(self, inputs: AttentionInputs) -> go.Figure:
        fbank = inputs.fbank.detach().cpu().numpy()
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=fbank,
                    x=self._time_axis(fbank.shape[1]),
                    y=self._freq_axis(fbank.shape[0]),
                    colorscale="Viridis",
                    colorbar={"title": "normalized fbank"},
                    hovertemplate="time=%{x:.3f}s<br>fbank_bin=%{y}<br>value=%{z:.6f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Filterbank bin",
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
    def _time_axis(length: int) -> np.ndarray:
        return np.arange(length, dtype=np.float64) * AST_FRAME_SHIFT_SECONDS

    @staticmethod
    def _freq_axis(length: int) -> np.ndarray:
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
        geometry: PatchGeometry,
    ) -> dict[str, Any]:
        Fs.ensure_dir(self.out_dir)
        stem = self._safe_stem(inputs.wav_path)
        outputs: dict[str, list[str] | str] = {}
        outputs["fbank"] = self._write_figure(
            figures.fbank, self.out_dir / f"{stem}_fbank"
        )
        if figures.heatmap_overlay is not None:
            outputs["heatmap_overlay"] = self._write_figure(
                figures.heatmap_overlay,
                self.out_dir / f"{stem}_attention_heatmap_overlay",
            )
        if figures.rectangle_overlay is not None:
            outputs["rectangle_overlay"] = self._write_figure(
                figures.rectangle_overlay,
                self.out_dir / f"{stem}_attention_rectangle_overlay",
            )
        top_patch_path = self.out_dir / f"{stem}_top_patches.csv"
        metadata_path = self.out_dir / f"{stem}_attention_metadata.json"
        self._write_top_patches(top_patch_path, score_result.top_patches)
        metadata = self._metadata(
            inputs=inputs,
            loaded=loaded,
            score_result=score_result,
            prediction=prediction,
            cli_config=cli_config,
            geometry=geometry,
            outputs=outputs,
            top_patch_path=top_patch_path,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.logger.info("Wrote %s", top_patch_path)
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
    def _write_top_patches(path: Path, patches: Sequence[PatchRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "patch_index",
                    "freq_index",
                    "time_index",
                    "freq_start_bin",
                    "freq_end_bin",
                    "time_start_frame",
                    "time_end_frame",
                    "time_start_sec",
                    "time_end_sec",
                    "attention_score",
                    "normalized_attention_score",
                ],
            )
            writer.writeheader()
            for patch in patches:
                writer.writerow(asdict(patch))

    @staticmethod
    def _metadata(
        *,
        inputs: AttentionInputs,
        loaded: LoadedCheckpoint,
        score_result: AttentionScoreResult,
        prediction: PredictionSummary,
        cli_config: AttentionCliConfig,
        geometry: PatchGeometry,
        outputs: Mapping[str, list[str] | str],
        top_patch_path: Path,
    ) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "wav": str(inputs.wav_path),
            "checkpoint": str(loaded.checkpoint_path),
            "device": str(loaded.device),
            "attention_method": score_result.method,
            "attention_metadata": score_result.metadata,
            "visualization": cli_config.visualization,
            "target_class": cli_config.target_class,
            "prediction": asdict(prediction),
            "patch_geometry": asdict(geometry),
            "top_k": cli_config.top_k,
            "top_patches_csv": str(top_patch_path),
            "outputs": outputs,
        }


class AstAttentionVisualizer(LoggingMixin):
    def __init__(self, config: AttentionCliConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        loaded = load_attention_checkpoint(
            self.config.checkpoint, self.config.device_override
        )
        inputs = AstAttentionInputLoader(loaded.data_cfg).load(self.config.wav)
        runner = AstAttentionModelRunner(loaded)
        run_result = runner.run(
            inputs.input_values,
            attention_method=self.config.attention_method,
            target_class=self.config.target_class,
        )
        mapper = PatchGridMapper.from_model(runner)
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
            visualization=self.config.visualization,
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
            "Generated AST attention visualization | wav=%s | method=%s | out_dir=%s",
            self.config.wav,
            self.config.attention_method,
            self.config.out_dir,
        )
        return metadata


def load_attention_checkpoint(
    checkpoint_path: Path, device_override: str | None
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
    checkpoint: Mapping[str, Any], checkpoint_path: Path
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
    checkpoint: Mapping[str, Any], checkpoint_path: Path
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


def _require_int_config_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"AST config {name} must be an int, got {value!r}")
    return value


def parse_formats(formats_arg: str) -> set[str]:
    formats = {part.strip() for part in formats_arg.split(",") if part.strip()}
    unsupported = formats - SUPPORTED_FORMATS
    if unsupported:
        raise ValueError(f"Unsupported format(s): {sorted(unsupported)}")
    if not formats:
        raise ValueError("--formats must include at least one of html,png,pdf")
    return formats


def parse_args(argv: Sequence[str] | None = None) -> AttentionCliConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--attention-method",
        default="last_cls_patch_head_mean",
        choices=ATTENTION_METHODS,
    )
    parser.add_argument("--visualization", default="both", choices=VISUALIZATION_MODES)
    parser.add_argument("--target-class", default="predicted")
    parser.add_argument("--head", default="0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--formats", default="html,png,pdf")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.attention_method != "last_cls_patch" and args.head != "0":
        raise ValueError(
            "--head is only supported with --attention-method last_cls_patch"
        )

    return AttentionCliConfig(
        checkpoint=Path(args.checkpoint),
        wav=Path(args.wav),
        out_dir=Path(args.out_dir),
        attention_method=args.attention_method,
        visualization=args.visualization,
        target_class=str(args.target_class),
        head=str(args.head),
        top_k=int(args.top_k),
        formats=parse_formats(str(args.formats)),
        device_override=args.device,
        install_chrome=bool(args.install_chrome),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    Fs.ensure_dir(config.out_dir)
    enable_file_logging(config.out_dir / "plot_ast_attention.log", mode="w")
    metadata = AstAttentionVisualizer(config).run()
    logger.info("Wrote AST attention metadata for %s", metadata["wav"])


if __name__ == "__main__":
    main()
