from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.plots.export import PlotlyExportMixin
from src.utils.logging import enable_file_logging, logger

FINAL_LABELS = ("Normal", "Airway", "Lung_Parenchymal")
APPROACHES = ("direct", "cascade")
APPROACH_DISPLAY = {"direct": "Direct 3-class", "cascade": "Cascade"}
METRICS = (
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
CV_CONTEXT_METRICS = (
    "f1_score",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "brier_score",
)
DEFAULT_REPORTS_ROOT = "Disease_Group_Results/reports"
DEFAULT_OUT_DIR = "Disease_Group_Results/reports/visual_summary"

REQUIRED_FOLD_METRIC_COLUMNS = {"fold"} | {
    f"{approach}_{metric}" for approach in APPROACHES for metric in METRICS
}
REQUIRED_PREDICTION_COLUMNS = {
    "fold",
    "audio_path",
    "true_label",
    "direct_pred_label",
    "direct_correct",
    "direct_prob_normal",
    "direct_prob_airway",
    "direct_prob_lung_parenchymal",
    "stage1_pred_label",
    "stage1_prob_normal",
    "stage1_prob_abnormal",
    "stage1_threshold",
    "stage1_threshold_source",
    "stage2_executed",
    "stage2_pred_label",
    "stage2_prob_airway",
    "stage2_prob_lung_parenchymal",
    "stage2_threshold",
    "stage2_threshold_source",
    "cascade_prob_normal",
    "cascade_prob_airway",
    "cascade_prob_lung_parenchymal",
    "cascade_final_pred_label",
    "cascade_correct",
    "cascade_error_type",
    "direct_checkpoint",
    "stage1_checkpoint",
    "stage2_checkpoint",
}


@dataclass(frozen=True)
class FoldMetricRecord:
    fold: str
    values: dict[str, float]


@dataclass(frozen=True)
class PredictionRecord:
    fold: str
    audio_path: str
    true_label: str
    direct_pred_label: str
    direct_correct: bool
    direct_probabilities: dict[str, float]
    stage1_pred_label: str
    stage1_prob_normal: float
    stage1_prob_abnormal: float
    stage1_threshold: float
    stage1_threshold_source: str
    stage2_executed: bool
    stage2_pred_label: str
    stage2_prob_airway: float | None
    stage2_prob_lung_parenchymal: float | None
    stage2_threshold: float | None
    stage2_threshold_source: str
    cascade_probabilities: dict[str, float]
    cascade_final_pred_label: str
    cascade_correct: bool
    cascade_error_type: str
    direct_checkpoint: str
    stage1_checkpoint: str
    stage2_checkpoint: str

    def predicted_label(self, approach: str) -> str:
        if approach == "direct":
            return self.direct_pred_label
        if approach == "cascade":
            return self.cascade_final_pred_label
        raise ValueError(f"Unsupported approach: {approach}")

    def is_correct(self, approach: str) -> bool:
        if approach == "direct":
            return self.direct_correct
        if approach == "cascade":
            return self.cascade_correct
        raise ValueError(f"Unsupported approach: {approach}")

    def confidence(self, approach: str) -> float:
        probabilities = (
            self.direct_probabilities
            if approach == "direct"
            else self.cascade_probabilities
        )
        return float(probabilities[self.predicted_label(approach)])


@dataclass(frozen=True)
class CvContextRecord:
    method: str
    metric_means: dict[str, float]
    metric_stds: dict[str, float]
    selected_f1_by_fold: dict[str, float]


@dataclass(frozen=True)
class LoadedReports:
    reports_root: Path
    fold_metrics_path: Path
    predictions_path: Path
    cascade_summary_path: Path
    cv_summary_paths: list[Path]
    markdown_paths: list[Path]
    fold_metrics: list[FoldMetricRecord]
    predictions: list[PredictionRecord]
    cascade_summary: dict[str, Any]
    cv_context: list[CvContextRecord]
    markdown_captions: dict[str, str]


@dataclass(frozen=True)
class FigureSpec:
    stem: str
    title: str
    caption: str
    source_files: list[str]
    source_columns: list[str]
    figure: go.Figure


class DiseaseGroupReportLoader:
    def __init__(self, reports_root: str | Path):
        self.reports_root = Path(reports_root)

    def load(self) -> LoadedReports:
        fold_metrics_path = (
            self.reports_root
            / "cascade_test_comparison"
            / "cascade_test_fold_metrics.csv"
        )
        predictions_path = (
            self.reports_root
            / "cascade_test_comparison"
            / "cascade_test_predictions.csv"
        )
        cascade_summary_path = (
            self.reports_root / "cascade_test_comparison" / "cascade_test_summary.json"
        )
        fold_metrics = self.load_fold_metrics(fold_metrics_path)
        predictions = self.load_predictions(predictions_path)
        cascade_summary = self._read_json(cascade_summary_path)
        cv_summary_paths = sorted(self.reports_root.glob("*_cv_summary.json"))
        cv_context = [self._load_cv_context(path) for path in cv_summary_paths]
        markdown_paths = sorted(self.reports_root.glob("*.md")) + sorted(
            (self.reports_root / "cascade_test_comparison").glob("*.md")
        )
        markdown_captions = {
            str(path): self._read_markdown_caption(path) for path in markdown_paths
        }
        return LoadedReports(
            reports_root=self.reports_root,
            fold_metrics_path=fold_metrics_path,
            predictions_path=predictions_path,
            cascade_summary_path=cascade_summary_path,
            cv_summary_paths=cv_summary_paths,
            markdown_paths=markdown_paths,
            fold_metrics=fold_metrics,
            predictions=predictions,
            cascade_summary=cascade_summary,
            cv_context=cv_context,
            markdown_captions=markdown_captions,
        )

    def load_fold_metrics(self, path: str | Path) -> list[FoldMetricRecord]:
        rows = self._read_csv(Path(path), REQUIRED_FOLD_METRIC_COLUMNS)
        records: list[FoldMetricRecord] = []
        for row_index, row in enumerate(rows, start=2):
            values = {
                column: self._parse_float(Path(path), row_index, column, row[column])
                for column in sorted(REQUIRED_FOLD_METRIC_COLUMNS - {"fold"})
            }
            records.append(FoldMetricRecord(fold=row["fold"], values=values))
        return records

    def load_predictions(self, path: str | Path) -> list[PredictionRecord]:
        csv_path = Path(path)
        rows = self._read_csv(csv_path, REQUIRED_PREDICTION_COLUMNS)
        records: list[PredictionRecord] = []
        for row_index, row in enumerate(rows, start=2):
            stage2_executed = self._parse_bool(
                csv_path, row_index, "stage2_executed", row["stage2_executed"]
            )
            records.append(
                PredictionRecord(
                    fold=row["fold"],
                    audio_path=row["audio_path"],
                    true_label=row["true_label"],
                    direct_pred_label=row["direct_pred_label"],
                    direct_correct=self._parse_bool(
                        csv_path, row_index, "direct_correct", row["direct_correct"]
                    ),
                    direct_probabilities={
                        "Normal": self._parse_float(
                            csv_path, row_index, "direct_prob_normal", row
                        ),
                        "Airway": self._parse_float(
                            csv_path, row_index, "direct_prob_airway", row
                        ),
                        "Lung_Parenchymal": self._parse_float(
                            csv_path,
                            row_index,
                            "direct_prob_lung_parenchymal",
                            row,
                        ),
                    },
                    stage1_pred_label=row["stage1_pred_label"],
                    stage1_prob_normal=self._parse_float(
                        csv_path, row_index, "stage1_prob_normal", row
                    ),
                    stage1_prob_abnormal=self._parse_float(
                        csv_path, row_index, "stage1_prob_abnormal", row
                    ),
                    stage1_threshold=self._parse_float(
                        csv_path, row_index, "stage1_threshold", row
                    ),
                    stage1_threshold_source=row["stage1_threshold_source"],
                    stage2_executed=stage2_executed,
                    stage2_pred_label=row["stage2_pred_label"],
                    stage2_prob_airway=self._parse_optional_float(
                        csv_path,
                        row_index,
                        "stage2_prob_airway",
                        row,
                        allow_blank=not stage2_executed,
                    ),
                    stage2_prob_lung_parenchymal=self._parse_optional_float(
                        csv_path,
                        row_index,
                        "stage2_prob_lung_parenchymal",
                        row,
                        allow_blank=not stage2_executed,
                    ),
                    stage2_threshold=self._parse_optional_float(
                        csv_path,
                        row_index,
                        "stage2_threshold",
                        row,
                        allow_blank=not stage2_executed,
                    ),
                    stage2_threshold_source=row["stage2_threshold_source"],
                    cascade_probabilities={
                        "Normal": self._parse_float(
                            csv_path, row_index, "cascade_prob_normal", row
                        ),
                        "Airway": self._parse_float(
                            csv_path, row_index, "cascade_prob_airway", row
                        ),
                        "Lung_Parenchymal": self._parse_float(
                            csv_path,
                            row_index,
                            "cascade_prob_lung_parenchymal",
                            row,
                        ),
                    },
                    cascade_final_pred_label=row["cascade_final_pred_label"],
                    cascade_correct=self._parse_bool(
                        csv_path, row_index, "cascade_correct", row["cascade_correct"]
                    ),
                    cascade_error_type=row["cascade_error_type"],
                    direct_checkpoint=row["direct_checkpoint"],
                    stage1_checkpoint=row["stage1_checkpoint"],
                    stage2_checkpoint=row["stage2_checkpoint"],
                )
            )
        self._validate_labels(csv_path, records)
        return records

    @staticmethod
    def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Required CSV not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{path} missing required columns: {sorted(missing)}")
            return [dict(row) for row in reader]

    @staticmethod
    def _parse_float(
        path: Path,
        row_index: int,
        column: str,
        value_or_row: str | Mapping[str, str],
    ) -> float:
        value = (
            value_or_row[column] if isinstance(value_or_row, Mapping) else value_or_row
        )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{row_index} column={column} expected float, got {value!r}"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"{path}:{row_index} column={column} expected finite float, got {value!r}"
            )
        return numeric

    @classmethod
    def _parse_optional_float(
        cls,
        path: Path,
        row_index: int,
        column: str,
        row: Mapping[str, str],
        *,
        allow_blank: bool,
    ) -> float | None:
        value = row[column]
        if value == "" and allow_blank:
            return None
        return cls._parse_float(path, row_index, column, value)

    @staticmethod
    def _parse_bool(path: Path, row_index: int, column: str, value: str) -> bool:
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(
            f"{path}:{row_index} column={column} expected true/false, got {value!r}"
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required JSON not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"JSON root must be an object: {path}")
        return raw

    @staticmethod
    def _read_markdown_caption(path: Path) -> str:
        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# ") or line.startswith("- Overall winner"):
                lines.append(line.strip("#- "))
            if len(lines) >= 3:
                break
        return " | ".join(lines) if lines else path.name

    def _load_cv_context(self, path: Path) -> CvContextRecord:
        raw = self._read_json(path)
        methods = raw.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ValueError(f"CV summary missing methods: {path}")
        method = methods[0]
        if not isinstance(method, Mapping):
            raise TypeError(f"CV summary method is not an object: {path}")
        name = str(method["name"])
        means = self._float_mapping(method.get("optimized_means"), path)
        stds = self._float_mapping(method.get("optimized_stds"), path)
        selected_f1_by_fold: dict[str, float] = {}
        folds = method.get("folds", [])
        if isinstance(folds, list):
            for fold in folds:
                if not isinstance(fold, Mapping):
                    continue
                selected = fold.get("selected_snapshot")
                if not isinstance(selected, Mapping):
                    continue
                optimized = selected.get("optimized_metrics")
                if not isinstance(optimized, Mapping):
                    continue
                f1_raw = optimized.get("f1_score")
                if isinstance(f1_raw, (int, float)):
                    selected_f1_by_fold[str(fold.get("name"))] = float(f1_raw)
        return CvContextRecord(
            method=name,
            metric_means=means,
            metric_stds=stds,
            selected_f1_by_fold=selected_f1_by_fold,
        )

    @staticmethod
    def _float_mapping(raw: object, path: Path) -> dict[str, float]:
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected mapping in {path}")
        result: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result[str(key)] = float(value)
        return result

    @staticmethod
    def _validate_labels(path: Path, records: Sequence[PredictionRecord]) -> None:
        valid = set(FINAL_LABELS)
        for record in records:
            for column, label in [
                ("true_label", record.true_label),
                ("direct_pred_label", record.direct_pred_label),
                ("cascade_final_pred_label", record.cascade_final_pred_label),
            ]:
                if label not in valid:
                    raise ValueError(
                        f"{path} column={column} has unknown label: {label}"
                    )


class MetricPlotBuilder:
    def __init__(self, data: LoadedReports):
        self.data = data

    def build(self) -> list[FigureSpec]:
        return [
            self.metric_mean_std_grouped_bar(),
            self.metric_foldwise_lines(),
            self.metric_delta_bar(),
            self.metric_heatmap_by_fold(),
            self.cv_context_metric_comparison(),
            self.checkpoint_f1_context(),
        ]

    def metric_mean_std_grouped_bar(self) -> FigureSpec:
        summary = metric_mean_std(self.data.fold_metrics)
        fig = go.Figure()
        colors = {"direct": "#2a6fbb", "cascade": "#d95f02"}
        for approach in APPROACHES:
            means = [summary[approach][metric]["mean"] for metric in METRICS]
            stds = [summary[approach][metric]["std"] for metric in METRICS]
            fig.add_trace(
                go.Bar(
                    name=APPROACH_DISPLAY[approach],
                    x=list(METRICS),
                    y=means,
                    error_y={"type": "data", "array": stds, "visible": True},
                    text=[
                        f"{mean:.3f}<br>+-{std:.3f}"
                        for mean, std in zip(means, stds, strict=True)
                    ],
                    textposition="outside",
                    marker={"color": colors[approach]},
                    hovertemplate=(
                        f"approach={APPROACH_DISPLAY[approach]}<br>"
                        "metric=%{x}<br>mean=%{y:.6f}<br>std=%{customdata:.6f}<extra></extra>"
                    ),
                    customdata=stds,
                )
            )
        fig.update_layout(
            title="Direct vs Cascade test metrics (mean +- std across folds)",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode="group",
            width=1400,
            height=720,
            legend={"orientation": "h", "y": 1.08},
        )
        fig.update_xaxes(tickangle=-30)
        return self._spec(
            "01_metric_mean_std_grouped_bar",
            "Direct vs cascade mean/std metrics",
            "Grouped bars show fold mean and standard deviation for each test metric; text labels report mean and std.",
            fig,
            ["cascade_test_fold_metrics.csv"],
            sorted(REQUIRED_FOLD_METRIC_COLUMNS),
        )

    def metric_foldwise_lines(self) -> FigureSpec:
        cols = 2
        rows = math.ceil(len(METRICS) / cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(METRICS),
            horizontal_spacing=0.08,
            vertical_spacing=0.08,
        )
        colors = {"direct": "#2a6fbb", "cascade": "#d95f02"}
        folds = [record.fold for record in self.data.fold_metrics]
        for metric_index, metric in enumerate(METRICS):
            row = metric_index // cols + 1
            col = metric_index % cols + 1
            for approach in APPROACHES:
                values = [
                    record.values[f"{approach}_{metric}"]
                    for record in self.data.fold_metrics
                ]
                fig.add_trace(
                    go.Scatter(
                        x=folds,
                        y=values,
                        mode="lines+markers+text",
                        name=APPROACH_DISPLAY[approach],
                        legendgroup=approach,
                        showlegend=metric_index == 0,
                        text=[f"{value:.3f}" for value in values],
                        textposition="top center",
                        marker={"color": colors[approach], "size": 8},
                        line={"color": colors[approach]},
                        hovertemplate=(
                            f"approach={APPROACH_DISPLAY[approach]}<br>"
                            f"metric={metric}<br>fold=%{{x}}<br>value=%{{y:.6f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_layout(
            title="Fold-wise direct vs cascade metrics",
            width=1450,
            height=1650,
            legend={"orientation": "h", "y": 1.04},
        )
        return self._spec(
            "02_metric_foldwise_lines",
            "Fold-wise metric lines",
            "Each subplot shows direct and cascade values for the five test folds with numeric point labels.",
            fig,
            ["cascade_test_fold_metrics.csv"],
            sorted(REQUIRED_FOLD_METRIC_COLUMNS),
        )

    def metric_delta_bar(self) -> FigureSpec:
        summary = metric_mean_std(self.data.fold_metrics)
        deltas = [
            summary["cascade"][metric]["mean"] - summary["direct"][metric]["mean"]
            for metric in METRICS
        ]
        fig = go.Figure(
            go.Bar(
                x=list(METRICS),
                y=deltas,
                text=[f"{delta:+.4f}" for delta in deltas],
                textposition="outside",
                marker={
                    "color": [
                        "#2ca02c" if delta >= 0 else "#d62728" for delta in deltas
                    ]
                },
                hovertemplate="metric=%{x}<br>cascade-direct=%{y:+.6f}<extra></extra>",
            )
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#555")
        fig.update_layout(
            title="Cascade minus direct metric delta",
            xaxis_title="Metric",
            yaxis_title="Cascade - Direct",
            width=1300,
            height=650,
        )
        fig.update_xaxes(tickangle=-30)
        return self._spec(
            "03_metric_delta_bar",
            "Cascade minus direct metric deltas",
            "Bars above zero favor cascade; bars below zero favor direct. Labels show signed mean differences.",
            fig,
            ["cascade_test_fold_metrics.csv"],
            sorted(REQUIRED_FOLD_METRIC_COLUMNS),
        )

    def metric_heatmap_by_fold(self) -> FigureSpec:
        folds = [record.fold for record in self.data.fold_metrics]
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Direct 3-class", "Cascade"],
            horizontal_spacing=0.12,
        )
        for col, approach in enumerate(APPROACHES, start=1):
            matrix = np.asarray(
                [
                    [record.values[f"{approach}_{metric}"] for metric in METRICS]
                    for record in self.data.fold_metrics
                ],
                dtype=float,
            )
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=list(METRICS),
                    y=folds,
                    text=[[f"{value:.3f}" for value in row] for row in matrix],
                    texttemplate="%{text}",
                    hovertemplate="fold=%{y}<br>metric=%{x}<br>value=%{z:.6f}<extra></extra>",
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=1.0,
                    colorbar={"title": "Score"} if col == 2 else None,
                    showscale=col == 2,
                ),
                row=1,
                col=col,
            )
        fig.update_layout(
            title="Fold x metric heatmap",
            width=1700,
            height=620,
        )
        fig.update_xaxes(tickangle=-35)
        return self._spec(
            "04_metric_heatmap_by_fold",
            "Fold-by-metric heatmap",
            "Heatmaps expose fold stability and metric tradeoffs; every cell is annotated with the numeric score.",
            fig,
            ["cascade_test_fold_metrics.csv"],
            sorted(REQUIRED_FOLD_METRIC_COLUMNS),
        )

    def cv_context_metric_comparison(self) -> FigureSpec:
        fig = go.Figure()
        for context in self.data.cv_context:
            means = [
                context.metric_means.get(metric, float("nan"))
                for metric in CV_CONTEXT_METRICS
            ]
            stds = [
                context.metric_stds.get(metric, 0.0) for metric in CV_CONTEXT_METRICS
            ]
            fig.add_trace(
                go.Bar(
                    name=context.method,
                    x=list(CV_CONTEXT_METRICS),
                    y=means,
                    error_y={"type": "data", "array": stds, "visible": True},
                    text=[
                        f"{mean:.3f}<br>+-{std:.3f}"
                        for mean, std in zip(means, stds, strict=True)
                    ],
                    textposition="outside",
                    hovertemplate=(
                        f"method={context.method}<br>metric=%{{x}}<br>"
                        "mean=%{y:.6f}<br>std=%{customdata:.6f}<extra></extra>"
                    ),
                    customdata=stds,
                )
            )
        fig.update_layout(
            title="CV context metrics from existing disease-group reports",
            xaxis_title="Optimized validation metric",
            yaxis_title="Mean score",
            barmode="group",
            width=1350,
            height=700,
            legend={"orientation": "h", "y": 1.1},
        )
        return self._spec(
            "17_cv_context_metric_comparison",
            "CV context metric comparison",
            "Existing CV summaries are shown as context for the test-set direct/cascade comparison.",
            fig,
            [path.name for path in self.data.cv_summary_paths],
            ["optimized_means", "optimized_stds"],
        )

    def checkpoint_f1_context(self) -> FigureSpec:
        selected = self.data.cascade_summary.get("selected_checkpoints", {})
        if not isinstance(selected, Mapping):
            raise TypeError("cascade summary selected_checkpoints must be a mapping")
        task_order = [
            ("Normal_vs_Airway_vs_LungParenchymal", "Direct 3-class"),
            ("Normal_vs_Abnormal", "Stage1 Normal/Abnormal"),
            ("Airway_vs_LungParenchymal", "Stage2 subtype"),
        ]
        fig = go.Figure()
        folds = sorted(str(fold) for fold in selected)
        for task, label in task_order:
            values: list[float] = []
            hover: list[str] = []
            for fold in folds:
                fold_payload = selected.get(fold, {})
                if not isinstance(fold_payload, Mapping):
                    values.append(float("nan"))
                    hover.append("missing")
                    continue
                checkpoint = fold_payload.get(task, {})
                if not isinstance(checkpoint, Mapping):
                    values.append(float("nan"))
                    hover.append("missing")
                    continue
                f1 = float(checkpoint.get("optimized_f1", float("nan")))
                values.append(f1)
                hover.append(
                    f"task={task}<br>fold={fold}<br>optimized_f1={f1:.6f}<br>"
                    f"checkpoint={Path(str(checkpoint.get('checkpoint_path', ''))).name}"
                )
            fig.add_trace(
                go.Scatter(
                    x=folds,
                    y=values,
                    mode="lines+markers+text",
                    name=label,
                    text=[f"{value:.3f}" for value in values],
                    textposition="top center",
                    hovertext=hover,
                    hoverinfo="text",
                )
            )
        fig.update_layout(
            title="Selected checkpoint validation optimized F1 by fold",
            xaxis_title="Fold",
            yaxis_title="Validation optimized F1",
            width=1250,
            height=650,
            legend={"orientation": "h", "y": 1.08},
        )
        return self._spec(
            "18_checkpoint_f1_context",
            "Selected checkpoint F1 context",
            "Fold-wise validation optimized F1 values explain which checkpoints fed the test-set comparison.",
            fig,
            ["cascade_test_summary.json"],
            ["selected_checkpoints.*.optimized_f1"],
        )

    @staticmethod
    def _spec(
        stem: str,
        title: str,
        caption: str,
        fig: go.Figure,
        source_files: list[str],
        source_columns: list[str],
    ) -> FigureSpec:
        return FigureSpec(stem, title, caption, source_files, source_columns, fig)


class ConfusionPlotBuilder:
    def __init__(self, data: LoadedReports):
        self.data = data

    def build(self) -> list[FigureSpec]:
        direct_counts = confusion_counts(self.data.predictions, "direct")
        cascade_counts = confusion_counts(self.data.predictions, "cascade")
        direct_norm = row_normalize(direct_counts)
        cascade_norm = row_normalize(cascade_counts)
        return [
            self.rich_confusion("direct", direct_counts, direct_norm),
            self.rich_confusion("cascade", cascade_counts, cascade_norm),
            self.confusion_delta_heatmap(direct_norm, cascade_norm),
            self.per_label_final_outcome(),
        ]

    def rich_confusion(
        self,
        approach: str,
        counts: np.ndarray,
        normalized: np.ndarray,
    ) -> FigureSpec:
        title = f"{APPROACH_DISPLAY[approach]} row-normalized confusion matrix"
        hover = [
            [
                f"true={FINAL_LABELS[row]}<br>pred={FINAL_LABELS[col]}<br>"
                f"ratio={normalized[row, col]:.4f}<br>count={int(counts[row, col])}"
                for col in range(len(FINAL_LABELS))
            ]
            for row in range(len(FINAL_LABELS))
        ]
        text = [
            [
                f"{normalized[row, col]:.2f}<br>n={int(counts[row, col])}"
                for col in range(len(FINAL_LABELS))
            ]
            for row in range(len(FINAL_LABELS))
        ]
        fig = go.Figure(
            go.Heatmap(
                z=normalized,
                x=list(FINAL_LABELS),
                y=list(FINAL_LABELS),
                text=text,
                texttemplate="%{text}",
                hovertext=hover,
                hoverinfo="text",
                colorscale="Blues",
                zmin=0,
                zmax=1,
                colorbar={"title": "Row ratio"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Predicted label",
            yaxis_title="True label",
            width=800,
            height=650,
        )
        return FigureSpec(
            f"0{5 if approach == 'direct' else 6}_{approach}_confusion_rich",
            title,
            "Cells show row-normalized ratio and raw count; rows sum to one by true label.",
            ["cascade_test_predictions.csv"],
            ["true_label", f"{approach}_pred_label"],
            fig,
        )

    def confusion_delta_heatmap(
        self,
        direct_norm: np.ndarray,
        cascade_norm: np.ndarray,
    ) -> FigureSpec:
        delta = cascade_norm - direct_norm
        fig = go.Figure(
            go.Heatmap(
                z=delta,
                x=list(FINAL_LABELS),
                y=list(FINAL_LABELS),
                text=[[f"{value:+.3f}" for value in row] for row in delta],
                texttemplate="%{text}",
                hovertemplate="true=%{y}<br>pred=%{x}<br>cascade-direct=%{z:+.6f}<extra></extra>",
                colorscale="RdBu",
                zmid=0.0,
                colorbar={"title": "Delta"},
            )
        )
        fig.update_layout(
            title="Cascade minus direct row-normalized confusion delta",
            xaxis_title="Predicted label",
            yaxis_title="True label",
            width=850,
            height=650,
        )
        return FigureSpec(
            "07_confusion_delta_heatmap",
            "Confusion delta heatmap",
            "Positive cells are more common under cascade; negative cells are more common under direct.",
            ["cascade_test_predictions.csv"],
            ["true_label", "direct_pred_label", "cascade_final_pred_label"],
            fig,
        )

    def per_label_final_outcome(self) -> FigureSpec:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Direct 3-class", "Cascade"],
            horizontal_spacing=0.12,
        )
        colors = {
            "Normal": "#4c78a8",
            "Airway": "#f58518",
            "Lung_Parenchymal": "#54a24b",
        }
        for col, approach in enumerate(APPROACHES, start=1):
            table = final_outcome_table(self.data.predictions, approach)
            for pred_label in FINAL_LABELS:
                counts = [
                    table[(true_label, pred_label)] for true_label in FINAL_LABELS
                ]
                totals = [
                    sum(table[(true_label, p)] for p in FINAL_LABELS)
                    for true_label in FINAL_LABELS
                ]
                ratios = [
                    count / total if total > 0 else 0.0
                    for count, total in zip(counts, totals, strict=True)
                ]
                fig.add_trace(
                    go.Bar(
                        x=list(FINAL_LABELS),
                        y=ratios,
                        name=pred_label,
                        legendgroup=pred_label,
                        showlegend=col == 1,
                        marker={"color": colors[pred_label]},
                        text=[
                            f"{ratio:.2f}<br>n={count}"
                            for ratio, count in zip(ratios, counts, strict=True)
                        ],
                        textposition="inside",
                        hovertemplate=(
                            f"approach={APPROACH_DISPLAY[approach]}<br>"
                            "true=%{x}<br>"
                            f"pred={pred_label}<br>ratio=%{{y:.6f}}<br>"
                            "count=%{customdata}<extra></extra>"
                        ),
                        customdata=counts,
                    ),
                    row=1,
                    col=col,
                )
        fig.update_layout(
            title="Final predicted label distribution by true label",
            barmode="stack",
            yaxis_title="Row ratio",
            width=1350,
            height=650,
            legend={"orientation": "h", "y": 1.12},
        )
        return FigureSpec(
            "08_per_label_final_outcome",
            "Per-label final outcome",
            "Stacked bars show where each true label ended up under direct and cascade predictions.",
            ["cascade_test_predictions.csv"],
            ["true_label", "direct_pred_label", "cascade_final_pred_label"],
            fig,
        )


class CascadeFlowPlotBuilder:
    def __init__(self, data: LoadedReports):
        self.data = data

    def build(self) -> list[FigureSpec]:
        return [
            self.cascade_sankey_flow(),
            self.cascade_stage_flow_stacked(),
            self.cascade_error_type_bar(),
            self.cascade_error_type_by_label(),
        ]

    def cascade_sankey_flow(self) -> FigureSpec:
        labels, source, target, values, hovers = build_sankey_components(
            self.data.predictions
        )
        fig = go.Figure(
            go.Sankey(
                node={"label": labels, "pad": 18, "thickness": 16},
                link={
                    "source": source,
                    "target": target,
                    "value": values,
                    "hovertemplate": "%{customdata}<extra></extra>",
                    "customdata": hovers,
                },
            )
        )
        fig.update_layout(
            title="Cascade flow: true label -> stage1 -> stage2/final",
            width=1350,
            height=760,
        )
        return FigureSpec(
            "09_cascade_sankey_flow",
            "Cascade Sankey flow",
            "Sankey links quantify how true labels pass through stage1 and stage2 into final labels.",
            ["cascade_test_predictions.csv"],
            [
                "true_label",
                "stage1_pred_label",
                "stage2_pred_label",
                "cascade_final_pred_label",
            ],
            fig,
        )

    def cascade_stage_flow_stacked(self) -> FigureSpec:
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
        flow = stage_flow_counts(self.data.predictions)
        fig = go.Figure()
        for true_label in FINAL_LABELS:
            values = [flow[true_label].get(category, 0) for category in categories]
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    name=true_label,
                    text=[str(value) for value in values],
                    textposition="outside",
                    hovertemplate=(
                        f"true={true_label}<br>bucket=%{{x}}<br>count=%{{y}}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title="Cascade stage-flow counts by true label",
            xaxis_title="Stage-flow bucket",
            yaxis_title="Count across folds",
            barmode="group",
            width=1500,
            height=680,
            legend={"orientation": "h", "y": 1.08},
        )
        fig.update_xaxes(tickangle=-35)
        return FigureSpec(
            "10_cascade_stage_flow_stacked",
            "Cascade stage-flow grouped bars",
            "Grouped bars expose stage1 decisions, stage2 execution, stage2 subtype predictions, and final labels.",
            ["cascade_test_predictions.csv"],
            categories,
            fig,
        )

    def cascade_error_type_bar(self) -> FigureSpec:
        counts = Counter(record.cascade_error_type for record in self.data.predictions)
        total = sum(counts.values())
        names = sorted(counts, key=lambda key: (-counts[key], key))
        values = [counts[name] for name in names]
        ratios = [value / total if total else 0.0 for value in values]
        fig = go.Figure(
            go.Bar(
                x=names,
                y=values,
                text=[
                    f"n={value}<br>{ratio:.1%}"
                    for value, ratio in zip(values, ratios, strict=True)
                ],
                textposition="outside",
                marker={
                    "color": [
                        "#2ca02c" if "correct" in name else "#d62728" for name in names
                    ]
                },
                hovertemplate="error_type=%{x}<br>count=%{y}<br>rate=%{customdata:.4f}<extra></extra>",
                customdata=ratios,
            )
        )
        fig.update_layout(
            title="Cascade error type counts and rates",
            xaxis_title="Cascade outcome type",
            yaxis_title="Count across folds",
            width=1500,
            height=700,
        )
        fig.update_xaxes(tickangle=-35)
        return FigureSpec(
            "11_cascade_error_type_bar",
            "Cascade error type counts",
            "Bars quantify every cascade success and error propagation type with count and percentage labels.",
            ["cascade_test_predictions.csv"],
            ["cascade_error_type", "true_label"],
            fig,
        )

    def cascade_error_type_by_label(self) -> FigureSpec:
        error_types = sorted(
            {record.cascade_error_type for record in self.data.predictions}
        )
        table = error_type_by_label(self.data.predictions, error_types)
        matrix = np.asarray(
            [
                [table[(true_label, err)] for err in error_types]
                for true_label in FINAL_LABELS
            ],
            dtype=float,
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        ratios = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals > 0,
        )
        text = [
            [
                f"{ratios[row, col]:.2f}<br>n={int(matrix[row, col])}"
                for col in range(len(error_types))
            ]
            for row in range(len(FINAL_LABELS))
        ]
        fig = go.Figure(
            go.Heatmap(
                z=ratios,
                x=error_types,
                y=list(FINAL_LABELS),
                text=text,
                texttemplate="%{text}",
                hovertemplate="true=%{y}<br>type=%{x}<br>ratio=%{z:.6f}<extra></extra>",
                colorscale="Oranges",
                zmin=0,
                zmax=1,
                colorbar={"title": "Row ratio"},
            )
        )
        fig.update_layout(
            title="Cascade error type distribution by true label",
            xaxis_title="Cascade outcome type",
            yaxis_title="True label",
            width=1650,
            height=620,
        )
        fig.update_xaxes(tickangle=-35)
        return FigureSpec(
            "12_cascade_error_type_by_label",
            "Cascade error type by true label",
            "Each cell shows the within-true-label ratio and raw count for each cascade outcome type.",
            ["cascade_test_predictions.csv"],
            ["true_label", "cascade_error_type"],
            fig,
        )


class ProbabilityPlotBuilder:
    def __init__(self, data: LoadedReports):
        self.data = data

    def build(self) -> list[FigureSpec]:
        return [
            self.probability_distribution(),
            self.stage1_probability_distribution(),
            self.stage2_probability_distribution(),
            self.confidence_correct_vs_wrong(),
        ]

    def probability_distribution(self) -> FigureSpec:
        fig = go.Figure()
        for approach in APPROACHES:
            for true_label in FINAL_LABELS:
                values = [
                    record.confidence(approach)
                    for record in self.data.predictions
                    if record.true_label == true_label
                ]
                fig.add_trace(
                    go.Violin(
                        x=[f"{APPROACH_DISPLAY[approach]}<br>{true_label}"]
                        * len(values),
                        y=values,
                        name=f"{APPROACH_DISPLAY[approach]} {true_label}",
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        jitter=0.25,
                        hovertemplate="group=%{x}<br>confidence=%{y:.6f}<extra></extra>",
                        showlegend=False,
                    )
                )
        fig.update_layout(
            title="Final prediction confidence distribution by true label",
            xaxis_title="Approach and true label",
            yaxis_title="Predicted-label probability",
            width=1500,
            height=720,
        )
        return FigureSpec(
            "13_probability_distribution",
            "Final probability distribution",
            "Violin plots show the predicted-label probability for direct and cascade outputs by true label.",
            ["cascade_test_predictions.csv"],
            [
                "direct_prob_*",
                "cascade_prob_*",
                "true_label",
                "direct_pred_label",
                "cascade_final_pred_label",
            ],
            fig,
        )

    def stage1_probability_distribution(self) -> FigureSpec:
        fig = go.Figure()
        for true_label in FINAL_LABELS:
            records = [
                record
                for record in self.data.predictions
                if record.true_label == true_label
            ]
            fig.add_trace(
                go.Box(
                    x=[true_label] * len(records),
                    y=[record.stage1_prob_abnormal for record in records],
                    name=true_label,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    hovertemplate="true=%{x}<br>P(Abnormal)=%{y:.6f}<extra></extra>",
                )
            )
        thresholds = sorted(
            {record.stage1_threshold for record in self.data.predictions}
        )
        for threshold in thresholds:
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#666",
                annotation_text=f"threshold {threshold:.3f}",
                annotation_position="right",
            )
        fig.update_layout(
            title="Stage1 P(Abnormal) distribution by true label",
            xaxis_title="True label",
            yaxis_title="P(Abnormal)",
            width=1050,
            height=720,
            showlegend=False,
        )
        return FigureSpec(
            "14_stage1_probability_distribution",
            "Stage1 P(Abnormal) distribution",
            "Box plots show stage1 abnormal probability for each true label; dashed lines mark fold thresholds.",
            ["cascade_test_predictions.csv"],
            ["true_label", "stage1_prob_abnormal", "stage1_threshold"],
            fig,
        )

    def stage2_probability_distribution(self) -> FigureSpec:
        records = [record for record in self.data.predictions if record.stage2_executed]
        fig = go.Figure()
        groups = sorted(
            {f"{record.true_label}->{record.stage2_pred_label}" for record in records}
        )
        for group in groups:
            values = [
                record.stage2_prob_lung_parenchymal
                for record in records
                if f"{record.true_label}->{record.stage2_pred_label}" == group
                and record.stage2_prob_lung_parenchymal is not None
            ]
            fig.add_trace(
                go.Box(
                    x=[group] * len(values),
                    y=values,
                    name=group,
                    boxpoints="all",
                    jitter=0.25,
                    hovertemplate="group=%{x}<br>P(Lung_Parenchymal)=%{y:.6f}<extra></extra>",
                )
            )
        thresholds = sorted(
            {
                record.stage2_threshold
                for record in records
                if record.stage2_threshold is not None
            }
        )
        for threshold in thresholds:
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#666",
                annotation_text=f"threshold {threshold:.3f}",
                annotation_position="right",
            )
        fig.update_layout(
            title="Stage2 P(Lung_Parenchymal) for stage2-executed samples",
            xaxis_title="True label -> stage2 predicted label",
            yaxis_title="P(Lung_Parenchymal)",
            width=1500,
            height=720,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-25)
        return FigureSpec(
            "15_stage2_probability_distribution",
            "Stage2 P(Lung_Parenchymal) distribution",
            "Only stage2-executed samples are included; thresholds are shown as dashed references.",
            ["cascade_test_predictions.csv"],
            [
                "stage2_executed",
                "true_label",
                "stage2_pred_label",
                "stage2_prob_lung_parenchymal",
                "stage2_threshold",
            ],
            fig,
        )

    def confidence_correct_vs_wrong(self) -> FigureSpec:
        fig = go.Figure()
        for approach in APPROACHES:
            for correct in (True, False):
                label = "correct" if correct else "wrong"
                values = [
                    record.confidence(approach)
                    for record in self.data.predictions
                    if record.is_correct(approach) is correct
                ]
                fig.add_trace(
                    go.Box(
                        x=[f"{APPROACH_DISPLAY[approach]}<br>{label}"] * len(values),
                        y=values,
                        name=f"{APPROACH_DISPLAY[approach]} {label}",
                        boxpoints="all",
                        jitter=0.25,
                        hovertemplate="group=%{x}<br>confidence=%{y:.6f}<extra></extra>",
                    )
                )
        fig.update_layout(
            title="Confidence distribution for correct vs wrong predictions",
            xaxis_title="Approach and correctness",
            yaxis_title="Predicted-label probability",
            width=1150,
            height=720,
            showlegend=False,
        )
        return FigureSpec(
            "16_confidence_correct_vs_wrong",
            "Confidence for correct vs wrong predictions",
            "Box plots compare confidence when each approach is correct versus wrong.",
            ["cascade_test_predictions.csv"],
            ["direct_correct", "cascade_correct", "direct_prob_*", "cascade_prob_*"],
            fig,
        )


class VisualSummaryWriter(PlotlyExportMixin):
    def __init__(self, out_dir: str | Path, formats: set[str]):
        self.out_dir = Path(out_dir)
        self.formats = formats

    def write(self, figures: Sequence[FigureSpec], data: LoadedReports) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_figures: list[dict[str, Any]] = []
        for spec in figures:
            written = self.write_outputs(
                spec.figure, self.out_dir / spec.stem, formats=self.formats
            )
            manifest_figures.append(
                {
                    "stem": spec.stem,
                    "title": spec.title,
                    "caption": spec.caption,
                    "source_files": spec.source_files,
                    "source_columns": spec.source_columns,
                    "outputs": [str(path) for path in written],
                }
            )
        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "reports_root": str(data.reports_root.resolve()),
            "out_dir": str(self.out_dir.resolve()),
            "input_files": self._input_file_manifest(data),
            "figures": manifest_figures,
            "markdown_captions": data.markdown_captions,
        }
        manifest_path = self.out_dir / "visual_summary_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        index_path = self.out_dir / "visual_summary_index.md"
        index_path.write_text(self._build_index(figures, data), encoding="utf-8")
        self.logger.info("Wrote %s", manifest_path)
        self.logger.info("Wrote %s", index_path)

    @staticmethod
    def _input_file_manifest(data: LoadedReports) -> list[dict[str, Any]]:
        paths = [
            data.fold_metrics_path,
            data.predictions_path,
            data.cascade_summary_path,
            *data.cv_summary_paths,
            *data.markdown_paths,
        ]
        payload: list[dict[str, Any]] = []
        for path in paths:
            if path.exists():
                stat = path.stat()
                payload.append(
                    {
                        "path": str(path),
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                )
        return payload

    def _build_index(self, figures: Sequence[FigureSpec], data: LoadedReports) -> str:
        summary = metric_mean_std(data.fold_metrics)
        lines = [
            "# Disease-Group Visual Summary",
            "",
            f"- reports root: `{data.reports_root}`",
            f"- prediction rows: `{len(data.predictions)}`",
            f"- figure count: `{len(figures)}`",
            (
                "- direct macro F1: "
                f"`{summary['direct']['macro_f1']['mean']:.6f} +- {summary['direct']['macro_f1']['std']:.6f}`"
            ),
            (
                "- cascade macro F1: "
                f"`{summary['cascade']['macro_f1']['mean']:.6f} +- {summary['cascade']['macro_f1']['std']:.6f}`"
            ),
            "",
            "## Figures",
            "",
        ]
        for spec in figures:
            lines.extend(
                [
                    f"### {spec.stem}",
                    "",
                    f"- title: {spec.title}",
                    f"- caption: {spec.caption}",
                    f"- source files: `{', '.join(spec.source_files)}`",
                    f"- source columns: `{', '.join(spec.source_columns)}`",
                    f"- outputs: `{spec.stem}.html`, `{spec.stem}.png`, `{spec.stem}.pdf`",
                    "",
                ]
            )
        return "\n".join(lines)


def metric_mean_std(
    fold_metrics: Sequence[FoldMetricRecord],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for approach in APPROACHES:
        summary[approach] = {}
        for metric in METRICS:
            values = np.asarray(
                [record.values[f"{approach}_{metric}"] for record in fold_metrics],
                dtype=float,
            )
            summary[approach][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size >= 2 else 0.0,
            }
    return summary


def confusion_counts(records: Sequence[PredictionRecord], approach: str) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(FINAL_LABELS)}
    matrix = np.zeros((len(FINAL_LABELS), len(FINAL_LABELS)), dtype=int)
    for record in records:
        matrix[
            label_to_index[record.true_label],
            label_to_index[record.predicted_label(approach)],
        ] += 1
    return matrix


def row_normalize(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=float),
        where=totals > 0,
    )


def final_outcome_table(
    records: Sequence[PredictionRecord],
    approach: str,
) -> dict[tuple[str, str], int]:
    table: dict[tuple[str, str], int] = {
        (true_label, pred_label): 0
        for true_label in FINAL_LABELS
        for pred_label in FINAL_LABELS
    }
    for record in records:
        table[(record.true_label, record.predicted_label(approach))] += 1
    return table


def stage_flow_counts(
    records: Sequence[PredictionRecord],
) -> dict[str, Counter[str]]:
    flow: dict[str, Counter[str]] = {label: Counter() for label in FINAL_LABELS}
    for record in records:
        flow[record.true_label][f"stage1_{record.stage1_pred_label}"] += 1
        if record.stage2_executed:
            flow[record.true_label]["stage2_executed"] += 1
            flow[record.true_label][f"stage2_{record.stage2_pred_label}"] += 1
        else:
            flow[record.true_label]["stage2_not_executed"] += 1
        flow[record.true_label][f"final_{record.cascade_final_pred_label}"] += 1
    return flow


def error_type_by_label(
    records: Sequence[PredictionRecord],
    error_types: Sequence[str],
) -> dict[tuple[str, str], int]:
    table = {
        (true_label, error_type): 0
        for true_label in FINAL_LABELS
        for error_type in error_types
    }
    for record in records:
        table[(record.true_label, record.cascade_error_type)] += 1
    return table


def build_sankey_components(
    records: Sequence[PredictionRecord],
) -> tuple[list[str], list[int], list[int], list[int], list[str]]:
    labels: list[str] = []
    node_index: dict[str, int] = {}

    def node(label: str) -> int:
        if label not in node_index:
            node_index[label] = len(labels)
            labels.append(label)
        return node_index[label]

    links: Counter[tuple[str, str]] = Counter()
    for record in records:
        true_node = f"True: {record.true_label}"
        stage1_node = f"Stage1: {record.stage1_pred_label}"
        links[(true_node, stage1_node)] += 1
        if record.stage2_executed:
            stage2_node = f"Stage2: {record.stage2_pred_label}"
            final_node = f"Final: {record.cascade_final_pred_label}"
            links[(stage1_node, stage2_node)] += 1
            links[(stage2_node, final_node)] += 1
        else:
            final_node = "Final: Normal"
            links[(stage1_node, final_node)] += 1

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    hovers: list[str] = []
    for (source_label, target_label), value in sorted(links.items()):
        sources.append(node(source_label))
        targets.append(node(target_label))
        values.append(value)
        hovers.append(f"{source_label} -> {target_label}<br>count={value}")
    return labels, sources, targets, values, hovers


def collect_figures(data: LoadedReports) -> list[FigureSpec]:
    figures: list[FigureSpec] = []
    figures.extend(MetricPlotBuilder(data).build()[:4])
    figures.extend(ConfusionPlotBuilder(data).build())
    figures.extend(CascadeFlowPlotBuilder(data).build())
    figures.extend(ProbabilityPlotBuilder(data).build())
    metric_context = MetricPlotBuilder(data).build()[4:]
    figures.extend(metric_context)
    return sorted(figures, key=lambda spec: spec.stem)


def parse_formats(raw: str) -> set[str]:
    formats = {item.strip() for item in raw.split(",") if item.strip()}
    unsupported = formats - {"html", "png", "pdf"}
    if unsupported:
        raise ValueError(f"Unsupported formats: {sorted(unsupported)}")
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", default="html,png,pdf")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    enable_file_logging(out_dir / "plot_disease_group_visual_summary.log", mode="w")
    loader = DiseaseGroupReportLoader(args.reports_root)
    data = loader.load()
    figures = collect_figures(data)
    VisualSummaryWriter(out_dir, parse_formats(args.formats)).write(figures, data)
    logger.info("Generated %d visual summary figures under %s", len(figures), out_dir)


if __name__ == "__main__":
    main()
