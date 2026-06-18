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

from src.cli.build_disease_dataset_catalog import CATALOG_COLUMNS, WARNING_COLUMNS
from src.plots.export import PlotlyExportMixin
from src.utils.logging import enable_file_logging, logger

DEFAULT_CATALOG_DIR = Path(
    "/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/"
    "DISEASE_CNUH_DATA/disease_dataset_catalog"
)
PRIMARY_LABELS = ("Normal", "Airway", "Lung_Parenchymal")
BINARY_LABELS = ("Normal", "Abnormal")
SPLITS = ("fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "test")
QUALITY_COLUMNS = (
    "has_excel_metadata",
    "has_split_membership",
    "label_matches_excel_source",
    "label_matches_split_folder",
)


@dataclass(frozen=True)
class CatalogRecord:
    filename_stem: str
    filename: str
    audio_path: str
    primary_label: str
    binary_label: str
    is_abnormal: bool
    excel_source_file: str
    excel_row_number: str
    auscultation_raw: str
    auscultation_code: str
    auscultation_label: str
    diagnosis_raw: str
    diagnosis_code: str
    diagnosis_label: str
    split: str
    split_role: str
    cv_validation_fold: str
    is_test: bool | None
    split_primary_label: str
    split_audio_path: str
    appears_in_abnormal_overlay: bool | None
    abnormal_overlay_path: str
    has_excel_metadata: bool
    has_split_membership: bool
    label_matches_excel_source: bool | None
    label_matches_split_folder: bool | None
    notes: str

    def value(self, field: str) -> str:
        raw = getattr(self, field)
        if raw is None:
            return "(missing)"
        if isinstance(raw, bool):
            return "True" if raw else "False"
        text = str(raw).strip()
        return text if text else "(missing)"


@dataclass(frozen=True)
class WarningRecord:
    warning_type: str
    filename_stem: str
    filename: str
    primary_label: str
    source_label: str
    path: str
    excel_source_file: str
    excel_row_number: str
    column: str
    raw_value: str
    message: str

    def value(self, field: str) -> str:
        text = str(getattr(self, field)).strip()
        return text if text else "(missing)"


@dataclass(frozen=True)
class LoadedCatalog:
    catalog_dir: Path
    catalog_path: Path
    summary_path: Path
    warnings_path: Path
    records: list[CatalogRecord]
    warnings: list[WarningRecord]
    summary_json: dict[str, Any]
    validation_warnings: list[str]


@dataclass(frozen=True)
class FigureSpec:
    stem: str
    title: str
    caption: str
    source_files: list[str]
    source_columns: list[str]
    figure: go.Figure


class DiseaseCatalogLoader:
    def __init__(self, catalog_dir: str | Path):
        self.catalog_dir = Path(catalog_dir)
        self.catalog_path = self.catalog_dir / "disease_dataset_catalog.csv"
        self.summary_path = self.catalog_dir / "disease_dataset_catalog_summary.json"
        self.warnings_path = self.catalog_dir / "disease_dataset_catalog_warnings.csv"

    def load(self) -> LoadedCatalog:
        for path in (self.catalog_path, self.summary_path, self.warnings_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required catalog input file not found: {path}"
                )
        records = self._load_records()
        warnings = self._load_warnings()
        summary_json = json.loads(self.summary_path.read_text(encoding="utf-8"))
        validation_warnings = CatalogAggregationBuilder(
            records, warnings
        ).compare_summary(summary_json)
        return LoadedCatalog(
            catalog_dir=self.catalog_dir,
            catalog_path=self.catalog_path,
            summary_path=self.summary_path,
            warnings_path=self.warnings_path,
            records=records,
            warnings=warnings,
            summary_json=summary_json,
            validation_warnings=validation_warnings,
        )

    def _load_records(self) -> list[CatalogRecord]:
        rows = self._read_csv(self.catalog_path, set(CATALOG_COLUMNS))
        records: list[CatalogRecord] = []
        for row_index, row in enumerate(rows, start=2):
            records.append(
                CatalogRecord(
                    filename_stem=row["filename_stem"],
                    filename=row["filename"],
                    audio_path=row["audio_path"],
                    primary_label=row["primary_label"],
                    binary_label=row["binary_label"],
                    is_abnormal=self._parse_bool(row, row_index, "is_abnormal"),
                    excel_source_file=row["excel_source_file"],
                    excel_row_number=row["excel_row_number"],
                    auscultation_raw=row["auscultation_raw"],
                    auscultation_code=row["auscultation_code"],
                    auscultation_label=row["auscultation_label"],
                    diagnosis_raw=row["diagnosis_raw"],
                    diagnosis_code=row["diagnosis_code"],
                    diagnosis_label=row["diagnosis_label"],
                    split=row["split"],
                    split_role=row["split_role"],
                    cv_validation_fold=row["cv_validation_fold"],
                    is_test=self._parse_optional_bool(row, row_index, "is_test"),
                    split_primary_label=row["split_primary_label"],
                    split_audio_path=row["split_audio_path"],
                    appears_in_abnormal_overlay=self._parse_optional_bool(
                        row, row_index, "appears_in_abnormal_overlay"
                    ),
                    abnormal_overlay_path=row["abnormal_overlay_path"],
                    has_excel_metadata=self._parse_bool(
                        row, row_index, "has_excel_metadata"
                    ),
                    has_split_membership=self._parse_bool(
                        row, row_index, "has_split_membership"
                    ),
                    label_matches_excel_source=self._parse_optional_bool(
                        row, row_index, "label_matches_excel_source"
                    ),
                    label_matches_split_folder=self._parse_optional_bool(
                        row, row_index, "label_matches_split_folder"
                    ),
                    notes=row["notes"],
                )
            )
        return records

    def _load_warnings(self) -> list[WarningRecord]:
        rows = self._read_csv(self.warnings_path, set(WARNING_COLUMNS))
        return [
            WarningRecord(
                warning_type=row["warning_type"],
                filename_stem=row["filename_stem"],
                filename=row["filename"],
                primary_label=row["primary_label"],
                source_label=row["source_label"],
                path=row["path"],
                excel_source_file=row["excel_source_file"],
                excel_row_number=row["excel_row_number"],
                column=row["column"],
                raw_value=row["raw_value"],
                message=row["message"],
            )
            for row in rows
        ]

    @staticmethod
    def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = required_columns - columns
            if missing:
                raise ValueError(
                    f"Required column(s) missing in {path}: {sorted(missing)}"
                )
            return [dict(row) for row in reader]

    def _parse_bool(self, row: Mapping[str, str], row_index: int, column: str) -> bool:
        value = self._parse_optional_bool(row, row_index, column)
        if value is None:
            raise ValueError(
                f"Missing boolean value in {self.catalog_path}:{row_index} column={column}"
            )
        return value

    def _parse_optional_bool(
        self, row: Mapping[str, str], row_index: int, column: str
    ) -> bool | None:
        raw = row[column].strip()
        if raw == "":
            return None
        if raw in {"True", "true", "1"}:
            return True
        if raw in {"False", "false", "0"}:
            return False
        raise ValueError(
            f"Invalid boolean value in {self.catalog_path}:{row_index} "
            f"column={column} value={raw!r}"
        )


class CatalogAggregationBuilder:
    def __init__(
        self, records: Sequence[CatalogRecord], warnings: Sequence[WarningRecord]
    ):
        self.records = list(records)
        self.warnings = list(warnings)

    def count_by(
        self, field: str, *, order: Sequence[str] | None = None
    ) -> Counter[str]:
        counter = Counter(record.value(field) for record in self.records)
        return reorder_counter(counter, order)

    def warning_count_by(self, field: str) -> Counter[str]:
        return Counter(warning.value(field) for warning in self.warnings)

    def crosstab(
        self,
        row_field: str,
        col_field: str,
        *,
        row_order: Sequence[str] | None = None,
        col_order: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        row_counter = self.count_by(row_field, order=row_order)
        col_counter = self.count_by(col_field, order=col_order)
        row_labels = list(row_counter)
        col_labels = list(col_counter)
        matrix = np.zeros((len(row_labels), len(col_labels)), dtype=np.int64)
        row_index = {label: index for index, label in enumerate(row_labels)}
        col_index = {label: index for index, label in enumerate(col_labels)}
        for record in self.records:
            matrix[
                row_index[record.value(row_field)], col_index[record.value(col_field)]
            ] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        ratios = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums != 0,
        )
        return row_labels, col_labels, matrix, ratios

    def stacked_counts(
        self,
        x_field: str,
        stack_field: str,
        *,
        x_order: Sequence[str] | None = None,
        stack_order: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str], dict[str, list[int]], dict[str, list[float]]]:
        x_labels, stack_labels, matrix, ratios = self.crosstab(
            x_field,
            stack_field,
            row_order=x_order,
            col_order=stack_order,
        )
        counts = {
            label: [int(value) for value in matrix[:, idx].tolist()]
            for idx, label in enumerate(stack_labels)
        }
        ratio_map = {
            label: [float(value) for value in ratios[:, idx].tolist()]
            for idx, label in enumerate(stack_labels)
        }
        return x_labels, stack_labels, counts, ratio_map

    def fold_vs_test_delta(self, field: str) -> list[dict[str, float | str]]:
        categories = sorted(
            {
                record.value(field)
                for record in self.records
                if record.split in set(SPLITS)
            }
        )
        fold_percentages: list[dict[str, float]] = []
        test_pct = {category: 0.0 for category in categories}
        for split in SPLITS:
            split_records = [record for record in self.records if record.split == split]
            if split == "test":
                test_pct = percentage_map(split_records, field, categories)
            else:
                fold_percentages.append(
                    percentage_map(split_records, field, categories)
                )
        if not fold_percentages:
            fold_mean = {category: 0.0 for category in categories}
        else:
            fold_mean = {
                category: float(np.mean([fold[category] for fold in fold_percentages]))
                for category in categories
            }
        return [
            {
                "category": category,
                "fold_mean_percent": fold_mean[category],
                "test_percent": test_pct[category],
                "delta_percentage_points": test_pct[category] - fold_mean[category],
            }
            for category in categories
        ]

    def coverage_summary(self) -> dict[str, Counter[str]]:
        summary: dict[str, Counter[str]] = {}
        for field in QUALITY_COLUMNS:
            summary[field] = self.count_by(field)
        return summary

    def recompute_summary(self) -> dict[str, Any]:
        return {
            "row_count": len(self.records),
            "primary_label_counts": dict(
                self.count_by("primary_label", order=PRIMARY_LABELS)
            ),
            "binary_label_counts": dict(
                self.count_by("binary_label", order=BINARY_LABELS)
            ),
            "split_counts": dict(self.count_by("split", order=SPLITS)),
            "auscultation_label_counts": dict(self.count_by("auscultation_label")),
            "diagnosis_label_counts": dict(self.count_by("diagnosis_label")),
            "warning_counts": dict(self.warning_count_by("warning_type")),
            "warning_count": len(self.warnings),
        }

    def compare_summary(self, summary_json: Mapping[str, Any]) -> list[str]:
        recomputed = self.recompute_summary()
        warnings: list[str] = []
        for key, value in recomputed.items():
            if key not in summary_json:
                warnings.append(f"summary_json missing key={key}")
                continue
            if summary_json[key] != value:
                warnings.append(
                    f"summary mismatch for {key}: csv={value} summary_json={summary_json[key]}"
                )
        return warnings


class DistributionPlotBuilder:
    def __init__(self, data: LoadedCatalog):
        self.data = data
        self.agg = CatalogAggregationBuilder(data.records, data.warnings)

    def build(self) -> list[FigureSpec]:
        return [
            self.dataset_overview_dashboard(),
            self.count_bar(
                "02_primary_label_distribution",
                "Primary label distribution",
                "primary_label",
                ["primary_label"],
                order=PRIMARY_LABELS,
            ),
            self.count_bar(
                "03_binary_label_distribution",
                "Binary label distribution",
                "binary_label",
                ["binary_label"],
                order=BINARY_LABELS,
            ),
            self.count_bar(
                "04_split_distribution",
                "Split distribution",
                "split",
                ["split"],
                order=SPLITS,
            ),
            self.stacked_bar(
                "05_primary_label_by_split_stacked",
                "Primary label composition by split",
                "split",
                "primary_label",
                ["split", "primary_label"],
                x_order=SPLITS,
                stack_order=PRIMARY_LABELS,
            ),
            self.stacked_bar(
                "06_binary_label_by_split_stacked",
                "Binary label composition by split",
                "split",
                "binary_label",
                ["split", "binary_label"],
                x_order=SPLITS,
                stack_order=BINARY_LABELS,
            ),
            self.abnormal_ratio_by_split(),
            self.count_bar(
                "08_auscultation_distribution",
                "Auscultation label distribution",
                "auscultation_label",
                ["auscultation_label"],
            ),
            self.count_bar(
                "09_diagnosis_distribution",
                "Diagnosis label distribution",
                "diagnosis_label",
                ["diagnosis_label"],
            ),
        ]

    def dataset_overview_dashboard(self) -> FigureSpec:
        recomputed = self.agg.recompute_summary()
        abnormal = recomputed["binary_label_counts"].get("Abnormal", 0)
        total = int(recomputed["row_count"])
        warning_count = int(recomputed["warning_count"])
        abnormal_ratio = abnormal / total if total else 0.0
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ],
            subplot_titles=(
                "Rows",
                "Warnings",
                "Abnormal ratio",
                "Primary label",
                "Binary label",
                "Split",
            ),
        )
        fig.add_trace(
            go.Indicator(mode="number", value=total, title={"text": "catalog rows"}),
            1,
            1,
        )
        fig.add_trace(
            go.Indicator(
                mode="number", value=warning_count, title={"text": "warning rows"}
            ),
            1,
            2,
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=100.0 * abnormal_ratio,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Abnormal / total"},
            ),
            1,
            3,
        )
        self._add_small_bar(fig, recomputed["primary_label_counts"], 2, 1)
        self._add_small_bar(fig, recomputed["binary_label_counts"], 2, 2)
        self._add_small_bar(fig, recomputed["split_counts"], 2, 3)
        fig.update_layout(
            title="Disease dataset catalog overview",
            width=1450,
            height=760,
            showlegend=False,
        )
        return FigureSpec(
            "01_dataset_overview_dashboard",
            "Dataset overview dashboard",
            "KPI cards and compact bars summarize rows, warnings, label balance, and split size.",
            ["disease_dataset_catalog.csv", "disease_dataset_catalog_summary.json"],
            ["primary_label", "binary_label", "split"],
            fig,
        )

    def count_bar(
        self,
        stem: str,
        title: str,
        field: str,
        source_columns: list[str],
        *,
        order: Sequence[str] | None = None,
    ) -> FigureSpec:
        counts = self.agg.count_by(field, order=order)
        fig = count_percent_bar(counts, title=title)
        return FigureSpec(
            stem,
            title,
            f"Bars show {field} counts and percentages over all catalog rows.",
            ["disease_dataset_catalog.csv"],
            source_columns,
            fig,
        )

    def stacked_bar(
        self,
        stem: str,
        title: str,
        x_field: str,
        stack_field: str,
        source_columns: list[str],
        *,
        x_order: Sequence[str] | None = None,
        stack_order: Sequence[str] | None = None,
    ) -> FigureSpec:
        x_labels, stack_labels, counts, ratios = self.agg.stacked_counts(
            x_field,
            stack_field,
            x_order=x_order,
            stack_order=stack_order,
        )
        fig = stacked_count_bar(x_labels, stack_labels, counts, ratios, title=title)
        return FigureSpec(
            stem,
            title,
            f"Stacked bars show {stack_field} composition within each {x_field}.",
            ["disease_dataset_catalog.csv"],
            source_columns,
            fig,
        )

    def abnormal_ratio_by_split(self) -> FigureSpec:
        x_labels, _, counts, ratios = self.agg.stacked_counts(
            "split",
            "binary_label",
            x_order=SPLITS,
            stack_order=BINARY_LABELS,
        )
        abnormal_counts = counts.get("Abnormal", [0 for _ in x_labels])
        abnormal_ratios = ratios.get("Abnormal", [0.0 for _ in x_labels])
        fold_values = [
            ratio
            for split, ratio in zip(x_labels, abnormal_ratios, strict=True)
            if split != "test"
        ]
        fold_mean = float(np.mean(fold_values)) if fold_values else 0.0
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=abnormal_ratios,
                text=[
                    f"{count}<br>{ratio:.1%}"
                    for count, ratio in zip(
                        abnormal_counts, abnormal_ratios, strict=True
                    )
                ],
                textposition="outside",
                customdata=abnormal_counts,
                hovertemplate="split=%{x}<br>abnormal_count=%{customdata}<br>ratio=%{y:.3%}<extra></extra>",
                marker_color="#c44e52",
            )
        )
        fig.add_hline(
            y=fold_mean,
            line_dash="dash",
            line_color="#4c72b0",
            annotation_text=f"fold mean {fold_mean:.1%}",
        )
        fig.update_layout(
            title="Abnormal ratio by split",
            xaxis_title="Split",
            yaxis_title="Abnormal ratio",
            width=1100,
            height=620,
        )
        return FigureSpec(
            "07_abnormal_ratio_by_split",
            "Abnormal ratio by split",
            "Bars show abnormal count and ratio; dashed line marks the validation-fold mean.",
            ["disease_dataset_catalog.csv"],
            ["split", "binary_label"],
            fig,
        )

    @staticmethod
    def _add_small_bar(
        fig: go.Figure, counts: Mapping[str, int], row: int, col: int
    ) -> None:
        labels = list(counts)
        values = [int(counts[label]) for label in labels]
        total = sum(values)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                text=[f"{value}<br>{safe_ratio(value, total):.1%}" for value in values],
                textposition="outside",
                hovertemplate="%{x}<br>count=%{y}<extra></extra>",
            ),
            row,
            col,
        )


class CrossTabPlotBuilder:
    def __init__(self, data: LoadedCatalog):
        self.data = data
        self.agg = CatalogAggregationBuilder(data.records, data.warnings)

    def build(self) -> list[FigureSpec]:
        specs = [
            (
                "10_primary_x_auscultation_heatmap",
                "Primary label x auscultation",
                "primary_label",
                "auscultation_label",
                PRIMARY_LABELS,
                None,
            ),
            (
                "11_primary_x_diagnosis_heatmap",
                "Primary label x diagnosis",
                "primary_label",
                "diagnosis_label",
                PRIMARY_LABELS,
                None,
            ),
            (
                "12_binary_x_auscultation_heatmap",
                "Binary label x auscultation",
                "binary_label",
                "auscultation_label",
                BINARY_LABELS,
                None,
            ),
            (
                "13_binary_x_diagnosis_heatmap",
                "Binary label x diagnosis",
                "binary_label",
                "diagnosis_label",
                BINARY_LABELS,
                None,
            ),
            (
                "14_auscultation_x_diagnosis_heatmap",
                "Auscultation x diagnosis",
                "auscultation_label",
                "diagnosis_label",
                None,
                None,
            ),
            (
                "15_split_x_primary_heatmap",
                "Split x primary label",
                "split",
                "primary_label",
                SPLITS,
                PRIMARY_LABELS,
            ),
            (
                "16_split_x_auscultation_heatmap",
                "Split x auscultation",
                "split",
                "auscultation_label",
                SPLITS,
                None,
            ),
            (
                "17_split_x_diagnosis_heatmap",
                "Split x diagnosis",
                "split",
                "diagnosis_label",
                SPLITS,
                None,
            ),
        ]
        return [
            self.heatmap(stem, title, row_field, col_field, row_order, col_order)
            for stem, title, row_field, col_field, row_order, col_order in specs
        ]

    def heatmap(
        self,
        stem: str,
        title: str,
        row_field: str,
        col_field: str,
        row_order: Sequence[str] | None,
        col_order: Sequence[str] | None,
    ) -> FigureSpec:
        rows, cols, counts, ratios = self.agg.crosstab(
            row_field,
            col_field,
            row_order=row_order,
            col_order=col_order,
        )
        fig = annotated_heatmap(rows, cols, counts, ratios, title=title)
        return FigureSpec(
            stem,
            title,
            f"Heatmap cells show raw count and row percentage for {row_field} by {col_field}.",
            ["disease_dataset_catalog.csv"],
            [row_field, col_field],
            fig,
        )


class DeltaPlotBuilder:
    def __init__(self, data: LoadedCatalog):
        self.data = data
        self.agg = CatalogAggregationBuilder(data.records, data.warnings)

    def build(self) -> list[FigureSpec]:
        return [
            self.delta_bar(
                "18_fold_vs_test_primary_delta",
                "Test minus fold-mean primary distribution",
                "primary_label",
                ["split", "primary_label"],
            ),
            self.delta_bar(
                "19_fold_vs_test_auscultation_delta",
                "Test minus fold-mean auscultation distribution",
                "auscultation_label",
                ["split", "auscultation_label"],
            ),
            self.delta_bar(
                "20_fold_vs_test_diagnosis_delta",
                "Test minus fold-mean diagnosis distribution",
                "diagnosis_label",
                ["split", "diagnosis_label"],
            ),
        ]

    def delta_bar(
        self,
        stem: str,
        title: str,
        field: str,
        source_columns: list[str],
    ) -> FigureSpec:
        rows = self.agg.fold_vs_test_delta(field)
        categories = [str(row["category"]) for row in rows]
        deltas = [float(row["delta_percentage_points"]) for row in rows]
        colors = ["#55a868" if value >= 0 else "#c44e52" for value in deltas]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=deltas,
                    marker_color=colors,
                    text=[f"{value:+.1f} pp" for value in deltas],
                    textposition="outside",
                    customdata=[
                        [row["fold_mean_percent"], row["test_percent"]] for row in rows
                    ],
                    hovertemplate=(
                        "category=%{x}<br>delta=%{y:+.3f} pp"
                        "<br>fold_mean=%{customdata[0]:.3f}%"
                        "<br>test=%{customdata[1]:.3f}%<extra></extra>"
                    ),
                )
            ]
        )
        fig.add_hline(y=0.0, line_color="black", line_width=1)
        fig.update_layout(
            title=title,
            xaxis_title=field,
            yaxis_title="Test - validation fold mean (percentage points)",
            width=max(1000, 80 * len(categories)),
            height=650,
        )
        return FigureSpec(
            stem,
            title,
            f"Bars show percentage-point difference between test distribution and validation-fold mean for {field}.",
            ["disease_dataset_catalog.csv"],
            source_columns,
            fig,
        )


class FlowPlotBuilder:
    def __init__(self, data: LoadedCatalog):
        self.data = data

    def build(self) -> list[FigureSpec]:
        return [self.label_auscultation_diagnosis(), self.primary_to_binary()]

    def label_auscultation_diagnosis(self) -> FigureSpec:
        source_target: Counter[tuple[str, str]] = Counter()
        for record in self.data.records:
            primary = f"Primary: {record.primary_label}"
            auscultation = f"Auscultation: {record.value('auscultation_label')}"
            diagnosis = f"Diagnosis: {record.value('diagnosis_label')}"
            source_target[(primary, auscultation)] += 1
            source_target[(auscultation, diagnosis)] += 1
        fig = sankey_from_links(
            source_target,
            title="Primary label -> auscultation -> diagnosis flow",
        )
        return FigureSpec(
            "21_label_to_auscultation_to_diagnosis_sankey",
            "Primary to auscultation to diagnosis Sankey",
            "Sankey links quantify how primary labels connect to auscultation and diagnosis annotations.",
            ["disease_dataset_catalog.csv"],
            ["primary_label", "auscultation_label", "diagnosis_label"],
            fig,
        )

    def primary_to_binary(self) -> FigureSpec:
        source_target = Counter(
            (f"Primary: {record.primary_label}", f"Binary: {record.binary_label}")
            for record in self.data.records
        )
        fig = sankey_from_links(
            source_target, title="Primary label -> binary label flow"
        )
        return FigureSpec(
            "22_primary_to_binary_sankey",
            "Primary to binary Sankey",
            "Sankey links show Airway and Lung_Parenchymal merging into the Abnormal binary group.",
            ["disease_dataset_catalog.csv"],
            ["primary_label", "binary_label"],
            fig,
        )


class QualityPlotBuilder:
    def __init__(self, data: LoadedCatalog):
        self.data = data
        self.agg = CatalogAggregationBuilder(data.records, data.warnings)

    def build(self) -> list[FigureSpec]:
        return [
            self.metadata_coverage(),
            self.warning_type_distribution(),
            self.warning_by_source_label(),
            self.abnormal_overlay_membership(),
            self.annotation_raw_code_quality(),
            self.catalog_file_index(),
        ]

    def metadata_coverage(self) -> FigureSpec:
        coverage = self.agg.coverage_summary()
        metrics = list(coverage)
        true_counts = [coverage[metric].get("True", 0) for metric in metrics]
        false_counts = [coverage[metric].get("False", 0) for metric in metrics]
        missing_counts = [coverage[metric].get("(missing)", 0) for metric in metrics]
        total = len(self.data.records)
        fig = go.Figure()
        for label, values, color in (
            ("True", true_counts, "#55a868"),
            ("False", false_counts, "#c44e52"),
            ("Missing", missing_counts, "#8172b2"),
        ):
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values,
                    name=label,
                    marker_color=color,
                    text=[
                        f"{value}<br>{safe_ratio(value, total):.1%}" for value in values
                    ],
                    textposition="inside",
                )
            )
        fig.update_layout(
            title="Metadata and consistency coverage",
            barmode="stack",
            xaxis_title="Quality field",
            yaxis_title="Rows",
            width=1250,
            height=650,
        )
        return FigureSpec(
            "23_metadata_coverage_summary",
            "Metadata coverage summary",
            "Stacked bars summarize metadata, split membership, and label consistency fields.",
            ["disease_dataset_catalog.csv"],
            list(QUALITY_COLUMNS),
            fig,
        )

    def warning_type_distribution(self) -> FigureSpec:
        counts = self.agg.warning_count_by("warning_type")
        fig = count_percent_bar(counts, title="Warning type distribution")
        return FigureSpec(
            "24_warning_type_distribution",
            "Warning type distribution",
            "Bars show warning counts and percentages over all warning rows.",
            ["disease_dataset_catalog_warnings.csv"],
            ["warning_type"],
            fig,
        )

    def warning_by_source_label(self) -> FigureSpec:
        counts = self.agg.warning_count_by("source_label")
        fig = count_percent_bar(counts, title="Warnings by source label")
        return FigureSpec(
            "25_warning_by_source_label",
            "Warnings by source label",
            "Bars show warning counts by source label from the warning table.",
            ["disease_dataset_catalog_warnings.csv"],
            ["source_label"],
            fig,
        )

    def abnormal_overlay_membership(self) -> FigureSpec:
        x_labels, stack_labels, counts, ratios = self.agg.stacked_counts(
            "primary_label",
            "appears_in_abnormal_overlay",
            x_order=PRIMARY_LABELS,
            stack_order=("False", "True", "(missing)"),
        )
        fig = stacked_count_bar(
            x_labels,
            stack_labels,
            counts,
            ratios,
            title="Abnormal overlay membership by primary label",
        )
        return FigureSpec(
            "26_abnormal_overlay_membership_summary",
            "Abnormal overlay membership summary",
            "Stacked bars show whether records appear in the split-local Abnormal overlay.",
            ["disease_dataset_catalog.csv"],
            ["primary_label", "appears_in_abnormal_overlay"],
            fig,
        )

    def annotation_raw_code_quality(self) -> FigureSpec:
        metrics = ["raw_present", "code_present", "label_present"]
        types = ["auscultation", "diagnosis"]
        values: dict[str, list[int]] = {}
        for annotation_type in types:
            values[annotation_type] = [
                sum(
                    bool(getattr(record, f"{annotation_type}_raw"))
                    for record in self.data.records
                ),
                sum(
                    bool(getattr(record, f"{annotation_type}_code"))
                    for record in self.data.records
                ),
                sum(
                    bool(getattr(record, f"{annotation_type}_label"))
                    for record in self.data.records
                ),
            ]
        total = len(self.data.records)
        fig = go.Figure()
        for annotation_type in types:
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values[annotation_type],
                    name=annotation_type,
                    text=[
                        f"{value}<br>{safe_ratio(value, total):.1%}"
                        for value in values[annotation_type]
                    ],
                    textposition="outside",
                )
            )
        fig.update_layout(
            title="Annotation raw/code/label coverage",
            barmode="group",
            xaxis_title="Coverage metric",
            yaxis_title="Rows",
            width=1100,
            height=620,
        )
        return FigureSpec(
            "27_annotation_raw_code_quality",
            "Annotation raw/code/label quality",
            "Grouped bars show coverage for raw annotation, parsed code, and decoded label fields.",
            ["disease_dataset_catalog.csv"],
            [
                "auscultation_raw",
                "auscultation_code",
                "auscultation_label",
                "diagnosis_raw",
                "diagnosis_code",
                "diagnosis_label",
            ],
            fig,
        )

    def catalog_file_index(self) -> FigureSpec:
        input_paths = [
            self.data.catalog_path,
            self.data.summary_path,
            self.data.warnings_path,
        ]
        rows = []
        for path in input_paths:
            stat = path.stat()
            rows.append(
                [
                    path.name,
                    str(path),
                    stat.st_size,
                    datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                ]
            )
        rows.append(["catalog rows", "", len(self.data.records), ""])
        rows.append(["warning rows", "", len(self.data.warnings), ""])
        fig = go.Figure(
            data=[
                go.Table(
                    header={"values": ["Item", "Path", "Value/Size", "Modified"]},
                    cells={"values": list(map(list, zip(*rows, strict=True)))},
                )
            ]
        )
        fig.update_layout(title="Catalog file index", width=1450, height=520)
        return FigureSpec(
            "28_catalog_file_index",
            "Catalog file index",
            "Table records input files, sizes, modified times, and current row/warning counts.",
            [
                "disease_dataset_catalog.csv",
                "disease_dataset_catalog_summary.json",
                "disease_dataset_catalog_warnings.csv",
            ],
            ["input_files"],
            fig,
        )


class CatalogVisualSummaryWriter(PlotlyExportMixin):
    def __init__(self, out_dir: str | Path, formats: set[str]):
        self.out_dir = Path(out_dir)
        self.formats = formats

    def write(self, figures: Sequence[FigureSpec], data: LoadedCatalog) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_figures: list[dict[str, Any]] = []
        for spec in figures:
            written = self.write_outputs(
                spec.figure,
                self.out_dir / spec.stem,
                formats=self.formats,
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
            "catalog_dir": str(data.catalog_dir.resolve()),
            "out_dir": str(self.out_dir.resolve()),
            "input_files": self._input_file_manifest(data),
            "figure_count": len(figures),
            "figures": manifest_figures,
            "recomputed_summary": CatalogAggregationBuilder(
                data.records, data.warnings
            ).recompute_summary(),
            "summary_validation_warnings": data.validation_warnings,
        }
        manifest_path = self.out_dir / "disease_dataset_visual_summary_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        index_path = self.out_dir / "disease_dataset_visual_summary_index.md"
        index_path.write_text(self._build_index(figures, data), encoding="utf-8")
        self.logger.info("Wrote %s", manifest_path)
        self.logger.info("Wrote %s", index_path)

    @staticmethod
    def _input_file_manifest(data: LoadedCatalog) -> list[dict[str, Any]]:
        payload = []
        for path in (data.catalog_path, data.summary_path, data.warnings_path):
            stat = path.stat()
            payload.append(
                {"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size}
            )
        return payload

    def _build_index(self, figures: Sequence[FigureSpec], data: LoadedCatalog) -> str:
        summary = CatalogAggregationBuilder(
            data.records, data.warnings
        ).recompute_summary()
        lines = [
            "# Disease Dataset Catalog Visual Summary",
            "",
            f"- catalog dir: `{data.catalog_dir}`",
            f"- row count: `{summary['row_count']}`",
            f"- warning count: `{summary['warning_count']}`",
            f"- figure count: `{len(figures)}`",
            f"- primary label counts: `{summary['primary_label_counts']}`",
            f"- binary label counts: `{summary['binary_label_counts']}`",
            f"- split counts: `{summary['split_counts']}`",
            f"- auscultation counts: `{summary['auscultation_label_counts']}`",
            f"- diagnosis counts: `{summary['diagnosis_label_counts']}`",
            f"- warning counts: `{summary['warning_counts']}`",
        ]
        if data.validation_warnings:
            lines.extend(["", "## Validation Warnings", ""])
            lines.extend(f"- {warning}" for warning in data.validation_warnings)
        lines.extend(["", "## Figures", ""])
        for spec in figures:
            lines.extend(
                [
                    f"### {spec.stem}",
                    "",
                    f"- title: {spec.title}",
                    f"- caption: {spec.caption}",
                    f"- source files: `{', '.join(spec.source_files)}`",
                    f"- source columns: `{', '.join(spec.source_columns)}`",
                    "- outputs: "
                    + ", ".join(f"`{spec.stem}.{fmt}`" for fmt in sorted(self.formats)),
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"


def collect_figures(data: LoadedCatalog) -> list[FigureSpec]:
    figures: list[FigureSpec] = []
    figures.extend(DistributionPlotBuilder(data).build())
    figures.extend(CrossTabPlotBuilder(data).build())
    figures.extend(DeltaPlotBuilder(data).build())
    figures.extend(FlowPlotBuilder(data).build())
    figures.extend(QualityPlotBuilder(data).build())
    return sorted(figures, key=lambda spec: spec.stem)


def count_percent_bar(
    counts: Mapping[str, int],
    *,
    title: str,
    width: int | None = None,
    height: int = 640,
) -> go.Figure:
    labels = list(counts)
    values = [int(counts[label]) for label in labels]
    total = sum(values)
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{value}<br>{safe_ratio(value, total):.1%}" for value in values],
                textposition="outside",
                customdata=[safe_ratio(value, total) for value in values],
                hovertemplate="category=%{x}<br>count=%{y}<br>percent=%{customdata:.3%}<extra></extra>",
                marker_color="#4c72b0",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Count",
        width=width or max(900, 95 * max(1, len(labels))),
        height=height,
    )
    return fig


def stacked_count_bar(
    x_labels: Sequence[str],
    stack_labels: Sequence[str],
    counts: Mapping[str, Sequence[int]],
    ratios: Mapping[str, Sequence[float]],
    *,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for label in stack_labels:
        values = list(counts[label])
        ratio_values = list(ratios[label])
        fig.add_trace(
            go.Bar(
                x=list(x_labels),
                y=values,
                name=label,
                text=[
                    "" if value == 0 else f"{value}<br>{ratio:.1%}"
                    for value, ratio in zip(values, ratio_values, strict=True)
                ],
                textposition="inside",
                customdata=ratio_values,
                hovertemplate="x=%{x}<br>category="
                + label
                + "<br>count=%{y}<br>within_x=%{customdata:.3%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Group",
        yaxis_title="Count",
        width=1150,
        height=680,
        legend_title="Category",
    )
    return fig


def annotated_heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    counts: np.ndarray,
    ratios: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    text = np.empty(counts.shape, dtype=object)
    total = float(counts.sum())
    customdata = np.empty(counts.shape + (2,), dtype=np.float64)
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            count = int(counts[row, col])
            ratio = float(ratios[row, col])
            text[row, col] = f"{count}<br>{ratio:.1%}" if count else "0<br>0.0%"
            customdata[row, col, 0] = count
            customdata[row, col, 1] = safe_ratio(count, total)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=ratios,
                x=list(col_labels),
                y=list(row_labels),
                text=text,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar={"title": "row %"},
                customdata=customdata,
                hovertemplate=(
                    "row=%{y}<br>column=%{x}<br>count=%{customdata[0]:.0f}"
                    "<br>row_percent=%{z:.3%}<br>total_percent=%{customdata[1]:.3%}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Column category",
        yaxis_title="Row category",
        width=max(950, 100 * len(col_labels)),
        height=max(520, 80 * len(row_labels) + 220),
    )
    return fig


def sankey_from_links(
    source_target: Mapping[tuple[str, str], int], *, title: str
) -> go.Figure:
    labels: list[str] = []
    label_to_index: dict[str, int] = {}
    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    for (source, target), count in sorted(source_target.items()):
        for label in (source, target):
            if label not in label_to_index:
                label_to_index[label] = len(labels)
                labels.append(label)
        sources.append(label_to_index[source])
        targets.append(label_to_index[target])
        values.append(int(count))
    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": labels, "pad": 14, "thickness": 16},
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "hovertemplate": "count=%{value}<extra></extra>",
                },
            )
        ]
    )
    fig.update_layout(title=title, width=1500, height=850)
    return fig


def percentage_map(
    records: Sequence[CatalogRecord], field: str, categories: Sequence[str]
) -> dict[str, float]:
    total = len(records)
    counts = Counter(record.value(field) for record in records)
    return {
        category: 100.0 * safe_ratio(counts.get(category, 0), total)
        for category in categories
    }


def reorder_counter(
    counter: Counter[str],
    order: Sequence[str] | None,
) -> Counter[str]:
    if order is None:
        ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    else:
        ordered = [(label, counter[label]) for label in order if label in counter]
        ordered.extend(
            sorted(
                [
                    (label, value)
                    for label, value in counter.items()
                    if label not in order
                ],
                key=lambda item: (-item[1], item[0]),
            )
        )
    return Counter(dict(ordered))


def safe_ratio(numerator: float, denominator: float) -> float:
    return (
        0.0
        if math.isclose(float(denominator), 0.0)
        else float(numerator) / float(denominator)
    )


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
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG_DIR))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--formats", default="html,png,pdf")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir)
    out_dir = Path(args.out_dir) if args.out_dir else catalog_dir / "visual_summary"
    enable_file_logging(out_dir / "plot_disease_dataset_catalog.log", mode="w")
    data = DiseaseCatalogLoader(catalog_dir).load()
    figures = collect_figures(data)
    CatalogVisualSummaryWriter(out_dir, parse_formats(args.formats)).write(
        figures, data
    )
    logger.info(
        "Generated %d disease dataset catalog figures under %s", len(figures), out_dir
    )


if __name__ == "__main__":
    main()
