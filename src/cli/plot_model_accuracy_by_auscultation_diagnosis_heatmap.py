from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import plotly.graph_objects as go

from src.plots.export import PlotlyExportMixin
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger

DEFAULT_CATALOG = Path(
    "/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/"
    "DISEASE_CNUH_DATA/disease_dataset_catalog/disease_dataset_catalog.csv"
)
DEFAULT_OUT_DIR = Path(
    "Disease_Group_Results/reports/model_accuracy_by_auscultation_diagnosis_heatmap"
)
DEFAULT_AST_PREDICTIONS = Path(
    "Disease_Group_Results/reports/cascade_test_comparison/"
    "AST_Partial_L1/cascade_test_predictions.csv"
)
DEFAULT_WHISPER_PREDICTIONS = Path(
    "Disease_Group_Results/reports/cascade_test_comparison/"
    "Whisper_Partial_L1/cascade_test_predictions.csv"
)
DEFAULT_RESNET50_PREDICTIONS = Path(
    "Disease_Group_Results/reports/cascade_test_comparison/"
    "ResNet50_Partial_L1/cascade_test_predictions.csv"
)

ROW_AUDIT_CSV = "model_accuracy_by_auscultation_diagnosis_rows.csv"
SUMMARY_CSV = "model_accuracy_by_auscultation_diagnosis_summary.csv"
SUMMARY_JSON = "model_accuracy_by_auscultation_diagnosis_summary.json"
AUSCULTATION_ORDER = ("non-specific", "crackle", "rhonchi", "wheeze")
DIAGNOSIS_ORDER = (
    "healthy",
    "lung cancer",
    "lung nodule",
    "pneumonia",
    "IPF",
    "ILD except IPF",
    "COPD",
    "asthma",
)
DIAGNOSIS_GROUP_ORDER = ("Normal", "Lung Parenchymal", "Airway")
DIAGNOSIS_GROUP_BY_DIAGNOSIS = {
    "healthy": "Normal",
    "lung cancer": "Lung Parenchymal",
    "lung nodule": "Lung Parenchymal",
    "pneumonia": "Lung Parenchymal",
    "IPF": "Lung Parenchymal",
    "ILD except IPF": "Lung Parenchymal",
    "COPD": "Airway",
    "asthma": "Airway",
}
RUN_LOG = "run.log"
X_AXIS_DOMAIN = (0.0, 0.92)
Y_AXIS_DOMAIN = (0.0, 0.84)
GROUP_BAND_Y0 = 0.855
GROUP_BAND_Y1 = 0.905
AUSCULTATION_DISPLAY_BY_RAW = {
    "non specific": "non-specific",
    "non-specific": "non-specific",
    "Crackle": "crackle",
    "crackle": "crackle",
    "wheezing": "wheeze",
    "wheeze": "wheeze",
    "rhonchi": "rhonchi",
}
DIAGNOSIS_DISPLAY_BY_RAW = {
    "lung cancer": "lung cancer",
    "lung nodule": "lung nodule",
    "healthy": "healthy",
    "COPD": "COPD",
    "asthma": "asthma",
    "IPF": "IPF",
    "pneumonia": "pneumonia",
    "ILD except IPF": "ILD except IPF",
    "ILD except IPF (CPFE included)": "ILD except IPF",
}

Strategy = Literal["direct", "cascade"]
STRATEGIES: tuple[Strategy, ...] = ("direct", "cascade")
SUPPORTED_FORMATS = {"html", "json", "png", "pdf"}
REQUIRED_CATALOG_COLUMNS = {
    "filename_stem",
    "split",
    "is_test",
    "auscultation_label",
    "diagnosis_label",
}
REQUIRED_PREDICTION_COLUMNS = {
    "fold",
    "audio_path",
    "true_label",
    "direct_pred_label",
    "direct_correct",
    "cascade_final_pred_label",
    "cascade_correct",
}


@dataclass(frozen=True)
class DiagnosisGroupSpan:
    group: str
    start_index: int
    end_index_exclusive: int
    x0_domain: float
    x1_domain: float
    center_domain: float
    diagnoses: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "start_index": self.start_index,
            "end_index_exclusive": self.end_index_exclusive,
            "x0_domain": self.x0_domain,
            "x1_domain": self.x1_domain,
            "center_domain": self.center_domain,
            "diagnoses": list(self.diagnoses),
        }


@dataclass(frozen=True)
class ModelReportSpec:
    model_family: str
    output_stem: str
    predictions_csv: Path


@dataclass(frozen=True)
class AccuracyHeatmapConfig:
    catalog: Path
    model_reports: tuple[ModelReportSpec, ...]
    out_dir: Path
    formats: set[str]
    install_chrome: bool


@dataclass(frozen=True)
class CatalogMetadata:
    filename_stem: str
    auscultation_label: str
    auscultation_display: str | None
    diagnosis_label: str
    diagnosis_display: str | None

    @property
    def is_supported_axis_cell(self) -> bool:
        return (
            self.auscultation_display is not None and self.diagnosis_display is not None
        )


@dataclass(frozen=True)
class PredictionRow:
    model_family: str
    model_stem: str
    fold: str
    audio_path: Path
    true_label: str
    direct_pred_label: str
    direct_correct: bool
    cascade_final_pred_label: str
    cascade_correct: bool

    @property
    def filename_stem(self) -> str:
        return self.audio_path.stem


@dataclass(frozen=True)
class JoinedPredictionRow:
    model_family: str
    model_stem: str
    strategy: Strategy
    fold: str
    audio_path: str
    filename_stem: str
    true_label: str
    pred_label: str
    correct: bool
    auscultation_label: str
    auscultation_display: str
    diagnosis_label: str
    diagnosis_display: str

    def to_row(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "strategy": self.strategy,
            "fold": self.fold,
            "audio_path": self.audio_path,
            "filename_stem": self.filename_stem,
            "true_label": self.true_label,
            "pred_label": self.pred_label,
            "correct": self.correct,
            "auscultation_label": self.auscultation_label,
            "auscultation_display": self.auscultation_display,
            "diagnosis_label": self.diagnosis_label,
            "diagnosis_display": self.diagnosis_display,
        }


@dataclass(frozen=True)
class AccuracyCell:
    model_family: str
    model_stem: str
    strategy: Strategy
    auscultation_display: str
    diagnosis_display: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total

    @property
    def accuracy_percent(self) -> float | None:
        if self.accuracy is None:
            return None
        return self.accuracy * 100.0

    @property
    def text(self) -> str:
        if self.accuracy is None:
            return "N/A<br>0/0"
        return f"{self.accuracy:.1%}<br>{self.correct}/{self.total}"

    def to_row(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "strategy": self.strategy,
            "auscultation_display": self.auscultation_display,
            "diagnosis_display": self.diagnosis_display,
            "correct": self.correct,
            "total": self.total,
            "accuracy": "" if self.accuracy is None else self.accuracy,
            "accuracy_percent": ""
            if self.accuracy_percent is None
            else self.accuracy_percent,
        }


@dataclass(frozen=True)
class BuildResult:
    joined_rows: list[JoinedPredictionRow]
    cells: list[AccuracyCell]
    excluded_prediction_row_counts: dict[str, int]
    input_prediction_row_counts: dict[str, int]


class CatalogMetadataLoader(LoggingMixin):
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, CatalogMetadata]:
        if not self.path.exists():
            raise FileNotFoundError(f"Catalog CSV does not exist: {self.path}")
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_CATALOG_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Catalog CSV missing required columns {sorted(missing)}: {self.path}"
                )
            records: dict[str, CatalogMetadata] = {}
            for row in reader:
                if not _is_test_catalog_row(row):
                    continue
                stem = row["filename_stem"]
                if stem in records:
                    raise ValueError(f"Duplicate test catalog filename_stem: {stem}")
                auscultation_label = str(row["auscultation_label"])
                diagnosis_label = str(row["diagnosis_label"])
                records[stem] = CatalogMetadata(
                    filename_stem=stem,
                    auscultation_label=auscultation_label,
                    auscultation_display=AUSCULTATION_DISPLAY_BY_RAW.get(
                        auscultation_label
                    ),
                    diagnosis_label=diagnosis_label,
                    diagnosis_display=DIAGNOSIS_DISPLAY_BY_RAW.get(diagnosis_label),
                )
        if not records:
            raise ValueError(f"No test catalog rows found in {self.path}")
        self.logger.info("Loaded %d test catalog rows", len(records))
        return records


class PredictionCsvReader(LoggingMixin):
    def __init__(self, spec: ModelReportSpec):
        self.spec = spec

    def read(self) -> list[PredictionRow]:
        path = self.spec.predictions_csv
        if not path.exists():
            raise FileNotFoundError(f"Prediction CSV does not exist: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_PREDICTION_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Prediction CSV missing required columns {sorted(missing)}: {path}"
                )
            rows = [
                PredictionRow(
                    model_family=self.spec.model_family,
                    model_stem=self.spec.output_stem,
                    fold=str(row["fold"]),
                    audio_path=Path(row["audio_path"]),
                    true_label=str(row["true_label"]),
                    direct_pred_label=str(row["direct_pred_label"]),
                    direct_correct=_parse_bool(row["direct_correct"]),
                    cascade_final_pred_label=str(row["cascade_final_pred_label"]),
                    cascade_correct=_parse_bool(row["cascade_correct"]),
                )
                for row in reader
            ]
        self.logger.info(
            "Loaded %d prediction rows for %s",
            len(rows),
            self.spec.model_family,
        )
        return rows


class AccuracyHeatmapBuilder(LoggingMixin):
    def __init__(self, config: AccuracyHeatmapConfig):
        self.config = config

    def build(self) -> BuildResult:
        catalog = CatalogMetadataLoader(self.config.catalog).load()
        joined_rows: list[JoinedPredictionRow] = []
        excluded: Counter[str] = Counter()
        input_counts: dict[str, int] = {}
        for spec in self.config.model_reports:
            prediction_rows = PredictionCsvReader(spec).read()
            input_counts[spec.model_family] = len(prediction_rows)
            for prediction in prediction_rows:
                metadata = catalog.get(prediction.filename_stem)
                if metadata is None:
                    raise ValueError(
                        "Prediction row does not join to test catalog: "
                        f"model={prediction.model_family} audio_path={prediction.audio_path}"
                    )
                if not metadata.is_supported_axis_cell:
                    reason = self._excluded_reason(metadata)
                    excluded[reason] += 1
                    continue
                joined_rows.extend(self._strategy_rows(prediction, metadata))
        cells = self._cells(joined_rows)
        self.logger.info(
            "Built %d joined strategy rows and %d summary cells",
            len(joined_rows),
            len(cells),
        )
        return BuildResult(
            joined_rows=joined_rows,
            cells=cells,
            excluded_prediction_row_counts=dict(sorted(excluded.items())),
            input_prediction_row_counts=input_counts,
        )

    @staticmethod
    def _excluded_reason(metadata: CatalogMetadata) -> str:
        if metadata.auscultation_display is None and metadata.diagnosis_display is None:
            return "unsupported_auscultation_and_diagnosis"
        if metadata.auscultation_display is None:
            return f"unsupported_auscultation:{metadata.auscultation_label}"
        return f"unsupported_diagnosis:{metadata.diagnosis_label}"

    @staticmethod
    def _strategy_rows(
        prediction: PredictionRow,
        metadata: CatalogMetadata,
    ) -> list[JoinedPredictionRow]:
        if metadata.auscultation_display is None or metadata.diagnosis_display is None:
            raise ValueError("Cannot create joined rows for unsupported axis labels")
        return [
            JoinedPredictionRow(
                model_family=prediction.model_family,
                model_stem=prediction.model_stem,
                strategy="direct",
                fold=prediction.fold,
                audio_path=str(prediction.audio_path),
                filename_stem=prediction.filename_stem,
                true_label=prediction.true_label,
                pred_label=prediction.direct_pred_label,
                correct=prediction.direct_correct,
                auscultation_label=metadata.auscultation_label,
                auscultation_display=metadata.auscultation_display,
                diagnosis_label=metadata.diagnosis_label,
                diagnosis_display=metadata.diagnosis_display,
            ),
            JoinedPredictionRow(
                model_family=prediction.model_family,
                model_stem=prediction.model_stem,
                strategy="cascade",
                fold=prediction.fold,
                audio_path=str(prediction.audio_path),
                filename_stem=prediction.filename_stem,
                true_label=prediction.true_label,
                pred_label=prediction.cascade_final_pred_label,
                correct=prediction.cascade_correct,
                auscultation_label=metadata.auscultation_label,
                auscultation_display=metadata.auscultation_display,
                diagnosis_label=metadata.diagnosis_label,
                diagnosis_display=metadata.diagnosis_display,
            ),
        ]

    def _cells(self, rows: Sequence[JoinedPredictionRow]) -> list[AccuracyCell]:
        cells: list[AccuracyCell] = []
        for spec in self.config.model_reports:
            for strategy in STRATEGIES:
                for auscultation in AUSCULTATION_ORDER:
                    for diagnosis in DIAGNOSIS_ORDER:
                        matching = [
                            row
                            for row in rows
                            if row.model_family == spec.model_family
                            and row.strategy == strategy
                            and row.auscultation_display == auscultation
                            and row.diagnosis_display == diagnosis
                        ]
                        cells.append(
                            AccuracyCell(
                                model_family=spec.model_family,
                                model_stem=spec.output_stem,
                                strategy=strategy,
                                auscultation_display=auscultation,
                                diagnosis_display=diagnosis,
                                correct=sum(1 for row in matching if row.correct),
                                total=len(matching),
                            )
                        )
        return cells


class AccuracyHeatmapWriter(PlotlyExportMixin):
    def __init__(self, config: AccuracyHeatmapConfig):
        self.config = config

    def write(self, result: BuildResult) -> dict[str, Any]:
        Fs.ensure_dir(self.config.out_dir)
        row_audit_path = self.config.out_dir / ROW_AUDIT_CSV
        summary_csv_path = self.config.out_dir / SUMMARY_CSV
        summary_json_path = self.config.out_dir / SUMMARY_JSON

        self._write_csv(row_audit_path, [row.to_row() for row in result.joined_rows])
        self._write_csv(summary_csv_path, [cell.to_row() for cell in result.cells])
        figure_outputs = self._write_figures(result.cells)

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "catalog": str(self.config.catalog),
            "out_dir": str(self.config.out_dir),
            "model_reports": [
                {
                    "model_family": spec.model_family,
                    "output_stem": spec.output_stem,
                    "predictions_csv": str(spec.predictions_csv),
                }
                for spec in self.config.model_reports
            ],
            "auscultation_order": list(AUSCULTATION_ORDER),
            "diagnosis_order": list(DIAGNOSIS_ORDER),
            "diagnosis_group_by_diagnosis": diagnosis_group_by_diagnosis(),
            "diagnosis_group_order": list(DIAGNOSIS_GROUP_ORDER),
            "diagnosis_group_spans": [
                span.to_row() for span in build_diagnosis_group_spans()
            ],
            "input_prediction_row_counts": result.input_prediction_row_counts,
            "total_input_prediction_rows": sum(
                result.input_prediction_row_counts.values()
            ),
            "total_joined_strategy_rows": len(result.joined_rows),
            "summary_cell_count": len(result.cells),
            "excluded_prediction_row_counts": result.excluded_prediction_row_counts,
            "row_audit_csv": str(row_audit_path),
            "summary_csv": str(summary_csv_path),
            "figures": {
                stem: [str(path) for path in paths]
                for stem, paths in figure_outputs.items()
            },
            "calculation_unit": "fold-level prediction row",
        }
        summary_json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", row_audit_path)
        self.logger.info("Wrote %s", summary_csv_path)
        self.logger.info("Wrote %s", summary_json_path)
        return summary

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            raise ValueError(f"Cannot write empty CSV: {path}")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_figures(
        self,
        cells: Sequence[AccuracyCell],
    ) -> dict[str, list[Path]]:
        outputs: dict[str, list[Path]] = {}
        for spec in self.config.model_reports:
            for strategy in STRATEGIES:
                stem = (
                    f"{spec.output_stem}_{strategy}_accuracy_by_"
                    "auscultation_x_diagnosis_heatmap"
                )
                fig = AccuracyFigureBuilder.build(
                    cells,
                    model_family=spec.model_family,
                    strategy=strategy,
                )
                outputs[stem] = self.write_outputs(
                    fig,
                    self.config.out_dir / stem,
                    formats=self.config.formats,
                    install_chrome=self.config.install_chrome,
                )
        return outputs


def diagnosis_group_by_diagnosis() -> dict[str, str]:
    return {
        diagnosis: DIAGNOSIS_GROUP_BY_DIAGNOSIS[diagnosis]
        for diagnosis in DIAGNOSIS_ORDER
    }


def build_diagnosis_group_spans() -> list[DiagnosisGroupSpan]:
    total = len(DIAGNOSIS_ORDER)
    spans: list[DiagnosisGroupSpan] = []
    for group in DIAGNOSIS_GROUP_ORDER:
        indices = [
            index
            for index, diagnosis in enumerate(DIAGNOSIS_ORDER)
            if DIAGNOSIS_GROUP_BY_DIAGNOSIS.get(diagnosis) == group
        ]
        if not indices:
            raise ValueError(f"No diagnoses found for diagnosis group: {group}")
        start_index = min(indices)
        end_index_exclusive = max(indices) + 1
        expected_indices = list(range(start_index, end_index_exclusive))
        if indices != expected_indices:
            raise ValueError(
                "Diagnosis group spans must be contiguous: "
                f"group={group} indices={indices}"
            )
        spans.append(
            DiagnosisGroupSpan(
                group=group,
                start_index=start_index,
                end_index_exclusive=end_index_exclusive,
                x0_domain=start_index / total,
                x1_domain=end_index_exclusive / total,
                center_domain=(start_index + end_index_exclusive) / (2 * total),
                diagnoses=tuple(DIAGNOSIS_ORDER[index] for index in indices),
            )
        )
    mapped_diagnoses = set(DIAGNOSIS_GROUP_BY_DIAGNOSIS)
    expected_diagnoses = set(DIAGNOSIS_ORDER)
    if mapped_diagnoses != expected_diagnoses:
        raise ValueError(
            "Diagnosis group mapping must match diagnosis order: "
            f"missing={sorted(expected_diagnoses - mapped_diagnoses)} "
            f"extra={sorted(mapped_diagnoses - expected_diagnoses)}"
        )
    return spans


class AccuracyFigureBuilder:
    @staticmethod
    def build(
        cells: Sequence[AccuracyCell],
        *,
        model_family: str,
        strategy: Strategy,
    ) -> go.Figure:
        cell_by_key = {
            (cell.auscultation_display, cell.diagnosis_display): cell
            for cell in cells
            if cell.model_family == model_family and cell.strategy == strategy
        }
        z: list[list[float | None]] = []
        text: list[list[str]] = []
        customdata: list[list[list[Any]]] = []
        for auscultation in AUSCULTATION_ORDER:
            z_row: list[float | None] = []
            text_row: list[str] = []
            custom_row: list[list[Any]] = []
            for diagnosis in DIAGNOSIS_ORDER:
                cell = cell_by_key[(auscultation, diagnosis)]
                z_row.append(cell.accuracy)
                text_row.append(cell.text)
                custom_row.append(
                    [
                        model_family,
                        strategy,
                        auscultation,
                        diagnosis,
                        "N/A" if cell.accuracy is None else f"{cell.accuracy:.3f}",
                        cell.correct,
                        cell.total,
                        "fold-level prediction row",
                    ]
                )
            z.append(z_row)
            text.append(text_row)
            customdata.append(custom_row)

        title = (
            f"{model_family} {strategy.title()} accuracy by auscultation and diagnosis"
        )
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=z,
                    x=list(DIAGNOSIS_ORDER),
                    y=list(AUSCULTATION_ORDER),
                    text=text,
                    texttemplate="%{text}",
                    customdata=customdata,
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    colorbar={"title": "Accuracy", "x": 0.965, "len": 0.82, "y": 0.41},
                    hovertemplate=(
                        "model=%{customdata[0]}<br>"
                        "strategy=%{customdata[1]}<br>"
                        "auscultation=%{customdata[2]}<br>"
                        "diagnosis=%{customdata[3]}<br>"
                        "accuracy=%{customdata[4]}<br>"
                        "correct=%{customdata[5]}<br>"
                        "total=%{customdata[6]}<br>"
                        "unit=%{customdata[7]}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title={"text": title, "x": 0.46, "xanchor": "center"},
            xaxis={"title": "Diagnosis", "domain": list(X_AXIS_DOMAIN)},
            yaxis={"title": "Auscultation", "domain": list(Y_AXIS_DOMAIN)},
            width=1250,
            height=700,
            margin={"t": 105, "r": 125, "b": 95, "l": 105},
            shapes=AccuracyFigureBuilder._diagnosis_group_shapes(),
            annotations=AccuracyFigureBuilder._diagnosis_group_annotations(),
        )
        return fig

    @staticmethod
    def _diagnosis_group_shapes() -> list[dict[str, Any]]:
        band_colors = {
            "Normal": "rgba(198, 219, 239, 0.45)",
            "Lung Parenchymal": "rgba(218, 218, 235, 0.45)",
            "Airway": "rgba(204, 235, 197, 0.45)",
        }
        shapes: list[dict[str, Any]] = []
        spans = build_diagnosis_group_spans()
        for span in spans:
            shapes.append(
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": _scale_x_domain(span.x0_domain),
                    "x1": _scale_x_domain(span.x1_domain),
                    "y0": GROUP_BAND_Y0,
                    "y1": GROUP_BAND_Y1,
                    "fillcolor": band_colors[span.group],
                    "line": {"color": "rgba(60, 60, 60, 0.55)", "width": 1},
                    "layer": "above",
                }
            )
        for span in spans[1:]:
            boundary_x = _scale_x_domain(span.x0_domain)
            shapes.append(
                {
                    "type": "line",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": boundary_x,
                    "x1": boundary_x,
                    "y0": Y_AXIS_DOMAIN[0],
                    "y1": GROUP_BAND_Y1,
                    "line": {"color": "rgba(40, 40, 40, 0.65)", "width": 1},
                    "layer": "above",
                }
            )
        return shapes

    @staticmethod
    def _diagnosis_group_annotations() -> list[dict[str, Any]]:
        return [
            {
                "xref": "paper",
                "yref": "paper",
                "x": _scale_x_domain(span.center_domain),
                "y": (GROUP_BAND_Y0 + GROUP_BAND_Y1) / 2,
                "text": span.group,
                "showarrow": False,
                "font": {"size": 14, "color": "#222222"},
                "align": "center",
                "xanchor": "center",
                "yanchor": "middle",
                "yshift": 2,
            }
            for span in build_diagnosis_group_spans()
        ]


def _scale_x_domain(value: float) -> float:
    x0, x1 = X_AXIS_DOMAIN
    return x0 + value * (x1 - x0)


def _is_test_catalog_row(row: Mapping[str, str]) -> bool:
    return str(row.get("split", "")) == "test" or _parse_bool(
        row.get("is_test", "False")
    )


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


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
        raise ValueError("--formats must include at least one output format")
    return formats


def default_model_reports(
    *,
    ast_predictions: Path,
    whisper_predictions: Path,
    resnet50_predictions: Path,
) -> tuple[ModelReportSpec, ...]:
    return (
        ModelReportSpec("AST", "ast", ast_predictions),
        ModelReportSpec("Whisper", "whisper", whisper_predictions),
        ModelReportSpec("ResNet50", "resnet50", resnet50_predictions),
    )


def parse_args(argv: Sequence[str] | None = None) -> AccuracyHeatmapConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--ast-predictions", type=Path, default=DEFAULT_AST_PREDICTIONS)
    parser.add_argument(
        "--whisper-predictions",
        type=Path,
        default=DEFAULT_WHISPER_PREDICTIONS,
    )
    parser.add_argument(
        "--resnet50-predictions",
        type=Path,
        default=DEFAULT_RESNET50_PREDICTIONS,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", action="append", default=None)
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args(argv)
    return AccuracyHeatmapConfig(
        catalog=args.catalog,
        model_reports=default_model_reports(
            ast_predictions=args.ast_predictions,
            whisper_predictions=args.whisper_predictions,
            resnet50_predictions=args.resnet50_predictions,
        ),
        out_dir=args.out_dir,
        formats=parse_formats(args.formats or ["html,json,png,pdf"]),
        install_chrome=bool(args.install_chrome),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    Fs.ensure_dir(config.out_dir)
    enable_file_logging(config.out_dir / RUN_LOG, mode="w")
    result = AccuracyHeatmapBuilder(config).build()
    summary = AccuracyHeatmapWriter(config).write(result)
    logger.info(
        "Wrote model accuracy by auscultation-diagnosis heatmaps | "
        "joined_strategy_rows=%d | summary_cells=%d | out_dir=%s",
        summary["total_joined_strategy_rows"],
        summary["summary_cell_count"],
        summary["out_dir"],
    )


if __name__ == "__main__":
    main()
