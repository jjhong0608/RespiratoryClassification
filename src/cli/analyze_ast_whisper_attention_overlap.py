from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.plots.export import PlotlyExportMixin
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger

DEFAULT_FAILURE_ROOT = "Disease_Group_Results/reports/ast_vs_whisper_airway_attention"
DEFAULT_BOTH_CORRECT_ROOT = (
    "Disease_Group_Results/reports/ast_vs_whisper_airway_attention_both_correct"
)
DEFAULT_OUT_DIR = (
    "Disease_Group_Results/reports/ast_vs_whisper_airway_attention_overlap"
)
AST_FRAME_SHIFT_SECONDS = 0.01
CASE_CSV = "attention_time_overlap_cases.csv"
SUMMARY_CSV = "attention_time_overlap_summary.csv"
SUMMARY_JSON = "attention_time_overlap_summary.json"
SelectionMode = Literal["ast_correct_whisper_normal", "both_correct_airway"]
SUPPORTED_FORMATS = {"html", "png", "pdf", "json"}
SUMMARY_METRICS = (
    "time_iou",
    "time_dice",
    "ast_overlap_ratio",
    "whisper_overlap_ratio",
    "whisper_outside_ast_ratio",
    "overlap_time_sec",
    "whisper_outside_ast_time_sec",
)


@dataclass(frozen=True)
class OverlapCliConfig:
    failure_root: Path
    both_correct_root: Path
    out_dir: Path
    top_k: int
    formats: set[str]
    install_chrome: bool


@dataclass(frozen=True)
class GroupSpec:
    selection: SelectionMode
    root: Path


@dataclass(frozen=True)
class TimeInterval:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def clipped(self, *, max_time_sec: float) -> TimeInterval | None:
        start = max(self.start_sec, 0.0)
        end = min(self.end_sec, max_time_sec)
        if end <= start:
            return None
        return TimeInterval(start, end)


@dataclass(frozen=True)
class CaseFiles:
    selection: SelectionMode
    case_dir: Path
    ast_top_patches_csv: Path
    whisper_top_time_spans_csv: Path
    ast_metadata_json: Path
    whisper_metadata_json: Path


@dataclass(frozen=True)
class AttentionMetadata:
    wav: str
    predicted_label: str
    max_time_sec: float | None


@dataclass(frozen=True)
class CaseOverlapResult:
    selection: SelectionMode
    fold: str
    audio_stem: str
    audio_path: str
    ast_pred_label: str
    whisper_pred_label: str
    ast_max_time_sec: float
    ast_top10_time_sec: float
    whisper_top10_time_sec: float
    whisper_inside_ast_time_sec: float
    whisper_outside_ast_time_sec: float
    whisper_outside_ast_ratio: float
    overlap_time_sec: float
    union_time_sec: float
    ast_overlap_ratio: float
    whisper_overlap_ratio: float
    time_iou: float
    time_dice: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SummaryRecord:
    selection: SelectionMode
    metric: str
    count: int
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min: float
    max: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisSummary:
    generated_at: str
    failure_root: str
    both_correct_root: str
    out_dir: str
    top_k: int
    total_cases: int
    case_csv: str
    summary_csv: str
    summary_records: list[dict[str, Any]]
    figures: dict[str, list[str]]


class IntervalSet:
    @staticmethod
    def validate(interval: TimeInterval) -> None:
        values = (interval.start_sec, interval.end_sec)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Interval has non-finite boundary: {interval}")
        if interval.end_sec <= interval.start_sec:
            raise ValueError(f"Interval must have positive duration: {interval}")

    @classmethod
    def union(cls, intervals: Sequence[TimeInterval]) -> list[TimeInterval]:
        if not intervals:
            raise ValueError("At least one interval is required")
        for interval in intervals:
            cls.validate(interval)
        sorted_intervals = sorted(intervals, key=lambda item: item.start_sec)
        merged: list[TimeInterval] = []
        for interval in sorted_intervals:
            if not merged or interval.start_sec > merged[-1].end_sec:
                merged.append(interval)
                continue
            previous = merged[-1]
            merged[-1] = TimeInterval(
                previous.start_sec,
                max(previous.end_sec, interval.end_sec),
            )
        return merged

    @staticmethod
    def duration(intervals: Sequence[TimeInterval]) -> float:
        return sum(interval.duration_sec for interval in intervals)

    @staticmethod
    def intersection_duration(
        left: Sequence[TimeInterval],
        right: Sequence[TimeInterval],
    ) -> float:
        i = 0
        j = 0
        total = 0.0
        while i < len(left) and j < len(right):
            start = max(left[i].start_sec, right[j].start_sec)
            end = min(left[i].end_sec, right[j].end_sec)
            if end > start:
                total += end - start
            if left[i].end_sec <= right[j].end_sec:
                i += 1
            else:
                j += 1
        return total

    @classmethod
    def clip_and_union(
        cls,
        intervals: Sequence[TimeInterval],
        *,
        max_time_sec: float,
    ) -> list[TimeInterval]:
        clipped = [
            interval.clipped(max_time_sec=max_time_sec) for interval in intervals
        ]
        valid = [interval for interval in clipped if interval is not None]
        if not valid:
            return []
        return cls.union(valid)


class CaseFileCollector(LoggingMixin):
    def __init__(self, specs: Sequence[GroupSpec]):
        self.specs = list(specs)

    def collect(self) -> list[CaseFiles]:
        cases: list[CaseFiles] = []
        for spec in self.specs:
            if not spec.root.exists():
                raise FileNotFoundError(
                    f"Attention result root does not exist: {spec.root}"
                )
            case_dirs = sorted(
                path
                for path in spec.root.iterdir()
                if path.is_dir() and path.name.startswith("fold_")
            )
            if not case_dirs:
                raise ValueError(f"No case directories found under {spec.root}")
            for case_dir in case_dirs:
                cases.append(self._case_files(spec.selection, case_dir))
        self.logger.info("Collected %d attention-overlap cases", len(cases))
        return cases

    @staticmethod
    def _case_files(selection: SelectionMode, case_dir: Path) -> CaseFiles:
        ast_dir = case_dir / "ast"
        whisper_dir = case_dir / "whisper"
        return CaseFiles(
            selection=selection,
            case_dir=case_dir,
            ast_top_patches_csv=_single_match(ast_dir, "*_top_patches.csv"),
            whisper_top_time_spans_csv=_single_match(
                whisper_dir,
                "*_top_time_spans.csv",
            ),
            ast_metadata_json=_single_match(ast_dir, "*_attention_metadata.json"),
            whisper_metadata_json=_single_match(
                whisper_dir,
                "*_attention_metadata.json",
            ),
        )


class MetadataReader:
    @staticmethod
    def read(path: Path) -> AttentionMetadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        prediction = _require_mapping(payload.get("prediction"), path, "prediction")
        predicted_label = str(prediction.get("predicted_label", ""))
        if not predicted_label:
            raise ValueError(f"Metadata missing prediction.predicted_label: {path}")
        return AttentionMetadata(
            wav=str(payload.get("wav", "")),
            predicted_label=predicted_label,
            max_time_sec=MetadataReader._ast_max_time_sec(payload),
        )

    @staticmethod
    def _ast_max_time_sec(payload: Mapping[str, Any]) -> float | None:
        patch_geometry = payload.get("patch_geometry")
        if not isinstance(patch_geometry, Mapping):
            return None
        max_length = patch_geometry.get("max_length")
        if not isinstance(max_length, (int, float)):
            return None
        return float(max_length) * AST_FRAME_SHIFT_SECONDS


class IntervalCsvReader:
    @staticmethod
    def read(path: Path, *, top_k: int) -> list[TimeInterval]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = {"rank", "time_start_sec", "time_end_sec"} - fieldnames
            if missing:
                raise ValueError(
                    f"Interval CSV missing columns {sorted(missing)}: {path}"
                )
            intervals: list[TimeInterval] = []
            for row in reader:
                rank = int(row["rank"])
                if rank > top_k:
                    continue
                intervals.append(
                    TimeInterval(
                        start_sec=float(row["time_start_sec"]),
                        end_sec=float(row["time_end_sec"]),
                    )
                )
        if not intervals:
            raise ValueError(f"No top-{top_k} intervals found in {path}")
        return intervals


class AttentionOverlapAnalyzer(LoggingMixin):
    def __init__(self, config: OverlapCliConfig):
        self.config = config

    def analyze(self) -> tuple[list[CaseOverlapResult], list[SummaryRecord]]:
        specs = [
            GroupSpec("ast_correct_whisper_normal", self.config.failure_root),
            GroupSpec("both_correct_airway", self.config.both_correct_root),
        ]
        case_files = CaseFileCollector(specs).collect()
        results = [self._analyze_case(case) for case in case_files]
        summaries = SummaryBuilder().summarize(results)
        return results, summaries

    def _analyze_case(self, case: CaseFiles) -> CaseOverlapResult:
        ast_metadata = MetadataReader.read(case.ast_metadata_json)
        whisper_metadata = MetadataReader.read(case.whisper_metadata_json)
        ast_intervals = IntervalCsvReader.read(
            case.ast_top_patches_csv,
            top_k=self.config.top_k,
        )
        whisper_intervals = IntervalCsvReader.read(
            case.whisper_top_time_spans_csv,
            top_k=self.config.top_k,
        )

        ast_union = IntervalSet.union(ast_intervals)
        whisper_union = IntervalSet.union(whisper_intervals)
        ast_max_time_sec = ast_metadata.max_time_sec or max(
            interval.end_sec for interval in ast_union
        )
        whisper_inside_ast = IntervalSet.clip_and_union(
            whisper_union,
            max_time_sec=ast_max_time_sec,
        )

        ast_duration = IntervalSet.duration(ast_union)
        whisper_duration = IntervalSet.duration(whisper_union)
        whisper_inside_duration = IntervalSet.duration(whisper_inside_ast)
        whisper_outside_duration = max(0.0, whisper_duration - whisper_inside_duration)
        overlap_duration = IntervalSet.intersection_duration(ast_union, whisper_union)
        union_duration = ast_duration + whisper_duration - overlap_duration

        fold, audio_stem = _parse_case_dir(case.case_dir)
        return CaseOverlapResult(
            selection=case.selection,
            fold=fold,
            audio_stem=audio_stem,
            audio_path=ast_metadata.wav,
            ast_pred_label=ast_metadata.predicted_label,
            whisper_pred_label=whisper_metadata.predicted_label,
            ast_max_time_sec=ast_max_time_sec,
            ast_top10_time_sec=ast_duration,
            whisper_top10_time_sec=whisper_duration,
            whisper_inside_ast_time_sec=whisper_inside_duration,
            whisper_outside_ast_time_sec=whisper_outside_duration,
            whisper_outside_ast_ratio=_safe_ratio(
                whisper_outside_duration,
                whisper_duration,
            ),
            overlap_time_sec=overlap_duration,
            union_time_sec=union_duration,
            ast_overlap_ratio=_safe_ratio(overlap_duration, ast_duration),
            whisper_overlap_ratio=_safe_ratio(overlap_duration, whisper_duration),
            time_iou=_safe_ratio(overlap_duration, union_duration),
            time_dice=_safe_ratio(
                2.0 * overlap_duration,
                ast_duration + whisper_duration,
            ),
        )


class SummaryBuilder:
    def summarize(self, results: Sequence[CaseOverlapResult]) -> list[SummaryRecord]:
        if not results:
            raise ValueError("At least one case result is required")
        records: list[SummaryRecord] = []
        selections = sorted({result.selection for result in results})
        for selection in selections:
            group = [result for result in results if result.selection == selection]
            for metric in SUMMARY_METRICS:
                values = [float(getattr(result, metric)) for result in group]
                records.append(self._record(selection, metric, values))
        return records

    @staticmethod
    def _record(
        selection: SelectionMode,
        metric: str,
        values: Sequence[float],
    ) -> SummaryRecord:
        ordered = sorted(values)
        std = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
        return SummaryRecord(
            selection=selection,
            metric=metric,
            count=len(ordered),
            mean=statistics.mean(ordered),
            std=std,
            median=statistics.median(ordered),
            q25=_percentile(ordered, 0.25),
            q75=_percentile(ordered, 0.75),
            min=min(ordered),
            max=max(ordered),
        )


class OverlapReportWriter(PlotlyExportMixin):
    def __init__(self, config: OverlapCliConfig):
        self.config = config

    def write(
        self,
        results: Sequence[CaseOverlapResult],
        summaries: Sequence[SummaryRecord],
    ) -> AnalysisSummary:
        Fs.ensure_dir(self.config.out_dir)
        case_csv = self.config.out_dir / CASE_CSV
        summary_csv = self.config.out_dir / SUMMARY_CSV
        summary_json = self.config.out_dir / SUMMARY_JSON
        self._write_csv(case_csv, [result.to_row() for result in results])
        self._write_csv(summary_csv, [summary.to_row() for summary in summaries])
        figures = self._write_figures(results)
        analysis_summary = AnalysisSummary(
            generated_at=datetime.now(UTC).isoformat(),
            failure_root=str(self.config.failure_root),
            both_correct_root=str(self.config.both_correct_root),
            out_dir=str(self.config.out_dir),
            top_k=self.config.top_k,
            total_cases=len(results),
            case_csv=str(case_csv),
            summary_csv=str(summary_csv),
            summary_records=[summary.to_row() for summary in summaries],
            figures={
                name: [str(path) for path in paths] for name, paths in figures.items()
            },
        )
        summary_json.write_text(
            json.dumps(asdict(analysis_summary), indent=2),
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", case_csv)
        self.logger.info("Wrote %s", summary_csv)
        self.logger.info("Wrote %s", summary_json)
        return analysis_summary

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
        results: Sequence[CaseOverlapResult],
    ) -> dict[str, list[Path]]:
        return {
            "overlap_metric_distributions": self.write_outputs(
                FigureBuilder.metric_distributions(results),
                self.config.out_dir / "overlap_metric_distributions",
                formats=self.config.formats,
                install_chrome=self.config.install_chrome,
            ),
            "overlap_scatter": self.write_outputs(
                FigureBuilder.overlap_scatter(results),
                self.config.out_dir / "overlap_scatter",
                formats=self.config.formats,
                install_chrome=self.config.install_chrome,
            ),
            "overlap_time_breakdown": self.write_outputs(
                FigureBuilder.time_breakdown(results),
                self.config.out_dir / "overlap_time_breakdown",
                formats=self.config.formats,
                install_chrome=self.config.install_chrome,
            ),
        }


class FigureBuilder:
    @staticmethod
    def metric_distributions(results: Sequence[CaseOverlapResult]) -> go.Figure:
        metrics = [
            ("time_iou", "Temporal IoU"),
            ("ast_overlap_ratio", "AST overlap ratio"),
            ("whisper_overlap_ratio", "Whisper overlap ratio"),
            ("whisper_outside_ast_ratio", "Whisper outside AST ratio"),
        ]
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=[label for _, label in metrics]
        )
        selections = _ordered_selections(results)
        for index, (metric, _) in enumerate(metrics):
            row = index // 2 + 1
            col = index % 2 + 1
            for selection in selections:
                group = [result for result in results if result.selection == selection]
                fig.add_trace(
                    go.Box(
                        y=[float(getattr(result, metric)) for result in group],
                        name=selection,
                        boxpoints="all",
                        jitter=0.25,
                        pointpos=0,
                        legendgroup=selection,
                        showlegend=index == 0,
                    ),
                    row=row,
                    col=col,
                )
        fig.update_layout(
            title="AST-Whisper top-10 attention time-overlap distributions",
            template="plotly_white",
            boxmode="group",
        )
        fig.update_yaxes(range=[0, 1])
        return fig

    @staticmethod
    def overlap_scatter(results: Sequence[CaseOverlapResult]) -> go.Figure:
        fig = go.Figure()
        for selection in _ordered_selections(results):
            group = [result for result in results if result.selection == selection]
            fig.add_trace(
                go.Scatter(
                    x=[result.ast_overlap_ratio for result in group],
                    y=[result.whisper_overlap_ratio for result in group],
                    mode="markers",
                    name=selection,
                    text=[f"{result.fold} | {result.audio_stem}" for result in group],
                    hovertemplate=(
                        "%{text}<br>"
                        "AST overlap=%{x:.3f}<br>"
                        "Whisper overlap=%{y:.3f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title="AST vs Whisper temporal overlap ratios",
            xaxis_title="AST top-10 time covered by Whisper",
            yaxis_title="Whisper top-10 time covered by AST",
            template="plotly_white",
        )
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        return fig

    @staticmethod
    def time_breakdown(results: Sequence[CaseOverlapResult]) -> go.Figure:
        metrics = [
            ("overlap_time_sec", "Overlap time (s)"),
            ("whisper_outside_ast_time_sec", "Whisper outside AST time (s)"),
        ]
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=[label for _, label in metrics]
        )
        selections = _ordered_selections(results)
        for index, (metric, _) in enumerate(metrics):
            for selection in selections:
                group = [result for result in results if result.selection == selection]
                fig.add_trace(
                    go.Box(
                        y=[float(getattr(result, metric)) for result in group],
                        name=selection,
                        boxpoints="all",
                        jitter=0.25,
                        pointpos=0,
                        legendgroup=selection,
                        showlegend=index == 0,
                    ),
                    row=1,
                    col=index + 1,
                )
        fig.update_layout(
            title="Top-10 attention time-overlap breakdown",
            template="plotly_white",
            boxmode="group",
        )
        return fig


def _single_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one file matching {pattern!r} under {root}, got {len(matches)}"
        )
    return matches[0]


def _require_mapping(value: object, path: Path, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Metadata missing mapping {key!r}: {path}")
    return value


def _parse_case_dir(case_dir: Path) -> tuple[str, str]:
    if "__" not in case_dir.name:
        raise ValueError(f"Case directory name must contain '__': {case_dir}")
    fold, audio_stem = case_dir.name.split("__", 1)
    if not fold or not audio_stem:
        raise ValueError(f"Invalid case directory name: {case_dir}")
    return fold, audio_stem


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for empty values")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _ordered_selections(results: Sequence[CaseOverlapResult]) -> list[SelectionMode]:
    preferred: list[SelectionMode] = [
        "ast_correct_whisper_normal",
        "both_correct_airway",
    ]
    present = {result.selection for result in results}
    return [selection for selection in preferred if selection in present]


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


def parse_args(argv: Sequence[str] | None = None) -> OverlapCliConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-root", default=DEFAULT_FAILURE_ROOT)
    parser.add_argument("--both-correct-root", default=DEFAULT_BOTH_CORRECT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--formats", action="append", default=None)
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    return OverlapCliConfig(
        failure_root=Path(args.failure_root),
        both_correct_root=Path(args.both_correct_root),
        out_dir=Path(args.out_dir),
        top_k=int(args.top_k),
        formats=parse_formats(args.formats or ["html,json"]),
        install_chrome=bool(args.install_chrome),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    Fs.ensure_dir(config.out_dir)
    enable_file_logging(config.out_dir / "run.log", mode="a")
    results, summaries = AttentionOverlapAnalyzer(config).analyze()
    summary = OverlapReportWriter(config).write(results, summaries)
    logger.info(
        "AST-Whisper attention overlap analysis complete | cases=%d | out_dir=%s",
        summary.total_cases,
        summary.out_dir,
    )


if __name__ == "__main__":
    main()
