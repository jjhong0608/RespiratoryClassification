from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.cli.plot_ast_attention import (
    AstAttentionVisualizer,
)
from src.cli.plot_ast_attention import (
    AttentionCliConfig as AstAttentionCliConfig,
)
from src.cli.plot_whisper_attention import (
    AttentionCliConfig as WhisperAttentionCliConfig,
)
from src.cli.plot_whisper_attention import (
    WhisperAttentionVisualizer,
)
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger

DEFAULT_AST_REPORT_DIR = (
    "Disease_Group_Results/reports/cascade_test_comparison/AST_Partial_L1"
)
DEFAULT_WHISPER_REPORT_DIR = (
    "Disease_Group_Results/reports/cascade_test_comparison/Whisper_Partial_L1"
)
DEFAULT_OUT_DIR = "Disease_Group_Results/reports/ast_vs_whisper_airway_attention"
PREDICTIONS_FILENAME = "cascade_test_predictions.csv"
MANIFEST_CSV = "selected_airway_cases.csv"
MANIFEST_JSON = "selected_airway_cases.json"
SelectionMode = Literal["ast_correct_whisper_normal", "both_correct_airway"]
SELECTION_MODES: tuple[SelectionMode, ...] = (
    "ast_correct_whisper_normal",
    "both_correct_airway",
)

REQUIRED_COLUMNS = {
    "fold",
    "audio_path",
    "true_label",
    "direct_pred_label",
    "direct_prob_airway",
    "direct_checkpoint",
}


@dataclass(frozen=True)
class ComparisonCliConfig:
    ast_report_dir: Path
    whisper_report_dir: Path
    out_dir: Path
    selection: SelectionMode
    formats: set[str]
    top_k: int
    device: str | None
    limit: int | None
    folds: set[str] | None
    audio_stems: set[str] | None
    dry_run: bool
    overwrite: bool
    install_chrome: bool


@dataclass(frozen=True)
class PredictionRecord:
    fold: str
    audio_path: Path
    true_label: str
    direct_pred_label: str
    direct_prob_normal: float | None
    direct_prob_airway: float | None
    direct_prob_lung_parenchymal: float | None
    direct_checkpoint: Path

    @property
    def key(self) -> tuple[str, str]:
        return self.fold, str(self.audio_path)

    @property
    def audio_stem(self) -> str:
        return self.audio_path.stem


@dataclass(frozen=True)
class AttentionComparisonCase:
    fold: str
    audio_path: Path
    audio_stem: str
    true_label: str
    selection: SelectionMode
    ast_direct_pred_label: str
    whisper_direct_pred_label: str
    ast_direct_prob_airway: float | None
    whisper_direct_prob_normal: float | None
    whisper_direct_prob_airway: float | None
    ast_direct_checkpoint: Path
    whisper_direct_checkpoint: Path
    case_output_dir: Path
    ast_output_dir: Path
    whisper_output_dir: Path

    def manifest_row(self, *, status: str = "selected") -> dict[str, Any]:
        return {
            "fold": self.fold,
            "audio_path": str(self.audio_path),
            "audio_stem": self.audio_stem,
            "true_label": self.true_label,
            "selection": self.selection,
            "ast_direct_pred_label": self.ast_direct_pred_label,
            "whisper_direct_pred_label": self.whisper_direct_pred_label,
            "ast_direct_prob_airway": self.ast_direct_prob_airway,
            "whisper_direct_prob_normal": self.whisper_direct_prob_normal,
            "whisper_direct_prob_airway": self.whisper_direct_prob_airway,
            "ast_direct_checkpoint": str(self.ast_direct_checkpoint),
            "whisper_direct_checkpoint": str(self.whisper_direct_checkpoint),
            "case_output_dir": str(self.case_output_dir),
            "ast_output_dir": str(self.ast_output_dir),
            "whisper_output_dir": str(self.whisper_output_dir),
            "status": status,
        }


@dataclass(frozen=True)
class RunSummary:
    generated_at: str
    ast_report_dir: str
    whisper_report_dir: str
    out_dir: str
    selection: SelectionMode
    total_selected_before_filters: int
    total_selected_after_filters: int
    executed: int
    skipped: int
    dry_run: bool
    formats: list[str]
    top_k: int
    device: str | None
    cases: list[dict[str, Any]]


class PredictionCsvReader:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.path = report_dir / PREDICTIONS_FILENAME

    def read(self) -> list[PredictionRecord]:
        if not self.path.exists():
            raise FileNotFoundError(f"Prediction CSV does not exist: {self.path}")
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing:
                raise ValueError(
                    f"Prediction CSV missing required columns {missing}: {self.path}"
                )
            return [self._parse_row(row) for row in reader]

    @staticmethod
    def _parse_row(row: Mapping[str, str]) -> PredictionRecord:
        audio_path = Path(row["audio_path"])
        checkpoint = Path(row["direct_checkpoint"])
        return PredictionRecord(
            fold=str(row["fold"]),
            audio_path=audio_path,
            true_label=str(row["true_label"]),
            direct_pred_label=str(row["direct_pred_label"]),
            direct_prob_normal=_optional_float(row.get("direct_prob_normal")),
            direct_prob_airway=_optional_float(row.get("direct_prob_airway")),
            direct_prob_lung_parenchymal=_optional_float(
                row.get("direct_prob_lung_parenchymal")
            ),
            direct_checkpoint=checkpoint,
        )


class AirwayCaseSelector(LoggingMixin):
    def __init__(self, config: ComparisonCliConfig):
        self.config = config

    def select(self) -> tuple[list[AttentionComparisonCase], int]:
        ast_rows = PredictionCsvReader(self.config.ast_report_dir).read()
        whisper_rows = PredictionCsvReader(self.config.whisper_report_dir).read()
        ast_by_key = self._index_by_key(ast_rows, "AST")
        whisper_by_key = self._index_by_key(whisper_rows, "Whisper")

        selected: list[AttentionComparisonCase] = []
        for key, ast_row in ast_by_key.items():
            whisper_row = whisper_by_key.get(key)
            if whisper_row is None:
                continue
            if not self._is_target_case(ast_row, whisper_row):
                continue
            selected.append(self._case_from_rows(ast_row, whisper_row))

        selected.sort(
            key=lambda item: (item.fold, item.audio_stem, str(item.audio_path))
        )
        total_before_filters = len(selected)
        selected = self._apply_filters(selected)
        if self.config.limit is not None:
            selected = selected[: self.config.limit]
        self._validate_case_files(selected)
        self.logger.info(
            "Selected %d Airway comparison cases after filters; %d before filters",
            len(selected),
            total_before_filters,
        )
        return selected, total_before_filters

    @staticmethod
    def _index_by_key(
        rows: Sequence[PredictionRecord],
        family_name: str,
    ) -> dict[tuple[str, str], PredictionRecord]:
        indexed: dict[tuple[str, str], PredictionRecord] = {}
        duplicates: list[tuple[str, str]] = []
        for row in rows:
            if row.key in indexed:
                duplicates.append(row.key)
            indexed[row.key] = row
        if duplicates:
            example = duplicates[0]
            raise ValueError(
                f"{family_name} prediction CSV has duplicate (fold, audio_path) key: {example}"
            )
        return indexed

    def _is_target_case(
        self,
        ast_row: PredictionRecord,
        whisper_row: PredictionRecord,
    ) -> bool:
        is_airway_pair = (
            ast_row.true_label == "Airway"
            and whisper_row.true_label == "Airway"
            and ast_row.direct_pred_label == "Airway"
        )
        if self.config.selection == "ast_correct_whisper_normal":
            return is_airway_pair and whisper_row.direct_pred_label == "Normal"
        if self.config.selection == "both_correct_airway":
            return is_airway_pair and whisper_row.direct_pred_label == "Airway"
        raise ValueError(f"Unsupported selection mode: {self.config.selection}")

    def _case_from_rows(
        self,
        ast_row: PredictionRecord,
        whisper_row: PredictionRecord,
    ) -> AttentionComparisonCase:
        case_dir = self.config.out_dir / _safe_case_dir_name(
            ast_row.fold,
            ast_row.audio_stem,
        )
        return AttentionComparisonCase(
            fold=ast_row.fold,
            audio_path=ast_row.audio_path,
            audio_stem=ast_row.audio_stem,
            true_label=ast_row.true_label,
            selection=self.config.selection,
            ast_direct_pred_label=ast_row.direct_pred_label,
            whisper_direct_pred_label=whisper_row.direct_pred_label,
            ast_direct_prob_airway=ast_row.direct_prob_airway,
            whisper_direct_prob_normal=whisper_row.direct_prob_normal,
            whisper_direct_prob_airway=whisper_row.direct_prob_airway,
            ast_direct_checkpoint=ast_row.direct_checkpoint,
            whisper_direct_checkpoint=whisper_row.direct_checkpoint,
            case_output_dir=case_dir,
            ast_output_dir=case_dir / "ast",
            whisper_output_dir=case_dir / "whisper",
        )

    def _apply_filters(
        self,
        cases: Sequence[AttentionComparisonCase],
    ) -> list[AttentionComparisonCase]:
        filtered = list(cases)
        if self.config.folds is not None:
            filtered = [case for case in filtered if case.fold in self.config.folds]
        if self.config.audio_stems is not None:
            filtered = [
                case for case in filtered if case.audio_stem in self.config.audio_stems
            ]
        return filtered

    @staticmethod
    def _validate_case_files(cases: Sequence[AttentionComparisonCase]) -> None:
        for case in cases:
            if not case.audio_path.exists():
                raise FileNotFoundError(
                    f"Selected WAV does not exist: {case.audio_path}"
                )
            if not case.ast_direct_checkpoint.exists():
                raise FileNotFoundError(
                    f"Selected AST checkpoint does not exist: {case.ast_direct_checkpoint}"
                )
            if not case.whisper_direct_checkpoint.exists():
                raise FileNotFoundError(
                    "Selected Whisper checkpoint does not exist: "
                    f"{case.whisper_direct_checkpoint}"
                )


class ManifestWriter(LoggingMixin):
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def write(
        self,
        *,
        config: ComparisonCliConfig,
        cases: Sequence[AttentionComparisonCase],
        statuses: Mapping[tuple[str, str], str],
        total_before_filters: int,
        executed: int,
        skipped: int,
    ) -> RunSummary:
        Fs.ensure_dir(self.out_dir)
        rows = [
            case.manifest_row(
                status=statuses.get((case.fold, str(case.audio_path)), "selected")
            )
            for case in cases
        ]
        self._write_csv(self.out_dir / MANIFEST_CSV, rows)
        summary = RunSummary(
            generated_at=datetime.now(UTC).isoformat(),
            ast_report_dir=str(config.ast_report_dir),
            whisper_report_dir=str(config.whisper_report_dir),
            out_dir=str(config.out_dir),
            selection=config.selection,
            total_selected_before_filters=total_before_filters,
            total_selected_after_filters=len(cases),
            executed=executed,
            skipped=skipped,
            dry_run=config.dry_run,
            formats=sorted(config.formats),
            top_k=config.top_k,
            device=config.device,
            cases=rows,
        )
        json_path = self.out_dir / MANIFEST_JSON
        json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        self.logger.info("Wrote %s", self.out_dir / MANIFEST_CSV)
        self.logger.info("Wrote %s", json_path)
        return summary

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


class AttentionComparisonRunner(LoggingMixin):
    def __init__(self, config: ComparisonCliConfig):
        self.config = config

    def run(self) -> RunSummary:
        Fs.ensure_dir(self.config.out_dir)
        cases, total_before_filters = AirwayCaseSelector(self.config).select()
        statuses: dict[tuple[str, str], str] = {}
        executed = 0
        skipped = 0
        if self.config.dry_run:
            statuses = {
                (case.fold, str(case.audio_path)): "dry_run_selected" for case in cases
            }
            return ManifestWriter(self.config.out_dir).write(
                config=self.config,
                cases=cases,
                statuses=statuses,
                total_before_filters=total_before_filters,
                executed=0,
                skipped=0,
            )

        for case in cases:
            key = (case.fold, str(case.audio_path))
            if self._is_case_complete(case) and not self.config.overwrite:
                self.logger.info(
                    "Skipping existing case output: %s", case.case_output_dir
                )
                statuses[key] = "skipped_existing"
                skipped += 1
                continue
            self._run_case(case)
            statuses[key] = "generated"
            executed += 1

        return ManifestWriter(self.config.out_dir).write(
            config=self.config,
            cases=cases,
            statuses=statuses,
            total_before_filters=total_before_filters,
            executed=executed,
            skipped=skipped,
        )

    @staticmethod
    def _is_case_complete(case: AttentionComparisonCase) -> bool:
        return _has_attention_metadata(case.ast_output_dir) and _has_attention_metadata(
            case.whisper_output_dir
        )

    def _run_case(self, case: AttentionComparisonCase) -> None:
        Fs.ensure_dir(case.ast_output_dir)
        Fs.ensure_dir(case.whisper_output_dir)
        self.logger.info(
            "Generating attention comparison | fold=%s | wav=%s",
            case.fold,
            case.audio_path,
        )
        ast_config = AstAttentionCliConfig(
            checkpoint=case.ast_direct_checkpoint,
            wav=case.audio_path,
            out_dir=case.ast_output_dir,
            attention_method="last_cls_patch_head_mean",
            visualization="both",
            target_class="predicted",
            head="0",
            top_k=self.config.top_k,
            formats=set(self.config.formats),
            device_override=self.config.device,
            install_chrome=self.config.install_chrome,
        )
        whisper_config = WhisperAttentionCliConfig(
            checkpoint=case.whisper_direct_checkpoint,
            wav=case.audio_path,
            out_dir=case.whisper_output_dir,
            attention_method="last_time_attention_head_mean",
            target_class="predicted",
            head="0",
            top_k=self.config.top_k,
            formats=set(self.config.formats),
            device_override=self.config.device,
            install_chrome=self.config.install_chrome,
        )
        AstAttentionVisualizer(ast_config).run()
        WhisperAttentionVisualizer(whisper_config).run()


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"Expected a numeric CSV value, got {value!r}")
    return float(value)


def _safe_case_dir_name(fold: str, audio_stem: str) -> str:
    raw = f"{fold}__{audio_stem}"
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in raw
    )


def _has_attention_metadata(path: Path) -> bool:
    return path.exists() and any(path.glob("*_attention_metadata.json"))


def parse_formats(formats_arg: str | Sequence[str]) -> set[str]:
    raw_items = [formats_arg] if isinstance(formats_arg, str) else list(formats_arg)
    formats = {
        part.strip()
        for item in raw_items
        for part in str(item).split(",")
        if part.strip()
    }
    unsupported = formats - {"html", "png", "pdf", "json"}
    if unsupported:
        raise ValueError(f"Unsupported format(s): {sorted(unsupported)}")
    if not formats:
        raise ValueError("--formats must include at least one output format")
    return formats


def parse_args(argv: Sequence[str] | None = None) -> ComparisonCliConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ast-report-dir", default=DEFAULT_AST_REPORT_DIR)
    parser.add_argument("--whisper-report-dir", default=DEFAULT_WHISPER_REPORT_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--selection",
        default="ast_correct_whisper_normal",
        choices=SELECTION_MODES,
    )
    parser.add_argument("--formats", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--audio-stem", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero when provided")

    return ComparisonCliConfig(
        ast_report_dir=Path(args.ast_report_dir),
        whisper_report_dir=Path(args.whisper_report_dir),
        out_dir=Path(args.out_dir),
        selection=args.selection,
        formats=parse_formats(args.formats or ["html,json"]),
        top_k=int(args.top_k),
        device=args.device,
        limit=args.limit,
        folds=set(args.fold) if args.fold else None,
        audio_stems=set(args.audio_stem) if args.audio_stem else None,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        install_chrome=bool(args.install_chrome),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    Fs.ensure_dir(config.out_dir)
    enable_file_logging(config.out_dir / "run.log", mode="a")
    summary = AttentionComparisonRunner(config).run()
    logger.info(
        "AST-vs-Whisper Airway attention comparison complete | selection=%s | "
        "selected=%d | executed=%d | skipped=%d | dry_run=%s",
        summary.selection,
        summary.total_selected_after_filters,
        summary.executed,
        summary.skipped,
        summary.dry_run,
    )


if __name__ == "__main__":
    main()
