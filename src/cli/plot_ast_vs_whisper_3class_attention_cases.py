from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.cli.plot_ast_attention import AstAttentionVisualizer
from src.cli.plot_ast_attention import AttentionCliConfig as AstAttentionCliConfig
from src.cli.plot_whisper_attention import (
    AttentionCliConfig as WhisperAttentionCliConfig,
)
from src.cli.plot_whisper_attention import WhisperAttentionVisualizer
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin, enable_file_logging, logger

DEFAULT_AST_REPORT_DIR = (
    "Disease_Group_Results/reports/cascade_test_comparison/AST_Partial_L1"
)
DEFAULT_WHISPER_REPORT_DIR = (
    "Disease_Group_Results/reports/cascade_test_comparison/Whisper_Partial_L1"
)
DEFAULT_OUT_DIR = "Disease_Group_Results/reports/ast_vs_whisper_3class_attention_cases"
PREDICTIONS_FILENAME = "cascade_test_predictions.csv"
MANIFEST_CSV = "selected_cases.csv"
MANIFEST_JSON = "selected_cases.json"
LABELS = ("Normal", "Airway", "Lung_Parenchymal")

SelectionSet = Literal[
    "ast_correct_high_confidence",
    "ast_wrong_high_confidence",
    "whisper_correct_high_confidence",
    "whisper_wrong_high_confidence",
]
ReferenceModel = Literal["AST", "Whisper"]
Correctness = Literal["correct", "wrong"]

PROBABILITY_COLUMN_BY_LABEL = {
    "Normal": "direct_prob_normal",
    "Airway": "direct_prob_airway",
    "Lung_Parenchymal": "direct_prob_lung_parenchymal",
}
REQUIRED_COLUMNS = {
    "fold",
    "audio_path",
    "true_label",
    "direct_pred_label",
    "direct_prob_normal",
    "direct_prob_airway",
    "direct_prob_lung_parenchymal",
    "direct_checkpoint",
}
SUPPORTED_FORMATS = {"html", "png", "pdf", "json"}


@dataclass(frozen=True)
class SelectionSpec:
    name: SelectionSet
    reference_model: ReferenceModel
    correctness: Correctness


SELECTION_SPECS: dict[SelectionSet, SelectionSpec] = {
    "ast_correct_high_confidence": SelectionSpec(
        "ast_correct_high_confidence",
        "AST",
        "correct",
    ),
    "ast_wrong_high_confidence": SelectionSpec(
        "ast_wrong_high_confidence",
        "AST",
        "wrong",
    ),
    "whisper_correct_high_confidence": SelectionSpec(
        "whisper_correct_high_confidence",
        "Whisper",
        "correct",
    ),
    "whisper_wrong_high_confidence": SelectionSpec(
        "whisper_wrong_high_confidence",
        "Whisper",
        "wrong",
    ),
}
DEFAULT_SELECTION_SETS: tuple[SelectionSet, ...] = tuple(SELECTION_SPECS)


@dataclass(frozen=True)
class ComparisonCliConfig:
    ast_report_dir: Path
    whisper_report_dir: Path
    out_dir: Path
    selection_sets: tuple[SelectionSet, ...]
    per_label: int
    formats: set[str]
    top_k: int
    device: str | None
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

    @property
    def is_correct(self) -> bool:
        return self.true_label == self.direct_pred_label

    def probability_for_label(self, label: str) -> float:
        if label == "Normal":
            value = self.direct_prob_normal
        elif label == "Airway":
            value = self.direct_prob_airway
        elif label == "Lung_Parenchymal":
            value = self.direct_prob_lung_parenchymal
        else:
            raise ValueError(f"Unsupported direct label: {label!r}")
        if value is None:
            raise ValueError(
                f"Missing probability for label={label!r} "
                f"fold={self.fold} audio_path={self.audio_path}"
            )
        return value

    @property
    def prediction_confidence(self) -> float:
        return self.probability_for_label(self.direct_pred_label)


@dataclass(frozen=True)
class AttentionCase:
    selection_set: SelectionSet
    reference_model: ReferenceModel
    correctness: Correctness
    true_label: str
    rank_within_label: int
    fold: str
    audio_path: Path
    audio_stem: str
    reference_pred_label: str
    reference_confidence: float
    ast_direct_pred_label: str
    whisper_direct_pred_label: str
    ast_direct_confidence: float
    whisper_direct_confidence: float
    ast_direct_checkpoint: Path
    whisper_direct_checkpoint: Path
    case_output_dir: Path
    ast_output_dir: Path
    whisper_output_dir: Path

    @property
    def case_key(self) -> tuple[str, str, str, str]:
        return self.selection_set, self.true_label, self.fold, str(self.audio_path)

    def manifest_row(self, *, status: str = "selected") -> dict[str, Any]:
        return {
            "selection_set": self.selection_set,
            "reference_model": self.reference_model,
            "correctness": self.correctness,
            "true_label": self.true_label,
            "rank_within_label": self.rank_within_label,
            "fold": self.fold,
            "audio_path": str(self.audio_path),
            "audio_stem": self.audio_stem,
            "reference_pred_label": self.reference_pred_label,
            "reference_confidence": self.reference_confidence,
            "ast_direct_pred_label": self.ast_direct_pred_label,
            "whisper_direct_pred_label": self.whisper_direct_pred_label,
            "ast_direct_confidence": self.ast_direct_confidence,
            "whisper_direct_confidence": self.whisper_direct_confidence,
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
    selection_sets: list[str]
    per_label: int
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
        return PredictionRecord(
            fold=str(row["fold"]),
            audio_path=Path(row["audio_path"]),
            true_label=str(row["true_label"]),
            direct_pred_label=str(row["direct_pred_label"]),
            direct_prob_normal=_optional_float(row.get("direct_prob_normal")),
            direct_prob_airway=_optional_float(row.get("direct_prob_airway")),
            direct_prob_lung_parenchymal=_optional_float(
                row.get("direct_prob_lung_parenchymal")
            ),
            direct_checkpoint=Path(row["direct_checkpoint"]),
        )


class ThreeClassCaseSelector(LoggingMixin):
    def __init__(self, config: ComparisonCliConfig):
        self.config = config

    def select(self) -> tuple[list[AttentionCase], int]:
        ast_rows = PredictionCsvReader(self.config.ast_report_dir).read()
        whisper_rows = PredictionCsvReader(self.config.whisper_report_dir).read()
        ast_by_key = self._index_by_key(ast_rows, "AST")
        whisper_by_key = self._index_by_key(whisper_rows, "Whisper")

        selected: list[AttentionCase] = []
        for selection_set in self.config.selection_sets:
            spec = SELECTION_SPECS[selection_set]
            selected.extend(self._select_for_spec(spec, ast_by_key, whisper_by_key))

        total_before_filters = len(selected)
        selected = self._apply_filters(selected)
        self._validate_case_files(selected)
        self.logger.info(
            "Selected %d 3-class attention cases after filters; %d before filters",
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

    def _select_for_spec(
        self,
        spec: SelectionSpec,
        ast_by_key: Mapping[tuple[str, str], PredictionRecord],
        whisper_by_key: Mapping[tuple[str, str], PredictionRecord],
    ) -> list[AttentionCase]:
        reference_rows = ast_by_key.values()
        if spec.reference_model == "Whisper":
            reference_rows = whisper_by_key.values()

        selected: list[AttentionCase] = []
        for label in LABELS:
            representatives = self._representative_rows(reference_rows, spec, label)
            if len(representatives) < self.config.per_label:
                raise ValueError(
                    "Insufficient unique audio for selection bucket: "
                    f"selection_set={spec.name} true_label={label} "
                    f"required={self.config.per_label} available={len(representatives)}"
                )
            for rank, reference_row in enumerate(
                representatives[: self.config.per_label],
                start=1,
            ):
                ast_row = ast_by_key.get(reference_row.key)
                whisper_row = whisper_by_key.get(reference_row.key)
                if ast_row is None or whisper_row is None:
                    raise ValueError(
                        "Selected row does not join between AST and Whisper reports: "
                        f"selection_set={spec.name} key={reference_row.key}"
                    )
                selected.append(
                    self._case_from_rows(
                        spec=spec,
                        rank=rank,
                        ast_row=ast_row,
                        whisper_row=whisper_row,
                        reference_row=reference_row,
                    )
                )
        return selected

    @staticmethod
    def _representative_rows(
        rows: Iterable[PredictionRecord],
        spec: SelectionSpec,
        true_label: str,
    ) -> list[PredictionRecord]:
        by_audio: dict[str, list[PredictionRecord]] = {}
        for row in rows:
            if row.true_label != true_label:
                continue
            if spec.correctness == "correct" and not row.is_correct:
                continue
            if spec.correctness == "wrong" and row.is_correct:
                continue
            _ = row.prediction_confidence
            by_audio.setdefault(str(row.audio_path), []).append(row)

        representatives = [
            sorted(audio_rows, key=_confidence_sort_key)[0]
            for audio_rows in by_audio.values()
        ]
        return sorted(representatives, key=_confidence_sort_key)

    def _case_from_rows(
        self,
        *,
        spec: SelectionSpec,
        rank: int,
        ast_row: PredictionRecord,
        whisper_row: PredictionRecord,
        reference_row: PredictionRecord,
    ) -> AttentionCase:
        case_dir = (
            self.config.out_dir
            / spec.name
            / reference_row.true_label
            / _safe_case_dir_name(
                f"rank_{rank:02d}",
                reference_row.fold,
                f"true_{reference_row.true_label}",
                f"pred_{reference_row.direct_pred_label}",
                reference_row.audio_stem,
            )
        )
        return AttentionCase(
            selection_set=spec.name,
            reference_model=spec.reference_model,
            correctness=spec.correctness,
            true_label=reference_row.true_label,
            rank_within_label=rank,
            fold=reference_row.fold,
            audio_path=reference_row.audio_path,
            audio_stem=reference_row.audio_stem,
            reference_pred_label=reference_row.direct_pred_label,
            reference_confidence=reference_row.prediction_confidence,
            ast_direct_pred_label=ast_row.direct_pred_label,
            whisper_direct_pred_label=whisper_row.direct_pred_label,
            ast_direct_confidence=ast_row.prediction_confidence,
            whisper_direct_confidence=whisper_row.prediction_confidence,
            ast_direct_checkpoint=ast_row.direct_checkpoint,
            whisper_direct_checkpoint=whisper_row.direct_checkpoint,
            case_output_dir=case_dir,
            ast_output_dir=case_dir / "ast",
            whisper_output_dir=case_dir / "whisper",
        )

    def _apply_filters(self, cases: Sequence[AttentionCase]) -> list[AttentionCase]:
        filtered = list(cases)
        if self.config.folds is not None:
            filtered = [case for case in filtered if case.fold in self.config.folds]
        if self.config.audio_stems is not None:
            filtered = [
                case for case in filtered if case.audio_stem in self.config.audio_stems
            ]
        return filtered

    @staticmethod
    def _validate_case_files(cases: Sequence[AttentionCase]) -> None:
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
        cases: Sequence[AttentionCase],
        statuses: Mapping[tuple[str, str, str, str], str],
        total_before_filters: int,
        executed: int,
        skipped: int,
    ) -> RunSummary:
        Fs.ensure_dir(self.out_dir)
        rows = [
            case.manifest_row(status=statuses.get(case.case_key, "selected"))
            for case in cases
        ]
        self._write_csv(self.out_dir / MANIFEST_CSV, rows)
        summary = RunSummary(
            generated_at=datetime.now(UTC).isoformat(),
            ast_report_dir=str(config.ast_report_dir),
            whisper_report_dir=str(config.whisper_report_dir),
            out_dir=str(config.out_dir),
            selection_sets=list(config.selection_sets),
            per_label=config.per_label,
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
        cases, total_before_filters = ThreeClassCaseSelector(self.config).select()
        statuses: dict[tuple[str, str, str, str], str] = {}
        executed = 0
        skipped = 0
        if self.config.dry_run:
            statuses = {case.case_key: "dry_run_selected" for case in cases}
            return ManifestWriter(self.config.out_dir).write(
                config=self.config,
                cases=cases,
                statuses=statuses,
                total_before_filters=total_before_filters,
                executed=0,
                skipped=0,
            )

        for case in cases:
            if self._is_case_complete(case) and not self.config.overwrite:
                self.logger.info(
                    "Skipping existing case output: %s", case.case_output_dir
                )
                statuses[case.case_key] = "skipped_existing"
                skipped += 1
                continue
            self._run_case(case)
            statuses[case.case_key] = "generated"
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
    def _is_case_complete(case: AttentionCase) -> bool:
        return _has_attention_metadata(case.ast_output_dir) and _has_attention_metadata(
            case.whisper_output_dir
        )

    def _run_case(self, case: AttentionCase) -> None:
        Fs.ensure_dir(case.ast_output_dir)
        Fs.ensure_dir(case.whisper_output_dir)
        self.logger.info(
            "Generating 3-class attention comparison | selection=%s | "
            "label=%s | rank=%d | fold=%s | wav=%s",
            case.selection_set,
            case.true_label,
            case.rank_within_label,
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


def _confidence_sort_key(row: PredictionRecord) -> tuple[float, str, str]:
    return (-row.prediction_confidence, row.fold, str(row.audio_path))


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"Expected a numeric CSV value, got {value!r}")
    return float(value)


def _safe_case_dir_name(*parts: str) -> str:
    raw = "__".join(parts)
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
    unsupported = formats - SUPPORTED_FORMATS
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
        "--selection-set",
        action="append",
        choices=tuple(SELECTION_SPECS),
        default=None,
        help="Selection set to run. Repeat to run multiple sets. Defaults to all sets.",
    )
    parser.add_argument("--per-label", type=int, default=5)
    parser.add_argument("--formats", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
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

    if args.per_label <= 0:
        raise ValueError("--per-label must be greater than zero")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    selection_sets = tuple(args.selection_set or DEFAULT_SELECTION_SETS)
    return ComparisonCliConfig(
        ast_report_dir=Path(args.ast_report_dir),
        whisper_report_dir=Path(args.whisper_report_dir),
        out_dir=Path(args.out_dir),
        selection_sets=selection_sets,
        per_label=int(args.per_label),
        formats=parse_formats(args.formats or ["html,json"]),
        top_k=int(args.top_k),
        device=args.device,
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
        "AST-vs-Whisper 3-class attention cases complete | "
        "selection_sets=%s | selected=%d | executed=%d | skipped=%d | dry_run=%s",
        ",".join(summary.selection_sets),
        summary.total_selected_after_filters,
        summary.executed,
        summary.skipped,
        summary.dry_run,
    )


if __name__ == "__main__":
    main()
