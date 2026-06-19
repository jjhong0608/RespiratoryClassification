from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging import LoggingMixin

_METRICS_FILE_PATTERN = re.compile(
    r"^eval_metrics__(?P<kind>last|best_loss|best_f1)(?:_(?P<score>[0-9.]+))?\.json$"
)


@dataclass(frozen=True)
class MetricsFile:
    path: str
    kind: str
    score_name: float | None
    f1_score: float | None
    precision: float | None
    recall: float | None
    balanced_accuracy: float | None
    pr_auc: float | None
    brier_score: float | None
    optimized_f1_score: float | None
    optimized_balanced_accuracy: float | None
    optimized_threshold: float | None
    threshold_source: str | None


@dataclass(frozen=True)
class FoldSummary:
    fold: str
    log_count: int
    json_count: int
    jsonl_count: int
    metrics_files: list[MetricsFile]
    representative_best_loss: MetricsFile | None
    representative_best_f1: MetricsFile | None
    last_metrics: MetricsFile | None


class CvResultAnalyzer(LoggingMixin):
    @staticmethod
    def parse_metrics_file(path: str | Path) -> MetricsFile:
        metrics_path = Path(path)
        match = _METRICS_FILE_PATTERN.match(metrics_path.name)
        if match is None:
            raise ValueError(f"Unsupported metrics filename: {metrics_path.name}")

        payload = CvResultAnalyzer._load_json(metrics_path)
        optimized_metrics = payload.get("optimized_metrics", {})
        threshold_optimization = payload.get("threshold_optimization", {})

        return MetricsFile(
            path=str(metrics_path),
            kind=match.group("kind"),
            score_name=CvResultAnalyzer._parse_optional_float(match.group("score")),
            f1_score=CvResultAnalyzer._parse_optional_float(payload.get("f1_score")),
            precision=CvResultAnalyzer._parse_optional_float(payload.get("precision")),
            recall=CvResultAnalyzer._parse_optional_float(payload.get("recall")),
            balanced_accuracy=CvResultAnalyzer._parse_optional_float(
                payload.get("balanced_accuracy")
            ),
            pr_auc=CvResultAnalyzer._parse_optional_float(payload.get("pr_auc")),
            brier_score=CvResultAnalyzer._parse_optional_float(
                payload.get("brier_score")
            ),
            optimized_f1_score=CvResultAnalyzer._parse_optional_float(
                optimized_metrics.get("f1_score")
            ),
            optimized_balanced_accuracy=CvResultAnalyzer._parse_optional_float(
                optimized_metrics.get("balanced_accuracy")
            ),
            optimized_threshold=CvResultAnalyzer._parse_optional_float(
                optimized_metrics.get("decision_threshold")
            ),
            threshold_source=CvResultAnalyzer._parse_optional_str(
                threshold_optimization.get("threshold_source")
            ),
        )

    @staticmethod
    def build_markdown(
        run_dir: Path,
        config_summary: dict[str, Any],
        fold_summaries: list[FoldSummary],
        artifact_counts: dict[str, int],
    ) -> str:
        lines = [
            "# Cross-Validation Report",
            "",
            f"- run_dir: `{run_dir}`",
            f"- config_keys: {len(config_summary)}",
            f"- artifacts: logs={artifact_counts.get('log', 0)}, "
            f"json={artifact_counts.get('json', 0)}, "
            f"jsonl={artifact_counts.get('jsonl', 0)}",
            "",
            "## Folds",
        ]
        for fold in fold_summaries:
            lines.append(f"### {fold.fold}")
            lines.append(
                f"- artifacts: logs={fold.log_count}, json={fold.json_count}, "
                f"jsonl={fold.jsonl_count}"
            )
            if fold.representative_best_f1 is not None:
                lines.extend(
                    CvResultAnalyzer._best_f1_lines(fold.representative_best_f1)
                )
            if fold.representative_best_loss is not None:
                lines.append(
                    f"- representative-best-loss: `{fold.representative_best_loss.path}`"
                )
            if fold.last_metrics is not None:
                lines.append(f"- last-metrics: `{fold.last_metrics.path}`")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _best_f1_lines(metrics: MetricsFile) -> list[str]:
        threshold_label = "validation-threshold-applied"
        if metrics.threshold_source == "validation_optimization":
            threshold_label = "validation-threshold-selected"
        if metrics.threshold_source == "fixed_default":
            threshold_label = "default-threshold-applied"
        return [
            f"- representative-best-f1: `{metrics.path}`",
            f"- threshold_policy: {threshold_label}",
            f"- optimized_f1: {CvResultAnalyzer._render_number(metrics.optimized_f1_score)}",
            "- optimized_balanced_accuracy: "
            f"{CvResultAnalyzer._render_number(metrics.optimized_balanced_accuracy)}",
            f"- optimized_threshold: {CvResultAnalyzer._render_number(metrics.optimized_threshold)}",
        ]

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object in {path}")
        return payload

    @staticmethod
    def _parse_optional_float(value: object) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _parse_optional_str(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _render_number(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"


def parse_metrics_file(path: str | Path) -> MetricsFile:
    return CvResultAnalyzer.parse_metrics_file(path)


def build_markdown(
    run_dir: Path,
    config_summary: dict[str, Any],
    fold_summaries: list[FoldSummary],
    artifact_counts: dict[str, int],
) -> str:
    return CvResultAnalyzer.build_markdown(
        run_dir,
        config_summary,
        fold_summaries,
        artifact_counts,
    )
