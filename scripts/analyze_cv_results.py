from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_METRICS_FILE_PATTERN = re.compile(
    r"^eval_metrics__(?P<kind>.+)_(?P<score>-?\d+(?:\.\d+)?)\.json$"
)


@dataclass(frozen=True)
class MetricsFile:
    path: str
    kind: str
    score_name: float
    f1_score: float | None
    precision: float | None
    recall: float | None
    balanced_accuracy: float | None
    pr_auc: float | None
    brier_score: float | None
    optimized_f1_score: float | None = None
    optimized_balanced_accuracy: float | None = None
    optimized_threshold: float | None = None
    threshold_source: str | None = None


@dataclass(frozen=True)
class FoldSummary:
    fold: str
    log_count: int
    json_count: int
    jsonl_count: int
    metrics_files: list[MetricsFile] = field(default_factory=list)
    representative_best_loss: MetricsFile | None = None
    representative_best_f1: MetricsFile | None = None
    last_metrics: MetricsFile | None = None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_metrics_file(path: str | Path) -> MetricsFile:
    metrics_path = Path(path)
    match = _METRICS_FILE_PATTERN.match(metrics_path.name)
    if match is None:
        raise ValueError(f"Unsupported metrics filename: {metrics_path.name}")

    payload = _read_json(metrics_path)
    optimized = payload.get("optimized_metrics", {})
    threshold_optimization = payload.get("threshold_optimization", {})
    return MetricsFile(
        path=str(metrics_path),
        kind=match.group("kind"),
        score_name=float(match.group("score")),
        f1_score=_optional_float(payload.get("f1_score")),
        precision=_optional_float(payload.get("precision")),
        recall=_optional_float(payload.get("recall")),
        balanced_accuracy=_optional_float(payload.get("balanced_accuracy")),
        pr_auc=_optional_float(payload.get("pr_auc")),
        brier_score=_optional_float(payload.get("brier_score")),
        optimized_f1_score=_optional_float(optimized.get("f1_score")),
        optimized_balanced_accuracy=_optional_float(optimized.get("balanced_accuracy")),
        optimized_threshold=_optional_float(optimized.get("decision_threshold")),
        threshold_source=_optional_str(threshold_optimization.get("threshold_source")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def build_markdown(
    run_dir: Path,
    config: dict[str, Any],
    fold_summaries: list[FoldSummary],
    artifact_counts: dict[str, int],
) -> str:
    del config

    lines = [
        "# Cross-validation summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- artifact_counts: log={artifact_counts.get('log', 0)}, json={artifact_counts.get('json', 0)}, jsonl={artifact_counts.get('jsonl', 0)}",
        "",
        "## Folds",
    ]
    for summary in fold_summaries:
        status = _threshold_status(summary.representative_best_f1)
        lines.append(
            f"- {summary.fold}: metrics_files={len(summary.metrics_files)} | threshold_behavior={status}"
        )
    return "\n".join(lines)


def _threshold_status(metrics_file: MetricsFile | None) -> str:
    if metrics_file is None:
        return "unavailable"
    if metrics_file.threshold_source == "checkpoint_validation":
        return "validation-threshold-applied"
    if metrics_file.threshold_source is None:
        return "threshold-unknown"
    return metrics_file.threshold_source
