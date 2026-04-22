from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricsFile:
    path: str
    kind: str
    score_name: float | None
    f1_score: float
    precision: float
    recall: float
    balanced_accuracy: float
    pr_auc: float
    brier_score: float
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


def _parse_kind_and_score(path: Path) -> tuple[str, float | None]:
    match = re.search(r"eval_metrics__(.+?)(?:_([0-9]+\.[0-9]+))?$", path.stem)
    if match is None:
        return path.stem, None
    kind = match.group(1)
    score_raw = match.group(2)
    return kind, float(score_raw) if score_raw is not None else None


def parse_metrics_file(path: str | Path) -> MetricsFile:
    metrics_path = Path(path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Metrics file must contain a JSON object")

    optimized_metrics = payload.get("optimized_metrics", {})
    threshold_optimization = payload.get("threshold_optimization", {})
    if not isinstance(optimized_metrics, Mapping):
        optimized_metrics = {}
    if not isinstance(threshold_optimization, Mapping):
        threshold_optimization = {}

    kind, score_name = _parse_kind_and_score(metrics_path)
    return MetricsFile(
        path=metrics_path.name,
        kind=kind,
        score_name=score_name,
        f1_score=float(payload.get("f1_score", 0.0)),
        precision=float(payload.get("precision", 0.0)),
        recall=float(payload.get("recall", 0.0)),
        balanced_accuracy=float(payload.get("balanced_accuracy", 0.0)),
        pr_auc=float(payload.get("pr_auc", 0.0)),
        brier_score=float(payload.get("brier_score", 0.0)),
        optimized_f1_score=_optional_float(optimized_metrics.get("f1_score")),
        optimized_balanced_accuracy=_optional_float(
            optimized_metrics.get("balanced_accuracy")
        ),
        optimized_threshold=_optional_float(
            optimized_metrics.get("decision_threshold")
        ),
        threshold_source=_optional_str(threshold_optimization.get("threshold_source")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid metric scores")
    if not isinstance(value, (int, float, str)):
        raise TypeError("Metric scores must be numeric or string values")
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def build_markdown(
    run_dir: Path,
    config: Mapping[str, Any],
    folds: Sequence[FoldSummary],
    counts: Mapping[str, int],
) -> str:
    lines = [f"# CV Summary: {run_dir.name}", ""]
    lines.append(f"- logs: {counts.get('log', 0)}")
    lines.append(f"- json: {counts.get('json', 0)}")
    lines.append(f"- jsonl: {counts.get('jsonl', 0)}")
    if config:
        lines.append(f"- config_keys: {sorted(config.keys())}")
    for fold in folds:
        lines.append("")
        lines.append(f"## {fold.fold}")
        if fold.representative_best_f1 is not None:
            lines.append(f"- best_f1_file: {fold.representative_best_f1.path}")
            if fold.representative_best_f1.threshold_source == "checkpoint_validation":
                lines.append("- threshold_status: validation-threshold-applied")
            else:
                lines.append("- threshold_status: fixed-threshold")
    return "\n".join(lines)
