from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_analyze_module() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_cv_results.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_cv_results", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metrics(path: Path) -> Path:
    payload = {
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "specificity": 0.8,
        "balanced_accuracy": 0.75,
        "f1_score": 0.7467,
        "roc_auc": 0.81,
        "pr_auc": 0.79,
        "brier_score": 0.2,
        "threshold_optimization": {
            "enabled": True,
            "applied": True,
            "selected_metric": "f1",
            "selected_threshold": 0.37,
            "selected_score": 0.8012,
            "reason": None,
            "threshold_source": "checkpoint_validation",
            "f1": {"metric": "f1", "threshold": 0.37, "score": 0.8012},
            "balanced_accuracy": None,
            "youden_j": None,
        },
        "optimized_metrics": {
            "f1_score": 0.8012,
            "balanced_accuracy": 0.78,
            "decision_threshold": 0.37,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_metrics_file_reads_best_f1_and_optimized_metrics(tmp_path: Path) -> None:
    module = _load_analyze_module()
    metrics_path = _write_metrics(tmp_path / "eval_metrics__best_f1_0.801200.json")

    metrics = module.parse_metrics_file(metrics_path)

    assert metrics.kind == "best_f1"
    assert metrics.score_name == 0.8012
    assert metrics.optimized_f1_score == 0.8012
    assert metrics.optimized_balanced_accuracy == 0.78
    assert metrics.optimized_threshold == 0.37
    assert metrics.threshold_source == "checkpoint_validation"


def test_build_markdown_uses_validation_threshold_applied_wording() -> None:
    module = _load_analyze_module()
    metrics_file = module.MetricsFile(
        path="eval_metrics__best_f1_0.801200.json",
        kind="best_f1",
        score_name=0.8012,
        f1_score=0.7467,
        precision=0.8,
        recall=0.7,
        balanced_accuracy=0.75,
        pr_auc=0.79,
        brier_score=0.2,
        optimized_f1_score=0.8012,
        optimized_balanced_accuracy=0.78,
        optimized_threshold=0.37,
        threshold_source="checkpoint_validation",
    )
    fold_summary = module.FoldSummary(
        fold="fold_0",
        log_count=0,
        json_count=1,
        jsonl_count=0,
        metrics_files=[metrics_file],
        representative_best_loss=None,
        representative_best_f1=metrics_file,
        last_metrics=None,
    )

    markdown = module.build_markdown(
        Path("/tmp/run"),
        {},
        [fold_summary],
        {"log": 0, "json": 1, "jsonl": 0},
    )

    assert "validation-threshold-applied" in markdown
