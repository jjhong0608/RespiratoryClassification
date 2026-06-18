from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.cli.disease_group_cascade_eval import (
    DIRECT_TASK,
    CascadeComparisonEvaluator,
    CheckpointSelector,
    compute_final_metrics,
    summarize_fold_metrics,
)


def _write_metric(
    path: Path, *, f1: float, bal: float, pr: float, brier: float
) -> None:
    path.write_text(
        json.dumps(
            {
                "optimized_metrics": {
                    "f1_score": f1,
                    "balanced_accuracy": bal,
                    "pr_auc": pr,
                    "brier_score": brier,
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint_name = path.name.replace("eval_metrics__", "").replace(".json", ".pt")
    (path.parent / checkpoint_name).write_bytes(b"checkpoint-placeholder")


def test_checkpoint_selector_uses_optimized_f1_and_tie_breakers(
    tmp_path: Path,
) -> None:
    fold_dir = tmp_path / DIRECT_TASK / "fold_0"
    fold_dir.mkdir(parents=True)
    _write_metric(
        fold_dir / "eval_metrics__best_f1_0.700000.json",
        f1=0.80,
        bal=0.99,
        pr=0.99,
        brier=0.01,
    )
    _write_metric(
        fold_dir / "eval_metrics__best_f1_0.800000.json",
        f1=0.90,
        bal=0.75,
        pr=0.70,
        brier=0.20,
    )
    _write_metric(
        fold_dir / "eval_metrics__best_f1_0.900000.json",
        f1=0.90,
        bal=0.75,
        pr=0.70,
        brier=0.10,
    )

    selected = CheckpointSelector(tmp_path).select(DIRECT_TASK, "fold_0")

    assert selected.checkpoint_path.name == "best_f1_0.900000.pt"
    assert selected.optimized_f1 == 0.90
    assert selected.brier_score == 0.10


@pytest.mark.parametrize(
    ("true_label", "stage1", "stage2", "expected"),
    [
        ("Normal", "Normal", None, "correct_normal"),
        ("Normal", "Abnormal", "Airway", "normal_false_abnormal_to_airway"),
        (
            "Normal",
            "Abnormal",
            "Lung_Parenchymal",
            "normal_false_abnormal_to_lung_parenchymal",
        ),
        ("Airway", "Normal", None, "airway_blocked_as_normal"),
        ("Airway", "Abnormal", "Airway", "airway_correct"),
        ("Airway", "Abnormal", "Lung_Parenchymal", "airway_to_lung_parenchymal"),
        (
            "Lung_Parenchymal",
            "Normal",
            None,
            "lung_parenchymal_blocked_as_normal",
        ),
        (
            "Lung_Parenchymal",
            "Abnormal",
            "Airway",
            "lung_parenchymal_to_airway",
        ),
        (
            "Lung_Parenchymal",
            "Abnormal",
            "Lung_Parenchymal",
            "lung_parenchymal_correct",
        ),
    ],
)
def test_cascade_error_type_covers_all_stage_paths(
    true_label: str,
    stage1: str,
    stage2: str | None,
    expected: str,
) -> None:
    assert (
        CascadeComparisonEvaluator.resolve_cascade_error_type(
            true_label,
            stage1,
            stage2,
        )
        == expected
    )


def test_abnormal_overlay_mismatch_raises(tmp_path: Path) -> None:
    test_root = tmp_path / "test"
    for label in ("Normal", "Airway", "Lung_Parenchymal", "Abnormal"):
        (test_root / label).mkdir(parents=True)
    (test_root / "Normal" / "normal.wav").write_bytes(b"")
    (test_root / "Airway" / "airway.wav").write_bytes(b"")
    (test_root / "Lung_Parenchymal" / "lung.wav").write_bytes(b"")
    (test_root / "Abnormal" / "airway.wav").write_bytes(b"")

    evaluator = CascadeComparisonEvaluator(
        results_root=tmp_path / "results",
        test_root=test_root,
        reports_dir=tmp_path / "reports",
        device="cpu",
        formats={"html"},
    )

    with pytest.raises(ValueError, match="Abnormal overlay does not match"):
        evaluator.build_test_manifest()


def test_metrics_and_fold_summary_are_aggregated() -> None:
    y_true = np.asarray([0, 1, 2, 2], dtype=int)
    y_pred = np.asarray([0, 1, 1, 2], dtype=int)
    y_prob = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.70, 0.25],
            [0.10, 0.20, 0.70],
        ],
        dtype=float,
    )

    metrics = compute_final_metrics(y_true, y_pred, y_prob)

    assert metrics["metrics"]["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert metrics["class_metrics"]["Lung_Parenchymal"]["support"] == 2

    summary = summarize_fold_metrics(
        [
            {
                "direct_accuracy": 0.6,
                "cascade_accuracy": 0.8,
                "direct_macro_precision": 0.6,
                "cascade_macro_precision": 0.8,
                "direct_macro_recall": 0.6,
                "cascade_macro_recall": 0.8,
                "direct_macro_f1": 0.6,
                "cascade_macro_f1": 0.8,
                "direct_weighted_f1": 0.6,
                "cascade_weighted_f1": 0.8,
                "direct_balanced_accuracy": 0.6,
                "cascade_balanced_accuracy": 0.8,
                "direct_specificity": 0.6,
                "cascade_specificity": 0.8,
                "direct_roc_auc": 0.6,
                "cascade_roc_auc": 0.8,
                "direct_pr_auc": 0.6,
                "cascade_pr_auc": 0.8,
                "direct_brier_score": 0.4,
                "cascade_brier_score": 0.2,
            },
            {
                "direct_accuracy": 0.8,
                "cascade_accuracy": 0.6,
                "direct_macro_precision": 0.8,
                "cascade_macro_precision": 0.6,
                "direct_macro_recall": 0.8,
                "cascade_macro_recall": 0.6,
                "direct_macro_f1": 0.8,
                "cascade_macro_f1": 0.6,
                "direct_weighted_f1": 0.8,
                "cascade_weighted_f1": 0.6,
                "direct_balanced_accuracy": 0.8,
                "cascade_balanced_accuracy": 0.6,
                "direct_specificity": 0.8,
                "cascade_specificity": 0.6,
                "direct_roc_auc": 0.8,
                "cascade_roc_auc": 0.6,
                "direct_pr_auc": 0.8,
                "cascade_pr_auc": 0.6,
                "direct_brier_score": 0.2,
                "cascade_brier_score": 0.4,
            },
        ]
    )

    assert summary["direct"]["accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["cascade"]["accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["direct"]["brier_score"]["mean"] == pytest.approx(0.3)
