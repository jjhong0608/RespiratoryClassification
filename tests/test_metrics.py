from __future__ import annotations

import math

import numpy as np
from src.evaluation.metrics import MetricsComputer


def test_multiclass_metrics_include_macro_aliases_and_per_class_values() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    y_pred = np.asarray([0, 1, 1, 3])
    y_prob = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.5, 0.3, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )

    metrics = MetricsComputer.compute(y_true, y_pred, y_prob).to_dict()

    assert metrics["macro_f1"] == metrics["f1_score"]
    assert metrics["macro_recall"] == metrics["recall"]
    assert metrics["weighted_f1"] is not None
    assert len(metrics["per_class_precision"]) == 4
    assert len(metrics["per_class_recall"]) == 4
    assert len(metrics["per_class_f1"]) == 4


def test_branch_binary_metrics_include_per_branch_and_means() -> None:
    y_true = np.asarray([0, 1, 0, 1])
    branch_probs = np.asarray(
        [
            [0.1, 0.2],
            [0.8, 0.7],
            [0.3, 0.4],
            [0.9, 0.6],
        ]
    )

    metrics = MetricsComputer.compute_branch_binary(y_true, branch_probs)

    assert len(metrics["per_branch"]) == 2
    assert metrics["mean_roc_auc"] == 1.0
    assert metrics["mean_pr_auc"] == 1.0
    assert metrics["mean_f1_score"] == 1.0


def test_branch_binary_one_class_auc_returns_nan_without_crashing() -> None:
    y_true = np.asarray([1, 1, 1])
    branch_probs = np.asarray([[0.6, 0.7], [0.8, 0.9], [0.55, 0.65]])

    metrics = MetricsComputer.compute_branch_binary(y_true, branch_probs)

    assert math.isnan(metrics["per_branch"][0]["roc_auc"])
    assert math.isnan(metrics["mean_roc_auc"])
