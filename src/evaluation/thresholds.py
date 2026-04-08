from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.metrics import EvalMetrics, MetricsComputer

ThresholdMetric = Literal["f1", "balanced_accuracy", "youden_j"]
ThresholdSource = Literal[
    "validation_optimization",
    "checkpoint_validation",
    "fixed_default",
    "disabled",
]


@dataclass(frozen=True)
class ThresholdOptimizationConfig:
    enabled: bool = False
    metric: ThresholdMetric = "f1"


@dataclass(frozen=True)
class ThresholdMetricResult:
    metric: ThresholdMetric
    threshold: float
    score: float


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    enabled: bool
    applied: bool
    selected_metric: ThresholdMetric
    selected_threshold: float
    selected_score: float | None
    reason: str | None
    threshold_source: ThresholdSource
    f1: ThresholdMetricResult | None
    balanced_accuracy: ThresholdMetricResult | None
    youden_j: ThresholdMetricResult | None

    @staticmethod
    def disabled(
        metric: ThresholdMetric,
        *,
        threshold: float = 0.5,
    ) -> ThresholdOptimizationResult:
        return ThresholdOptimizationResult(
            enabled=False,
            applied=False,
            selected_metric=metric,
            selected_threshold=threshold,
            selected_score=None,
            reason="threshold optimization disabled",
            threshold_source="disabled",
            f1=None,
            balanced_accuracy=None,
            youden_j=None,
        )

    @staticmethod
    def fallback(
        metric: ThresholdMetric,
        *,
        reason: str,
        threshold: float = 0.5,
        threshold_source: ThresholdSource = "fixed_default",
    ) -> ThresholdOptimizationResult:
        return ThresholdOptimizationResult(
            enabled=True,
            applied=False,
            selected_metric=metric,
            selected_threshold=threshold,
            selected_score=None,
            reason=reason,
            threshold_source=threshold_source,
            f1=None,
            balanced_accuracy=None,
            youden_j=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThresholdOptimizer:
    def __init__(self, y_true: np.ndarray, y_score: np.ndarray):
        self.y_true = np.asarray(y_true, dtype=int).reshape(-1)
        self.y_score = np.asarray(y_score, dtype=float).reshape(-1)
        if self.y_true.shape != self.y_score.shape:
            raise ValueError("y_true and y_score must have the same shape")

    @staticmethod
    def predict(y_score: np.ndarray, threshold: float) -> np.ndarray:
        scores = np.asarray(y_score, dtype=float).reshape(-1)
        return (scores >= float(threshold)).astype(int)

    def optimize(self, metric: ThresholdMetric) -> ThresholdOptimizationResult:
        if self.y_true.size == 0:
            return ThresholdOptimizationResult.fallback(
                metric, reason="validation set is empty"
            )
        if np.unique(self.y_true).size != 2:
            return ThresholdOptimizationResult.fallback(
                metric,
                reason="validation set must contain both classes for threshold optimization",
            )

        try:
            f1 = self._optimize_f1()
            balanced_accuracy = self._optimize_balanced_accuracy()
            youden_j = self._optimize_youden_j()
        except ValueError as exc:
            return ThresholdOptimizationResult.fallback(metric, reason=str(exc))

        selected = {
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "youden_j": youden_j,
        }[metric]
        return ThresholdOptimizationResult(
            enabled=True,
            applied=True,
            selected_metric=metric,
            selected_threshold=selected.threshold,
            selected_score=selected.score,
            reason=None,
            threshold_source="validation_optimization",
            f1=f1,
            balanced_accuracy=balanced_accuracy,
            youden_j=youden_j,
        )

    def _optimize_f1(self) -> ThresholdMetricResult:
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_score
        )
        if thresholds.size == 0:
            raise ValueError("precision_recall_curve returned no thresholds")
        numer = 2.0 * precision[:-1] * recall[:-1]
        denom = precision[:-1] + recall[:-1]
        scores = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)
        threshold, score = self._select_threshold(thresholds, scores)
        return ThresholdMetricResult(metric="f1", threshold=threshold, score=score)

    def _optimize_balanced_accuracy(self) -> ThresholdMetricResult:
        fpr, tpr, thresholds = roc_curve(
            self.y_true, self.y_score, drop_intermediate=False
        )
        specificity = 1.0 - fpr
        scores = 0.5 * (tpr + specificity)
        threshold, score = self._select_threshold(thresholds, scores)
        return ThresholdMetricResult(
            metric="balanced_accuracy",
            threshold=threshold,
            score=score,
        )

    def _optimize_youden_j(self) -> ThresholdMetricResult:
        fpr, tpr, thresholds = roc_curve(
            self.y_true, self.y_score, drop_intermediate=False
        )
        scores = tpr - fpr
        threshold, score = self._select_threshold(thresholds, scores)
        return ThresholdMetricResult(
            metric="youden_j", threshold=threshold, score=score
        )

    @staticmethod
    def _select_threshold(
        thresholds: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[float, float]:
        thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        valid = np.isfinite(thresholds) & np.isfinite(scores)
        if not np.any(valid):
            raise ValueError("no finite thresholds available")
        thresholds = thresholds[valid]
        scores = scores[valid]

        best_score = float(np.max(scores))
        best_mask = np.isclose(scores, best_score, rtol=1e-12, atol=1e-12)
        candidate_thresholds = thresholds[best_mask]
        distances = np.abs(candidate_thresholds - 0.5)
        min_distance = float(np.min(distances))
        closest_mask = np.isclose(distances, min_distance, rtol=1e-12, atol=1e-12)
        selected_threshold = float(np.max(candidate_thresholds[closest_mask]))
        return selected_threshold, best_score


def compute_threshold_optimized_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: ThresholdMetric,
) -> tuple[ThresholdOptimizationResult, EvalMetrics]:
    target_arr = np.asarray(y_true, dtype=int).reshape(-1)
    score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    optimization = ThresholdOptimizer(target_arr, score_arr).optimize(metric)
    metrics = compute_metrics_at_threshold(
        target_arr,
        score_arr,
        optimization.selected_threshold,
    )
    return optimization, metrics


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> EvalMetrics:
    target_arr = np.asarray(y_true, dtype=int).reshape(-1)
    score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    predictions = ThresholdOptimizer.predict(score_arr, threshold)
    return MetricsComputer.compute(target_arr, predictions, score_arr)


def _parse_metric_result(
    raw: object,
    metric: ThresholdMetric,
) -> ThresholdMetricResult | None:
    if not isinstance(raw, Mapping):
        return None
    threshold_raw = raw.get("threshold")
    score_raw = raw.get("score")
    try:
        threshold = float(threshold_raw)
        score = float(score_raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(threshold) or not np.isfinite(score):
        return None
    return ThresholdMetricResult(metric=metric, threshold=threshold, score=score)


def load_checkpoint_threshold_optimization(
    raw: object,
    metric: ThresholdMetric,
    *,
    default_threshold: float = 0.5,
) -> ThresholdOptimizationResult:
    if not isinstance(raw, Mapping):
        return ThresholdOptimizationResult.fallback(
            metric,
            reason="checkpoint missing val_threshold_optimization metadata",
            threshold=default_threshold,
            threshold_source="fixed_default",
        )

    metric_results = {
        "f1": _parse_metric_result(raw.get("f1"), "f1"),
        "balanced_accuracy": _parse_metric_result(
            raw.get("balanced_accuracy"),
            "balanced_accuracy",
        ),
        "youden_j": _parse_metric_result(raw.get("youden_j"), "youden_j"),
    }
    selected_metric_result = metric_results[metric]
    if selected_metric_result is not None:
        return ThresholdOptimizationResult(
            enabled=True,
            applied=True,
            selected_metric=metric,
            selected_threshold=selected_metric_result.threshold,
            selected_score=selected_metric_result.score,
            reason=None,
            threshold_source="checkpoint_validation",
            f1=metric_results["f1"],
            balanced_accuracy=metric_results["balanced_accuracy"],
            youden_j=metric_results["youden_j"],
        )

    selected_metric_raw = raw.get("selected_metric")
    selected_threshold_raw = raw.get("selected_threshold")
    selected_score_raw = raw.get("selected_score")
    raw_applied = raw.get("applied")
    try:
        selected_threshold = float(selected_threshold_raw)
    except (TypeError, ValueError):
        selected_threshold = None
    if (
        selected_metric_raw == metric
        and selected_threshold is not None
        and np.isfinite(selected_threshold)
        and raw_applied is not False
    ):
        try:
            selected_score = (
                None if selected_score_raw is None else float(selected_score_raw)
            )
        except (TypeError, ValueError):
            selected_score = None
        if selected_score is not None and not np.isfinite(selected_score):
            selected_score = None
        return ThresholdOptimizationResult(
            enabled=True,
            applied=True,
            selected_metric=metric,
            selected_threshold=selected_threshold,
            selected_score=selected_score,
            reason=None,
            threshold_source="checkpoint_validation",
            f1=metric_results["f1"],
            balanced_accuracy=metric_results["balanced_accuracy"],
            youden_j=metric_results["youden_j"],
        )

    raw_reason = raw.get("reason")
    if isinstance(raw_reason, str) and raw_reason:
        reason = f"checkpoint validation threshold unavailable: {raw_reason}"
    else:
        reason = (
            "checkpoint threshold metadata missing requested metric "
            f"'{metric}'"
        )
    return ThresholdOptimizationResult(
        enabled=True,
        applied=False,
        selected_metric=metric,
        selected_threshold=default_threshold,
        selected_score=None,
        reason=reason,
        threshold_source="fixed_default",
        f1=metric_results["f1"],
        balanced_accuracy=metric_results["balanced_accuracy"],
        youden_j=metric_results["youden_j"],
    )
