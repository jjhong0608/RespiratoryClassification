from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

ThresholdMetric = Literal["f1", "balanced_accuracy", "youden_j"]


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
    f1: ThresholdMetricResult | None
    balanced_accuracy: ThresholdMetricResult | None
    youden_j: ThresholdMetricResult | None

    @staticmethod
    def fallback(
        metric: ThresholdMetric,
        *,
        reason: str,
        threshold: float = 0.5,
    ) -> ThresholdOptimizationResult:
        return ThresholdOptimizationResult(
            enabled=True,
            applied=False,
            selected_metric=metric,
            selected_threshold=threshold,
            selected_score=None,
            reason=reason,
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
