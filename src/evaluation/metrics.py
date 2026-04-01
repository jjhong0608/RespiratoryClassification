from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    f1_score: float
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "confusion_matrix": self.confusion_matrix,
        }


class MetricsComputer:
    @staticmethod
    def compute(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None
    ) -> EvalMetrics:
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape[0] >= 2:
            total = float(cm.sum())
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = float(tn) / float(max(1, tn + fp))
            else:
                specificity_per_class: list[float] = []
                for c in range(cm.shape[0]):
                    tp = float(cm[c, c])
                    fp = float(cm[:, c].sum() - tp)
                    fn = float(cm[c, :].sum() - tp)
                    tn = total - tp - fp - fn
                    specificity_per_class.append(tn / max(1.0, tn + fp))
                specificity = float(np.mean(specificity_per_class))
        else:
            specificity = float("nan")

        classes = np.unique(y_true)
        is_binary = len(classes) == 2
        average = "binary" if is_binary else "macro"

        roc_auc: float | None = None
        pr_auc: float | None = None
        brier: float | None = None
        if y_prob is not None:
            if y_prob.ndim == 1:
                try:
                    roc_auc = float(roc_auc_score(y_true, y_prob))
                except ValueError:
                    roc_auc = None
                try:
                    pr_auc = float(average_precision_score(y_true, y_prob))
                except ValueError:
                    pr_auc = None
                try:
                    brier = float(brier_score_loss(y_true, y_prob))
                except ValueError:
                    brier = None
            elif y_prob.ndim == 2:
                try:
                    roc_auc = float(
                        roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average="macro"
                        )
                    )
                except ValueError:
                    roc_auc = None
                try:
                    y_onehot = label_binarize(
                        y_true, classes=list(range(y_prob.shape[1]))
                    )
                    pr_auc = float(
                        average_precision_score(y_onehot, y_prob, average="macro")
                    )
                except ValueError:
                    pr_auc = None
                try:
                    y_onehot = label_binarize(
                        y_true, classes=list(range(y_prob.shape[1]))
                    )
                    brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))
                except Exception:
                    brier = None

        return EvalMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            ),
            recall=float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            ),
            specificity=specificity,
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            f1_score=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier_score=brier,
            confusion_matrix=cm.astype(int).tolist(),
        )
