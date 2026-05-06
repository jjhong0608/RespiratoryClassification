from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
    macro_f1: float | None = None
    macro_recall: float | None = None
    weighted_f1: float | None = None
    per_class_precision: list[float] | None = None
    per_class_recall: list[float] | None = None
    per_class_f1: list[float] | None = None
    branch_binary: dict[str, Any] | None = None

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
            "macro_f1": self.macro_f1,
            "macro_recall": self.macro_recall,
            "weighted_f1": self.weighted_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "branch_binary": self.branch_binary,
        }


@dataclass(frozen=True)
class MultiLabelMetrics:
    macro_AP: float
    micro_AP: float
    macro_f1_at_0_5: float
    micro_f1_at_0_5: float
    macro_recall_at_0_5: float
    micro_recall_at_0_5: float
    per_class_AP: list[float]
    valid_ap_class_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_AP": self.macro_AP,
            "micro_AP": self.micro_AP,
            "macro_f1_at_0.5": self.macro_f1_at_0_5,
            "micro_f1_at_0.5": self.micro_f1_at_0_5,
            "macro_recall_at_0.5": self.macro_recall_at_0_5,
            "micro_recall_at_0.5": self.micro_recall_at_0_5,
            "per_class_AP": self.per_class_AP,
            "valid_ap_class_indices": self.valid_ap_class_indices,
        }


class MultiLabelMetricsComputer:
    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        threshold: float = 0.5,
    ) -> MultiLabelMetrics:
        if y_true.ndim != 2 or y_prob.ndim != 2:
            raise ValueError("multi-label targets and probabilities must be 2D")
        if y_true.shape != y_prob.shape:
            raise ValueError(
                "multi-label targets and probabilities must share shape; "
                f"got y_true={y_true.shape} y_prob={y_prob.shape}"
            )
        per_class_ap: list[float] = []
        valid_class_indices: list[int] = []
        for class_index in range(y_true.shape[1]):
            class_targets = y_true[:, class_index]
            if float(class_targets.sum()) <= 0.0:
                logger.warning(
                    "Skipping class=%d for macro AP because validation has zero positives",
                    class_index,
                )
                per_class_ap.append(float("nan"))
                continue
            per_class_ap.append(
                float(average_precision_score(class_targets, y_prob[:, class_index]))
            )
            valid_class_indices.append(class_index)
        if valid_class_indices:
            macro_ap = float(np.nanmean(np.asarray(per_class_ap, dtype=np.float64)))
        else:
            macro_ap = float("nan")
        try:
            micro_ap = float(average_precision_score(y_true.ravel(), y_prob.ravel()))
        except ValueError:
            micro_ap = float("nan")
        y_pred = (y_prob >= threshold).astype(np.int64)
        return MultiLabelMetrics(
            macro_AP=macro_ap,
            micro_AP=micro_ap,
            macro_f1_at_0_5=float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            micro_f1_at_0_5=float(
                f1_score(y_true, y_pred, average="micro", zero_division=0)
            ),
            macro_recall_at_0_5=float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            micro_recall_at_0_5=float(
                recall_score(y_true, y_pred, average="micro", zero_division=0)
            ),
            per_class_AP=per_class_ap,
            valid_ap_class_indices=valid_class_indices,
        )


class MetricsComputer:
    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
        *,
        branch_binary_probabilities: np.ndarray | None = None,
        branch_binary_targets: np.ndarray | None = None,
    ) -> EvalMetrics:
        if y_prob is not None and y_prob.ndim == 1:
            labels = [0, 1]
        elif y_prob is not None and y_prob.ndim == 2:
            labels = list(range(y_prob.shape[1]))
        else:
            labels = sorted(
                set(np.asarray(y_true, dtype=int).tolist())
                | set(np.asarray(y_pred, dtype=int).tolist())
            )
        cm = confusion_matrix(y_true, y_pred, labels=labels)

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

        is_binary = y_prob is not None and y_prob.ndim == 1
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

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_recall = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        per_class_precision = precision_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        per_class_recall = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        branch_binary = None
        if (
            branch_binary_probabilities is not None
            and branch_binary_targets is not None
        ):
            branch_binary = MetricsComputer.compute_branch_binary(
                branch_binary_targets,
                branch_binary_probabilities,
            )

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
            macro_f1=macro_f1,
            macro_recall=macro_recall,
            weighted_f1=weighted_f1,
            per_class_precision=[float(value) for value in per_class_precision],
            per_class_recall=[float(value) for value in per_class_recall],
            per_class_f1=[float(value) for value in per_class_f1],
            branch_binary=branch_binary,
        )

    @staticmethod
    def compute_branch_binary(
        y_true_binary: np.ndarray,
        branch_probabilities: np.ndarray,
    ) -> dict[str, Any]:
        if branch_probabilities.ndim != 2:
            raise ValueError(
                "branch_probabilities must have shape (N, num_branches), "
                f"got {tuple(branch_probabilities.shape)}"
            )
        if y_true_binary.ndim != 1:
            raise ValueError(
                f"y_true_binary must be 1D, got {tuple(y_true_binary.shape)}"
            )
        if branch_probabilities.shape[0] != y_true_binary.shape[0]:
            raise ValueError(
                "branch binary probabilities and targets must have matching rows"
            )

        per_branch: list[dict[str, float | int]] = []
        one_binary_class = np.unique(y_true_binary).size < 2
        for branch_index in range(branch_probabilities.shape[1]):
            branch_probs = branch_probabilities[:, branch_index]
            branch_preds = (branch_probs >= 0.5).astype(np.int64)
            if one_binary_class:
                logger.warning(
                    "Branch binary ROC-AUC is undefined for branch=%d because only "
                    "one binary class is present",
                    branch_index,
                )
                roc_auc = float("nan")
                logger.warning(
                    "Branch binary PR-AUC is undefined for branch=%d because only "
                    "one binary class is present",
                    branch_index,
                )
                pr_auc = float("nan")
            else:
                roc_auc = float(roc_auc_score(y_true_binary, branch_probs))
                pr_auc = float(average_precision_score(y_true_binary, branch_probs))
            try:
                brier = float(brier_score_loss(y_true_binary, branch_probs))
            except ValueError:
                brier = float("nan")
            per_branch.append(
                {
                    "branch_index": branch_index,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "f1_score": float(
                        f1_score(y_true_binary, branch_preds, zero_division=0)
                    ),
                    "brier_score": brier,
                }
            )

        def _nanmean(key: str) -> float:
            values = np.asarray([float(item[key]) for item in per_branch])
            if np.all(np.isnan(values)):
                return float("nan")
            return float(np.nanmean(values))

        return {
            "per_branch": per_branch,
            "mean_roc_auc": _nanmean("roc_auc"),
            "mean_pr_auc": _nanmean("pr_auc"),
            "mean_f1_score": _nanmean("f1_score"),
            "mean_brier_score": _nanmean("brier_score"),
        }
