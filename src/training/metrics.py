from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RunningConfusionMatrix:
    num_classes: int

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        self.matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        preds_i = preds.view(-1).to(torch.long)
        targets_i = targets.view(-1).to(torch.long)
        if preds_i.numel() != targets_i.numel():
            raise ValueError("preds/targets size mismatch")
        idx = self.num_classes * targets_i + preds_i
        bincount = torch.bincount(idx, minlength=self.num_classes**2)
        self.matrix += bincount.view(self.num_classes, self.num_classes)

    @property
    def total(self) -> int:
        return int(self.matrix.sum().item())

    @property
    def correct(self) -> int:
        return int(torch.diag(self.matrix).sum().item())

    @property
    def accuracy(self) -> float:
        return float(self.correct) / float(max(1, self.total))

    def macro_precision(self) -> float:
        tp = torch.diag(self.matrix).to(torch.float64)
        fp = self.matrix.sum(dim=0).to(torch.float64) - tp
        precision_per_class = tp / torch.clamp(tp + fp, min=1.0)
        return float(precision_per_class.mean().item())

    def macro_recall(self) -> float:
        tp = torch.diag(self.matrix).to(torch.float64)
        fn = self.matrix.sum(dim=1).to(torch.float64) - tp
        recall_per_class = tp / torch.clamp(tp + fn, min=1.0)
        return float(recall_per_class.mean().item())

    def binary_precision(self, positive: int = 1) -> float:
        if self.num_classes != 2:
            raise ValueError("binary_precision only valid for num_classes == 2")
        tp = float(self.matrix[positive, positive].item())
        fp = float(self.matrix[:, positive].sum().item() - tp)
        return tp / max(1.0, tp + fp)

    def binary_recall(self, positive: int = 1) -> float:
        if self.num_classes != 2:
            raise ValueError("binary_recall only valid for num_classes == 2")
        tp = float(self.matrix[positive, positive].item())
        fn = float(self.matrix[positive, :].sum().item() - tp)
        return tp / max(1.0, tp + fn)

    def binary_counts(self, positive: int = 1) -> tuple[int, int, int, int]:
        if self.num_classes != 2:
            raise ValueError("binary_counts only valid for num_classes == 2")
        tp = int(self.matrix[positive, positive].item())
        fp = int(self.matrix[:, positive].sum().item() - tp)
        fn = int(self.matrix[positive, :].sum().item() - tp)
        tn = int(self.total - tp - fp - fn)
        return tp, fp, fn, tn
