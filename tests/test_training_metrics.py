from __future__ import annotations

import torch
from src.training.metrics import RunningConfusionMatrix


def test_binary_counts_precision_recall() -> None:
    cm = RunningConfusionMatrix(num_classes=2)
    preds = torch.tensor([1, 1, 0, 0, 1])
    targets = torch.tensor([1, 0, 0, 0, 1])
    cm.update(preds, targets)

    tp, fp, fn, tn = cm.binary_counts()
    assert (tp, fp, fn, tn) == (2, 1, 0, 2)
    assert cm.binary_precision() == 2 / 3
    assert cm.binary_recall() == 1.0
