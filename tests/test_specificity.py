from __future__ import annotations

import numpy as np
import pytest
from src.evaluation.metrics import MetricsComputer


def test_specificity_binary_differs_from_balanced_accuracy() -> None:
    # TN=4, FP=0, FN=2, TP=2 -> specificity=1.0, recall=0.5, balanced_accuracy=0.75
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=int)

    metrics = MetricsComputer.compute(y_true, y_pred, y_prob=None)
    assert metrics.specificity == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
