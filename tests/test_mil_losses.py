from __future__ import annotations

import pytest
import torch

from src.training.losses import FocalLoss


def test_focal_loss_pos_weight_only_scales_positive_examples() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = FocalLoss(gamma=0.0, pos_weight=3.0)(logits, targets)
    expected = 2.0 * float(torch.log(torch.tensor(2.0)))

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert loss.item() == pytest.approx(expected)


def test_focal_loss_does_not_downweight_negative_examples() -> None:
    logits = torch.tensor([0.0], dtype=torch.float32)
    targets = torch.tensor([0.0], dtype=torch.float32)

    baseline = FocalLoss(gamma=0.0)(logits, targets)
    weighted = FocalLoss(gamma=0.0, pos_weight=5.0)(logits, targets)

    assert torch.isfinite(weighted)
    assert weighted.item() == pytest.approx(baseline.item())
