from __future__ import annotations

import torch
from src.training.supcon import SupervisedContrastiveLoss


def test_supervised_contrastive_loss_prefers_correct_labels() -> None:
    features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.1, 0.9], [0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    good_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    bad_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    criterion = SupervisedContrastiveLoss(temperature=0.07, normalize=True)

    good_loss = criterion(features, good_labels)
    bad_loss = criterion(features, bad_labels)

    assert good_loss.item() < bad_loss.item()


def test_supervised_contrastive_loss_requires_two_views() -> None:
    criterion = SupervisedContrastiveLoss(temperature=0.07, normalize=True)
    features = torch.randn(4, 1, 8)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    try:
        criterion(features, labels)
    except ValueError as exc:
        assert "at least 2 views" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a single-view batch")
