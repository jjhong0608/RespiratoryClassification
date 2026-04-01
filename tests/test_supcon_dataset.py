from __future__ import annotations

import torch
from src.data.contrastive import ContrastiveViewDataset, SpectrogramAugmenter
from src.utils.config import ContrastiveAugmentationConfig
from torch.utils.data import TensorDataset


def test_contrastive_view_dataset_returns_deterministic_views_when_seeded() -> None:
    base = TensorDataset(torch.ones(2, 80, 12), torch.tensor([0, 1]))
    augmenter = SpectrogramAugmenter(
        ContrastiveAugmentationConfig(
            time_mask_param=0,
            time_mask_count=0,
            freq_mask_param=0,
            freq_mask_count=0,
            gaussian_noise_std=0.1,
        )
    )
    dataset = ContrastiveViewDataset(
        base,
        augmenter,
        deterministic_seed_base=123,
    )

    first_a, first_b, first_label = dataset[0]
    second_a, second_b, second_label = dataset[0]

    assert first_label == 0
    assert second_label == 0
    assert torch.equal(first_a, second_a)
    assert torch.equal(first_b, second_b)
    assert not torch.equal(first_a, first_b)
    assert first_a.shape == torch.Size([80, 12])


def test_contrastive_view_dataset_preserves_shape_without_augmentation() -> None:
    base = TensorDataset(torch.randn(1, 80, 10), torch.tensor([1]))
    augmenter = SpectrogramAugmenter(
        ContrastiveAugmentationConfig(
            time_mask_param=0,
            time_mask_count=0,
            freq_mask_param=0,
            freq_mask_count=0,
            gaussian_noise_std=0.0,
        )
    )
    dataset = ContrastiveViewDataset(base, augmenter)

    view_1, view_2, label = dataset[0]

    assert label == 1
    assert torch.equal(view_1, view_2)
    assert view_1.shape == torch.Size([80, 10])
