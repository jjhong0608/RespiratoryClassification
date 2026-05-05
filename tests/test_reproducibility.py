from __future__ import annotations

import random

import numpy as np
import torch
from src.training.imbalance import (
    build_sqrt_inverse_class_sampler,
    build_weighted_sampler,
)
from src.utils.config import SamplerConfig
from src.utils.reproducibility import Reproducibility


def test_seed_everything_repeats_python_numpy_and_torch_values() -> None:
    Reproducibility.seed_everything(123)
    first = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
    )

    Reproducibility.seed_everything(123)
    second = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.allclose(first[2], second[2])


def test_seeded_generators_repeat_sampler_order() -> None:
    targets = [0, 1, 1, 0, 1]
    first_sampler = build_weighted_sampler(
        targets,
        generator=Reproducibility.create_generators(7).sampler,
    )
    second_sampler = build_weighted_sampler(
        targets,
        generator=Reproducibility.create_generators(7).sampler,
    )

    assert list(first_sampler) == list(second_sampler)


def test_seeded_generators_repeat_sqrt_inverse_sampler_order() -> None:
    targets = [0, 0, 0, 1, 1, 2]
    sampler_cfg = SamplerConfig(
        enabled=True,
        type="sqrt_inverse_class",
        replacement=True,
        num_samples="dataset_size",
        source="train",
    )
    first_sampler, _ = build_sqrt_inverse_class_sampler(
        targets,
        num_classes=3,
        cfg=sampler_cfg,
        generator=Reproducibility.create_generators(11).sampler,
    )
    second_sampler, _ = build_sqrt_inverse_class_sampler(
        targets,
        num_classes=3,
        cfg=sampler_cfg,
        generator=Reproducibility.create_generators(11).sampler,
    )

    assert list(first_sampler) == list(second_sampler)
