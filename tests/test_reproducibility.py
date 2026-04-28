from __future__ import annotations

import random

import numpy as np
import torch
from src.training.imbalance import build_weighted_sampler
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
