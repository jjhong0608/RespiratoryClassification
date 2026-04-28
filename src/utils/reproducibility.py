from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeededGenerators:
    train_loader: torch.Generator
    val_loader: torch.Generator
    sampler: torch.Generator


class Reproducibility:
    @staticmethod
    def seed_everything(seed: int) -> SeededGenerators:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return Reproducibility.create_generators(seed)

    @staticmethod
    def create_generators(seed: int) -> SeededGenerators:
        return SeededGenerators(
            train_loader=Reproducibility._generator(seed),
            val_loader=Reproducibility._generator(seed + 1),
            sampler=Reproducibility._generator(seed + 2),
        )

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
