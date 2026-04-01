from __future__ import annotations

from collections.abc import Sized
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.config import ContrastiveAugmentationConfig


class SpectrogramAugmenter:
    def __init__(self, cfg: ContrastiveAugmentationConfig):
        self.cfg = cfg

    @staticmethod
    def _randint(
        low: int,
        high: int,
        *,
        generator: torch.Generator | None,
    ) -> int:
        if high <= low:
            return low
        return int(torch.randint(low, high, (1,), generator=generator).item())

    def _apply_gaussian_noise(
        self,
        spec: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.cfg.gaussian_noise_std == 0.0:
            return spec

        noise = torch.randn(
            spec.shape,
            generator=generator,
            device=spec.device,
            dtype=spec.dtype,
        )
        return spec + noise * self.cfg.gaussian_noise_std

    def _apply_time_mask(
        self,
        spec: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.cfg.time_mask_param <= 0 or self.cfg.time_mask_count <= 0:
            return spec
        out = spec
        time_dim = out.shape[-1]
        max_width = min(self.cfg.time_mask_param, time_dim)
        for _ in range(self.cfg.time_mask_count):
            width = self._randint(0, max_width + 1, generator=generator)
            if width == 0:
                continue
            start = self._randint(0, time_dim - width + 1, generator=generator)
            out[..., start : start + width] = 0.0
        return out

    def _apply_frequency_mask(
        self,
        spec: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.cfg.freq_mask_param <= 0 or self.cfg.freq_mask_count <= 0:
            return spec
        out = spec
        freq_dim = out.shape[-2]
        max_width = min(self.cfg.freq_mask_param, freq_dim)
        for _ in range(self.cfg.freq_mask_count):
            width = self._randint(0, max_width + 1, generator=generator)
            if width == 0:
                continue
            start = self._randint(0, freq_dim - width + 1, generator=generator)
            out[..., start : start + width, :] = 0.0
        return out

    def __call__(
        self,
        spec: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        out = spec.clone()
        out = self._apply_gaussian_noise(out, generator=generator)
        out = self._apply_time_mask(out, generator=generator)
        out = self._apply_frequency_mask(out, generator=generator)
        return out


class ContrastiveViewDataset(Dataset[tuple[Tensor, Tensor, int]]):
    def __init__(
        self,
        dataset: Dataset[tuple[Tensor, int]],
        augmenter: SpectrogramAugmenter,
        *,
        deterministic_seed_base: int | None = None,
    ):
        self.dataset = dataset
        self.augmenter = augmenter
        self.deterministic_seed_base = deterministic_seed_base

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        x, y = self.dataset[idx]
        label = int(y.item()) if hasattr(y, "item") else int(y)
        if self.deterministic_seed_base is None:
            gen_1 = None
            gen_2 = None
        else:
            base_seed = self.deterministic_seed_base + (idx * 2)
            gen_1 = self._make_generator(base_seed)
            gen_2 = self._make_generator(base_seed + 1)
        view_1 = self.augmenter(x, generator=gen_1)
        view_2 = self.augmenter(x, generator=gen_2)
        return view_1, view_2, label
