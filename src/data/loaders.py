from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import ClipSample, RespiratoryClipDataset
from src.utils.config import DataConfig


@dataclass(frozen=True)
class ClipBatch:
    input_values: Tensor
    labels: Tensor
    audio_paths: tuple[str, ...]
    label_names: tuple[str, ...]

    def to(self, device: torch.device) -> ClipBatch:
        return ClipBatch(
            input_values=self.input_values.to(device),
            labels=self.labels.to(device),
            audio_paths=self.audio_paths,
            label_names=self.label_names,
        )


def build_dataset(cfg: DataConfig, *, split: str = "train") -> RespiratoryClipDataset:
    split_to_roots = {
        "train": cfg.train_dirs,
        "val": cfg.val_dirs,
        "eval": cfg.eval_dirs,
    }
    if split not in split_to_roots:
        raise ValueError(f"Unsupported split: {split}")
    dataset = RespiratoryClipDataset(cfg, split_to_roots[split])
    if len(dataset) == 0:
        raise ValueError(
            f"No usable .wav files found for split={split} under {split_to_roots[split]}"
        )
    return dataset


def clip_collate_fn(samples: list[ClipSample]) -> ClipBatch:
    if not samples:
        raise ValueError("Cannot collate an empty clip batch")
    return ClipBatch(
        input_values=torch.stack([sample.input_values for sample in samples], dim=0),
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.long),
        audio_paths=tuple(sample.audio_path for sample in samples),
        label_names=tuple(sample.label_name for sample in samples),
    )


def build_clip_loader(
    dataset: RespiratoryClipDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader[ClipBatch]:
    return cast(
        DataLoader[ClipBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=clip_collate_fn,
            generator=generator,
        ),
    )
