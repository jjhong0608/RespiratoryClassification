from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import BagSample, RespiratoryBagDataset
from src.data.segment import SegmentMetadata
from src.utils.config import DataConfig


@dataclass(frozen=True)
class BagBatch:
    segments: Tensor
    instance_mask: Tensor
    labels: Tensor
    audio_paths: tuple[str, ...]
    label_names: tuple[str, ...]
    segment_metadata: tuple[tuple[SegmentMetadata, ...], ...]

    def to(self, device: torch.device) -> BagBatch:
        return BagBatch(
            segments=self.segments.to(device),
            instance_mask=self.instance_mask.to(device),
            labels=self.labels.to(device),
            audio_paths=self.audio_paths,
            label_names=self.label_names,
            segment_metadata=self.segment_metadata,
        )


def build_dataset(cfg: DataConfig, *, split: str = "train") -> RespiratoryBagDataset:
    split_to_roots = {
        "train": cfg.train_dirs,
        "val": cfg.val_dirs,
        "eval": cfg.eval_dirs,
    }
    if split not in split_to_roots:
        raise ValueError(f"Unsupported split: {split}")
    dataset = RespiratoryBagDataset(cfg, split_to_roots[split])
    if len(dataset) == 0:
        raise ValueError(
            f"No usable .wav files found for split={split} under {split_to_roots[split]}"
        )
    return dataset


def bag_collate_fn(samples: list[BagSample]) -> BagBatch:
    if not samples:
        raise ValueError("Cannot collate an empty bag batch")

    batch_size = len(samples)
    max_instances = max(sample.instances.shape[0] for sample in samples)
    n_mels = samples[0].instances.shape[1]
    n_frames = samples[0].instances.shape[2]
    segments = torch.zeros(
        batch_size,
        max_instances,
        n_mels,
        n_frames,
        dtype=samples[0].instances.dtype,
    )
    mask = torch.zeros(batch_size, max_instances, dtype=torch.bool)
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.float32)

    audio_paths: list[str] = []
    label_names: list[str] = []
    metadata: list[tuple[SegmentMetadata, ...]] = []
    for bag_index, sample in enumerate(samples):
        count = sample.instances.shape[0]
        segments[bag_index, :count] = sample.instances
        mask[bag_index, :count] = True
        audio_paths.append(sample.audio_path)
        label_names.append(sample.label_name)
        metadata.append(sample.segment_metadata)

    return BagBatch(
        segments=segments,
        instance_mask=mask,
        labels=labels,
        audio_paths=tuple(audio_paths),
        label_names=tuple(label_names),
        segment_metadata=tuple(metadata),
    )


def build_bag_loader(
    dataset: RespiratoryBagDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader[BagBatch]:
    return cast(
        DataLoader[BagBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=bag_collate_fn,
        ),
    )
