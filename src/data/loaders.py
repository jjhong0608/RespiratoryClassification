from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import RecordingBagSample, RespiratoryRecordingBagDataset
from src.utils.config import DataConfig


@dataclass(frozen=True)
class BagBatch:
    input_values: Tensor
    labels: Tensor
    instance_mask: Tensor
    num_instances: Tensor
    instance_start_sec: Tensor
    instance_end_sec: Tensor
    instance_index: Tensor
    recording_paths: tuple[str, ...]
    recording_ids: tuple[str, ...]
    label_names: tuple[str, ...]

    def to(self, device: torch.device) -> BagBatch:
        return BagBatch(
            input_values=self.input_values.to(device),
            labels=self.labels.to(device),
            instance_mask=self.instance_mask.to(device),
            num_instances=self.num_instances.to(device),
            instance_start_sec=self.instance_start_sec.to(device),
            instance_end_sec=self.instance_end_sec.to(device),
            instance_index=self.instance_index.to(device),
            recording_paths=self.recording_paths,
            recording_ids=self.recording_ids,
            label_names=self.label_names,
        )


def build_dataset(
    cfg: DataConfig,
    *,
    split: str = "train",
) -> RespiratoryRecordingBagDataset:
    roots = cfg.roots_for_split(split)
    dataset = RespiratoryRecordingBagDataset(cfg, roots)
    if len(dataset) == 0:
        raise ValueError(f"No usable .wav files found for split={split} under {roots}")
    return dataset


def bag_collate_fn(samples: list[RecordingBagSample]) -> BagBatch:
    if not samples:
        raise ValueError("Cannot collate an empty bag batch")

    batch_size = len(samples)
    max_instances = max(int(sample.input_values.shape[0]) for sample in samples)
    max_length = int(samples[0].input_values.shape[1])
    num_mel_bins = int(samples[0].input_values.shape[2])

    input_values = torch.zeros(
        (batch_size, max_instances, max_length, num_mel_bins),
        dtype=samples[0].input_values.dtype,
    )
    instance_mask = torch.zeros((batch_size, max_instances), dtype=torch.bool)
    num_instances = torch.zeros(batch_size, dtype=torch.long)
    instance_start_sec = torch.full(
        (batch_size, max_instances), -1.0, dtype=torch.float32
    )
    instance_end_sec = torch.full(
        (batch_size, max_instances), -1.0, dtype=torch.float32
    )
    instance_index = torch.full((batch_size, max_instances), -1, dtype=torch.long)

    for row, sample in enumerate(samples):
        length = int(sample.input_values.shape[0])
        input_values[row, :length] = sample.input_values
        instance_mask[row, :length] = True
        num_instances[row] = length
        instance_start_sec[row, :length] = sample.instance_start_sec
        instance_end_sec[row, :length] = sample.instance_end_sec
        instance_index[row, :length] = sample.instance_index

    return BagBatch(
        input_values=input_values,
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.long),
        instance_mask=instance_mask,
        num_instances=num_instances,
        instance_start_sec=instance_start_sec,
        instance_end_sec=instance_end_sec,
        instance_index=instance_index,
        recording_paths=tuple(sample.recording_path for sample in samples),
        recording_ids=tuple(sample.recording_id for sample in samples),
        label_names=tuple(sample.label_name for sample in samples),
    )


def build_bag_loader(
    dataset: RespiratoryRecordingBagDataset,
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
