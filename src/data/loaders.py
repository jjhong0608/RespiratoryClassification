from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
from typing import cast

from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from src.data.audio import AudioPreprocessConfig
from src.data.dataset import DatasetConfig, RespiratorySoundDataset


def build_dataset(
    roots: Sequence[str],
    cfg: AudioPreprocessConfig,
    label_to_index: Mapping[str, int],
) -> Dataset[tuple[Tensor, int]]:
    datasets = [
        RespiratorySoundDataset(
            DatasetConfig(roots=[root], label_to_index=label_to_index, preprocess=cfg)
        )
        for root in roots
    ]
    ds: Dataset[tuple[Tensor, int]] = (
        datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    )
    if len(cast(Sized, ds)) == 0:
        raise ValueError(f"No .wav files found under: {list(roots)}")
    return ds


def build_loader(
    dataset: Dataset[tuple[Tensor, int]],
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
