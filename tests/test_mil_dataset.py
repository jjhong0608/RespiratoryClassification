from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from src.data.loaders import build_bag_loader, build_dataset
from src.utils.config import (
    AudioConfig,
    BandPassConfig,
    DataConfig,
    PreprocessingConfig,
    SegmentationConfig,
)


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _data_config(root: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root)],
        val_dirs=[],
        eval_dirs=[],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=2.0),
        preprocessing=PreprocessingConfig(
            feature_type="log_mel",
            source_type="original",
            n_mels=80,
            bandpass=BandPassConfig(enabled=False),
        ),
        segment=SegmentationConfig(
            mode="sliding_window",
            length_sec=1.0,
            stride_sec=0.5,
            pad_last=True,
            drop_last=False,
        ),
    )


def test_bag_dataset_returns_segment_bag_and_metadata(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    _write_wav(normal_dir / "normal.wav", duration_sec=1.2, sample_rate=16000)
    _write_wav(wheeze_dir / "wheeze.wav", duration_sec=1.7, sample_rate=16000)

    dataset = build_dataset(_data_config(tmp_path))

    sample = dataset[0]

    assert sample.instances.ndim == 3
    assert sample.instances.shape[1] == 80
    assert sample.instances.shape[0] >= 2
    assert len(sample.segment_metadata) == sample.instances.shape[0]
    assert sample.audio_path.endswith(".wav")


def test_bag_loader_pads_variable_instance_counts(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    _write_wav(normal_dir / "short.wav", duration_sec=0.9, sample_rate=16000)
    _write_wav(wheeze_dir / "long.wav", duration_sec=1.9, sample_rate=16000)

    data_cfg = _data_config(tmp_path)
    dataset = build_dataset(data_cfg)
    loader = build_bag_loader(dataset, batch_size=2, num_workers=0, shuffle=False)

    batch = next(iter(loader))

    assert batch.segments.shape[0] == 2
    assert batch.instance_mask.shape[0] == 2
    assert batch.instance_mask.dtype == torch.bool
    assert batch.instance_mask[0].sum().item() != batch.instance_mask[1].sum().item()
