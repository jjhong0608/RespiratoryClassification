from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from src.data.loaders import build_bag_loader, build_dataset
from src.utils.config import (
    AstFbankConfig,
    AudioConfig,
    BandPassConfig,
    DataConfig,
    DataLoaderConfig,
    DataSplitsConfig,
    FeatureConfig,
    InstanceConfig,
    MetadataConfig,
    SplitConfig,
)


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _data_config(roots: list[str], *, source_type: str = "original") -> DataConfig:
    return DataConfig(
        metadata=MetadataConfig(
            label_to_index={"normal": 0, "wheeze": 1},
            splits=DataSplitsConfig(train=SplitConfig(roots=roots)),
        ),
        audio=AudioConfig(sample_rate=16000),
        instance=InstanceConfig(window_sec=2.0, hop_sec=1.0, tail_policy="cover_end"),
        features=FeatureConfig(
            source_type=source_type,
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=48,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
        loader=DataLoaderConfig(num_workers=0),
    )


def test_recording_bag_dataset_returns_instance_stack_and_metadata(
    tmp_path: Path,
) -> None:
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    _write_wav(normal_dir / "short.wav", duration_sec=0.9, sample_rate=16000)
    _write_wav(wheeze_dir / "long.wav", duration_sec=3.4, sample_rate=16000)

    dataset = build_dataset(_data_config([str(tmp_path)]))
    sample_by_name = {Path(sample.recording_path).name: sample for sample in dataset}

    short_sample = sample_by_name["short.wav"]
    long_sample = sample_by_name["long.wav"]

    assert len(dataset) == 2
    assert short_sample.input_values.shape == (1, 48, 32)
    assert short_sample.instance_start_sec.tolist() == [0.0]
    assert short_sample.instance_end_sec.tolist() == pytest.approx([0.9])
    assert long_sample.input_values.shape == (3, 48, 32)
    assert long_sample.instance_start_sec.tolist() == pytest.approx([0.0, 1.0, 1.4])
    assert long_sample.instance_end_sec.tolist() == pytest.approx([2.0, 3.0, 3.4])
    assert dataset.num_mel_bins == 32
    assert dataset.max_length == 48


def test_bag_loader_pads_variable_instance_counts_and_mask(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    _write_wav(normal_dir / "short.wav", duration_sec=0.9, sample_rate=16000)
    _write_wav(wheeze_dir / "long.wav", duration_sec=3.4, sample_rate=16000)

    dataset = build_dataset(_data_config([str(tmp_path)]))
    loader = build_bag_loader(dataset, batch_size=2, num_workers=0, shuffle=False)

    batch = next(iter(loader))

    assert batch.input_values.shape == (2, 3, 48, 32)
    assert batch.labels.dtype == torch.long
    assert batch.instance_mask.tolist() == [[True, False, False], [True, True, True]]
    assert batch.instance_index.tolist() == [[0, -1, -1], [0, 1, 2]]
    assert batch.recording_paths[0].endswith(".wav")


def test_source_type_changes_recording_bag_features(tmp_path: Path) -> None:
    sample_rate = 16000
    t = torch.arange(sample_rate, dtype=torch.float32) / float(sample_rate)
    tone = 0.2 * torch.sin(2.0 * torch.pi * 400.0 * t)
    clicks = torch.zeros_like(tone)
    clicks[:: (sample_rate // 8)] = 0.6
    audio = tone + clicks
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    sf.write(normal_dir / "sample.wav", audio.numpy(), sample_rate)
    sf.write(wheeze_dir / "sample.wav", audio.numpy(), sample_rate)

    original_dataset = build_dataset(
        _data_config([str(tmp_path)], source_type="original")
    )
    harmonic_dataset = build_dataset(
        _data_config([str(tmp_path)], source_type="harmonic")
    )

    assert not torch.allclose(
        original_dataset[0].input_values,
        harmonic_dataset[0].input_values,
    )
