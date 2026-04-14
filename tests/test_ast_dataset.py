from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from src.data.loaders import build_clip_loader, build_dataset
from src.utils.config import (
    AstFbankConfig,
    AudioConfig,
    BandPassConfig,
    DataConfig,
    PreprocessingConfig,
)


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(path, audio.astype(np.float32), sample_rate)


def _data_config(roots: list[str], *, source_type: str = "original") -> DataConfig:
    return DataConfig(
        train_dirs=roots,
        val_dirs=[],
        eval_dirs=[],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=2.0),
        preprocessing=PreprocessingConfig(
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
    )


def test_clip_dataset_returns_fixed_shape_ast_input(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    ignored_dir = tmp_path / "ignored"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    ignored_dir.mkdir()
    _write_wav(normal_dir / "normal.wav", duration_sec=1.2, sample_rate=16000)
    _write_wav(wheeze_dir / "wheeze.wav", duration_sec=1.7, sample_rate=16000)
    _write_wav(ignored_dir / "ignored.wav", duration_sec=1.0, sample_rate=16000)

    dataset = build_dataset(_data_config([str(tmp_path)]))

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample.input_values.shape == (48, 32)
    assert sample.audio_path.endswith(".wav")
    assert dataset.num_mel_bins == 32
    assert dataset.max_length == 48


def test_clip_dataset_combines_multiple_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root, label, name in [
        (first_root, "normal", "a.wav"),
        (second_root, "wheeze", "b.wav"),
    ]:
        label_dir = root / label
        label_dir.mkdir(parents=True)
        _write_wav(label_dir / name, duration_sec=1.0, sample_rate=16000)

    dataset = build_dataset(
        _data_config([str(first_root), str(second_root)]),
        split="train",
    )

    assert len(dataset) == 2
    assert sorted(dataset.targets) == [0, 1]


def test_clip_loader_stacks_fixed_shape_inputs(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    wheeze_dir = tmp_path / "wheeze"
    normal_dir.mkdir()
    wheeze_dir.mkdir()
    _write_wav(normal_dir / "short.wav", duration_sec=0.9, sample_rate=16000)
    _write_wav(wheeze_dir / "long.wav", duration_sec=1.9, sample_rate=16000)

    dataset = build_dataset(_data_config([str(tmp_path)]))
    loader = build_clip_loader(dataset, batch_size=2, num_workers=0, shuffle=False)

    batch = next(iter(loader))

    assert batch.input_values.shape == (2, 48, 32)
    assert batch.labels.dtype == torch.long
    assert batch.audio_paths[0].endswith(".wav")


def test_source_type_changes_clip_features(tmp_path: Path) -> None:
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
