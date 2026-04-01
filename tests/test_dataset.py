from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel
from src.data.dataset import DatasetConfig, RespiratorySoundDataset
from src.data.loaders import build_dataset


def test_dataset_reads_wav_and_label(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "normal").mkdir(parents=True)

    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    wav_path = root / "normal" / "a.wav"
    sf.write(wav_path, audio, sr)

    cfg = DatasetConfig(
        roots=[str(root)],
        label_to_index={"normal": 0},
        preprocess=AudioPreprocessConfig(sample_rate=sr, n_mels=80, clip_seconds=0.2),
    )
    ds = RespiratorySoundDataset(cfg)
    x, y = ds[0]
    assert y == 0
    assert x.shape[0] == 80


def test_log_mel_has_expected_frame_count() -> None:
    cfg = AudioPreprocessConfig(sample_rate=16000, n_mels=80, clip_seconds=0.2)
    transform = WhisperLikeLogMel(cfg)
    audio = torch.zeros(cfg.n_samples, dtype=torch.float32)
    mel = transform(audio)
    assert mel.shape[-1] == cfg.n_frames


def test_dataset_skips_unknown_labels_with_aggregated_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "data"
    (root / "normal").mkdir(parents=True)
    (root / "unknown_a").mkdir(parents=True)
    (root / "unknown_b").mkdir(parents=True)

    sr = 16000
    audio = np.zeros(int(sr * 0.2), dtype=np.float32)
    sf.write(root / "normal" / "ok.wav", audio, sr)
    sf.write(root / "unknown_a" / "skip1.wav", audio, sr)
    sf.write(root / "unknown_a" / "skip2.wav", audio, sr)
    sf.write(root / "unknown_b" / "skip3.wav", audio, sr)

    cfg = DatasetConfig(
        roots=[str(root)],
        label_to_index={"normal": 0},
        preprocess=AudioPreprocessConfig(sample_rate=sr, n_mels=80, clip_seconds=0.2),
    )

    dataset = RespiratorySoundDataset(cfg)
    captured = capsys.readouterr()
    log_text = captured.out + captured.err

    assert len(dataset) == 1
    assert dataset.targets == [0]
    assert dataset.file_paths == [root / "normal" / "ok.wav"]
    assert "unknown_a" in log_text
    assert "count=2" in log_text
    assert "unknown_b" in log_text
    assert "count=1" in log_text


def test_build_dataset_raises_when_all_files_have_unknown_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    (root / "unexpected").mkdir(parents=True)

    sr = 16000
    audio = np.zeros(int(sr * 0.2), dtype=np.float32)
    sf.write(root / "unexpected" / "skip.wav", audio, sr)

    with pytest.raises(ValueError, match="No usable \\.wav files found under"):
        build_dataset(
            [str(root)],
            AudioPreprocessConfig(sample_rate=sr, n_mels=80, clip_seconds=0.2),
            {"normal": 0},
        )
