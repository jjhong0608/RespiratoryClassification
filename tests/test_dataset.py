from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel
from src.data.dataset import DatasetConfig, RespiratorySoundDataset


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
