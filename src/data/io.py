from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch import Tensor


class WaveformLoader:
    def __init__(self, target_sample_rate: int):
        self.target_sample_rate = target_sample_rate
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _resampler_for(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sample_rate,
            )
        return self._resamplers[orig_sr]

    def load(self, path: str | Path) -> Tensor:
        file_path = Path(path)
        data, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)
        audio = torch.from_numpy(data)
        if sr != self.target_sample_rate:
            audio = self._resampler_for(sr)(audio)
        return audio
