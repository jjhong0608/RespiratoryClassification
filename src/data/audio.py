from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class AudioPreprocessConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    clip_seconds: float = 30.0
    source_type: Literal["original", "harmonic", "percussive"] = "original"
    bandpass_enabled: bool = False
    bandpass_low_freq: float | None = None
    bandpass_high_freq: float | None = None
    bandpass_q: float = 0.707

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.hop_length

    @property
    def n_audio_ctx(self) -> int:
        # Encoder uses a Conv1d with stride=2; output length is floor((L + 1) / 2)
        # for kernel_size=3, padding=1, stride=2.
        return (self.n_frames + 1) // 2


class WhisperLikeLogMel:
    HPSS_TIME_KERNEL = 31
    HPSS_FREQ_KERNEL = 31

    def __init__(self, cfg: AudioPreprocessConfig):
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=0.0,
            f_max=cfg.sample_rate / 2,
            n_mels=cfg.n_mels,
            power=2.0,
            mel_scale="slaney",
            norm="slaney",
        )
        self._stft_window: Tensor | None = None

    def pad_or_trim(self, audio: Tensor) -> Tensor:
        audio = self._trim(audio)
        return self._pad(audio)

    def _trim(self, audio: Tensor) -> Tensor:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform (T,), got {tuple(audio.shape)}")
        n_samples = self.cfg.n_samples
        if audio.numel() > n_samples:
            return audio[:n_samples]
        return audio

    def _pad(self, audio: Tensor) -> Tensor:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform (T,), got {tuple(audio.shape)}")
        n_samples = self.cfg.n_samples
        if audio.numel() < n_samples:
            return torch.nn.functional.pad(audio, (0, n_samples - audio.numel()))
        return audio

    def _apply_bandpass(self, audio: Tensor) -> Tensor:
        if not self.cfg.bandpass_enabled:
            return audio
        if self.cfg.bandpass_low_freq is None or self.cfg.bandpass_high_freq is None:
            raise ValueError("bandpass requires both low and high cutoff frequencies")
        filtered = torchaudio.functional.highpass_biquad(
            audio,
            sample_rate=self.cfg.sample_rate,
            cutoff_freq=self.cfg.bandpass_low_freq,
            Q=self.cfg.bandpass_q,
        )
        filtered = torchaudio.functional.lowpass_biquad(
            filtered,
            sample_rate=self.cfg.sample_rate,
            cutoff_freq=self.cfg.bandpass_high_freq,
            Q=self.cfg.bandpass_q,
        )
        return filtered

    @staticmethod
    def _odd_kernel(kernel: int, dim_len: int) -> int:
        if dim_len <= 1:
            return 1
        k = min(kernel, dim_len)
        if k % 2 == 0:
            k = max(1, k - 1)
        return k

    @staticmethod
    def _median_filter_2d(x: Tensor, *, kernel_size: int, dim: int) -> Tensor:
        # x: (freq, time)
        if dim == 1:
            pad = kernel_size // 2
            padded = F.pad(x[None, None, ...], (pad, pad, 0, 0), mode="reflect")[0, 0]
            windows = padded.unfold(dimension=1, size=kernel_size, step=1)
            return windows.median(dim=-1).values
        if dim == 0:
            pad = kernel_size // 2
            padded = F.pad(x[None, None, ...], (0, 0, pad, pad), mode="reflect")[0, 0]
            windows = padded.unfold(dimension=0, size=kernel_size, step=1)
            return windows.median(dim=-1).values
        raise ValueError(f"Unsupported dim: {dim}")

    def _window(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._stft_window is None or self._stft_window.device != device:
            self._stft_window = torch.hann_window(
                self.cfg.win_length, device=device, dtype=torch.float32
            )
        return self._stft_window.to(dtype=dtype)

    def _hpss_fallback(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        window = self._window(audio.device, audio.dtype)
        stft = torch.stft(
            audio,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        mag = stft.abs()
        k_t = self._odd_kernel(self.HPSS_TIME_KERNEL, mag.shape[1])
        k_f = self._odd_kernel(self.HPSS_FREQ_KERNEL, mag.shape[0])
        harm_med = self._median_filter_2d(mag, kernel_size=k_t, dim=1)
        perc_med = self._median_filter_2d(mag, kernel_size=k_f, dim=0)
        denom = harm_med + perc_med + 1e-10
        harm_mask = harm_med / denom
        perc_mask = perc_med / denom
        harm_stft = stft * harm_mask
        perc_stft = stft * perc_mask
        harmonic = torch.istft(
            harm_stft,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=True,
            length=audio.numel(),
        )
        percussive = torch.istft(
            perc_stft,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=True,
            length=audio.numel(),
        )
        return harmonic, percussive

    def separate_harmonic_percussive(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        if hasattr(torchaudio.functional, "hpss"):
            window = self._window(audio.device, audio.dtype)
            stft = torch.stft(
                audio,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=window,
                return_complex=True,
                center=True,
            )
            harmonic_stft, percussive_stft = torchaudio.functional.hpss(stft)
            harmonic = torch.istft(
                harmonic_stft,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=window,
                center=True,
                length=audio.numel(),
            )
            percussive = torch.istft(
                percussive_stft,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=window,
                center=True,
                length=audio.numel(),
            )
            return harmonic, percussive
        return self._hpss_fallback(audio)

    def _apply_source_type(self, audio: Tensor) -> Tensor:
        if self.cfg.source_type == "original":
            return audio

        harmonic, percussive = self.separate_harmonic_percussive(audio)

        if self.cfg.source_type == "harmonic":
            return harmonic
        if self.cfg.source_type == "percussive":
            return percussive
        raise ValueError(f"Unsupported source_type: {self.cfg.source_type}")

    def __call__(self, audio: Tensor) -> Tensor:
        audio = self._trim(audio)
        audio = self._apply_bandpass(audio)
        audio = self._apply_source_type(audio)
        audio = self._pad(audio)
        mel = self.mel(audio)  # (n_mels, n_frames) with torchaudio's STFT defaults
        expected_frames = self.cfg.n_frames
        if mel.shape[-1] > expected_frames:
            mel = mel[..., :expected_frames]
        elif mel.shape[-1] < expected_frames:
            mel = torch.nn.functional.pad(mel, (0, expected_frames - mel.shape[-1]))
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
