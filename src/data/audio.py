from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torchaudio.compliance import kaldi


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
    bandpass_low_hz: float | None = None
    bandpass_high_hz: float | None = None
    bandpass_q: float = 0.707

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.hop_length

    @property
    def n_audio_ctx(self) -> int:
        return (self.n_frames + 1) // 2


@dataclass(frozen=True)
class AstFbankFeatureConfig:
    sample_rate: int = 16000
    clip_seconds: float = 30.0
    num_mel_bins: int = 128
    max_length: int = 1024
    do_normalize: bool = True
    mean: float = -4.2677393
    std: float = 4.5689974

    @property
    def n_mels(self) -> int:
        return self.num_mel_bins

    @property
    def n_frames(self) -> int:
        return self.max_length

    @property
    def n_audio_ctx(self) -> int:
        return (self.max_length + 1) // 2


@dataclass(frozen=True)
class ResNetSpectrogramFeatureConfig:
    sample_rate: int = 16000
    clip_seconds: float = 15.0
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    use_hpss: bool = True
    hpss_margin: float = 1.0
    bandpass_enabled: bool = False
    bandpass_low_hz: float | None = None
    bandpass_high_hz: float | None = None
    bandpass_q: float = 0.707
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.hop_length

    @property
    def n_audio_ctx(self) -> int:
        return (self.n_frames + 1) // 2

    @property
    def resolved_f_max(self) -> float:
        return self.f_max if self.f_max is not None else self.sample_rate / 2.0


class SegmentFeatureExtractor(Protocol):
    @property
    def n_mels(self) -> int: ...

    @property
    def n_frames(self) -> int: ...

    @property
    def n_audio_ctx(self) -> int: ...

    def __call__(self, audio: Tensor) -> Tensor: ...


class WaveformPreprocessor:
    HPSS_TIME_KERNEL = 31
    HPSS_FREQ_KERNEL = 31

    def __init__(self, cfg: AudioPreprocessConfig):
        self.cfg = cfg
        self._stft_window: Tensor | None = None

    def trim(self, audio: Tensor) -> Tensor:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform (T,), got {tuple(audio.shape)}")
        n_samples = self.cfg.n_samples
        if audio.numel() > n_samples:
            return audio[:n_samples]
        return audio

    def pad(self, audio: Tensor) -> Tensor:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform (T,), got {tuple(audio.shape)}")
        n_samples = self.cfg.n_samples
        if audio.numel() < n_samples:
            return F.pad(audio, (0, n_samples - audio.numel()))
        return audio

    def prepare(self, audio: Tensor) -> Tensor:
        audio = self.trim(audio)
        audio = self._apply_bandpass(audio)
        return self._apply_source_type(audio)

    def prepare_fixed_length(self, audio: Tensor) -> Tensor:
        return self.pad(self.prepare(audio))

    def _apply_bandpass(self, audio: Tensor) -> Tensor:
        if not self.cfg.bandpass_enabled:
            return audio
        if self.cfg.bandpass_low_hz is None or self.cfg.bandpass_high_hz is None:
            raise ValueError("bandpass requires both low_hz and high_hz")
        filtered = torchaudio.functional.highpass_biquad(
            audio,
            sample_rate=self.cfg.sample_rate,
            cutoff_freq=self.cfg.bandpass_low_hz,
            Q=self.cfg.bandpass_q,
        )
        return torchaudio.functional.lowpass_biquad(
            filtered,
            sample_rate=self.cfg.sample_rate,
            cutoff_freq=self.cfg.bandpass_high_hz,
            Q=self.cfg.bandpass_q,
        )

    @staticmethod
    def _odd_kernel(kernel: int, dim_len: int) -> int:
        if dim_len <= 1:
            return 1
        value = min(kernel, dim_len)
        if value % 2 == 0:
            value = max(1, value - 1)
        return value

    @staticmethod
    def _median_filter_2d(x: Tensor, *, kernel_size: int, dim: int) -> Tensor:
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
                self.cfg.win_length,
                device=device,
                dtype=torch.float32,
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
        magnitude = stft.abs()
        time_kernel = self._odd_kernel(self.HPSS_TIME_KERNEL, magnitude.shape[1])
        freq_kernel = self._odd_kernel(self.HPSS_FREQ_KERNEL, magnitude.shape[0])
        harmonic_med = self._median_filter_2d(magnitude, kernel_size=time_kernel, dim=1)
        percussive_med = self._median_filter_2d(
            magnitude,
            kernel_size=freq_kernel,
            dim=0,
        )
        denom = harmonic_med + percussive_med + 1e-10
        harmonic_stft = stft * (harmonic_med / denom)
        percussive_stft = stft * (percussive_med / denom)
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


class WhisperLikeLogMel:
    def __init__(self, cfg: AudioPreprocessConfig):
        self.cfg = cfg
        self.preprocessor = WaveformPreprocessor(cfg)
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

    def __call__(self, audio: Tensor) -> Tensor:
        fixed_audio = self.preprocessor.prepare_fixed_length(audio)
        mel = self.mel(fixed_audio)
        expected_frames = self.cfg.n_frames
        if mel.shape[-1] > expected_frames:
            mel = mel[..., :expected_frames]
        elif mel.shape[-1] < expected_frames:
            mel = F.pad(mel, (0, expected_frames - mel.shape[-1]))
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    @property
    def n_mels(self) -> int:
        return self.cfg.n_mels

    @property
    def n_frames(self) -> int:
        return self.cfg.n_frames

    @property
    def n_audio_ctx(self) -> int:
        return self.cfg.n_audio_ctx


class ResNetSpectrogramImage:
    def __init__(self, cfg: ResNetSpectrogramFeatureConfig):
        self.cfg = cfg
        self.preprocessor = WaveformPreprocessor(
            AudioPreprocessConfig(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                n_mels=cfg.n_mels,
                clip_seconds=cfg.clip_seconds,
                source_type="original",
                bandpass_enabled=cfg.bandpass_enabled,
                bandpass_low_hz=cfg.bandpass_low_hz,
                bandpass_high_hz=cfg.bandpass_high_hz,
                bandpass_q=cfg.bandpass_q,
            )
        )
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.resolved_f_max,
            n_mels=cfg.n_mels,
            power=2.0,
            mel_scale="slaney",
            norm="slaney",
        )
        self._librosa_hpss = None
        if cfg.use_hpss:
            try:
                import librosa
            except ImportError as exc:
                raise RuntimeError(
                    "ResNet50 HPSS frontend requires librosa. Install librosa or "
                    "set data.preprocessing.resnet_spectrogram.use_hpss=false."
                ) from exc
            self._librosa_hpss = librosa.decompose.hpss

    def __call__(self, audio: Tensor) -> Tensor:
        fixed_audio = self.preprocessor.prepare_fixed_length(audio)
        mel = self._match_expected_frames(self.mel(fixed_audio))
        channels = self._build_channels(mel)
        image = F.interpolate(
            channels.unsqueeze(0),
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return self._normalize_image(image)

    def _match_expected_frames(self, mel: Tensor) -> Tensor:
        expected_frames = self.cfg.n_frames
        if mel.shape[-1] > expected_frames:
            return mel[..., :expected_frames]
        if mel.shape[-1] < expected_frames:
            return F.pad(mel, (0, expected_frames - mel.shape[-1]))
        return mel

    @staticmethod
    def _to_log_scale(mel: Tensor) -> Tensor:
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    def _build_channels(self, mel: Tensor) -> Tensor:
        full = self._to_log_scale(mel)
        if not self.cfg.use_hpss:
            return torch.stack([full, full, full], dim=0)
        if self._librosa_hpss is None:
            raise RuntimeError("librosa HPSS function is not initialized")
        mel_np = mel.detach().cpu().numpy().astype(np.float32)
        harmonic_np, percussive_np = self._librosa_hpss(
            mel_np,
            margin=self.cfg.hpss_margin,
        )
        harmonic = torch.from_numpy(np.asarray(harmonic_np)).to(
            device=mel.device,
            dtype=mel.dtype,
        )
        percussive = torch.from_numpy(np.asarray(percussive_np)).to(
            device=mel.device,
            dtype=mel.dtype,
        )
        return torch.stack(
            [
                full,
                self._to_log_scale(harmonic),
                self._to_log_scale(percussive),
            ],
            dim=0,
        )

    def _normalize_image(self, image: Tensor) -> Tensor:
        if image.shape != (3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError(
                "ResNet50 frontend must produce shape "
                f"(3, {self.cfg.image_size}, {self.cfg.image_size}), "
                f"got {tuple(image.shape)}"
            )
        mean = torch.tensor(
            self.cfg.image_mean,
            dtype=image.dtype,
            device=image.device,
        ).view(3, 1, 1)
        std = torch.tensor(
            self.cfg.image_std,
            dtype=image.dtype,
            device=image.device,
        ).view(3, 1, 1)
        return (image - mean) / std

    @property
    def n_mels(self) -> int:
        return self.cfg.n_mels

    @property
    def n_frames(self) -> int:
        return self.cfg.n_frames

    @property
    def n_audio_ctx(self) -> int:
        return self.cfg.n_audio_ctx

    @property
    def input_channels(self) -> int:
        return 3

    @property
    def image_size(self) -> int:
        return self.cfg.image_size


class AstLikeFbank:
    def __init__(self, cfg: AstFbankFeatureConfig):
        self.cfg = cfg

    def __call__(self, audio: Tensor) -> Tensor:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform (T,), got {tuple(audio.shape)}")
        waveform = audio.to(dtype=torch.float32).unsqueeze(0)
        fbank = kaldi.fbank(
            waveform,
            sample_frequency=float(self.cfg.sample_rate),
            window_type="hanning",
            num_mel_bins=self.cfg.num_mel_bins,
        )
        if int(fbank.shape[0]) > self.cfg.max_length:
            fbank = fbank[: self.cfg.max_length, :]
        if self.cfg.do_normalize:
            fbank = (fbank - self.cfg.mean) / (self.cfg.std * 2.0)
        difference = self.cfg.max_length - int(fbank.shape[0])
        if difference > 0:
            fbank = F.pad(fbank, (0, 0, 0, difference))
        return fbank.transpose(0, 1)

    @property
    def n_mels(self) -> int:
        return self.cfg.n_mels

    @property
    def n_frames(self) -> int:
        return self.cfg.n_frames

    @property
    def n_audio_ctx(self) -> int:
        return self.cfg.n_audio_ctx


def build_segment_feature_extractor(
    *,
    feature_type: Literal["log_mel", "ast_fbank", "resnet_spectrogram"],
    log_mel_cfg: AudioPreprocessConfig,
    ast_fbank_cfg: AstFbankFeatureConfig,
    resnet_spectrogram_cfg: ResNetSpectrogramFeatureConfig | None = None,
) -> SegmentFeatureExtractor:
    if feature_type == "log_mel":
        return WhisperLikeLogMel(log_mel_cfg)
    if feature_type == "ast_fbank":
        return AstLikeFbank(ast_fbank_cfg)
    if feature_type == "resnet_spectrogram":
        if resnet_spectrogram_cfg is None:
            raise ValueError(
                "resnet_spectrogram_cfg is required for feature_type='resnet_spectrogram'"
            )
        return ResNetSpectrogramImage(resnet_spectrogram_cfg)
    raise ValueError(f"Unsupported feature_type: {feature_type}")
