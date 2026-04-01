from __future__ import annotations

import torch
from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel


def _build_mixed_signal(n_samples: int, sample_rate: int) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / float(sample_rate)
    tonal = 0.1 * torch.sin(2.0 * torch.pi * 220.0 * t)
    clicks = torch.zeros_like(tonal)
    stride = max(1, sample_rate // 8)
    clicks[::stride] = 0.5
    return tonal + clicks


def _build_dual_tone_signal(n_samples: int, sample_rate: int) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / float(sample_rate)
    in_band = torch.sin(2.0 * torch.pi * 400.0 * t)
    out_band = torch.sin(2.0 * torch.pi * 3000.0 * t)
    return 0.5 * (in_band + out_band)


def test_source_type_modes_return_expected_shape() -> None:
    base = AudioPreprocessConfig(sample_rate=16000, n_mels=80, clip_seconds=0.4)
    audio = _build_mixed_signal(base.n_samples, base.sample_rate)
    for mode in ("original", "harmonic", "percussive"):
        cfg = AudioPreprocessConfig(
            sample_rate=base.sample_rate,
            n_mels=base.n_mels,
            clip_seconds=base.clip_seconds,
            source_type=mode,
        )
        mel = WhisperLikeLogMel(cfg)(audio)
        assert mel.shape[0] == cfg.n_mels
        assert mel.shape[1] == cfg.n_frames


def test_harmonic_and_percussive_features_differ() -> None:
    cfg_h = AudioPreprocessConfig(
        sample_rate=16000,
        n_mels=80,
        clip_seconds=0.4,
        source_type="harmonic",
    )
    cfg_p = AudioPreprocessConfig(
        sample_rate=16000,
        n_mels=80,
        clip_seconds=0.4,
        source_type="percussive",
    )
    audio = _build_mixed_signal(cfg_h.n_samples, cfg_h.sample_rate)
    mel_h = WhisperLikeLogMel(cfg_h)(audio)
    mel_p = WhisperLikeLogMel(cfg_p)(audio)
    assert not torch.allclose(mel_h, mel_p)


def test_bandpass_filter_preserves_shape() -> None:
    cfg = AudioPreprocessConfig(
        sample_rate=16000,
        n_mels=80,
        clip_seconds=0.4,
        bandpass_enabled=True,
        bandpass_low_freq=250.0,
        bandpass_high_freq=1000.0,
    )
    audio = _build_mixed_signal(cfg.n_samples, cfg.sample_rate)
    mel = WhisperLikeLogMel(cfg)(audio)
    assert mel.shape == (cfg.n_mels, cfg.n_frames)


def test_bandpass_filter_attenuates_out_of_band_energy() -> None:
    cfg_off = AudioPreprocessConfig(
        sample_rate=16000,
        n_mels=80,
        clip_seconds=0.4,
    )
    cfg_on = AudioPreprocessConfig(
        sample_rate=16000,
        n_mels=80,
        clip_seconds=0.4,
        bandpass_enabled=True,
        bandpass_low_freq=250.0,
        bandpass_high_freq=1000.0,
    )
    audio = _build_dual_tone_signal(cfg_on.n_samples, cfg_on.sample_rate)
    pre_off = WhisperLikeLogMel(cfg_off)
    pre_on = WhisperLikeLogMel(cfg_on)
    filtered = pre_on._apply_bandpass(pre_on.pad_or_trim(audio))
    unfiltered = pre_off.pad_or_trim(audio)

    spec_off = torch.fft.rfft(unfiltered)
    spec_on = torch.fft.rfft(filtered)
    freqs = torch.fft.rfftfreq(unfiltered.numel(), d=1.0 / cfg_on.sample_rate)

    idx_400 = int(torch.argmin(torch.abs(freqs - 400.0)))
    idx_3000 = int(torch.argmin(torch.abs(freqs - 3000.0)))
    ratio_off = spec_off.abs()[idx_3000] / spec_off.abs()[idx_400]
    ratio_on = spec_on.abs()[idx_3000] / spec_on.abs()[idx_400]

    assert ratio_on < ratio_off
