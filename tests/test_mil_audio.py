from __future__ import annotations

import torch
from src.data.audio import (
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    WaveformPreprocessor,
)
from torch.testing import assert_close
from torchaudio.compliance import kaldi


def _sine_wave(
    *,
    frequency_hz: float,
    sample_rate: int,
    duration_sec: float,
    amplitude: float = 0.2,
) -> torch.Tensor:
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32)
    t = t / float(sample_rate)
    return amplitude * torch.sin(2.0 * torch.pi * frequency_hz * t)


def test_ast_like_fbank_matches_manual_kaldi_normalization() -> None:
    cfg = AstFbankFeatureConfig(
        sample_rate=16000,
        clip_seconds=1.0,
        num_mel_bins=64,
        max_length=96,
        do_normalize=True,
        mean=-4.2677393,
        std=4.5689974,
    )
    audio = _sine_wave(
        frequency_hz=440.0,
        sample_rate=cfg.sample_rate,
        duration_sec=0.8,
    )
    extractor = AstLikeFbank(cfg)

    actual = extractor(audio)

    manual = kaldi.fbank(
        audio.unsqueeze(0),
        sample_frequency=float(cfg.sample_rate),
        window_type="hanning",
        num_mel_bins=cfg.num_mel_bins,
    )
    if int(manual.shape[0]) > cfg.max_length:
        manual = manual[: cfg.max_length, :]
    manual = (manual - cfg.mean) / (cfg.std * 2.0)
    difference = cfg.max_length - int(manual.shape[0])
    if difference > 0:
        manual = torch.nn.functional.pad(manual, (0, 0, 0, difference))
    manual = manual.transpose(0, 1)

    assert_close(actual, manual)


def test_ast_like_fbank_uses_zero_padding_after_normalization() -> None:
    cfg = AstFbankFeatureConfig(
        sample_rate=16000,
        clip_seconds=1.0,
        num_mel_bins=32,
        max_length=96,
        do_normalize=True,
        mean=-4.2677393,
        std=4.5689974,
    )
    audio = _sine_wave(
        frequency_hz=440.0,
        sample_rate=cfg.sample_rate,
        duration_sec=0.2,
    )
    extractor = AstLikeFbank(cfg)

    features = extractor(audio)
    raw = kaldi.fbank(
        audio.unsqueeze(0),
        sample_frequency=float(cfg.sample_rate),
        window_type="hanning",
        num_mel_bins=cfg.num_mel_bins,
    )
    real_frames = int(raw.shape[0])

    assert real_frames < cfg.max_length
    assert_close(features[:, real_frames:], torch.zeros_like(features[:, real_frames:]))


def test_ast_like_fbank_pads_and_truncates_to_fixed_frame_length() -> None:
    cfg = AstFbankFeatureConfig(
        sample_rate=16000,
        clip_seconds=1.0,
        num_mel_bins=32,
        max_length=48,
    )
    extractor = AstLikeFbank(cfg)

    short_audio = _sine_wave(
        frequency_hz=320.0,
        sample_rate=cfg.sample_rate,
        duration_sec=0.2,
    )
    long_audio = _sine_wave(
        frequency_hz=320.0,
        sample_rate=cfg.sample_rate,
        duration_sec=2.0,
    )

    short_features = extractor(short_audio)
    long_features = extractor(long_audio)

    assert short_features.shape == (32, 48)
    assert long_features.shape == (32, 48)


def test_ast_like_fbank_reports_audio_context_from_max_length() -> None:
    cfg = AstFbankFeatureConfig(max_length=129)

    extractor = AstLikeFbank(cfg)

    assert extractor.n_mels == 128
    assert extractor.n_frames == 129
    assert extractor.n_audio_ctx == 65


def test_bandpass_is_applied_before_ast_feature_extraction() -> None:
    sample_rate = 16000
    low = _sine_wave(
        frequency_hz=80.0,
        sample_rate=sample_rate,
        duration_sec=1.0,
        amplitude=0.2,
    )
    high = _sine_wave(
        frequency_hz=500.0,
        sample_rate=sample_rate,
        duration_sec=1.0,
        amplitude=0.2,
    )
    audio = low + high
    preprocessor = WaveformPreprocessor(
        AudioPreprocessConfig(
            sample_rate=sample_rate,
            clip_seconds=1.0,
            bandpass_enabled=True,
            bandpass_low_hz=250.0,
            bandpass_high_hz=1000.0,
        )
    )
    extractor = AstLikeFbank(
        AstFbankFeatureConfig(
            sample_rate=sample_rate,
            clip_seconds=1.0,
            num_mel_bins=64,
            max_length=64,
        )
    )

    filtered_audio = preprocessor.prepare(audio)
    original_features = extractor(audio)
    filtered_features = extractor(filtered_audio)

    assert not torch.allclose(filtered_audio, audio)
    assert not torch.allclose(filtered_features, original_features)


def test_source_type_is_applied_before_ast_feature_extraction() -> None:
    sample_rate = 16000
    tone = _sine_wave(
        frequency_hz=400.0,
        sample_rate=sample_rate,
        duration_sec=1.0,
        amplitude=0.2,
    )
    clicks = torch.zeros_like(tone)
    clicks[:: (sample_rate // 8)] = 0.6
    audio = tone + clicks
    preprocessor = WaveformPreprocessor(
        AudioPreprocessConfig(
            sample_rate=sample_rate,
            clip_seconds=1.0,
            source_type="harmonic",
        )
    )
    extractor = AstLikeFbank(
        AstFbankFeatureConfig(
            sample_rate=sample_rate,
            clip_seconds=1.0,
            num_mel_bins=64,
            max_length=64,
        )
    )

    harmonic_audio = preprocessor.prepare(audio)
    original_features = extractor(audio)
    harmonic_features = extractor(harmonic_audio)

    assert not torch.allclose(harmonic_audio, audio)
    assert not torch.allclose(harmonic_features, original_features)
