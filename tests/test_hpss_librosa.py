from __future__ import annotations

import librosa
import numpy as np
import torch
from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel


def _build_tonal_click_signal(n_samples: int, sample_rate: int) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / float(sample_rate)
    tonal = 0.2 * torch.sin(2.0 * torch.pi * 440.0 * t)
    clicks = torch.zeros_like(tonal)
    stride = max(1, sample_rate // 10)
    clicks[::stride] = 0.8
    return tonal + clicks


def _build_wheeze_like_signal(n_samples: int, sample_rate: int) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / float(sample_rate)
    inst_freq = 450.0 + 60.0 * torch.sin(2.0 * torch.pi * 3.0 * t)
    phase = 2.0 * torch.pi * torch.cumsum(inst_freq / float(sample_rate), dim=0)
    wheeze = 0.15 * torch.sin(phase)
    clicks = torch.zeros_like(wheeze)
    stride = max(1, sample_rate // 12)
    clicks[::stride] = 0.6
    return wheeze + clicks


def _crop_edges(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() <= (2 * n):
        return x
    return x[n:-n]


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    denom = torch.linalg.norm(reference)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.linalg.norm(reference - estimate) / denom)


def _librosa_hpss(
    audio: torch.Tensor,
    cfg: AudioPreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    harmonic, percussive = librosa.effects.hpss(
        y=audio.detach().cpu().numpy().astype(np.float32),
        kernel_size=(31, 31),
        margin=(1.0, 1.0),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        center=True,
        pad_mode="reflect",
        window="hann",
    )
    return (
        torch.from_numpy(np.asarray(harmonic, dtype=np.float32)),
        torch.from_numpy(np.asarray(percussive, dtype=np.float32)),
    )


def test_project_hpss_reconstructs_like_librosa() -> None:
    cfg = AudioPreprocessConfig(sample_rate=16000, n_mels=80, clip_seconds=1.0)
    pre = WhisperLikeLogMel(cfg)
    audio = _build_tonal_click_signal(cfg.n_samples, cfg.sample_rate)

    ours_h, ours_p = pre.separate_harmonic_percussive(audio)
    librosa_h, librosa_p = _librosa_hpss(audio, cfg)

    ours_recon_error = _relative_error(audio, ours_h + ours_p)
    librosa_recon_error = _relative_error(audio, librosa_h + librosa_p)

    assert ours_recon_error < 0.15
    assert librosa_recon_error < 0.15


def test_project_hpss_matches_librosa_component_assignment() -> None:
    cfg = AudioPreprocessConfig(sample_rate=16000, n_mels=80, clip_seconds=1.0)
    pre = WhisperLikeLogMel(cfg)
    audio = _build_tonal_click_signal(cfg.n_samples, cfg.sample_rate)

    ours_h, ours_p = pre.separate_harmonic_percussive(audio)
    librosa_h, librosa_p = _librosa_hpss(audio, cfg)

    crop = cfg.n_fft
    ours_h = _crop_edges(ours_h, crop)
    ours_p = _crop_edges(ours_p, crop)
    librosa_h = _crop_edges(librosa_h, crop)
    librosa_p = _crop_edges(librosa_p, crop)

    matched_h = _cosine_similarity(ours_h, librosa_h)
    matched_p = _cosine_similarity(ours_p, librosa_p)
    crossed_h = _cosine_similarity(ours_h, librosa_p)
    crossed_p = _cosine_similarity(ours_p, librosa_h)

    assert matched_h > 0.55
    assert matched_p > 0.55
    assert matched_h > crossed_h
    assert matched_p > crossed_p


def test_project_hpss_matches_librosa_on_wheeze_like_signal() -> None:
    cfg = AudioPreprocessConfig(sample_rate=16000, n_mels=80, clip_seconds=1.0)
    pre = WhisperLikeLogMel(cfg)
    audio = _build_wheeze_like_signal(cfg.n_samples, cfg.sample_rate)

    ours_h, ours_p = pre.separate_harmonic_percussive(audio)
    librosa_h, librosa_p = _librosa_hpss(audio, cfg)

    crop = cfg.n_fft
    ours_h = _crop_edges(ours_h, crop)
    ours_p = _crop_edges(ours_p, crop)
    librosa_h = _crop_edges(librosa_h, crop)
    librosa_p = _crop_edges(librosa_p, crop)

    matched_h = _cosine_similarity(ours_h, librosa_h)
    matched_p = _cosine_similarity(ours_p, librosa_p)

    assert matched_h > 0.5
    assert matched_p > 0.5
