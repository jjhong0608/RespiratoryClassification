from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
import pytest
import soundfile as sf
import torch
from src.cli.plot_mels import (
    FeatureMapPlotter,
    _parse_formats,
    _validate_cli_args,
)
from src.data.audio import AstFbankFeatureConfig, AudioPreprocessConfig


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


def _write_wav(path: Path, audio: torch.Tensor, *, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.detach().cpu().numpy(), sample_rate)


def _build_plotter(
    *,
    input_path: Path,
    out_path: Path,
    preprocess: AudioPreprocessConfig | None = None,
    feature_type: str = "log_mel",
    ast_fbank_cfg: AstFbankFeatureConfig | None = None,
    formats: set[str] | None = None,
) -> FeatureMapPlotter:
    return FeatureMapPlotter(
        input_path=input_path,
        out_path=out_path,
        preprocess=preprocess or AudioPreprocessConfig(clip_seconds=1.0),
        feature_type=feature_type,
        ast_fbank_cfg=ast_fbank_cfg,
        formats=formats or {"html"},
    )


def _base_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "input": "audio.wav",
        "out": "plots/out",
        "formats": "html",
        "feature_type": "log_mel",
        "sample_rate": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "win_length": 400,
        "n_mels": 80,
        "clip_seconds": 30.0,
        "source_type": "original",
        "bandpass_enabled": False,
        "bandpass_low_freq": 250.0,
        "bandpass_high_freq": 1000.0,
        "bandpass_q": 0.707,
        "ast_num_mel_bins": 128,
        "ast_max_length": 1024,
        "ast_mean": -4.2677393,
        "ast_std": 4.5689974,
        "ast_no_normalize": False,
        "install_chrome": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plotter_collects_directory_inputs_recursively(tmp_path: Path) -> None:
    audio = _sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.5)
    wav_a = tmp_path / "input" / "a.wav"
    wav_b = tmp_path / "input" / "nested" / "b.wav"
    _write_wav(wav_a, audio)
    _write_wav(wav_b, audio)

    plotter = _build_plotter(
        input_path=tmp_path / "input",
        out_path=tmp_path / "plots",
    )

    assert plotter.collect_inputs() == [wav_a, wav_b]


def test_output_base_for_directory_preserves_relative_paths(tmp_path: Path) -> None:
    audio = _sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.5)
    wav_path = tmp_path / "input" / "class_a" / "sample.wav"
    _write_wav(wav_path, audio)

    plotter = _build_plotter(
        input_path=tmp_path / "input",
        out_path=tmp_path / "plots",
    )

    assert (
        plotter.output_base_for(wav_path) == tmp_path / "plots" / "class_a" / "sample"
    )


def test_log_mel_feature_map_shape_matches_expected_frames(tmp_path: Path) -> None:
    audio = _sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.8)
    wav_path = tmp_path / "tone.wav"
    preprocess = AudioPreprocessConfig(
        sample_rate=16000,
        clip_seconds=1.0,
        n_mels=64,
        hop_length=160,
    )
    _write_wav(wav_path, audio)

    plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot",
        preprocess=preprocess,
    )

    feature_map = plotter.load_feature_map(wav_path)

    assert feature_map.shape == (64, preprocess.n_frames)


def test_ast_fbank_feature_map_uses_expected_shape_and_labels(tmp_path: Path) -> None:
    audio = _sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.8)
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, audio)
    ast_cfg = AstFbankFeatureConfig(
        sample_rate=16000,
        clip_seconds=1.0,
        num_mel_bins=64,
        max_length=96,
        do_normalize=True,
    )
    plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot",
        preprocess=AudioPreprocessConfig(sample_rate=16000, clip_seconds=1.0),
        feature_type="ast_fbank",
        ast_fbank_cfg=ast_cfg,
    )

    feature_map = plotter.load_feature_map(wav_path)
    fig = plotter.build_figure(feature_map, wav_path)

    assert feature_map.shape == (64, 96)
    assert fig.layout.yaxis.title.text == "Filterbank bin"
    assert fig.data[0].colorbar.title.text == "normalized fbank"
    assert "feature=ast_fbank" in fig.layout.title.text


def test_ast_fbank_time_axis_uses_10ms_frame_shift(tmp_path: Path) -> None:
    audio = _sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.8)
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, audio)
    plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot",
        feature_type="ast_fbank",
        ast_fbank_cfg=AstFbankFeatureConfig(max_length=48),
    )

    fig = plotter.build_figure(plotter.load_feature_map(wav_path), wav_path)
    x_axis = fig.data[0].x

    assert x_axis[0] == pytest.approx(0.0)
    assert x_axis[1] == pytest.approx(0.01)


def test_ast_fbank_source_type_changes_feature_map(tmp_path: Path) -> None:
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
    wav_path = tmp_path / "mixed.wav"
    _write_wav(wav_path, audio)
    ast_cfg = AstFbankFeatureConfig(
        sample_rate=sample_rate,
        clip_seconds=1.0,
        num_mel_bins=64,
        max_length=64,
    )

    original_plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot_original",
        preprocess=AudioPreprocessConfig(sample_rate=sample_rate, clip_seconds=1.0),
        feature_type="ast_fbank",
        ast_fbank_cfg=ast_cfg,
    )
    harmonic_plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot_harmonic",
        preprocess=AudioPreprocessConfig(
            sample_rate=sample_rate,
            clip_seconds=1.0,
            source_type="harmonic",
        ),
        feature_type="ast_fbank",
        ast_fbank_cfg=ast_cfg,
    )

    original = original_plotter.load_feature_map(wav_path)
    harmonic = harmonic_plotter.load_feature_map(wav_path)

    assert not torch.allclose(harmonic, original)


def test_ast_fbank_bandpass_changes_feature_map(tmp_path: Path) -> None:
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
    wav_path = tmp_path / "bandmix.wav"
    _write_wav(wav_path, low + high)
    ast_cfg = AstFbankFeatureConfig(
        sample_rate=sample_rate,
        clip_seconds=1.0,
        num_mel_bins=64,
        max_length=64,
    )

    plain_plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot_plain",
        preprocess=AudioPreprocessConfig(sample_rate=sample_rate, clip_seconds=1.0),
        feature_type="ast_fbank",
        ast_fbank_cfg=ast_cfg,
    )
    filtered_plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plot_filtered",
        preprocess=AudioPreprocessConfig(
            sample_rate=sample_rate,
            clip_seconds=1.0,
            bandpass_enabled=True,
            bandpass_low_hz=250.0,
            bandpass_high_hz=1000.0,
        ),
        feature_type="ast_fbank",
        ast_fbank_cfg=ast_cfg,
    )

    plain = plain_plotter.load_feature_map(wav_path)
    filtered = filtered_plotter.load_feature_map(wav_path)

    assert not torch.allclose(filtered, plain)


def test_plotter_run_dispatches_html_and_static_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "tone.wav"
    audio = _sine_wave(frequency_hz=440.0, sample_rate=16000, duration_sec=0.5)
    _write_wav(wav_path, audio)
    html_paths: list[Path] = []
    image_paths: list[Path] = []

    def _write_html(self: go.Figure, path: str | Path) -> None:
        html_paths.append(Path(path))

    def _write_image(self: go.Figure, path: str | Path) -> None:
        image_paths.append(Path(path))

    monkeypatch.setattr(go.Figure, "write_html", _write_html)
    monkeypatch.setattr(go.Figure, "write_image", _write_image)
    plotter = _build_plotter(
        input_path=wav_path,
        out_path=tmp_path / "plots" / "tone",
        formats={"html", "png", "pdf"},
    )

    written = plotter.run()

    assert sorted(path.suffix for path in written) == [".html", ".pdf", ".png"]
    assert html_paths == [tmp_path / "plots" / "tone.html"]
    assert sorted(path.suffix for path in image_paths) == [".pdf", ".png"]


def test_parse_formats_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="Unsupported format"):
        _parse_formats("html,jpg")


def test_validate_cli_args_rejects_ast_frontend_with_non_16khz_sample_rate() -> None:
    args = _base_args(feature_type="ast_fbank", sample_rate=8000)

    with pytest.raises(ValueError, match="sample-rate"):
        _validate_cli_args(args)


def test_validate_cli_args_rejects_non_positive_ast_std_when_normalized() -> None:
    args = _base_args(feature_type="ast_fbank", ast_std=0.0)

    with pytest.raises(ValueError, match="ast-std"):
        _validate_cli_args(args)
