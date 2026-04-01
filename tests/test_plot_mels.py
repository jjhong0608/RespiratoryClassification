from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.io as pio
import soundfile as sf
from src.cli.plot_mels import MelSpectrogramPlotter
from src.data.audio import AudioPreprocessConfig
from src.data.dataset import DatasetConfig, RespiratorySoundDataset


def _write_wav(path: Path, *, sr: int = 16000, stereo: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    if stereo:
        audio = np.stack([audio, audio * 0.5], axis=-1)
    sf.write(path, audio, sr)


def test_plot_mels_collect_inputs_recursively(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    wav_a = root / "a.wav"
    wav_b = root / "nested" / "b.wav"
    _write_wav(wav_a)
    _write_wav(wav_b)

    plotter = MelSpectrogramPlotter(
        input_path=root,
        out_path=tmp_path / "plots",
        preprocess=AudioPreprocessConfig(
            sample_rate=16000, n_mels=80, clip_seconds=0.3
        ),
        formats={"html"},
    )

    assert plotter.collect_inputs() == [wav_a, wav_b]


def test_plot_mels_matches_dataset_preprocessing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    wav_path = root / "normal" / "sample.wav"
    _write_wav(wav_path, sr=8000, stereo=True)
    preprocess = AudioPreprocessConfig(sample_rate=16000, n_mels=80, clip_seconds=0.3)

    dataset = RespiratorySoundDataset(
        DatasetConfig(
            roots=[str(root)],
            label_to_index={"normal": 0},
            preprocess=preprocess,
        )
    )
    expected_mel, _ = dataset[0]

    plotter = MelSpectrogramPlotter(
        input_path=wav_path,
        out_path=tmp_path / "out",
        preprocess=preprocess,
        formats={"html"},
    )

    actual_mel = plotter.load_mel(wav_path)

    assert np.allclose(actual_mel.numpy(), expected_mel.numpy())


def test_plot_mels_directory_output_preserves_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    wav_path = root / "label" / "sample.wav"
    _write_wav(wav_path)

    plotter = MelSpectrogramPlotter(
        input_path=root,
        out_path=tmp_path / "plots",
        preprocess=AudioPreprocessConfig(
            sample_rate=16000, n_mels=80, clip_seconds=0.3
        ),
        formats={"html"},
    )

    out_base = plotter.output_base_for(wav_path)

    assert out_base == tmp_path / "plots" / "label" / "sample"


def test_plot_mels_run_writes_html_outputs_for_directory(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    wav_a = root / "x.wav"
    wav_b = root / "nested" / "y.wav"
    _write_wav(wav_a)
    _write_wav(wav_b)

    plotter = MelSpectrogramPlotter(
        input_path=root,
        out_path=tmp_path / "plots",
        preprocess=AudioPreprocessConfig(
            sample_rate=16000, n_mels=80, clip_seconds=0.3
        ),
        formats={"html"},
    )

    written = plotter.run()

    assert tmp_path.joinpath("plots", "x.html").exists()
    assert tmp_path.joinpath("plots", "nested", "y.html").exists()
    assert len(written) == 2


def test_plot_mels_run_dispatches_png_and_pdf_exports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path)
    recorded: list[Path] = []

    def fake_write_image(fig, path, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        recorded.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")

    monkeypatch.setattr(pio, "write_image", fake_write_image)

    plotter = MelSpectrogramPlotter(
        input_path=wav_path,
        out_path=tmp_path / "single_plot",
        preprocess=AudioPreprocessConfig(
            sample_rate=16000, n_mels=80, clip_seconds=0.3
        ),
        formats={"png", "pdf"},
    )

    written = plotter.run()

    assert tmp_path.joinpath("single_plot.png") in recorded
    assert tmp_path.joinpath("single_plot.pdf") in recorded
    assert set(written) == {
        tmp_path / "single_plot.png",
        tmp_path / "single_plot.pdf",
    }
