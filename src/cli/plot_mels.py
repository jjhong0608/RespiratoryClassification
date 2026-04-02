from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from torch import Tensor

from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel
from src.data.io import WaveformLoader
from src.plots.export import PlotlyExportMixin
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging


class MelSpectrogramPlotter(PlotlyExportMixin):
    def __init__(
        self,
        *,
        input_path: str | Path,
        out_path: str | Path,
        preprocess: AudioPreprocessConfig,
        formats: set[str],
        install_chrome: bool = False,
    ):
        self.input_path = Path(input_path)
        self.out_path = Path(out_path)
        self.preprocess = preprocess
        self.formats = formats
        self.install_chrome = install_chrome
        self._transform = WhisperLikeLogMel(preprocess)
        self._waveform_loader = WaveformLoader(preprocess.sample_rate)

    def collect_inputs(self) -> list[Path]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".wav":
                raise ValueError(f"Expected a .wav file, got: {self.input_path}")
            return [self.input_path]
        if self.input_path.is_dir():
            files = sorted(self.input_path.rglob("*.wav"))
            if not files:
                raise ValueError(f"No .wav files found under: {self.input_path}")
            return files
        raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

    def output_base_for(self, wav_path: Path) -> Path:
        if self.input_path.is_file():
            return self.out_path.with_suffix("")
        if self.out_path.suffix:
            raise ValueError(
                "Directory input requires `--out` to be a directory path without suffix"
            )
        relative = wav_path.relative_to(self.input_path).with_suffix("")
        return self.out_path / relative

    def load_mel(self, wav_path: Path) -> Tensor:
        audio = self._waveform_loader.load(wav_path)
        return self._transform(audio)

    def build_figure(self, mel: Tensor, wav_path: Path) -> go.Figure:
        mel_np = mel.detach().cpu().numpy()
        time_axis = (
            np.arange(mel_np.shape[1], dtype=np.float32)
            * float(self.preprocess.hop_length)
            / float(self.preprocess.sample_rate)
        )
        mel_bins = np.arange(mel_np.shape[0], dtype=np.int32)

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=mel_np,
                    x=time_axis,
                    y=mel_bins,
                    colorscale="Viridis",
                    colorbar={"title": "log-mel"},
                )
            ]
        )
        fig.update_layout(
            title=(
                f"Mel Spectrogram | file={wav_path.name} | "
                f"source={self.preprocess.source_type} | "
                f"shape={mel_np.shape[0]}x{mel_np.shape[1]}"
            ),
            xaxis_title="Time (s)",
            yaxis_title="Mel bin",
            width=1100,
            height=600,
        )
        return fig

    def run(self) -> list[Path]:
        written: list[Path] = []
        for wav_path in self.collect_inputs():
            mel = self.load_mel(wav_path)
            fig = self.build_figure(mel, wav_path)
            out_base = self.output_base_for(wav_path)
            self.logger.info(f"Plotting {wav_path} -> {out_base}")
            written.extend(
                self.write_outputs(
                    fig,
                    out_base,
                    formats=self.formats,
                    install_chrome=self.install_chrome,
                )
            )
        return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="A .wav file or directory")
    parser.add_argument(
        "--out",
        required=True,
        help="Output base path for a single file, or output directory for a directory input",
    )
    parser.add_argument(
        "--formats",
        default="html,png,pdf",
        help="Comma-separated list: html,png,pdf",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument(
        "--source-type",
        default="original",
        choices=["original", "harmonic", "percussive"],
    )
    parser.add_argument("--bandpass-enabled", action="store_true")
    parser.add_argument("--bandpass-low-freq", type=float, default=250.0)
    parser.add_argument("--bandpass-high-freq", type=float, default=1000.0)
    parser.add_argument("--bandpass-q", type=float, default=0.707)
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    log_dir = (
        out_path if input_path.is_dir() and not out_path.suffix else out_path.parent
    )
    Fs.ensure_dir(log_dir)
    enable_file_logging(log_dir / "plot_mels.log", mode="w")

    preprocess = AudioPreprocessConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        clip_seconds=args.clip_seconds,
        source_type=args.source_type,
        bandpass_enabled=bool(args.bandpass_enabled),
        bandpass_low_hz=args.bandpass_low_freq,
        bandpass_high_hz=args.bandpass_high_freq,
        bandpass_q=args.bandpass_q,
    )
    formats = {part.strip() for part in str(args.formats).split(",") if part.strip()}
    unsupported = formats - {"html", "png", "pdf"}
    if unsupported:
        raise ValueError(f"Unsupported format(s): {sorted(unsupported)}")

    plotter = MelSpectrogramPlotter(
        input_path=input_path,
        out_path=out_path,
        preprocess=preprocess,
        formats=formats,
        install_chrome=args.install_chrome,
    )
    plotter.run()


if __name__ == "__main__":
    main()
