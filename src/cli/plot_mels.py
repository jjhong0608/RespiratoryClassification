from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from torch import Tensor

from src.data.audio import (
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    WaveformPreprocessor,
    WhisperLikeLogMel,
)
from src.data.io import WaveformLoader
from src.plots.export import PlotlyExportMixin
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging

FeatureType = Literal["log_mel", "ast_fbank"]


class FeatureMapPlotter(PlotlyExportMixin):
    AST_FRAME_SHIFT_SECONDS = 0.01

    def __init__(
        self,
        *,
        input_path: str | Path,
        out_path: str | Path,
        preprocess: AudioPreprocessConfig,
        feature_type: FeatureType,
        ast_fbank_cfg: AstFbankFeatureConfig | None,
        formats: set[str],
        install_chrome: bool = False,
    ):
        self.input_path = Path(input_path)
        self.out_path = Path(out_path)
        self.preprocess = preprocess
        self.feature_type = feature_type
        self.ast_fbank_cfg = ast_fbank_cfg
        self.formats = formats
        self.install_chrome = install_chrome
        self._waveform_loader = WaveformLoader(preprocess.sample_rate)
        self._log_mel_transform: WhisperLikeLogMel | None = None
        self._waveform_preprocessor: WaveformPreprocessor | None = None
        self._ast_transform: AstLikeFbank | None = None
        if feature_type == "log_mel":
            self._log_mel_transform = WhisperLikeLogMel(preprocess)
        elif feature_type == "ast_fbank":
            if ast_fbank_cfg is None:
                raise ValueError(
                    "ast_fbank_cfg is required when feature_type='ast_fbank'"
                )
            self._waveform_preprocessor = WaveformPreprocessor(preprocess)
            self._ast_transform = AstLikeFbank(ast_fbank_cfg)
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

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

    def load_feature_map(self, wav_path: Path) -> Tensor:
        audio = self._waveform_loader.load(wav_path)
        if self.feature_type == "log_mel":
            if self._log_mel_transform is None:
                raise RuntimeError("log-mel transform is not initialized")
            return self._log_mel_transform(audio)
        if self._waveform_preprocessor is None or self._ast_transform is None:
            raise RuntimeError("AST feature extractor is not initialized")
        prepared_audio = self._waveform_preprocessor.prepare_fixed_length(audio)
        return self._ast_transform(prepared_audio)

    def _frame_shift_seconds(self) -> float:
        if self.feature_type == "log_mel":
            return float(self.preprocess.hop_length) / float(
                self.preprocess.sample_rate
            )
        return self.AST_FRAME_SHIFT_SECONDS

    def _y_axis_title(self) -> str:
        if self.feature_type == "log_mel":
            return "Mel bin"
        return "Filterbank bin"

    def _colorbar_title(self) -> str:
        if self.feature_type == "log_mel":
            return "log-mel"
        if self.ast_fbank_cfg is not None and self.ast_fbank_cfg.do_normalize:
            return "normalized fbank"
        return "fbank"

    def build_figure(self, feature_map: Tensor, wav_path: Path) -> go.Figure:
        feature_np = feature_map.detach().cpu().numpy()
        time_axis = (
            np.arange(feature_np.shape[1], dtype=np.float32)
            * self._frame_shift_seconds()
        )
        bins = np.arange(feature_np.shape[0], dtype=np.int32)

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=feature_np,
                    x=time_axis,
                    y=bins,
                    colorscale="Viridis",
                    colorbar={"title": self._colorbar_title()},
                )
            ]
        )
        fig.update_layout(
            title=(
                f"Feature Map | file={wav_path.name} | "
                f"feature={self.feature_type} | "
                f"source={self.preprocess.source_type} | "
                f"shape={feature_np.shape[0]}x{feature_np.shape[1]}"
            ),
            xaxis_title="Time (s)",
            yaxis_title=self._y_axis_title(),
            width=1100,
            height=600,
        )
        return fig

    def run(self) -> list[Path]:
        written: list[Path] = []
        for wav_path in self.collect_inputs():
            feature_map = self.load_feature_map(wav_path)
            fig = self.build_figure(feature_map, wav_path)
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


MelSpectrogramPlotter = FeatureMapPlotter


def _validate_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _build_audio_preprocess(args: argparse.Namespace) -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
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


def _build_ast_fbank_config(args: argparse.Namespace) -> AstFbankFeatureConfig:
    return AstFbankFeatureConfig(
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        num_mel_bins=args.ast_num_mel_bins,
        max_length=args.ast_max_length,
        do_normalize=not bool(args.ast_no_normalize),
        mean=args.ast_mean,
        std=args.ast_std,
    )


def _validate_cli_args(args: argparse.Namespace) -> None:
    _validate_positive("--sample-rate", args.sample_rate)
    _validate_positive("--clip-seconds", args.clip_seconds)
    _validate_positive("--n-fft", args.n_fft)
    _validate_positive("--hop-length", args.hop_length)
    _validate_positive("--win-length", args.win_length)
    _validate_positive("--n-mels", args.n_mels)
    if args.feature_type == "ast_fbank":
        if args.sample_rate != 16000:
            raise ValueError(
                "`--sample-rate` must be 16000 when `--feature-type ast_fbank` is selected"
            )
        _validate_positive("--ast-num-mel-bins", args.ast_num_mel_bins)
        _validate_positive("--ast-max-length", args.ast_max_length)
        if not args.ast_no_normalize and args.ast_std <= 0:
            raise ValueError(
                "`--ast-std` must be greater than zero when AST normalization is enabled"
            )


def _parse_formats(formats_arg: str) -> set[str]:
    formats = {part.strip() for part in formats_arg.split(",") if part.strip()}
    unsupported = formats - {"html", "png", "pdf"}
    if unsupported:
        raise ValueError(f"Unsupported format(s): {sorted(unsupported)}")
    return formats


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
    parser.add_argument(
        "--feature-type",
        default="log_mel",
        choices=["log_mel", "ast_fbank"],
        help="Segment feature frontend to visualize",
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
    parser.add_argument("--ast-num-mel-bins", type=int, default=128)
    parser.add_argument("--ast-max-length", type=int, default=1024)
    parser.add_argument("--ast-mean", type=float, default=-4.2677393)
    parser.add_argument("--ast-std", type=float, default=4.5689974)
    parser.add_argument(
        "--ast-no-normalize",
        action="store_true",
        help="Disable AST-style normalization after Kaldi fbank extraction.",
    )
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args()

    _validate_cli_args(args)

    input_path = Path(args.input)
    out_path = Path(args.out)
    log_dir = (
        out_path if input_path.is_dir() and not out_path.suffix else out_path.parent
    )
    Fs.ensure_dir(log_dir)
    enable_file_logging(log_dir / "plot_mels.log", mode="w")

    preprocess = _build_audio_preprocess(args)
    ast_fbank_cfg = (
        _build_ast_fbank_config(args) if args.feature_type == "ast_fbank" else None
    )
    formats = _parse_formats(str(args.formats))

    plotter = FeatureMapPlotter(
        input_path=input_path,
        out_path=out_path,
        preprocess=preprocess,
        feature_type=args.feature_type,
        ast_fbank_cfg=ast_fbank_cfg,
        formats=formats,
        install_chrome=args.install_chrome,
    )
    plotter.run()


if __name__ == "__main__":
    main()
