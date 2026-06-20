from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from src.data.audio import (
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    ResNetSpectrogramFeatureConfig,
    ResNetSpectrogramImage,
    SegmentFeatureExtractor,
    WaveformPreprocessor,
    WhisperLikeLogMel,
)
from src.data.io import WaveformLoader
from src.utils.config import DataConfig
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class ClipSample:
    input_values: Tensor
    label: int
    label_name: str
    audio_path: str


class _RespiratoryClipRootDataset(LoggingMixin, Dataset[ClipSample]):
    def __init__(self, cfg: DataConfig, root: str):
        self.cfg = cfg
        self.root = root
        self._paths: list[Path] = []
        self._targets: list[int] = []
        self._label_names: list[str] = []

        unknown_by_label: dict[str, list[Path]] = defaultdict(list)
        root_path = Path(root)
        if root_path.exists():
            for path in sorted(root_path.rglob("*.wav")):
                label_name = path.parent.name
                label = cfg.label_to_index.get(label_name)
                if label is None:
                    unknown_by_label[label_name].append(path)
                    continue
                self._paths.append(path)
                self._targets.append(int(label))
                self._label_names.append(label_name)
        self._log_unknown_labels(unknown_by_label)

        self._waveform_loader = WaveformLoader(cfg.audio.sample_rate)
        self._preprocessor = WaveformPreprocessor(
            self._build_audio_preprocess_config(cfg)
        )
        self._feature_extractor = self._build_feature_extractor(cfg)

    @staticmethod
    def _build_audio_preprocess_config(
        cfg: DataConfig,
    ) -> AudioPreprocessConfig:
        return AudioPreprocessConfig(
            sample_rate=cfg.audio.sample_rate,
            clip_seconds=cfg.audio.clip_duration_sec,
            source_type=cfg.preprocessing.source_type,
            bandpass_enabled=cfg.preprocessing.bandpass.enabled,
            bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
            bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
            bandpass_q=cfg.preprocessing.bandpass.q,
        )

    @staticmethod
    def _build_feature_extractor(cfg: DataConfig) -> SegmentFeatureExtractor:
        if cfg.preprocessing.feature_type == "ast_fbank":
            return AstLikeFbank(
                AstFbankFeatureConfig(
                    sample_rate=cfg.audio.sample_rate,
                    clip_seconds=cfg.audio.clip_duration_sec,
                    num_mel_bins=cfg.preprocessing.ast_fbank.num_mel_bins,
                    max_length=cfg.preprocessing.ast_fbank.max_length,
                    do_normalize=cfg.preprocessing.ast_fbank.do_normalize,
                    mean=cfg.preprocessing.ast_fbank.mean,
                    std=cfg.preprocessing.ast_fbank.std,
                )
            )
        if cfg.preprocessing.feature_type == "log_mel":
            return WhisperLikeLogMel(
                AudioPreprocessConfig(
                    sample_rate=cfg.audio.sample_rate,
                    n_fft=cfg.preprocessing.log_mel.n_fft,
                    hop_length=cfg.preprocessing.log_mel.hop_length,
                    win_length=cfg.preprocessing.log_mel.win_length,
                    n_mels=cfg.preprocessing.log_mel.n_mels,
                    clip_seconds=cfg.audio.clip_duration_sec,
                    source_type=cfg.preprocessing.source_type,
                    bandpass_enabled=cfg.preprocessing.bandpass.enabled,
                    bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
                    bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
                    bandpass_q=cfg.preprocessing.bandpass.q,
                )
            )
        if cfg.preprocessing.feature_type == "resnet_spectrogram":
            return ResNetSpectrogramImage(
                ResNetSpectrogramFeatureConfig(
                    sample_rate=cfg.audio.sample_rate,
                    clip_seconds=cfg.audio.clip_duration_sec,
                    n_fft=cfg.preprocessing.resnet_spectrogram.n_fft,
                    hop_length=cfg.preprocessing.resnet_spectrogram.hop_length,
                    win_length=cfg.preprocessing.resnet_spectrogram.win_length,
                    n_mels=cfg.preprocessing.resnet_spectrogram.n_mels,
                    f_min=cfg.preprocessing.resnet_spectrogram.f_min,
                    f_max=cfg.preprocessing.resnet_spectrogram.f_max,
                    use_hpss=cfg.preprocessing.resnet_spectrogram.use_hpss,
                    hpss_margin=cfg.preprocessing.resnet_spectrogram.hpss_margin,
                    bandpass_enabled=cfg.preprocessing.bandpass.enabled,
                    bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
                    bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
                    bandpass_q=cfg.preprocessing.bandpass.q,
                    image_size=cfg.preprocessing.resnet_spectrogram.image_size,
                    image_mean=cfg.preprocessing.resnet_spectrogram.image_mean,
                    image_std=cfg.preprocessing.resnet_spectrogram.image_std,
                )
            )
        raise ValueError(f"Unsupported feature_type: {cfg.preprocessing.feature_type}")

    def _log_unknown_labels(
        self, unknown_by_label: Mapping[str, Sequence[Path]]
    ) -> None:
        known = sorted(self.cfg.label_to_index.keys())
        for label_name, paths in sorted(unknown_by_label.items()):
            self.logger.warning(
                "Skipping clips for unknown label=%s | count=%d | example=%s | known=%s",
                label_name,
                len(paths),
                paths[0],
                known,
            )

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def targets(self) -> list[int]:
        return list(self._targets)

    @property
    def file_paths(self) -> list[Path]:
        return list(self._paths)

    def __getitem__(self, idx: int) -> ClipSample:
        path = self._paths[idx]
        waveform = self._waveform_loader.load(path)
        if self.cfg.preprocessing.feature_type == "ast_fbank":
            clip_waveform = self._preprocessor.prepare(waveform)
            feature_map = self._feature_extractor(clip_waveform).transpose(0, 1)
        else:
            feature_map = self._feature_extractor(waveform)
        return ClipSample(
            input_values=feature_map.contiguous(),
            label=self._targets[idx],
            label_name=self._label_names[idx],
            audio_path=str(path),
        )

    @property
    def num_mel_bins(self) -> int:
        return self._feature_extractor.n_mels

    @property
    def max_length(self) -> int:
        return self._feature_extractor.n_frames

    @property
    def n_audio_ctx(self) -> int:
        return self._feature_extractor.n_audio_ctx

    @property
    def input_channels(self) -> int:
        return int(getattr(self._feature_extractor, "input_channels", 1))

    @property
    def image_size(self) -> int | None:
        value = getattr(self._feature_extractor, "image_size", None)
        return int(value) if value is not None else None


class RespiratoryClipDataset(LoggingMixin, Dataset[ClipSample]):
    def __init__(self, cfg: DataConfig, roots: Sequence[str]):
        self.cfg = cfg
        self.roots = list(roots)
        self._datasets = [
            _RespiratoryClipRootDataset(cfg, root)
            for root in self.roots
            if Path(root).exists()
        ]
        self._concat: ConcatDataset | None
        self._concat = ConcatDataset(self._datasets) if self._datasets else None

    def __len__(self) -> int:
        if self._concat is None:
            return 0
        return len(self._concat)

    def __getitem__(self, idx: int) -> ClipSample:
        if self._concat is None:
            raise IndexError("Dataset is empty")
        return self._concat[idx]

    @property
    def targets(self) -> list[int]:
        targets: list[int] = []
        for dataset in self._datasets:
            targets.extend(dataset.targets)
        return targets

    @property
    def file_paths(self) -> list[Path]:
        paths: list[Path] = []
        for dataset in self._datasets:
            paths.extend(dataset.file_paths)
        return paths

    @property
    def num_mel_bins(self) -> int:
        if not self._datasets:
            raise ValueError("Dataset is empty")
        return self._datasets[0].num_mel_bins

    @property
    def max_length(self) -> int:
        if not self._datasets:
            raise ValueError("Dataset is empty")
        return self._datasets[0].max_length

    @property
    def n_audio_ctx(self) -> int:
        if not self._datasets:
            raise ValueError("Dataset is empty")
        return self._datasets[0].n_audio_ctx

    @property
    def input_channels(self) -> int:
        if not self._datasets:
            raise ValueError("Dataset is empty")
        return self._datasets[0].input_channels

    @property
    def image_size(self) -> int | None:
        if not self._datasets:
            raise ValueError("Dataset is empty")
        return self._datasets[0].image_size

    @property
    def feature_type(self) -> str:
        return self.cfg.preprocessing.feature_type
