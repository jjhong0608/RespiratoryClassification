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
    WaveformPreprocessor,
)
from src.data.augmentation import AugmentationPipeline
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
    def __init__(
        self,
        cfg: DataConfig,
        root: str,
        *,
        split: str = "train",
        apply_augmentation: bool = False,
    ):
        self.cfg = cfg
        self.root = root
        self.split = split
        self.apply_augmentation = bool(apply_augmentation)
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
        self._feature_extractor = AstLikeFbank(
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
        self._augmentation_pipeline = (
            AugmentationPipeline(cfg.augmentation, sample_rate=cfg.audio.sample_rate)
            if self.apply_augmentation and cfg.augmentation.enabled
            else None
        )

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

    @property
    def waveform_augmentation_enabled(self) -> bool:
        return (
            self._augmentation_pipeline is not None
            and self._augmentation_pipeline.waveform_enabled
        )

    @property
    def fbank_augmentation_enabled(self) -> bool:
        return (
            self._augmentation_pipeline is not None
            and self._augmentation_pipeline.fbank_enabled
        )

    def __getitem__(self, idx: int) -> ClipSample:
        path = self._paths[idx]
        waveform = self._waveform_loader.load(path)
        clip_waveform = self._preprocessor.prepare(waveform)
        augmentation_choice = "independent"
        if self._augmentation_pipeline is not None:
            augmentation_choice = self._augmentation_pipeline.sample_choice()
            clip_waveform = self._augmentation_pipeline.apply_waveform(
                clip_waveform,
                augmentation_choice,
            )
        feature_map = (
            self._feature_extractor(clip_waveform).transpose(0, 1).contiguous()
        )
        if self._augmentation_pipeline is not None:
            feature_map = self._augmentation_pipeline.apply_fbank(
                feature_map,
                augmentation_choice,
            )
        return ClipSample(
            input_values=feature_map,
            label=self._targets[idx],
            label_name=self._label_names[idx],
            audio_path=str(path),
        )


class RespiratoryClipDataset(LoggingMixin, Dataset[ClipSample]):
    def __init__(
        self,
        cfg: DataConfig,
        roots: Sequence[str],
        *,
        split: str = "train",
        apply_augmentation: bool = False,
    ):
        self.cfg = cfg
        self.roots = list(roots)
        self.split = split
        self.apply_augmentation = bool(apply_augmentation)
        self._datasets = [
            _RespiratoryClipRootDataset(
                cfg,
                root,
                split=split,
                apply_augmentation=apply_augmentation,
            )
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
    def root_datasets(self) -> tuple[_RespiratoryClipRootDataset, ...]:
        return tuple(self._datasets)

    @property
    def num_mel_bins(self) -> int:
        return self.cfg.preprocessing.ast_fbank.num_mel_bins

    @property
    def max_length(self) -> int:
        return self.cfg.preprocessing.ast_fbank.max_length
