from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.audio import (
    AudioPreprocessConfig,
    WaveformPreprocessor,
    WhisperLikeLogMel,
)
from src.data.io import WaveformLoader
from src.data.segment import BagSegmenter, SegmentMetadata
from src.utils.config import DataConfig
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class BagSample:
    instances: Tensor
    label: int
    label_name: str
    audio_path: str
    segment_metadata: tuple[SegmentMetadata, ...]


class RespiratoryBagDataset(LoggingMixin, Dataset[BagSample]):
    def __init__(self, cfg: DataConfig, roots: Sequence[str]):
        self.cfg = cfg
        self.roots = list(roots)
        self._paths: list[Path] = []
        self._targets: list[int] = []
        self._label_names: list[str] = []
        unknown_by_label: dict[str, list[Path]] = defaultdict(list)
        for root in self.roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
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

        bag_preprocess = AudioPreprocessConfig(
            sample_rate=cfg.audio.sample_rate,
            n_fft=cfg.preprocessing.n_fft,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            n_mels=cfg.preprocessing.n_mels,
            clip_seconds=cfg.audio.clip_duration_sec,
            source_type=cfg.preprocessing.source_type,
            bandpass_enabled=cfg.preprocessing.bandpass.enabled,
            bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
            bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
            bandpass_q=cfg.preprocessing.bandpass.q,
        )
        segment_preprocess = AudioPreprocessConfig(
            sample_rate=cfg.audio.sample_rate,
            n_fft=cfg.preprocessing.n_fft,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            n_mels=cfg.preprocessing.n_mels,
            clip_seconds=cfg.segment.effective_length_sec(cfg.audio.clip_duration_sec),
            source_type="original",
            bandpass_enabled=False,
        )
        self._waveform_loader = WaveformLoader(cfg.audio.sample_rate)
        self._bag_preprocessor = WaveformPreprocessor(bag_preprocess)
        self._feature_extractor = WhisperLikeLogMel(segment_preprocess)
        self._segmenter = BagSegmenter(audio=cfg.audio, segment=cfg.segment)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def targets(self) -> list[int]:
        return list(self._targets)

    @property
    def file_paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def segment_audio_ctx(self) -> int:
        return self._feature_extractor.cfg.n_audio_ctx

    @property
    def segment_n_mels(self) -> int:
        return self._feature_extractor.cfg.n_mels

    def _log_unknown_labels(
        self, unknown_by_label: Mapping[str, Sequence[Path]]
    ) -> None:
        known = sorted(self.cfg.label_to_index.keys())
        for label_name, paths in sorted(unknown_by_label.items()):
            self.logger.warning(
                "Skipping MIL bags for unknown label=%s | count=%d | example=%s | known=%s",
                label_name,
                len(paths),
                paths[0],
                known,
            )

    def __getitem__(self, idx: int) -> BagSample:
        path = self._paths[idx]
        waveform = self._waveform_loader.load(path)
        bag_waveform = self._bag_preprocessor.prepare(waveform)
        segments, metadata = self._segmenter(bag_waveform)
        instances = [self._feature_extractor(segment) for segment in segments]
        return BagSample(
            instances=torch.stack(instances, dim=0),
            label=self._targets[idx],
            label_name=self._label_names[idx],
            audio_path=str(path),
            segment_metadata=tuple(metadata),
        )
