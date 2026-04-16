from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.data.audio import (
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    WaveformPreprocessor,
)
from src.data.io import WaveformLoader
from src.data.segmentation import compute_sliding_window_segments
from src.utils.config import DataConfig
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class RecordingBagSample:
    input_values: Tensor
    label: int
    label_name: str
    recording_path: str
    recording_id: str
    instance_start_sec: Tensor
    instance_end_sec: Tensor
    instance_index: Tensor


class RespiratoryRecordingBagDataset(LoggingMixin, Dataset[RecordingBagSample]):
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

        self._waveform_loader = WaveformLoader(cfg.audio.sample_rate)
        self._window_samples = cfg.instance.window_samples(cfg.audio.sample_rate)
        self._preprocessor = WaveformPreprocessor(
            AudioPreprocessConfig(
                sample_rate=cfg.audio.sample_rate,
                clip_seconds=cfg.instance.window_sec,
                source_type=cfg.features.source_type,
                bandpass_enabled=cfg.features.bandpass.enabled,
                bandpass_low_hz=cfg.features.bandpass.low_hz,
                bandpass_high_hz=cfg.features.bandpass.high_hz,
                bandpass_q=cfg.features.bandpass.q,
            )
        )
        self._feature_extractor = AstLikeFbank(
            AstFbankFeatureConfig(
                sample_rate=cfg.audio.sample_rate,
                clip_seconds=cfg.instance.window_sec,
                num_mel_bins=cfg.features.ast_fbank.num_mel_bins,
                max_length=cfg.features.ast_fbank.max_length,
                do_normalize=cfg.features.ast_fbank.do_normalize,
                mean=cfg.features.ast_fbank.mean,
                std=cfg.features.ast_fbank.std,
            )
        )

    def _log_unknown_labels(
        self, unknown_by_label: Mapping[str, Sequence[Path]]
    ) -> None:
        known = sorted(self.cfg.label_to_index.keys())
        for label_name, paths in sorted(unknown_by_label.items()):
            self.logger.warning(
                "Skipping recordings for unknown label=%s | count=%d | example=%s | known=%s",
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
    def recording_paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def num_mel_bins(self) -> int:
        return self.cfg.features.ast_fbank.num_mel_bins

    @property
    def max_length(self) -> int:
        return self.cfg.features.ast_fbank.max_length

    def __getitem__(self, idx: int) -> RecordingBagSample:
        path = self._paths[idx]
        waveform = self._waveform_loader.load(path)
        processed_waveform = self._preprocessor.transform(waveform)
        segments = compute_sliding_window_segments(
            num_samples=int(processed_waveform.numel()),
            sample_rate=self.cfg.audio.sample_rate,
            window_sec=self.cfg.instance.window_sec,
            hop_sec=self.cfg.instance.hop_sec,
            tail_policy=self.cfg.instance.tail_policy,
        )

        features: list[Tensor] = []
        start_sec: list[float] = []
        end_sec: list[float] = []
        indices: list[int] = []
        for segment in segments:
            clip = processed_waveform[segment.start_sample : segment.end_sample]
            if clip.numel() < self._window_samples:
                clip = F.pad(clip, (0, self._window_samples - clip.numel()))
            feature_map = self._feature_extractor(clip).transpose(0, 1).contiguous()
            features.append(feature_map)
            start_sec.append(segment.start_sec)
            end_sec.append(segment.end_sec)
            indices.append(segment.index)

        return RecordingBagSample(
            input_values=torch.stack(features, dim=0),
            label=self._targets[idx],
            label_name=self._label_names[idx],
            recording_path=str(path),
            recording_id=path.stem,
            instance_start_sec=torch.tensor(start_sec, dtype=torch.float32),
            instance_end_sec=torch.tensor(end_sec, dtype=torch.float32),
            instance_index=torch.tensor(indices, dtype=torch.long),
        )
