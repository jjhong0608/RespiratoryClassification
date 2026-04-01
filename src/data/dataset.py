from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel
from src.data.io import WaveformLoader
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class DatasetConfig:
    roots: Sequence[str]
    label_to_index: Mapping[str, int]
    preprocess: AudioPreprocessConfig


class RespiratorySoundDataset(LoggingMixin, Dataset[tuple[Tensor, int]]):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._paths: list[Path] = []
        self._targets: list[int] = []
        unknown_by_label: dict[str, list[Path]] = defaultdict(list)
        for root in cfg.roots:
            root_p = Path(root)
            if not root_p.exists():
                continue
            for path in sorted(root_p.rglob("*.wav")):
                label = self._label_for_path(path)
                if label is None:
                    unknown_by_label[path.parent.name].append(path)
                    continue
                self._paths.append(path)
                self._targets.append(label)
        self._log_unknown_labels(unknown_by_label)
        self._transform = WhisperLikeLogMel(cfg.preprocess)
        self._waveform_loader = WaveformLoader(cfg.preprocess.sample_rate)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def file_paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def targets(self) -> list[int]:
        return list(self._targets)

    def _load_wav(self, path: Path) -> Tensor:
        return self._waveform_loader.load(path)

    def _label_for_path(self, path: Path) -> int | None:
        label_name = path.parent.name
        label = self.cfg.label_to_index.get(label_name)
        if label is None:
            return None
        return int(label)

    def _log_unknown_labels(
        self, unknown_by_label: Mapping[str, Sequence[Path]]
    ) -> None:
        known_labels = sorted(self.cfg.label_to_index.keys())
        for label_name in sorted(unknown_by_label):
            paths = unknown_by_label[label_name]
            first_path = paths[0]
            self.logger.warning(
                "Skipping files for unknown label=%s | count=%d | example=%s | known=%s",
                label_name,
                len(paths),
                first_path,
                known_labels,
            )

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path = self._paths[idx]
        y = self._targets[idx]
        audio = self._load_wav(path)
        x = self._transform(audio)
        return x, y
