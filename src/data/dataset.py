from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from src.data.audio import AudioPreprocessConfig, WhisperLikeLogMel
from src.data.io import WaveformLoader


@dataclass(frozen=True)
class DatasetConfig:
    roots: Sequence[str]
    label_to_index: Mapping[str, int]
    preprocess: AudioPreprocessConfig


class RespiratorySoundDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._paths: list[Path] = []
        for root in cfg.roots:
            root_p = Path(root)
            if not root_p.exists():
                continue
            self._paths.extend(sorted(root_p.rglob("*.wav")))
        self._targets = [self._label_for_path(path) for path in self._paths]
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

    def _label_for_path(self, path: Path) -> int:
        label_name = path.parent.name
        try:
            return int(self.cfg.label_to_index[label_name])
        except KeyError as e:
            raise KeyError(
                f"Unknown label '{label_name}' from {path}; "
                f"known={sorted(self.cfg.label_to_index.keys())}"
            ) from e

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path = self._paths[idx]
        y = self._targets[idx]
        audio = self._load_wav(path)
        x = self._transform(audio)
        return x, y
