from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.audio import (
    AstFbankFeatureConfig,
    AstLikeFbank,
    AudioPreprocessConfig,
    WaveformPreprocessor,
)
from src.data.augmentation import AugmentationPipeline
from src.data.io import WaveformLoader
from src.utils.config import Fsd50kDataConfig
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class Fsd50kSample:
    input_values: Tensor
    clip_id: str
    audio_path: str
    labels: Tensor | None = None


@dataclass(frozen=True)
class Fsd50kBatch:
    input_values: Tensor
    clip_ids: tuple[str, ...]
    audio_paths: tuple[str, ...]
    labels: Tensor | None = None

    def to(self, device: torch.device) -> Fsd50kBatch:
        return Fsd50kBatch(
            input_values=self.input_values.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            clip_ids=self.clip_ids,
            audio_paths=self.audio_paths,
        )


@dataclass(frozen=True)
class Fsd50kVocabulary:
    label_to_index: Mapping[str, int]
    index_to_label: tuple[str, ...]
    mid_to_index: Mapping[str, int]
    index_to_mid: tuple[str, ...]


@dataclass(frozen=True)
class Fsd50kRecord:
    clip_id: str
    audio_path: Path
    labels: tuple[str, ...]
    split: Literal["train", "val", "eval"]


def _first_present(row: Mapping[str, str], candidates: Sequence[str]) -> str | None:
    lower_map = {key.lower(): key for key in row}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None and str(row.get(key, "")).strip():
            return str(row[key]).strip()
    return None


def _split_labels(value: str) -> tuple[str, ...]:
    cleaned = value.replace(";", ",")
    return tuple(part.strip() for part in cleaned.split(",") if part.strip())


class Fsd50kDataset(LoggingMixin, Dataset[Fsd50kSample]):
    def __init__(
        self, cfg: Fsd50kDataConfig, *, split: Literal["train", "val", "eval"]
    ):
        self.cfg = cfg
        self.split = split
        self.vocabulary = parse_fsd50k_vocabulary(Path(cfg.vocabulary_csv))
        dev_rows = _read_csv_rows(Path(cfg.dev_csv))
        eval_rows = _read_csv_rows(Path(cfg.eval_csv))
        self.records = build_fsd50k_records(
            cfg,
            self.vocabulary,
            dev_rows=dev_rows,
            eval_rows=eval_rows,
        )
        self._records = [record for record in self.records if record.split == split]
        if not self._records:
            raise ValueError(f"No FSD50K records found for split={split}")
        self._waveform_loader = WaveformLoader(cfg.audio.sample_rate)
        self._preprocessor = WaveformPreprocessor(
            AudioPreprocessConfig(
                sample_rate=cfg.audio.sample_rate,
                clip_seconds=cfg.audio.clip_duration_sec,
                source_type=cfg.preprocessing.source_type,
                bandpass_enabled=cfg.preprocessing.bandpass.enabled,
                bandpass_low_hz=cfg.preprocessing.bandpass.low_hz,
                bandpass_high_hz=cfg.preprocessing.bandpass.high_hz,
                bandpass_q=cfg.preprocessing.bandpass.q,
            )
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
        self.apply_augmentation = split == "train" and cfg.augmentation.enabled
        self._augmentation_pipeline = (
            AugmentationPipeline(cfg.augmentation, sample_rate=cfg.audio.sample_rate)
            if self.apply_augmentation
            else None
        )

    @property
    def num_classes(self) -> int:
        return len(self.vocabulary.index_to_label)

    @property
    def num_mel_bins(self) -> int:
        return self.cfg.preprocessing.ast_fbank.num_mel_bins

    @property
    def max_length(self) -> int:
        return self.cfg.preprocessing.ast_fbank.max_length

    @property
    def targets(self) -> Tensor:
        targets = []
        for record in self._records:
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in record.labels:
                target[self._label_index(label)] = 1.0
            targets.append(target)
        return torch.stack(targets, dim=0)

    def _label_index(self, label: str) -> int:
        if label in self.vocabulary.label_to_index:
            return int(self.vocabulary.label_to_index[label])
        if label in self.vocabulary.mid_to_index:
            return int(self.vocabulary.mid_to_index[label])
        raise ValueError(f"FSD50K label is missing from vocabulary: {label}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Fsd50kSample:
        record = self._records[idx]
        waveform = self._waveform_loader.load(record.audio_path)
        crop_mode: Literal["center", "random"] = (
            "random" if self.split == "train" else "center"
        )
        clip_waveform = self._preprocessor.prepare_cropped(
            waveform,
            crop_mode=crop_mode,
        )
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
        labels = None
        if self.cfg.mode == "supervised":
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in record.labels:
                labels[self._label_index(label)] = 1.0
        return Fsd50kSample(
            input_values=feature_map,
            labels=labels,
            clip_id=record.clip_id,
            audio_path=str(record.audio_path),
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


_VOCABULARY_INDEX_COLUMNS = ("index", "idx", "id")
_VOCABULARY_LABEL_COLUMNS = ("display_name", "label", "name", "class")
_VOCABULARY_MID_COLUMNS = ("mid", "mids", "audioset_mid")


def _has_vocabulary_header(row: Sequence[str]) -> bool:
    normalized = {cell.strip().lower() for cell in row if cell.strip()}
    known_columns = {
        *[column.lower() for column in _VOCABULARY_INDEX_COLUMNS],
        *[column.lower() for column in _VOCABULARY_LABEL_COLUMNS],
        *[column.lower() for column in _VOCABULARY_MID_COLUMNS],
    }
    return bool(normalized & known_columns)


def _read_vocabulary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if not rows:
        return []
    if _has_vocabulary_header(rows[0]):
        header = rows[0]
        return [
            {header[index]: value for index, value in enumerate(row)}
            for row in rows[1:]
        ]
    return [
        {
            "index": row[0].strip() if len(row) > 0 else "",
            "display_name": row[1].strip() if len(row) > 1 else "",
            "mid": row[2].strip() if len(row) > 2 else "",
        }
        for row in rows
    ]


def parse_fsd50k_vocabulary(path: Path) -> Fsd50kVocabulary:
    rows = _read_vocabulary_rows(path)
    if not rows:
        raise ValueError(f"FSD50K vocabulary is empty: {path}")
    label_to_index: dict[str, int] = {}
    mid_to_index: dict[str, int] = {}
    index_to_label: list[str] = []
    index_to_mid: list[str] = []
    for fallback_index, row in enumerate(rows):
        raw_index = _first_present(row, _VOCABULARY_INDEX_COLUMNS)
        index = fallback_index if raw_index is None else int(raw_index)
        label = _first_present(row, _VOCABULARY_LABEL_COLUMNS)
        mid = _first_present(row, _VOCABULARY_MID_COLUMNS)
        if label is None and mid is None:
            raise ValueError(f"Vocabulary row has no label or MID: {row}")
        while len(index_to_label) <= index:
            index_to_label.append("")
            index_to_mid.append("")
        canonical_label = label or str(mid)
        index_to_label[index] = canonical_label
        index_to_mid[index] = mid or canonical_label
        label_to_index[canonical_label] = index
        if mid is not None:
            mid_to_index[mid] = index
    if any(not label for label in index_to_label):
        raise ValueError("FSD50K vocabulary indices must be contiguous")
    return Fsd50kVocabulary(
        label_to_index=label_to_index,
        index_to_label=tuple(index_to_label),
        mid_to_index=mid_to_index,
        index_to_mid=tuple(index_to_mid),
    )


def _clip_id_from_row(row: Mapping[str, str]) -> str:
    value = _first_present(row, ("fname", "clip_id", "id", "file", "filename"))
    if value is None:
        raise ValueError(f"FSD50K metadata row has no clip id column: {row}")
    return Path(value).stem


def _labels_from_row(row: Mapping[str, str]) -> tuple[str, ...]:
    mids = _first_present(row, ("mids", "mid"))
    if mids is not None:
        return _split_labels(mids)
    labels = _first_present(row, ("labels", "label", "display_name", "tags"))
    if labels is None:
        raise ValueError(f"FSD50K metadata row has no labels column: {row}")
    return _split_labels(labels)


def _metadata_split(row: Mapping[str, str]) -> str | None:
    split = _first_present(row, ("split", "subset"))
    return split.lower() if split is not None else None


def _resolve_audio_path(audio_dir: Path, clip_id: str) -> Path:
    candidate = audio_dir / clip_id
    if candidate.suffix:
        return candidate
    wav_path = audio_dir / f"{clip_id}.wav"
    if wav_path.exists():
        return wav_path
    return wav_path


def _validate_labels(labels: Sequence[str], vocabulary: Fsd50kVocabulary) -> None:
    missing = [
        label
        for label in labels
        if label not in vocabulary.label_to_index
        and label not in vocabulary.mid_to_index
    ]
    if missing:
        raise ValueError(
            f"FSD50K labels missing from vocabulary: {sorted(set(missing))}"
        )


def build_fsd50k_records(
    cfg: Fsd50kDataConfig,
    vocabulary: Fsd50kVocabulary,
    *,
    dev_rows: Sequence[Mapping[str, str]],
    eval_rows: Sequence[Mapping[str, str]],
) -> tuple[Fsd50kRecord, ...]:
    dev_audio_dir = Path(cfg.dev_audio_dir)
    eval_audio_dir = Path(cfg.eval_audio_dir)
    split_values = [_metadata_split(row) for row in dev_rows]
    has_official_split = any(
        split in {"train", "val", "valid", "validation"} for split in split_values
    )
    val_indices: set[int] = set()
    if not has_official_split:
        generator = torch.Generator().manual_seed(int(cfg.split_seed))
        order = torch.randperm(len(dev_rows), generator=generator).tolist()
        val_count = max(1, int(round(float(cfg.val_ratio) * float(len(dev_rows)))))
        val_indices = set(int(index) for index in order[:val_count])

    records: list[Fsd50kRecord] = []
    for row_index, row in enumerate(dev_rows):
        clip_id = _clip_id_from_row(row)
        labels = _labels_from_row(row)
        _validate_labels(labels, vocabulary)
        split_value = _metadata_split(row)
        if split_value in {"val", "valid", "validation"}:
            split: Literal["train", "val", "eval"] = "val"
        elif split_value == "train":
            split = "train"
        else:
            split = "val" if row_index in val_indices else "train"
        records.append(
            Fsd50kRecord(
                clip_id=clip_id,
                audio_path=_resolve_audio_path(dev_audio_dir, clip_id),
                labels=labels,
                split=split,
            )
        )
    for row in eval_rows:
        clip_id = _clip_id_from_row(row)
        labels = _labels_from_row(row)
        _validate_labels(labels, vocabulary)
        records.append(
            Fsd50kRecord(
                clip_id=clip_id,
                audio_path=_resolve_audio_path(eval_audio_dir, clip_id),
                labels=labels,
                split="eval",
            )
        )
    return tuple(records)


def build_fsd50k_dataset(
    cfg: Fsd50kDataConfig,
    *,
    split: Literal["train", "val", "eval"],
) -> Fsd50kDataset:
    return Fsd50kDataset(cfg, split=split)


def fsd50k_collate_fn(samples: list[Fsd50kSample]) -> Fsd50kBatch:
    if not samples:
        raise ValueError("Cannot collate an empty FSD50K batch")
    labels = [sample.labels for sample in samples]
    batch_labels = None
    if all(label is not None for label in labels):
        batch_labels = torch.stack(
            [label for label in labels if label is not None], dim=0
        )
    return Fsd50kBatch(
        input_values=torch.stack([sample.input_values for sample in samples], dim=0),
        labels=batch_labels,
        clip_ids=tuple(sample.clip_id for sample in samples),
        audio_paths=tuple(sample.audio_path for sample in samples),
    )


def build_fsd50k_loader(
    dataset: Fsd50kDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Fsd50kBatch]:
    return cast(
        DataLoader[Fsd50kBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=fsd50k_collate_fn,
            generator=generator,
        ),
    )
