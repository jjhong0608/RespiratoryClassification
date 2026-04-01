from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
handler = RichHandler(
    rich_tracebacks=True,
    show_path=True,
    omit_repeated_times=False,
)
formatter = logging.Formatter("%(funcName)s - %(message)s")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG)
logging.root.handlers.clear()


@dataclass(frozen=True)
class RecordingLabel:
    wav_path: Path
    txt_path: Path
    label: str


class LoggingMixin:
    @property
    def logger(self) -> logging.Logger:
        return logger


class IcbhiSymlinkBuilder(LoggingMixin):
    LABELS = ("normal", "crackle", "wheeze", "mixed")

    def __init__(self, source_dir: Path, target_dir: Path):
        self.source_dir = source_dir.resolve()
        self.target_dir = target_dir.resolve()

    def _parse_label(self, txt_path: Path) -> str:
        has_crackle = False
        has_wheeze = False
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Annotation file is empty: {txt_path}")
        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Expected at least 4 columns in {txt_path}:{line_no}, got {len(parts)}"
                )
            try:
                crackle = int(parts[2])
                wheeze = int(parts[3])
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer crackle/wheeze flag in {txt_path}:{line_no}"
                ) from exc
            if crackle not in {0, 1}:
                raise ValueError(
                    f"Invalid crackle flag in {txt_path}:{line_no}: {crackle}"
                )
            if wheeze not in {0, 1}:
                raise ValueError(
                    f"Invalid wheeze flag in {txt_path}:{line_no}: {wheeze}"
                )
            has_crackle = has_crackle or crackle == 1
            has_wheeze = has_wheeze or wheeze == 1

        if has_crackle and has_wheeze:
            return "mixed"
        if has_crackle:
            return "crackle"
        if has_wheeze:
            return "wheeze"
        return "normal"

    def collect_recordings(self) -> list[RecordingLabel]:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        if not self.source_dir.is_dir():
            raise NotADirectoryError(
                f"Source path is not a directory: {self.source_dir}"
            )

        recordings: list[RecordingLabel] = []
        wav_paths = sorted(self.source_dir.glob("*.wav"))
        if not wav_paths:
            raise ValueError(f"No .wav files found in: {self.source_dir}")

        for wav_path in wav_paths:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                raise FileNotFoundError(
                    f"Missing annotation for {wav_path.name}: {txt_path}"
                )
            label = self._parse_label(txt_path)
            recordings.append(
                RecordingLabel(wav_path=wav_path, txt_path=txt_path, label=label)
            )
        return recordings

    def prepare_output_dirs(self) -> None:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        for label in self.LABELS:
            (self.target_dir / label).mkdir(parents=True, exist_ok=True)

    def _safe_link(self, source_path: Path, dest_path: Path) -> bool:
        relative_source = Path(os.path.relpath(source_path, start=dest_path.parent))
        if dest_path.is_symlink():
            current_target = Path(os.readlink(dest_path))
            if current_target == relative_source:
                return False
            raise FileExistsError(
                f"Conflicting symlink already exists: {dest_path} -> {current_target}"
            )
        if dest_path.exists():
            raise FileExistsError(
                f"Destination already exists and is not a symlink: {dest_path}"
            )
        dest_path.symlink_to(relative_source)
        return True

    def build(self) -> None:
        recordings = self.collect_recordings()
        self.prepare_output_dirs()
        created_counts = {label: 0 for label in self.LABELS}
        reused_counts = {label: 0 for label in self.LABELS}

        self.logger.info(
            "Source dir=%s | target dir=%s | recordings=%d",
            self.source_dir,
            self.target_dir,
            len(recordings),
        )

        for recording in recordings:
            dest_path = self.target_dir / recording.label / recording.wav_path.name
            created = self._safe_link(recording.wav_path, dest_path)
            if created:
                created_counts[recording.label] += 1
            else:
                reused_counts[recording.label] += 1

        for label in self.LABELS:
            total_links = sum(1 for _ in (self.target_dir / label).glob("*.wav"))
            self.logger.info(
                "Label=%s | created=%d | reused=%d | total_links=%d",
                label,
                created_counts[label],
                reused_counts[label],
                total_links,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create recording-level symbolic links for ICBHI data into "
            "normal/crackle/wheeze/mixed label folders."
        )
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Flat directory containing paired ICBHI .wav and .txt files.",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Output directory where label folders and symbolic links will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    IcbhiSymlinkBuilder(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
    ).build()


if __name__ == "__main__":
    main()
