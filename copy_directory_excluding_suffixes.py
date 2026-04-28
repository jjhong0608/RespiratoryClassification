from __future__ import annotations

import argparse
import logging
import shutil
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
class CopySummary:
    copied_files: int
    skipped_files: int


class DirectoryCopyExcludingSuffixes:
    def __init__(
        self,
        source_dir: Path | str,
        target_dir: Path | str,
        excluded_suffixes: list[str] | tuple[str, ...],
    ) -> None:
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.target_dir = Path(target_dir).expanduser().resolve()
        self.excluded_suffixes = self._normalize_suffixes(excluded_suffixes)

    def copy(self) -> CopySummary:
        self._validate()
        self.target_dir.mkdir(parents=True, exist_ok=True)

        copied_files = 0
        skipped_files = 0
        for source_path in self.source_dir.rglob("*"):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(self.source_dir)
            if self._should_skip(source_path):
                skipped_files += 1
                logger.debug("Skipped %s", relative_path)
                continue

            target_path = self.target_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_files += 1
            logger.debug("Copied %s to %s", source_path, target_path)

        summary = CopySummary(copied_files=copied_files, skipped_files=skipped_files)
        logger.info(
            "Copy complete | source=%s | target=%s | excluded_suffixes=%s | "
            "copied_files=%d | skipped_files=%d",
            self.source_dir,
            self.target_dir,
            list(self.excluded_suffixes),
            summary.copied_files,
            summary.skipped_files,
        )
        return summary

    def _validate(self) -> None:
        if not self.source_dir.exists():
            raise ValueError(f"SOURCE_DIR does not exist: {self.source_dir}")
        if not self.source_dir.is_dir():
            raise ValueError(f"SOURCE_DIR is not a directory: {self.source_dir}")
        if not self.excluded_suffixes:
            raise ValueError("At least one excluded suffix must be provided")

    def _should_skip(self, source_path: Path) -> bool:
        name = source_path.name.lower()
        return name.endswith(self.excluded_suffixes)

    @staticmethod
    def _normalize_suffixes(
        suffixes: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = []
        for suffix in suffixes:
            stripped = suffix.strip()
            if not stripped:
                continue
            if not stripped.startswith("."):
                stripped = f".{stripped}"
            normalized.append(stripped.lower())
        return tuple(dict.fromkeys(normalized))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively copy SOURCE_DIR contents into TARGET_DIR while excluding "
            "files with the provided suffixes."
        )
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("excluded_suffixes", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copier = DirectoryCopyExcludingSuffixes(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        excluded_suffixes=args.excluded_suffixes,
    )
    copier.copy()


if __name__ == "__main__":
    main()
