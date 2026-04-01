from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class RepairStats:
    scanned: int = 0
    repaired: int = 0
    valid: int = 0
    unresolved: int = 0


class DatasetSymlinkRepairer(LoggingMixin):
    def __init__(self, root: Path, repo_root: Path):
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()

    @staticmethod
    def _target_suffix(path: Path) -> Path | None:
        parts = path.parts
        if "DataProcessing" not in parts:
            return None
        idx = parts.index("DataProcessing")
        return Path(*parts[idx:])

    def _candidate_target(self, raw_target: Path) -> Path | None:
        suffix = self._target_suffix(raw_target)
        if suffix is None:
            return None
        candidate = self.repo_root.parent / suffix
        if candidate.exists():
            return candidate
        return None

    def repair(self) -> RepairStats:
        stats = RepairStats()
        for link_path in sorted(self.root.rglob("*.wav")):
            if not link_path.is_symlink():
                continue
            stats = RepairStats(
                scanned=stats.scanned + 1,
                repaired=stats.repaired,
                valid=stats.valid,
                unresolved=stats.unresolved,
            )
            if link_path.exists():
                stats = RepairStats(
                    scanned=stats.scanned,
                    repaired=stats.repaired,
                    valid=stats.valid + 1,
                    unresolved=stats.unresolved,
                )
                continue

            raw_target = Path(os.readlink(link_path))
            candidate = self._candidate_target(raw_target)
            if candidate is None:
                self.logger.warning(
                    "Unresolved broken symlink: %s -> %s", link_path, raw_target
                )
                stats = RepairStats(
                    scanned=stats.scanned,
                    repaired=stats.repaired,
                    valid=stats.valid,
                    unresolved=stats.unresolved + 1,
                )
                continue

            new_target = Path(os.path.relpath(candidate, start=link_path.parent))
            link_path.unlink()
            link_path.symlink_to(new_target)
            self.logger.info("Repaired symlink: %s -> %s", link_path, new_target)
            stats = RepairStats(
                scanned=stats.scanned,
                repaired=stats.repaired + 1,
                valid=stats.valid,
                unresolved=stats.unresolved,
            )

        self.logger.info(
            "Symlink repair finished | scanned=%d | repaired=%d | valid=%d | unresolved=%d",
            stats.scanned,
            stats.repaired,
            stats.valid,
            stats.unresolved,
        )
        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair broken dataset symlinks.")
    parser.add_argument(
        "--root", required=True, type=Path, help="Dataset root to scan."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Whisper repository root. Defaults to the current project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DatasetSymlinkRepairer(root=args.root, repo_root=args.repo_root).repair()


if __name__ == "__main__":
    main()
