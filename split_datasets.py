from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.utils.logging import LoggingMixin


class SplitCommand(LoggingMixin):
    def run(self, base_dir: Path, output_dir: Path, folds: int, test_ratio: float, seed: int) -> None:
        if folds <= 1:
            raise ValueError("Folds must be greater than 1.")
        if not (0.0 <= test_ratio < 1.0):
            raise ValueError("Test ratio must be between 0.0 and 1.0.")

        # Set seed for reproducibility
        random.seed(seed)
        label_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
        if not label_dirs:
            raise ValueError("Base directory must contain label subdirectories.")
        assignments: Dict[str, List[Path]] = {}
        for label_dir in label_dirs:
            files = sorted(label_dir.glob("*.wav"))
            if not files:
                self.logger.warning("No wav files found in %s", label_dir)
            assignments[label_dir.name] = files

        # 1. Prepare Test Split
        for label_name, files in assignments.items():
            # Always shuffle for random distribution
            random.shuffle(files)

            if test_ratio > 0:
                test_root = output_dir / "test"
                (test_root / label_name).mkdir(parents=True, exist_ok=True)

                num_test = int(len(files) * test_ratio)
                test_files = files[:num_test]
                train_files = files[num_test:]

                # Move/Copy test files
                for file_path in test_files:
                    destination = test_root / label_name / file_path.name
                    shutil.copy2(file_path, destination)

                # Update assignments to only include remaining train files
                assignments[label_name] = train_files

        if test_ratio > 0:
            self.logger.info("Created test split at %s/test (ratio=%.2f)", output_dir, test_ratio)

        # 2. Prepare CV Folds
        for fold_index in range(folds):
            fold_root = output_dir / f"fold_{fold_index}"
            for label_name in assignments.keys():
                (fold_root / label_name).mkdir(parents=True, exist_ok=True)
            for label_name, files in assignments.items():
                for idx, file_path in enumerate(files):
                    if idx % folds == fold_index:
                        destination = fold_root / label_name / file_path.name
                        shutil.copy2(file_path, destination)
            self.logger.info("Prepared fold %d at %s", fold_index, fold_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create k-fold dataset splits.")
    parser.add_argument("--multi-fold", type=int, default=5, help="Number of folds to create.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Ratio of data to hold out for testing (0.0-1.0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Directory containing label subfolders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = SplitCommand()
    command.run(args.base_dir, args.output_dir, args.multi_fold, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
