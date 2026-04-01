from __future__ import annotations

import os
from pathlib import Path

from src.cli.repair_dataset_symlinks import DatasetSymlinkRepairer


def test_repairer_rewrites_broken_relative_dataprocessing_symlink(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "Whisper"
    dataset_dir = repo_root / "datasets" / "binary" / "crackle"
    source_file = (
        tmp_path
        / "DataProcessing"
        / "audio"
        / "LungSoundNAClassification"
        / "class"
        / "crackle_ILD"
        / "example.wav"
    )
    dataset_dir.mkdir(parents=True)
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"RIFF")

    link_path = dataset_dir / "example.wav"
    broken_target = Path(
        "../../../DataProcessing/audio/LungSoundNAClassification/class/crackle_ILD/example.wav"
    )
    link_path.symlink_to(broken_target)
    assert link_path.is_symlink()
    assert not link_path.exists()

    repairer = DatasetSymlinkRepairer(root=repo_root / "datasets", repo_root=repo_root)
    stats = repairer.repair()

    assert stats.repaired == 1
    assert link_path.exists()
    expected = Path(os.path.relpath(source_file, start=link_path.parent))
    assert Path(os.readlink(link_path)) == expected
