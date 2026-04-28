from __future__ import annotations

from pathlib import Path

import pytest
from copy_directory_excluding_suffixes import DirectoryCopyExcludingSuffixes


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_recursively_copies_allowed_files_and_preserves_structure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "keep.txt", "root")
    _write(source / "nested" / "keep.json", "nested")
    _write(source / "nested" / "skip.wav", "audio")

    summary = DirectoryCopyExcludingSuffixes(source, target, [".wav"]).copy()

    assert summary.copied_files == 2
    assert summary.skipped_files == 1
    assert (target / "keep.txt").read_text(encoding="utf-8") == "root"
    assert (target / "nested" / "keep.json").read_text(encoding="utf-8") == "nested"
    assert not (target / "nested" / "skip.wav").exists()


def test_skips_multiple_excluded_suffixes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "keep.txt", "keep")
    _write(source / "skip.wav", "wav")
    _write(source / "skip.mp3", "mp3")
    _write(source / "skip.npy", "npy")

    summary = DirectoryCopyExcludingSuffixes(
        source,
        target,
        [".wav", ".mp3", ".npy"],
    ).copy()

    assert summary.copied_files == 1
    assert summary.skipped_files == 3
    assert (target / "keep.txt").exists()
    assert not (target / "skip.wav").exists()
    assert not (target / "skip.mp3").exists()
    assert not (target / "skip.npy").exists()


def test_accepts_suffixes_with_and_without_leading_dot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "skip.wav", "wav")
    _write(source / "skip.mp3", "mp3")
    _write(source / "keep.txt", "keep")

    summary = DirectoryCopyExcludingSuffixes(source, target, ["wav", ".mp3"]).copy()

    assert summary.copied_files == 1
    assert summary.skipped_files == 2
    assert (target / "keep.txt").exists()
    assert not (target / "skip.wav").exists()
    assert not (target / "skip.mp3").exists()


def test_matches_suffixes_case_insensitively(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "skip.WAV", "wav")
    _write(source / "keep.txt", "keep")

    summary = DirectoryCopyExcludingSuffixes(source, target, [".wav"]).copy()

    assert summary.copied_files == 1
    assert summary.skipped_files == 1
    assert (target / "keep.txt").exists()
    assert not (target / "skip.WAV").exists()


def test_supports_compound_suffixes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "skip.tar.gz", "archive")
    _write(source / "keep.gz", "compressed")

    summary = DirectoryCopyExcludingSuffixes(source, target, [".tar.gz"]).copy()

    assert summary.copied_files == 1
    assert summary.skipped_files == 1
    assert (target / "keep.gz").exists()
    assert not (target / "skip.tar.gz").exists()


def test_overwrites_existing_target_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write(source / "same.txt", "new")
    _write(target / "same.txt", "old")

    summary = DirectoryCopyExcludingSuffixes(source, target, [".wav"]).copy()

    assert summary.copied_files == 1
    assert summary.skipped_files == 0
    assert (target / "same.txt").read_text(encoding="utf-8") == "new"


def test_raises_when_source_directory_does_not_exist(tmp_path: Path) -> None:
    source = tmp_path / "missing"
    target = tmp_path / "target"

    with pytest.raises(ValueError, match="SOURCE_DIR does not exist"):
        DirectoryCopyExcludingSuffixes(source, target, [".wav"]).copy()


def test_raises_when_source_path_is_not_directory(tmp_path: Path) -> None:
    source = tmp_path / "file.txt"
    target = tmp_path / "target"
    _write(source, "not a directory")

    with pytest.raises(ValueError, match="SOURCE_DIR is not a directory"):
        DirectoryCopyExcludingSuffixes(source, target, [".wav"]).copy()


def test_raises_when_no_excluded_suffixes_are_provided(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"

    with pytest.raises(ValueError, match="At least one excluded suffix"):
        DirectoryCopyExcludingSuffixes(source, target, []).copy()
