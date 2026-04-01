from __future__ import annotations

from pathlib import Path

from src.training.trainer import CheckpointManager
from src.utils.logging import enable_file_logging


def test_checkpoint_manager_logs_saves(tmp_path: Path) -> None:
    log_path = enable_file_logging(tmp_path / "run.log")
    ckpt = CheckpointManager(tmp_path, top_k=1)
    ckpt.save_last({"x": 1})
    ckpt.maybe_save_best(0.5, {"x": 2})
    ckpt.maybe_save_best(0.4, {"x": 3})

    text = log_path.read_text(encoding="utf-8")
    assert "Saved last checkpoint" in text
    assert "Saved best checkpoint" in text
    assert "Removed checkpoint" in text
