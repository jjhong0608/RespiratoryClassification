from __future__ import annotations

from pathlib import Path

from src.utils.logging import LoggingMixin, enable_file_logging


class _Dummy(LoggingMixin):
    pass


def test_enable_file_logging_writes_file(tmp_path: Path) -> None:
    log_path = enable_file_logging(tmp_path / "run.log")
    obj = _Dummy()
    obj.logger.info("hello")

    text = log_path.read_text(encoding="utf-8")
    assert "hello" in text


def test_enable_file_logging_truncates_with_write_mode(tmp_path: Path) -> None:
    p = tmp_path / "run.log"
    p.write_text("old\n", encoding="utf-8")
    enable_file_logging(p, mode="w")
    obj = _Dummy()
    obj.logger.info("new")
    text = p.read_text(encoding="utf-8")
    assert "old" not in text
    assert "new" in text
