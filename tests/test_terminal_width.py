from __future__ import annotations

import logging
from pathlib import Path

import src.utils.logging as logging_utils
from src.utils.logging import LoggingMixin, configure_terminal_width
from src.utils.progress import tqdm_kwargs


class _TerminalWidthLogger(LoggingMixin):
    pass


def test_tqdm_kwargs_respects_optional_width() -> None:
    assert tqdm_kwargs(None) == {}
    assert tqdm_kwargs(120) == {"ncols": 120}


def test_configure_terminal_width_updates_existing_loggers() -> None:
    mixin_logger = _TerminalWidthLogger().logger
    try:
        configure_terminal_width(120)

        assert logging_utils.current_terminal_width() == 120
        assert logging_utils.handler.console.width == 120
        assert logging_utils.logger.handlers[0] is logging_utils.handler
        assert mixin_logger.handlers[0] is logging_utils.handler

        configure_terminal_width(None)
        assert logging_utils.current_terminal_width() is None
        assert logging_utils.logger.handlers[0] is logging_utils.handler
        assert mixin_logger.handlers[0] is logging_utils.handler
    finally:
        configure_terminal_width(None)


def test_configure_terminal_width_preserves_file_handler(tmp_path: Path) -> None:
    log_path = logging_utils.enable_file_logging(tmp_path / "run.log", mode="w")
    try:
        configure_terminal_width(100)

        file_handlers = [
            handler
            for handler in logging_utils.logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        assert file_handlers
        assert file_handlers[0].baseFilename == str(log_path)
    finally:
        configure_terminal_width(None)
