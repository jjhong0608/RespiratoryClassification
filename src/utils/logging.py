import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(funcName)s - %(message)s")


def _build_rich_handler(width: int | None) -> RichHandler:
    console = Console(width=width) if width is not None else None
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=True,
        omit_repeated_times=False,
    )
    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(logging.DEBUG)
    return rich_handler


handler = _build_rich_handler(None)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG)
logging.root.handlers.clear()

_file_handler: logging.Handler | None = None
_configured_loggers: list[logging.Logger] = [logger]
_terminal_width: int | None = None


@dataclass(frozen=True)
class LoggerConfig:
    name: str = __name__
    level: int = logging.DEBUG


def _attach_handlers(target: logging.Logger) -> None:
    target.handlers.clear()
    target.addHandler(handler)
    if _file_handler is not None:
        target.addHandler(_file_handler)
    target.propagate = False
    target.setLevel(logging.DEBUG)
    if target not in _configured_loggers:
        _configured_loggers.append(target)


def configure_terminal_width(width: int | None) -> None:
    """Rebuild Rich terminal handlers with a fixed width or auto-width."""
    global handler, _terminal_width

    _terminal_width = width
    handler = _build_rich_handler(width)
    for target in list(_configured_loggers):
        _attach_handlers(target)


def current_terminal_width() -> int | None:
    return _terminal_width


def enable_file_logging(log_path: str | Path, *, mode: str = "a") -> Path:
    global _file_handler

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _file_handler is not None:
        with suppress(Exception):
            logger.removeHandler(_file_handler)
        with suppress(Exception):
            _file_handler.close()
        _file_handler = None

    file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(funcName)s - %(message)s")
    )
    logger.addHandler(file_handler)
    _file_handler = file_handler
    return path


class LoggingMixin:
    _logger: logging.Logger | None = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
            _attach_handlers(self._logger)
        return self._logger
