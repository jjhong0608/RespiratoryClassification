import logging
from contextlib import suppress
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

_file_handler: logging.Handler | None = None


@dataclass(frozen=True)
class LoggerConfig:
    name: str = __name__
    level: int = logging.DEBUG


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
            self._logger.handlers.clear()
            self._logger.addHandler(handler)
            if _file_handler is not None:
                self._logger.addHandler(_file_handler)
            self._logger.propagate = False
            self._logger.setLevel(logging.DEBUG)
        return self._logger
