from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tqdm import tqdm


def tqdm_kwargs(width: int | None) -> dict[str, int]:
    if width is None:
        return {}
    return {"ncols": width}


def iter_progress[T](
    iterable: Iterable[T],
    *,
    terminal_width: int | None,
    **kwargs: Any,
) -> Iterable[T]:
    return tqdm(iterable, **tqdm_kwargs(terminal_width), **kwargs)
