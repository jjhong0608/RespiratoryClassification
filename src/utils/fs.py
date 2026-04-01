from __future__ import annotations

import shutil
from pathlib import Path


class Fs:
    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def copy_file(src: str | Path, dst_dir: str | Path) -> Path:
        src_p = Path(src)
        dst_dir_p = Fs.ensure_dir(dst_dir)
        dst_p = dst_dir_p / src_p.name
        shutil.copy2(src_p, dst_p)
        return dst_p
