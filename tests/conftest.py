from __future__ import annotations

import sys
from pathlib import Path

from src.models.model import PatchBranchConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def small_patch_branches() -> tuple[PatchBranchConfig, ...]:
    return (
        PatchBranchConfig(patch_size=(8, 8), stride=(4, 8)),
        PatchBranchConfig(patch_size=(4, 16), stride=(2, 16)),
        PatchBranchConfig(patch_size=(2, 32), stride=(1, 32)),
        PatchBranchConfig(patch_size=(16, 4), stride=(8, 4)),
    )
