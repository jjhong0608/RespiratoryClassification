from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RespiratoryModelOutput:
    logits: Tensor
    pooled_embedding: Tensor | None = None
