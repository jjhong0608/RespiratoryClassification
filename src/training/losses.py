from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss with BCE-style positive-class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is None:
            self.register_buffer("_pos_weight", None, persistent=False)
        else:
            self.register_buffer(
                "_pos_weight",
                torch.as_tensor(pos_weight, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
        loss = torch.pow(1.0 - p_t, self.gamma) * bce_loss

        if self._pos_weight is not None:
            pos_weight = cast(torch.Tensor, self._pos_weight).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            sample_weight = torch.where(targets > 0.5, pos_weight, 1.0)
            loss = loss * sample_weight

        return loss.mean()
