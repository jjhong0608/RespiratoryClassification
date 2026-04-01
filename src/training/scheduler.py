from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps >= total_steps:
            warmup_steps = max(0, total_steps - 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        super().__init__(optimizer, lr_lambda=lr_lambda)
