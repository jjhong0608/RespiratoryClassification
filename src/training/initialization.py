from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.checkpoint import torch_load_compat
from src.utils.config import TrainingInitializationConfig


@dataclass(frozen=True)
class InitializationSummary:
    checkpoint_path: str | None
    loaded_model_state: bool
    loaded_optimizer_state: bool
    strict: bool
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


def initialize_from_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainingInitializationConfig,
    map_location: str | torch.device = "cpu",
) -> InitializationSummary:
    if cfg.checkpoint_path is None:
        return InitializationSummary(
            checkpoint_path=None,
            loaded_model_state=False,
            loaded_optimizer_state=False,
            strict=cfg.strict,
        )

    device = (
        map_location
        if isinstance(map_location, torch.device)
        else torch.device(map_location)
    )
    checkpoint = torch_load_compat(
        cfg.checkpoint_path,
        device=device,
        weights_only=True,
    )

    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    if cfg.load_model_state:
        if "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint is missing model_state_dict")
        result = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=cfg.strict,
        )
        if isinstance(result, tuple):
            missing_keys = tuple(str(key) for key in result[0])
            unexpected_keys = tuple(str(key) for key in result[1])
        else:
            missing_keys = tuple(result.missing_keys)
            unexpected_keys = tuple(result.unexpected_keys)

    if cfg.load_optimizer_state:
        if optimizer is None:
            raise ValueError("load_optimizer_state=true but optimizer is None")
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError("checkpoint is missing optimizer_state_dict")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return InitializationSummary(
        checkpoint_path=cfg.checkpoint_path,
        loaded_model_state=cfg.load_model_state,
        loaded_optimizer_state=cfg.load_optimizer_state,
        strict=cfg.strict,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
