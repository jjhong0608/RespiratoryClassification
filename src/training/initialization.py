from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from src.utils.checkpoint import torch_load_compat
from src.utils.config import TrainingInitializationConfig


@dataclass(frozen=True)
class ShapeMismatchSummary:
    key: str
    checkpoint_shape: tuple[int, ...]
    model_shape: tuple[int, ...]


@dataclass(frozen=True)
class InitializationSummary:
    checkpoint_path: str | None
    loaded_model_state: bool
    loaded_optimizer_state: bool
    strict: bool
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    skipped_mismatched_shapes: tuple[ShapeMismatchSummary, ...] = ()


def _shape_tuple(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _filter_compatible_state_dict(
    checkpoint_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    tuple[str, ...],
    tuple[str, ...],
    tuple[ShapeMismatchSummary, ...],
]:
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_mismatched_shapes: list[ShapeMismatchSummary] = []
    missing_keys = tuple(key for key in model_state if key not in checkpoint_state)
    unexpected_keys = tuple(key for key in checkpoint_state if key not in model_state)

    for key, checkpoint_value in checkpoint_state.items():
        model_value = model_state.get(key)
        if model_value is None:
            continue
        checkpoint_shape = _shape_tuple(checkpoint_value)
        model_shape = _shape_tuple(model_value)
        if checkpoint_shape != model_shape:
            skipped_mismatched_shapes.append(
                ShapeMismatchSummary(
                    key=key,
                    checkpoint_shape=checkpoint_shape,
                    model_shape=model_shape,
                )
            )
            continue
        filtered_state[key] = checkpoint_value

    return (
        filtered_state,
        missing_keys,
        unexpected_keys,
        tuple(skipped_mismatched_shapes),
    )


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
    if cfg.skip_mismatched_shapes and cfg.load_optimizer_state:
        raise ValueError(
            "load_optimizer_state=true is not supported when "
            "skip_mismatched_shapes=true"
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
    skipped_mismatched_shapes: tuple[ShapeMismatchSummary, ...] = ()
    if cfg.load_model_state:
        if "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint is missing model_state_dict")
        checkpoint_state = checkpoint["model_state_dict"]
        if not isinstance(checkpoint_state, Mapping):
            raise TypeError("checkpoint model_state_dict must be a mapping")
        if cfg.skip_mismatched_shapes:
            filtered_state, missing_keys, unexpected_keys, skipped_mismatched_shapes = (
                _filter_compatible_state_dict(checkpoint_state, model.state_dict())
            )
            model.load_state_dict(filtered_state, strict=False)
        else:
            result = model.load_state_dict(
                checkpoint_state,
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
        skipped_mismatched_shapes=skipped_mismatched_shapes,
    )
