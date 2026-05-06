from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from src.models.model import MultiScaleRdtAstModel
from src.utils.checkpoint import torch_load_compat


@dataclass(frozen=True)
class TransferLoadSummary:
    checkpoint_path: str
    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class FreezeSummary:
    frozen_parameter_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class TransferOptimizerSummary:
    param_group_count: int
    frozen_encoder_params: int
    body_params: int
    head_params: int


def _extract_checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state = checkpoint.get(
        "base_model_state_dict", checkpoint.get("model_state_dict")
    )
    if not isinstance(raw_state, dict):
        raise KeyError(
            "checkpoint is missing model_state_dict or base_model_state_dict"
        )
    result: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if isinstance(value, torch.Tensor):
            result[str(key)] = value
    return result


def _normalize_checkpoint_key(key: str) -> str:
    if key.startswith("base_model."):
        return key.removeprefix("base_model.")
    return key


def load_compatible_model_state(
    *,
    model: nn.Module,
    checkpoint_path: str,
    reset_classifier: bool,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> TransferLoadSummary:
    checkpoint = torch_load_compat(
        checkpoint_path,
        device=map_location
        if isinstance(map_location, torch.device)
        else torch.device(map_location),
        weights_only=True,
    )
    raw_state = _extract_checkpoint_state(checkpoint)
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for raw_key, value in raw_state.items():
        key = _normalize_checkpoint_key(raw_key)
        if key.startswith("decoders."):
            skipped.append(raw_key)
            continue
        if reset_classifier and key.startswith("classifier."):
            skipped.append(raw_key)
            continue
        if key not in model_state:
            skipped.append(raw_key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append(raw_key)
            continue
        filtered_state[key] = value
    result = model.load_state_dict(filtered_state, strict=strict)
    return TransferLoadSummary(
        checkpoint_path=checkpoint_path,
        loaded_keys=tuple(sorted(filtered_state.keys())),
        skipped_keys=tuple(sorted(skipped)),
        missing_keys=tuple(str(key) for key in result.missing_keys),
        unexpected_keys=tuple(str(key) for key in result.unexpected_keys),
    )


def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def apply_cnuh_transfer_freeze(
    model: MultiScaleRdtAstModel,
    *,
    freeze_encoder: bool,
    freeze_modules: tuple[str, ...],
) -> FreezeSummary:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if freeze_encoder:
        for module_name in freeze_modules:
            if module_name == "patch_tokenizers":
                _set_module_requires_grad(model.encoder.patch_tokenizers, False)
            elif module_name == "position_embeddings":
                _set_module_requires_grad(model.encoder.position_embeddings, False)
            elif module_name == "scale_embeddings":
                model.encoder.scale_embeddings.requires_grad = False
            elif module_name == "shared_stem":
                _set_module_requires_grad(model.encoder.shared_stem, False)
            elif module_name == "scale_specific_adapters":
                _set_module_requires_grad(model.encoder.adapters, False)
            elif module_name == "frequency_attention_poolers":
                _set_module_requires_grad(model.encoder.frequency_poolers, False)
            else:
                raise ValueError(f"Unsupported transfer freeze module: {module_name}")
    frozen: list[str] = []
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
    return FreezeSummary(
        frozen_parameter_names=tuple(frozen),
        trainable_parameter_names=tuple(trainable),
    )


def _module_params(modules: tuple[nn.Module, ...]) -> list[nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def build_fsd50k_multilabel_optimizer(
    model: MultiScaleRdtAstModel,
    *,
    encoder_lr: float,
    body_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[AdamW, TransferOptimizerSummary]:
    for parameter in model.branch_binary_head.parameters():
        parameter.requires_grad = False
    encoder_params = _module_params((model.encoder,))
    body_modules: list[nn.Module] = [
        model.branch_mil_heads,
        model.evidence_pooler,
        model.fusion_projector,
    ]
    if model.rdt_block is not None:
        body_modules.append(model.rdt_block)
    body_params = _module_params(tuple(body_modules))
    head_params = _module_params((model.classifier,))
    param_groups = [
        {
            "params": encoder_params,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder",
        },
        {
            "params": body_params,
            "lr": body_lr,
            "weight_decay": weight_decay,
            "name": "body",
        },
        {
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head",
        },
    ]
    return (
        AdamW(param_groups),
        TransferOptimizerSummary(
            param_group_count=len(param_groups),
            frozen_encoder_params=0,
            body_params=sum(parameter.numel() for parameter in body_params),
            head_params=sum(parameter.numel() for parameter in head_params),
        ),
    )


def build_cnuh_transfer_optimizer(
    model: MultiScaleRdtAstModel,
    *,
    frozen_encoder_lr: float,
    body_lr: float,
    head_lr: float,
    weight_decay: float = 0.01,
) -> tuple[AdamW, TransferOptimizerSummary]:
    encoder_params = _module_params((model.encoder,))
    body_modules: list[nn.Module] = [
        model.branch_mil_heads,
        model.branch_binary_head,
        model.evidence_pooler,
        model.fusion_projector,
    ]
    if model.rdt_block is not None:
        body_modules.append(model.rdt_block)
    body_params = _module_params(tuple(body_modules))
    head_params = _module_params((model.classifier,))
    param_groups = []
    if encoder_params and frozen_encoder_lr > 0:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": frozen_encoder_lr,
                "weight_decay": weight_decay,
                "name": "frozen_encoder",
            }
        )
    if body_params:
        param_groups.append(
            {
                "params": body_params,
                "lr": body_lr,
                "weight_decay": weight_decay,
                "name": "body",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        )
    if not param_groups:
        raise ValueError(
            "No trainable parameters available for CNUH transfer optimizer"
        )
    return (
        AdamW(param_groups),
        TransferOptimizerSummary(
            param_group_count=len(param_groups),
            frozen_encoder_params=sum(
                parameter.numel() for parameter in encoder_params
            ),
            body_params=sum(parameter.numel() for parameter in body_params),
            head_params=sum(parameter.numel() for parameter in head_params),
        ),
    )


def checkpoint_exists(path: str) -> bool:
    return Path(path).exists()
