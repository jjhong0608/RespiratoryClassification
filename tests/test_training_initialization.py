from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
from src.training.initialization import initialize_from_checkpoint
from src.utils.config import TrainingInitializationConfig
from torch import nn


def test_initialize_from_checkpoint_noops_without_checkpoint_path() -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    summary = initialize_from_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=TrainingInitializationConfig(checkpoint_path=None),
    )

    assert summary.checkpoint_path is None
    assert summary.loaded_model_state is False
    assert summary.loaded_optimizer_state is False
    assert summary.missing_keys == ()
    assert summary.unexpected_keys == ()
    assert summary.skipped_mismatched_shapes == ()


def test_initialize_from_checkpoint_loads_matching_model_strictly(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    checkpoint_path = tmp_path / "matching.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        },
        checkpoint_path,
    )
    target = nn.Linear(2, 1)

    summary = initialize_from_checkpoint(
        model=target,
        optimizer=None,
        cfg=TrainingInitializationConfig(
            checkpoint_path=str(checkpoint_path),
            load_model_state=True,
            strict=True,
            load_optimizer_state=False,
        ),
    )

    assert summary.loaded_model_state is True
    assert summary.strict is True
    assert summary.missing_keys == ()
    assert summary.unexpected_keys == ()
    assert summary.skipped_mismatched_shapes == ()
    assert torch.equal(target.weight, model.weight)


def test_initialize_from_checkpoint_reports_non_strict_key_mismatch(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(2, 1))
    checkpoint_path = tmp_path / "mismatch.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)
    target = nn.Sequential(nn.Linear(2, 1), nn.Linear(1, 1))

    summary = initialize_from_checkpoint(
        model=target,
        optimizer=None,
        cfg=TrainingInitializationConfig(
            checkpoint_path=str(checkpoint_path),
            load_model_state=True,
            strict=False,
            load_optimizer_state=False,
        ),
    )

    assert summary.loaded_model_state is True
    assert summary.strict is False
    assert summary.missing_keys == ("1.weight", "1.bias")
    assert summary.unexpected_keys == ()
    assert summary.skipped_mismatched_shapes == ()


def test_initialize_from_checkpoint_raises_on_shape_mismatch_by_default(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
    checkpoint_path = tmp_path / "shape_mismatch.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)
    target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

    with pytest.raises(RuntimeError, match="size mismatch"):
        initialize_from_checkpoint(
            model=target,
            optimizer=None,
            cfg=TrainingInitializationConfig(
                checkpoint_path=str(checkpoint_path),
                load_model_state=True,
                strict=False,
                load_optimizer_state=False,
            ),
        )


def test_initialize_from_checkpoint_skips_mismatched_shapes_when_enabled(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
    checkpoint_path = tmp_path / "filtered.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)
    target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    original_mismatched_weight = target[1].weight.detach().clone()
    original_mismatched_bias = target[1].bias.detach().clone()

    summary = initialize_from_checkpoint(
        model=target,
        optimizer=None,
        cfg=TrainingInitializationConfig(
            checkpoint_path=str(checkpoint_path),
            load_model_state=True,
            strict=False,
            load_optimizer_state=False,
            skip_mismatched_shapes=True,
        ),
    )

    assert summary.loaded_model_state is True
    assert summary.missing_keys == ()
    assert summary.unexpected_keys == ()
    assert torch.equal(target[0].weight, source[0].weight)
    assert torch.equal(target[0].bias, source[0].bias)
    assert torch.equal(target[1].weight, original_mismatched_weight)
    assert torch.equal(target[1].bias, original_mismatched_bias)
    assert [item.key for item in summary.skipped_mismatched_shapes] == [
        "1.weight",
        "1.bias",
    ]
    assert summary.skipped_mismatched_shapes[0].checkpoint_shape == (3, 2)
    assert summary.skipped_mismatched_shapes[0].model_shape == (1, 2)


def test_initialize_from_checkpoint_does_not_load_optimizer_when_disabled(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "no_optimizer.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    with mock.patch.object(
        optimizer,
        "load_state_dict",
        wraps=optimizer.load_state_dict,
    ) as wrapped:
        summary = initialize_from_checkpoint(
            model=model,
            optimizer=optimizer,
            cfg=TrainingInitializationConfig(
                checkpoint_path=str(checkpoint_path),
                load_model_state=True,
                strict=True,
                load_optimizer_state=False,
            ),
        )

    assert summary.loaded_optimizer_state is False
    assert summary.skipped_mismatched_shapes == ()
    wrapped.assert_not_called()


def test_initialize_from_checkpoint_requires_optimizer_when_loading_optimizer_state(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    checkpoint_path = tmp_path / "optimizer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="optimizer is None"):
        initialize_from_checkpoint(
            model=model,
            optimizer=None,
            cfg=TrainingInitializationConfig(
                checkpoint_path=str(checkpoint_path),
                load_model_state=False,
                strict=False,
                load_optimizer_state=True,
            ),
        )


def test_initialize_from_checkpoint_rejects_optimizer_with_shape_filter(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "optimizer_filtered.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="skip_mismatched_shapes"):
        initialize_from_checkpoint(
            model=model,
            optimizer=optimizer,
            cfg=TrainingInitializationConfig(
                checkpoint_path=str(checkpoint_path),
                load_model_state=True,
                strict=False,
                load_optimizer_state=True,
                skip_mismatched_shapes=True,
            ),
        )
