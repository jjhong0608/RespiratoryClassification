from __future__ import annotations

import numpy as np
import pytest
import torch
from src.evaluation.thresholds import ThresholdOptimizationConfig, ThresholdOptimizer
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import WhisperEncoderDims
from src.training.trainer import Trainer, TrainerConfig
from src.utils.logging import enable_file_logging
from torch.utils.data import DataLoader, TensorDataset


def test_threshold_optimizer_selects_expected_thresholds() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)

    f1_result = ThresholdOptimizer(y_true, y_score).optimize("f1")
    balanced_result = ThresholdOptimizer(y_true, y_score).optimize("balanced_accuracy")
    youden_result = ThresholdOptimizer(y_true, y_score).optimize("youden_j")

    assert f1_result.applied is True
    assert f1_result.selected_threshold == pytest.approx(0.35)
    assert f1_result.selected_score == pytest.approx(0.8)

    assert balanced_result.applied is True
    assert balanced_result.selected_threshold == pytest.approx(0.35)
    assert balanced_result.selected_score == pytest.approx(0.75)

    assert youden_result.applied is True
    assert youden_result.selected_threshold == pytest.approx(0.35)
    assert youden_result.selected_score == pytest.approx(0.5)


def test_threshold_optimizer_falls_back_when_validation_has_one_class() -> None:
    y_true = np.array([1, 1, 1], dtype=int)
    y_score = np.array([0.4, 0.7, 0.9], dtype=float)

    result = ThresholdOptimizer(y_true, y_score).optimize("f1")

    assert result.applied is False
    assert result.selected_threshold == pytest.approx(0.5)
    assert result.reason is not None


def test_trainer_saves_threshold_optimization_result(tmp_path) -> None:
    log_path = enable_file_logging(tmp_path / "run.log", mode="w")

    dims = WhisperEncoderDims(
        n_mels=80,
        n_audio_ctx=2,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
    )
    model = WhisperEncoderClassifier(
        WhisperClassifierConfig(encoder=dims, num_classes=2, pooling="mean")
    )

    x = torch.zeros((4, 80, 4), dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    trainer = Trainer(
        TrainerConfig(
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            top_k=1,
            num_classes=2,
            run_dir=tmp_path,
            threshold_optimization=ThresholdOptimizationConfig(
                enabled=True,
                metric="f1",
            ),
        )
    )
    trainer.fit(model, loader, loader)

    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["threshold_optimization_cfg"]["enabled"] is True
    result = checkpoint["threshold_optimization_result"]
    assert result["selected_metric"] == "f1"
    assert 0.0 <= result["selected_threshold"] <= 1.0

    text = log_path.read_text(encoding="utf-8")
    assert "Val Threshold[f1]:" in text
