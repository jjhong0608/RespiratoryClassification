from __future__ import annotations

from pathlib import Path

import torch
from src.models.model import WhisperClassifierConfig, WhisperEncoderClassifier
from src.models.whisper_encoder import WhisperEncoderDims
from src.training.trainer import Trainer, TrainerConfig
from src.utils.logging import enable_file_logging
from torch.utils.data import DataLoader, TensorDataset


def test_trainer_logs_lr_per_epoch(tmp_path: Path) -> None:
    enable_file_logging(tmp_path / "run.log")

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
    y = torch.zeros((4,), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    trainer = Trainer(
        TrainerConfig(
            device="cpu",
            epochs=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.5,
            max_grad_norm=1.0,
            top_k=1,
            num_classes=2,
            run_dir=tmp_path,
        )
    )
    trainer.fit(model, loader, loader)

    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "LR:" in text
