from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from src.cli.evaluate import evaluate_checkpoint
from src.data.loaders import build_bag_loader, build_dataset
from src.models.model import MILModelConfig, RespiratoryMILModel
from src.models.whisper_encoder import WhisperEncoderDims
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import (
    AnalysisConfig,
    AnalysisOutputConfig,
    AudioConfig,
    BandPassConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    PreprocessingConfig,
    SegmentationConfig,
)


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 320.0 * t)
    sf.write(path, tone.astype(np.float32), sample_rate)


def _data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.5),
        preprocessing=PreprocessingConfig(
            feature_type="log_mel",
            source_type="original",
            n_mels=80,
            bandpass=BandPassConfig(enabled=False),
        ),
        segment=SegmentationConfig(
            mode="sliding_window",
            length_sec=0.5,
            stride_sec=0.25,
            pad_last=True,
            drop_last=False,
        ),
    )


def _prepare_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 0.7, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 1.3, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 0.9, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 1.4, 16000)
    return _data_cfg(root)


def test_mil_trainer_and_evaluator_smoke(tmp_path: Path) -> None:
    torch.manual_seed(0)
    data_cfg = _prepare_dataset(tmp_path / "bags")
    train_dataset = build_dataset(data_cfg, split="train")
    val_dataset = build_dataset(data_cfg, split="val")
    train_loader = build_bag_loader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=False,
    )
    val_loader = build_bag_loader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=False,
    )

    model = RespiratoryMILModel(
        MILModelConfig(
            encoder=WhisperEncoderDims(
                n_mels=train_dataset.segment_n_mels,
                n_audio_ctx=train_dataset.segment_audio_ctx,
                n_audio_state=8,
                n_audio_head=2,
                n_audio_layer=1,
            ),
            instance_head_type="linear",
            instance_hidden_dim=8,
            instance_dropout=0.0,
            aggregator="attention",
            topk_k=1,
            attention_hidden_dim=8,
            attention_dropout=0.0,
            attention_gated=True,
            logsumexp_temperature=1.0,
            softmax_weighted_temperature=1.0,
            noisy_or_clamp_eps=1e-6,
        )
    )
    run_dir = tmp_path / "run"
    trainer = Trainer(
        TrainerConfig(
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            top_k=1,
            run_dir=run_dir,
            pos_weight=None,
            analysis=AnalysisConfig(
                outputs=AnalysisOutputConfig(
                    save_segment_scores=True,
                    save_attention_weights=True,
                    save_topk_indices=False,
                    save_bag_metadata=True,
                )
            ),
        )
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        extra_state={
            "model_cfg": asdict(model.cfg),
            "label_to_index": dict(data_cfg.label_to_index),
        },
    )

    checkpoint_path = run_dir / "last.pt"
    assert checkpoint_path.exists()
    assert (run_dir / "diagnostics" / "val_epoch_001.jsonl").exists()

    eval_cfg = EvalConfig(
        experiment=ExperimentConfig(
            name="eval",
            task="normal_vs_wheeze",
            mode="mil",
            seed=0,
            device="cpu",
            output_dir=str(tmp_path / "eval"),
        ),
        checkpoint_path=str(checkpoint_path),
        data=data_cfg,
        analysis=AnalysisConfig(
            outputs=AnalysisOutputConfig(
                save_segment_scores=True,
                save_attention_weights=True,
                save_topk_indices=False,
                save_bag_metadata=True,
            )
        ),
    )

    metrics, rows, diagnostics = evaluate_checkpoint(
        eval_cfg,
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )

    assert "accuracy" in metrics
    assert len(rows) == 2
    assert len(diagnostics) == 2
