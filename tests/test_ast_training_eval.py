from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from src.cli.cv import main as cv_main
from src.cli.evaluate import evaluate_checkpoint
from src.data.loaders import build_bag_loader, build_dataset
from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
)
from src.training.trainer import Trainer, TrainerConfig
from src.utils.checkpoint import load_checkpoint
from src.utils.config import (
    AstArchitectureConfig,
    AstEncoderConfig,
    AstFbankConfig,
    AudioConfig,
    BandPassConfig,
    DataConfig,
    DataLoaderConfig,
    DataSplitsConfig,
    DiagnosticsConfig,
    EarlyStoppingConfig,
    EvalSectionConfig,
    EvaluationRunConfig,
    ExperimentConfig,
    FeatureConfig,
    GatedAttentionConfig,
    InstanceConfig,
    InstanceHeadConfig,
    LoggingConfig,
    MetadataConfig,
    MilConfig,
    ModelConfig,
    SplitConfig,
    ThresholdOptimizationConfig,
)


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 320.0 * t)
    sf.write(path, tone.astype(np.float32), sample_rate)


def _logging_cfg() -> LoggingConfig:
    return LoggingConfig(
        diagnostics=DiagnosticsConfig(
            save_bag_logits=True,
            save_bag_probabilities=True,
            save_bag_embedding=False,
            save_instance_logits=True,
            save_instance_probabilities=True,
            save_instance_embeddings=False,
            save_attention_weights=True,
            save_instance_metadata=True,
            top_k_instances=2,
        )
    )


def _model_cfg(*, mil_type: str = "gated_attention") -> ModelConfig:
    return ModelConfig(
        encoder=AstEncoderConfig(
            pretrained_name_or_path=None,
            pooling="cls",
            adaptation=replace(
                AstEncoderConfig().adaptation, mode="partial", num_layers=1
            ),
            architecture=AstArchitectureConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
            ),
        ),
        instance_head=InstanceHeadConfig(
            projection_dim=24,
            dropout=0.1,
            normalize=True,
        ),
        mil=MilConfig(
            type=mil_type,  # type: ignore[arg-type]
            gated_attention=GatedAttentionConfig(attention_dim=16, dropout=0.1),
        ),
        classifier=replace(
            ModelConfig(encoder=AstEncoderConfig()).classifier,
            type="linear",
            hidden_dim=24,
            dropout=0.0,
        ),
    )


def _binary_data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        metadata=MetadataConfig(
            label_to_index={"normal": 0, "wheeze": 1},
            splits=DataSplitsConfig(
                train=SplitConfig(roots=[str(root / "train")]),
                val=SplitConfig(roots=[str(root / "val")]),
                eval=SplitConfig(roots=[str(root / "val")]),
            ),
        ),
        audio=AudioConfig(sample_rate=16000),
        instance=InstanceConfig(window_sec=2.0, hop_sec=1.0, tail_policy="cover_end"),
        features=FeatureConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=32,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
        loader=DataLoaderConfig(num_workers=0),
    )


def _multiclass_data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        metadata=MetadataConfig(
            label_to_index={"normal": 0, "wheeze": 1, "crackle": 2},
            splits=DataSplitsConfig(
                train=SplitConfig(roots=[str(root / "train")]),
                val=SplitConfig(roots=[str(root / "val")]),
                eval=SplitConfig(roots=[str(root / "val")]),
            ),
        ),
        audio=AudioConfig(sample_rate=16000),
        instance=InstanceConfig(window_sec=2.0, hop_sec=1.0, tail_policy="cover_end"),
        features=FeatureConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=32,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
        loader=DataLoaderConfig(num_workers=0),
    )


def _prepare_binary_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 1.1, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 3.1, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 1.3, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 3.4, 16000)
    return _binary_data_cfg(root)


def _prepare_multiclass_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze", "crackle"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 1.0, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 3.2, 16000)
    _write_wav(root / "train" / "crackle" / "crackle_a.wav", 2.7, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 1.1, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 3.3, 16000)
    _write_wav(root / "val" / "crackle" / "crackle_b.wav", 2.9, 16000)
    return _multiclass_data_cfg(root)


def _train_smoke_run(
    tmp_path: Path,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    loss_type: str,
    pos_weight: float | None = None,
) -> tuple[Path, DataConfig]:
    torch.manual_seed(0)
    train_dataset = build_dataset(data_cfg, split="train")
    val_dataset = build_dataset(data_cfg, split="val")
    train_loader = build_bag_loader(
        train_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )
    val_loader = build_bag_loader(
        val_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    model = build_ast_model(
        model_cfg,
        num_mel_bins=train_dataset.num_mel_bins,
        max_length=train_dataset.max_length,
        num_classes=len(data_cfg.label_to_index),
    )
    adaptation_summary = apply_encoder_adaptation(
        model,
        model.cfg.encoder.adaptation,
    )
    optimizer, optimizer_summary = build_grouped_optimizer(
        model,
        encoder_lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.0,
    )

    run_dir = tmp_path / "run"
    trainer = Trainer(
        TrainerConfig(
            device="cpu",
            epochs=1,
            encoder_lr=1e-4,
            head_lr=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            top_k=1,
            run_dir=run_dir,
            num_classes=len(data_cfg.label_to_index),
            loss_type=loss_type,
            gamma=2.0,
            pos_weight=pos_weight,
            logging=_logging_cfg(),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                monitor="val_loss",
                patience=5,
                min_delta=1e-4,
            ),
        )
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        extra_state={
            "model_cfg": model.cfg,
            "label_to_index": dict(data_cfg.label_to_index),
            "adaptation_summary": adaptation_summary,
            "optimizer_summary": optimizer_summary,
        },
    )
    return run_dir / "last.pt", data_cfg


def _eval_cfg(
    tmp_path: Path,
    checkpoint_path: Path,
    data_cfg: DataConfig,
) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        experiment=ExperimentConfig(
            name="eval",
            task="respiratory_classification",
            mode="recording_mil",
            seed=0,
            device="cpu",
            output_dir=str(tmp_path / "eval"),
        ),
        data=data_cfg,
        eval=EvalSectionConfig(
            batch_size=2,
            checkpoint_path=str(checkpoint_path),
            threshold_optimization=ThresholdOptimizationConfig(
                enabled=True,
                metric="f1",
            ),
        ),
        logging=_logging_cfg(),
    )


def test_ast_mil_binary_trainer_and_evaluator_smoke(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "bags")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(mil_type="gated_attention"),
        loss_type="focal",
        pos_weight=2.0,
    )
    run_dir = checkpoint_path.parent

    assert checkpoint_path.exists()
    assert sorted(run_dir.glob("best_loss_*.pt"))
    assert sorted(run_dir.glob("best_f1_*.pt"))
    assert (run_dir / "diagnostics" / "val_epoch_001.jsonl").exists()

    checkpoint = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    assert checkpoint["val_threshold_optimization"]["threshold_source"] == (
        "validation_optimization"
    )

    metrics, rows, diagnostics = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )

    assert metrics["threshold_optimization"]["applied"] is True
    assert metrics["threshold_optimization"]["threshold_source"] == (
        "checkpoint_validation"
    )
    assert len(rows) == 2
    assert rows[0].class_probabilities is None
    assert len(diagnostics) == 2
    assert "bag_logits" in diagnostics[0]
    assert "instance_probabilities" in diagnostics[0]
    assert "attention_weights" in diagnostics[0]
    assert "top_k_instances" in diagnostics[0]


def test_ast_mil_multiclass_trainer_and_evaluator_smoke(tmp_path: Path) -> None:
    data_cfg = _prepare_multiclass_dataset(tmp_path / "multiclass")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(mil_type="linear_softmax"),
        loss_type="cross_entropy",
    )

    metrics, rows, diagnostics = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )

    assert metrics["accuracy"] >= 0.0
    assert metrics["threshold_optimization"]["enabled"] is False
    assert "binary classification" in metrics["threshold_optimization"]["reason"]
    assert rows[0].class_probabilities is not None
    assert len(rows[0].class_probabilities or ()) == 3
    assert diagnostics[0]["instance_probabilities"]
    assert "top_k_instances" in diagnostics[0]


def test_evaluate_checkpoint_rejects_frontend_dim_mismatch(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "frontend")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(mil_type="gated_attention"),
        loss_type="focal",
        pos_weight=2.0,
    )
    mismatched_cfg = replace(
        data_cfg,
        features=replace(
            data_cfg.features,
            ast_fbank=replace(
                data_cfg.features.ast_fbank,
                max_length=data_cfg.features.ast_fbank.max_length + 8,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Evaluation frontend dims do not match the checkpoint encoder dims",
    ):
        evaluate_checkpoint(
            _eval_cfg(tmp_path, checkpoint_path, mismatched_cfg),
            checkpoint_path,
        )


def test_cv_cli_smoke_writes_fold_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "cv_data"
    for fold_name, label in [("fold_0", "normal"), ("fold_1", "wheeze")]:
        label_dir = dataset_root / fold_name / label
        label_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(label_dir / f"{fold_name}.wav", 3.0, 16000)

    config = {
        "experiment": {
            "name": "cv_smoke",
            "task": "normal_vs_wheeze",
            "mode": "recording_mil",
            "seed": 0,
            "device": "cpu",
            "output_dir": str(tmp_path / "outputs"),
        },
        "data": {
            "metadata": {
                "label_to_index": {"normal": 0, "wheeze": 1},
                "splits": {
                    "train": {"roots": []},
                    "val": {"roots": []},
                    "eval": {"roots": []},
                },
            },
            "audio": {"sample_rate": 16000},
            "instance": {
                "window_sec": 2.0,
                "hop_sec": 1.0,
                "tail_policy": "cover_end",
            },
            "features": {
                "source_type": "original",
                "bandpass": {"enabled": False},
                "ast_fbank": {
                    "num_mel_bins": 32,
                    "max_length": 32,
                    "do_normalize": True,
                    "mean": -4.2677393,
                    "std": 4.5689974,
                },
            },
            "loader": {"num_workers": 0},
        },
        "model": {
            "encoder": {
                "type": "ast",
                "pretrained_name_or_path": None,
                "pooling": "cls",
                "adaptation": {"mode": "partial", "num_layers": 1},
                "architecture": {
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 64,
                },
            },
            "instance_head": {
                "projection_dim": 24,
                "dropout": 0.1,
                "normalize": True,
            },
            "mil": {
                "type": "gated_attention",
                "gated_attention": {"attention_dim": 16, "dropout": 0.1},
            },
            "classifier": {
                "type": "linear",
                "hidden_dim": 24,
                "dropout": 0.0,
            },
        },
        "train": {
            "batch_size": 1,
            "epochs": 1,
            "top_k": 1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "encoder_lr": 1e-4,
                "head_lr": 1e-3,
                "weight_decay": 0.0,
            },
            "scheduler": {"warmup_ratio": 0.0},
            "loss": {
                "type": "focal",
                "auto_pos_weight": False,
                "pos_weight": 1.0,
                "gamma": 2.0,
            },
            "sampler": {"weighted_random": False},
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 3,
                "min_delta": 1e-4,
            },
        },
        "eval": {
            "batch_size": 1,
            "threshold_optimization": {"enabled": True, "metric": "f1"},
        },
        "logging": {
            "diagnostics": {
                "save_bag_logits": True,
                "save_bag_probabilities": True,
                "save_bag_embedding": False,
                "save_instance_logits": True,
                "save_instance_probabilities": True,
                "save_instance_embeddings": False,
                "save_attention_weights": True,
                "save_instance_metadata": True,
                "top_k_instances": 2,
            }
        },
        "cv": {
            "folds": [
                {
                    "name": "fold_0",
                    "train": {"roots": [str(dataset_root / "fold_1")]},
                    "val": {"roots": [str(dataset_root / "fold_0")]},
                }
            ]
        },
    }
    config_path = tmp_path / "cv.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["cv", "--config", str(config_path)])
    cv_main()

    assert (tmp_path / "outputs" / "cv_smoke" / "fold_0" / "last.pt").exists()
