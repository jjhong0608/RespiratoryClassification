from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile as sf
import torch
from src.cli.cv import main as cv_main
from src.cli.evaluate import evaluate_checkpoint
from src.data.loaders import ClipBatch, build_clip_loader, build_dataset
from src.models.model import AstModelOutput, ClassGateEvidenceAuxiliaryConfig
from src.training.ast_setup import (
    apply_encoder_adaptation,
    build_ast_model,
    build_grouped_optimizer,
)
from src.training.trainer import (
    CheckpointManager,
    Trainer,
    TrainerConfig,
    resolve_branch_binary_aux_weight,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.config import (
    AnalysisConfig,
    AnalysisOutputConfig,
    AstFbankConfig,
    AttentionEntropyLossConfig,
    AudioConfig,
    BandPassConfig,
    BranchAuxiliaryLossConfig,
    BranchBinaryAuxiliaryLossConfig,
    BranchBinaryAuxiliaryMonitorConfig,
    BranchBinaryAuxiliaryScheduleConfig,
    BranchBinaryPosWeightConfig,
    CheckpointingConfig,
    CheckpointMonitorConfig,
    ClassGateDiversityRegularizationConfig,
    ClassifierConfig,
    DataConfig,
    EarlyStoppingConfig,
    EncoderAdaptationConfig,
    EvalConfig,
    ExperimentConfig,
    GateEntropyRegularizationConfig,
    LabelSmoothingConfig,
    ModelConfig,
    ModelEncoderConfig,
    MultiScaleRdtArchitectureConfig,
    PreprocessingConfig,
    RdtConfig,
)

from conftest import small_patch_branches


def _write_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 320.0 * t)
    sf.write(path, tone.astype(np.float32), sample_rate)


def _analysis_cfg() -> AnalysisConfig:
    return AnalysisConfig(
        outputs=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=True,
            save_clip_metadata=True,
        )
    )


def _small_architecture(
    *,
    rdt_enabled: bool,
    rdt_steps: int,
) -> MultiScaleRdtArchitectureConfig:
    return MultiScaleRdtArchitectureConfig(
        hidden_size=32,
        num_attention_heads=4,
        mlp_ratio=2.0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        shared_stem_depth=1,
        adapter_depth=1,
        patch_branches=small_patch_branches(),
        rdt=RdtConfig(
            enabled=rdt_enabled,
            steps=rdt_steps,
            top_tokens_per_branch=2,
            gated_residual=True,
            layerscale_init=0.01,
        ),
    )


def _model_cfg(
    *,
    classifier_type: Literal["linear", "mlp"] = "linear",
    rdt_enabled: bool = True,
    rdt_steps: int = 3,
) -> ModelConfig:
    return ModelConfig(
        encoder=ModelEncoderConfig(
            adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
            architecture=_small_architecture(
                rdt_enabled=rdt_enabled,
                rdt_steps=rdt_steps,
            ),
        ),
        classifier=ClassifierConfig(
            type=classifier_type,
            hidden_dim=32,
            dropout=0.0,
            pooling="latent_mean",
        ),
    )


def _binary_data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.5),
        preprocessing=PreprocessingConfig(
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
    )


def _multiclass_data_cfg(root: Path) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1, "crackle": 2},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.5),
        preprocessing=PreprocessingConfig(
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
    )


def _prepare_binary_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 0.7, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 1.3, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 0.9, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 1.4, 16000)
    return _binary_data_cfg(root)


def _prepare_multiclass_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze", "crackle"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 0.7, 16000)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 1.3, 16000)
    _write_wav(root / "train" / "crackle" / "crackle_a.wav", 1.1, 16000)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 0.9, 16000)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 1.4, 16000)
    _write_wav(root / "val" / "crackle" / "crackle_b.wav", 1.2, 16000)
    return _multiclass_data_cfg(root)


def _trainer_cfg(
    *,
    num_classes: int,
    run_dir: Path,
    loss_type: str,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    branch_auxiliary_enabled: bool = False,
    branch_auxiliary_weight: float = 0.3,
    branch_auxiliary_weights: tuple[float, ...] | None = None,
    branch_binary_auxiliary_enabled: bool = False,
    branch_binary_auxiliary_weight: float = 0.3,
    branch_binary_auxiliary_schedule: BranchBinaryAuxiliaryScheduleConfig | None = None,
    branch_binary_auxiliary_monitor: BranchBinaryAuxiliaryMonitorConfig | None = None,
    branch_binary_pos_weight: float | None = None,
    main_index_to_binary_target: tuple[int, ...] | None = None,
    class_weights: tuple[float, ...] | None = None,
    label_smoothing: LabelSmoothingConfig | None = None,
    attention_entropy_enabled: bool = False,
    attention_entropy_weight: float = 0.0,
    gate_entropy_regularization_enabled: bool = False,
    gate_entropy_regularization_weight: float = 0.0,
    gate_entropy_regularization_target: Literal[
        "evidence_gate",
        "class_evidence_gate",
        "true_class_evidence_gate",
    ] = "evidence_gate",
    class_gate_evidence_auxiliary_enabled: bool = False,
    class_gate_evidence_auxiliary_weight: float = 0.1,
    class_gate_diversity_regularization_enabled: bool = False,
    class_gate_diversity_regularization_weight: float = 0.0,
) -> TrainerConfig:
    return TrainerConfig(
        device="cpu",
        epochs=1,
        encoder_lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        top_k=1,
        run_dir=run_dir,
        num_classes=num_classes,
        loss_type=loss_type,
        gamma=gamma,
        pos_weight=pos_weight,
        class_weights=class_weights,
        label_smoothing=label_smoothing or LabelSmoothingConfig(),
        branch_auxiliary=BranchAuxiliaryLossConfig(
            enabled=branch_auxiliary_enabled,
            weight=branch_auxiliary_weight,
            aggregation="mean",
            weights=branch_auxiliary_weights,
        ),
        branch_binary_auxiliary=BranchBinaryAuxiliaryLossConfig(
            enabled=branch_binary_auxiliary_enabled,
            weight=branch_binary_auxiliary_weight,
            label_to_index={
                "normal": 0,
                "crackle": 1,
                "wheeze": 1,
                "rhonchi": 1,
            },
            pos_weight=BranchBinaryPosWeightConfig(
                enabled=branch_binary_pos_weight is not None
            ),
            schedule=(
                branch_binary_auxiliary_schedule
                or BranchBinaryAuxiliaryScheduleConfig()
            ),
            monitor=(
                branch_binary_auxiliary_monitor or BranchBinaryAuxiliaryMonitorConfig()
            ),
            aggregation="mean",
        ),
        branch_binary_pos_weight=branch_binary_pos_weight,
        main_index_to_binary_target=main_index_to_binary_target,
        attention_entropy=AttentionEntropyLossConfig(
            enabled=attention_entropy_enabled,
            weight=attention_entropy_weight,
        ),
        gate_entropy_regularization=GateEntropyRegularizationConfig(
            enabled=gate_entropy_regularization_enabled,
            weight=gate_entropy_regularization_weight,
            target=gate_entropy_regularization_target,
        ),
        class_gate_evidence_auxiliary=ClassGateEvidenceAuxiliaryConfig(
            enabled=class_gate_evidence_auxiliary_enabled,
            weight=class_gate_evidence_auxiliary_weight,
        ),
        class_gate_diversity_regularization=ClassGateDiversityRegularizationConfig(
            enabled=class_gate_diversity_regularization_enabled,
            weight=class_gate_diversity_regularization_weight,
            target="class_evidence_gate",
            metric="js_divergence",
        ),
        analysis=_analysis_cfg(),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
        ),
    )


def _train_smoke_run(
    tmp_path: Path,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    loss_type: str,
    pos_weight: float | None = None,
    branch_auxiliary_enabled: bool = False,
) -> tuple[Path, DataConfig]:
    torch.manual_seed(0)
    train_dataset = build_dataset(data_cfg, split="train")
    val_dataset = build_dataset(data_cfg, split="val")
    train_loader = build_clip_loader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=False,
    )
    val_loader = build_clip_loader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
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
        _trainer_cfg(
            num_classes=len(data_cfg.label_to_index),
            run_dir=run_dir,
            loss_type=loss_type,
            pos_weight=pos_weight,
            branch_auxiliary_enabled=branch_auxiliary_enabled,
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
) -> EvalConfig:
    return EvalConfig(
        experiment=ExperimentConfig(
            name="eval",
            task="respiratory_classification",
            mode="clip",
            seed=0,
            device="cpu",
            output_dir=str(tmp_path / "eval"),
        ),
        checkpoint_path=str(checkpoint_path),
        data=data_cfg,
        analysis=_analysis_cfg(),
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _cv_payload(root: Path, output_dir: Path) -> dict:
    return {
        "experiment": {
            "name": "cv_smoke",
            "task": "normal_vs_wheeze",
            "mode": "clip",
            "seed": 1,
            "device": "cpu",
            "output_dir": str(output_dir),
        },
        "data": {
            "train_dirs": [],
            "val_dirs": [],
            "eval_dirs": [],
            "label_to_index": {"normal": 0, "wheeze": 1},
            "batch_size": 2,
            "num_workers": 0,
            "audio": {
                "sample_rate": 16000,
                "clip_duration_sec": 1.5,
            },
            "preprocessing": {
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
        },
        "model": {
            "encoder": {
                "type": "multiscale_rdt_ast",
                "adaptation": {"mode": "full", "num_layers": 0},
                "architecture": {
                    "hidden_size": 32,
                    "num_attention_heads": 4,
                    "mlp_ratio": 2.0,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1,
                    "layer_norm_eps": 1e-6,
                    "shared_stem_depth": 1,
                    "adapter_depth": 1,
                    "patch_branches": [
                        {"patch_size": [8, 8], "stride": [4, 8]},
                        {"patch_size": [4, 16], "stride": [2, 16]},
                        {"patch_size": [2, 32], "stride": [1, 32]},
                        {"patch_size": [16, 4], "stride": [8, 4]},
                    ],
                    "rdt": {
                        "enabled": True,
                        "steps": 3,
                        "top_tokens_per_branch": 2,
                        "gated_residual": True,
                        "layerscale_init": 0.01,
                    },
                },
            },
            "classifier": {
                "type": "linear",
                "hidden_dim": 32,
                "dropout": 0.0,
                "pooling": "latent_mean",
            },
        },
        "train": {
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
                "type": "bce",
                "auto_pos_weight": False,
                "pos_weight": None,
                "gamma": 2.0,
                "branch_auxiliary": {
                    "enabled": True,
                    "weight": 0.3,
                    "aggregation": "mean",
                },
            },
            "sampler": {"weighted_random": False},
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 5,
                "min_delta": 1e-4,
            },
        },
        "analysis": {
            "outputs": {
                "save_logits": True,
                "save_probabilities": True,
                "save_embeddings": True,
                "save_clip_metadata": True,
            },
        },
        "folds": [
            {
                "name": "fold_0",
                "train_dirs": [str(root / "fold_1")],
                "val_dirs": [str(root / "fold_0")],
            }
        ],
    }


def _prepare_cv_dataset(root: Path) -> Path:
    for fold_name in ("fold_0", "fold_1"):
        for label in ("normal", "wheeze"):
            (root / fold_name / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "fold_0" / "normal" / "normal_fold0.wav", 0.8, 16000)
    _write_wav(root / "fold_0" / "wheeze" / "wheeze_fold0.wav", 1.0, 16000)
    _write_wav(root / "fold_1" / "normal" / "normal_fold1.wav", 0.9, 16000)
    _write_wav(root / "fold_1" / "wheeze" / "wheeze_fold1.wav", 1.2, 16000)
    return root


def test_trainer_total_loss_matches_final_loss_when_branch_auxiliary_disabled() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.3]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_branch_auxiliary_disabled_allows_missing_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_total_loss_includes_branch_auxiliary_when_enabled() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=0.5,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.3]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is not None
    expected = final_loss + (0.5 * components.branch_auxiliary)
    assert torch.isclose(components.total, expected)


def test_trainer_branch_auxiliary_weights_override_scalar_weight() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=9.0,
            branch_auxiliary_weights=(0.1, 0.2, 0.3, 0.4),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.3]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    assert output.branch_logits is not None
    branch_losses = trainer._compute_branch_auxiliary_losses(
        criterion,
        output.branch_logits,
        labels,
    )
    weight_tensor = torch.tensor(
        [0.1, 0.2, 0.3, 0.4],
        dtype=branch_losses.dtype,
    )
    expected_auxiliary = (branch_losses * weight_tensor).sum() / weight_tensor.sum()

    assert components.branch_auxiliary is not None
    assert torch.isclose(components.branch_auxiliary, expected_auxiliary)
    assert torch.isclose(components.total, final_loss + expected_auxiliary)


def test_trainer_raises_when_branch_auxiliary_enabled_without_branch_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="branch auxiliary loss enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_cross_entropy_label_smoothing_matches_torch_loss() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            label_smoothing=LabelSmoothingConfig(enabled=True, value=0.2),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    logits = torch.tensor(
        [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    assert torch.isclose(criterion(logits, labels), expected(logits, labels))


def test_weighted_cross_entropy_label_smoothing_matches_torch_loss() -> None:
    class_weights = torch.tensor([0.5, 1.0, 1.25, 1.25], dtype=torch.float32)
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=tuple(float(value.item()) for value in class_weights),
            label_smoothing=LabelSmoothingConfig(enabled=True, value=0.05),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )
    logits = torch.tensor(
        [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    assert torch.isclose(criterion(logits, labels), expected(logits, labels))


def test_cross_entropy_label_smoothing_disabled_uses_zero() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            label_smoothing=LabelSmoothingConfig(enabled=False, value=0.2),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    expected = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    logits = torch.tensor(
        [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    assert torch.isclose(criterion(logits, labels), expected(logits, labels))


def test_trainer_total_loss_includes_weighted_ce_and_branch_binary_auxiliary() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_weights=(0.5, 1.0, 1.25, 1.25),
            branch_binary_auxiliary_enabled=True,
            branch_binary_auxiliary_weight=0.3,
            branch_binary_pos_weight=1.5,
            main_index_to_binary_target=(0, 1, 1, 1),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    branch_binary_criterion = trainer._branch_binary_criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.zeros(2, 4, 4),
        branch_binary_logits=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, 0.0, 0.5]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    branch_binary_loss = trainer._compute_branch_binary_auxiliary_loss(
        branch_binary_criterion,
        output,
        labels,
    )

    assert components.branch_auxiliary is None
    assert components.branch_binary_auxiliary is not None
    assert torch.isclose(components.main, final_loss)
    assert torch.isclose(components.branch_binary_auxiliary, branch_binary_loss)
    assert torch.isclose(components.total, final_loss + (0.3 * branch_binary_loss))


def test_branch_binary_auxiliary_cosine_schedule_resolves_epoch_weights() -> None:
    cfg = BranchBinaryAuxiliaryLossConfig(
        enabled=True,
        weight=0.3,
        schedule=BranchBinaryAuxiliaryScheduleConfig(
            enabled=True,
            type="cosine_floor",
            max_weight=0.4,
            min_weight=0.1,
            total_epochs=120,
        ),
    )

    weights = [
        resolve_branch_binary_aux_weight(cfg, epoch=epoch, total_epochs=120)
        for epoch in range(1, 121)
    ]
    expected_epoch_60 = 0.1 + (0.4 - 0.1) * 0.5 * (1.0 + np.cos(np.pi * (59.0 / 119.0)))

    assert weights[0] == pytest.approx(0.4)
    assert weights[59] == pytest.approx(expected_epoch_60)
    assert weights[-1] == pytest.approx(0.1)
    assert weights == sorted(weights, reverse=True)


def test_branch_binary_auxiliary_schedule_disabled_uses_fixed_weight() -> None:
    cfg = BranchBinaryAuxiliaryLossConfig(
        enabled=True,
        weight=0.7,
        schedule=BranchBinaryAuxiliaryScheduleConfig(enabled=False),
    )
    disabled_cfg = BranchBinaryAuxiliaryLossConfig(enabled=False, weight=0.7)

    assert resolve_branch_binary_aux_weight(cfg, epoch=10, total_epochs=20) == 0.7
    assert (
        resolve_branch_binary_aux_weight(
            disabled_cfg,
            epoch=10,
            total_epochs=20,
        )
        == 0.0
    )


def test_branch_binary_auxiliary_schedule_and_monitor_totals_are_separate() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_binary_auxiliary_enabled=True,
            branch_binary_auxiliary_weight=0.3,
            branch_binary_auxiliary_schedule=BranchBinaryAuxiliaryScheduleConfig(
                enabled=True,
                type="cosine_floor",
                max_weight=0.4,
                min_weight=0.1,
                total_epochs=120,
            ),
            branch_binary_auxiliary_monitor=BranchBinaryAuxiliaryMonitorConfig(
                loss_weight=0.3
            ),
            main_index_to_binary_target=(0, 1, 1, 1),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    branch_binary_criterion = trainer._branch_binary_criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        branch_binary_logits=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, 0.0, 0.5]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    components = trainer._compute_total_loss(
        criterion,
        output,
        labels,
        epoch=120,
        use_monitor_total=True,
    )
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    branch_binary_loss = trainer._compute_branch_binary_auxiliary_loss(
        branch_binary_criterion,
        output,
        labels,
    )

    assert components.branch_binary_aux_weight == pytest.approx(0.1)
    assert components.branch_binary_aux_monitor_weight == pytest.approx(0.3)
    assert torch.isclose(
        components.total_scheduled,
        final_loss + (0.1 * branch_binary_loss),
    )
    assert torch.isclose(
        components.total_monitor,
        final_loss + (0.3 * branch_binary_loss),
    )
    assert torch.isclose(components.total, components.total_monitor)


def test_trainer_raises_when_branch_binary_auxiliary_missing_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=4,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            branch_binary_auxiliary_enabled=True,
            main_index_to_binary_target=(0, 1, 1, 1),
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[2.0, 0.1, -0.3, 0.0], [0.0, 0.2, 1.5, -0.4]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        branch_binary_logits=None,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="branch_binary_logits"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_attention_entropy_disabled_adds_no_term() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
            attention_entropy_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_attention_weights=(
            torch.tensor([[0.75, 0.25], [0.60, 0.40]], dtype=torch.float32),
            torch.tensor([[0.20, 0.80], [0.55, 0.45]], dtype=torch.float32),
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss)


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_trainer_focal_loss_builder_accepts_round_g_gamma_values(
    gamma: float,
) -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="focal",
            gamma=gamma,
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=0.1,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2]]),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)

    assert trainer.cfg.gamma == gamma
    assert torch.isfinite(components.total)
    assert components.branch_auxiliary is not None
    assert torch.isfinite(components.branch_auxiliary)


def test_trainer_attention_entropy_enabled_adds_weighted_entropy() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=False,
            attention_entropy_enabled=True,
            attention_entropy_weight=0.25,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_attention_weights=(
            torch.tensor([[0.75, 0.25], [0.60, 0.40]], dtype=torch.float32),
            torch.tensor([[0.20, 0.80], [0.55, 0.45]], dtype=torch.float32),
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    entropy_loss = trainer._compute_attention_entropy_loss(output)

    assert components.branch_auxiliary is None
    assert torch.isclose(components.total, final_loss + (0.25 * entropy_loss))


def test_trainer_raises_when_attention_entropy_enabled_without_attention() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            attention_entropy_enabled=True,
            attention_entropy_weight=0.1,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_attention_weights=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="attention entropy loss enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_gate_entropy_regularization_disabled_adds_no_term() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        evidence_gate_entropy=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.gate_entropy is None
    assert components.gate_entropy_regularization is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_evidence_auxiliary_disabled_adds_no_term() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=False,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)

    assert components.evidence_auxiliary_loss is None
    assert torch.isclose(components.total, final_loss)


def test_trainer_evidence_auxiliary_enabled_adds_weighted_ce() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=True,
            class_gate_evidence_auxiliary_weight=0.2,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=torch.tensor(
            [[0.1, 0.8, -0.2], [0.7, 0.1, 0.0]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    assert output.class_evidence_logits is not None
    evidence_loss = trainer._compute_main_loss(
        criterion,
        output.class_evidence_logits,
        labels,
    )

    assert components.evidence_auxiliary_loss is not None
    assert torch.isclose(components.evidence_auxiliary_loss, evidence_loss)
    assert torch.isclose(components.total, final_loss + (0.2 * evidence_loss))


def test_trainer_raises_when_evidence_auxiliary_enabled_without_logits() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=True,
            class_gate_evidence_auxiliary_weight=0.2,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_logits=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="class gate evidence auxiliary loss enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_gate_entropy_regularization_subtracts_weighted_entropy() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        evidence_gate_entropy=torch.tensor([0.2, 0.6], dtype=torch.float32),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    gate_entropy, regularization = trainer._compute_gate_entropy_regularization(output)

    assert torch.isclose(gate_entropy, torch.tensor(0.4))
    assert torch.isclose(regularization, torch.tensor(-0.1))
    assert components.gate_entropy is not None
    assert torch.isclose(components.gate_entropy, gate_entropy)
    assert components.gate_entropy_regularization is not None
    assert torch.isclose(components.gate_entropy_regularization, regularization)
    assert torch.isclose(components.total, final_loss + regularization)


def test_trainer_gate_entropy_regularization_supports_class_gate_targets() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.5,
            gate_entropy_regularization_target="true_class_evidence_gate",
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_entropy=torch.tensor(
            [[0.2, 0.6, 0.4], [0.9, 0.3, 0.7]],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    gate_entropy, regularization = trainer._compute_gate_entropy_regularization(
        output,
        labels,
    )

    assert torch.isclose(gate_entropy, torch.tensor(0.75))
    assert torch.isclose(regularization, torch.tensor(-0.375))
    assert torch.isclose(components.total, final_loss + regularization)


def test_trainer_gate_entropy_regularization_supports_all_class_gate_target() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
            gate_entropy_regularization_target="class_evidence_gate",
        )
    )
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_entropy=torch.tensor(
            [[0.2, 0.6, 0.4], [0.9, 0.3, 0.7]],
            dtype=torch.float32,
        ),
    )

    gate_entropy, regularization = trainer._compute_gate_entropy_regularization(output)

    assert torch.isclose(gate_entropy, torch.tensor(0.5166667), atol=1e-6)
    assert torch.isclose(regularization, torch.tensor(-0.1291667), atol=1e-6)


def test_trainer_gate_entropy_regularization_combines_with_branch_auxiliary() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            branch_auxiliary_enabled=True,
            branch_auxiliary_weight=0.1,
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        branch_logits=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2]]),
        evidence_gate_entropy=torch.tensor([0.2, 0.6], dtype=torch.float32),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    branch_loss = trainer._compute_branch_auxiliary_loss(
        criterion,
        output.branch_logits,
        labels,
    )
    _, regularization = trainer._compute_gate_entropy_regularization(output)

    assert components.branch_auxiliary is not None
    assert torch.isclose(
        components.total,
        final_loss + (0.1 * branch_loss) + regularization,
    )


def test_trainer_class_gate_diversity_regularization_subtracts_js_divergence() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor(
            [[0.4, -0.7, 0.1], [0.2, 0.3, -0.1]],
            dtype=torch.float32,
        ),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_weights=torch.tensor(
            [
                [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]],
            ],
            dtype=torch.float32,
        ),
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    components = trainer._compute_total_loss(criterion, output, labels)
    final_loss = trainer._compute_main_loss(criterion, output.logits, labels)
    diversity, regularization = trainer._compute_class_gate_diversity_regularization(
        output
    )

    assert diversity > 0
    assert torch.isclose(regularization, -0.3 * diversity)
    assert components.class_gate_diversity is not None
    assert components.class_gate_diversity_regularization is not None
    assert torch.isclose(components.total, final_loss + regularization)


def test_trainer_raises_when_class_gate_diversity_enabled_without_weights() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.zeros(2, 3),
        pooled_embedding=torch.zeros(2, 32),
        class_evidence_gate_weights=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="class gate diversity regularization enabled",
    ):
        trainer._compute_total_loss(criterion, output, labels)


def test_trainer_raises_when_gate_entropy_enabled_without_evidence_entropy() -> None:
    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.1,
        )
    )
    criterion = trainer._criterion_on(torch.device("cpu"))
    output = AstModelOutput(
        logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
        pooled_embedding=torch.zeros(2, 32),
        evidence_gate_entropy=None,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="gate entropy regularization enabled"):
        trainer._compute_total_loss(criterion, output, labels)


def test_epoch_loss_components_include_gate_entropy_regularization() -> None:
    class GateEntropyModel(torch.nn.Module):
        def forward(self, input_values: torch.Tensor) -> AstModelOutput:
            batch_size = input_values.shape[0]
            return AstModelOutput(
                logits=torch.tensor([0.4, -0.7], dtype=torch.float32),
                pooled_embedding=torch.zeros(batch_size, 32),
                evidence_gate_entropy=torch.tensor([0.2, 0.6], dtype=torch.float32),
            )

    trainer = Trainer(
        _trainer_cfg(
            num_classes=2,
            run_dir=Path("unused"),
            loss_type="bce",
            gate_entropy_regularization_enabled=True,
            gate_entropy_regularization_weight=0.25,
        )
    )
    batch = ClipBatch(
        input_values=torch.zeros(2, 1),
        labels=torch.tensor([1, 0], dtype=torch.long),
        audio_paths=("a.wav", "b.wav"),
        label_names=("wheeze", "normal"),
    )

    result = trainer._epoch(
        GateEntropyModel(),
        [batch],
        optimizer=None,
        scheduler=None,
        device=torch.device("cpu"),
        epoch=1,
    )

    assert result.loss_components["gate_entropy"] == pytest.approx(0.4)
    assert result.loss_components["gate_entropy_regularization"] == pytest.approx(-0.1)


def test_epoch_loss_components_include_class_gate_terms() -> None:
    class ClassGateModel(torch.nn.Module):
        def forward(self, input_values: torch.Tensor) -> AstModelOutput:
            batch_size = input_values.shape[0]
            return AstModelOutput(
                logits=torch.zeros(batch_size, 3, dtype=torch.float32),
                pooled_embedding=torch.zeros(batch_size, 32),
                class_evidence_logits=torch.tensor(
                    [[0.1, 0.8, -0.2], [0.7, 0.1, 0.0]],
                    dtype=torch.float32,
                ),
                class_evidence_gate_weights=torch.tensor(
                    [
                        [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
                        [[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]],
                    ],
                    dtype=torch.float32,
                ),
            )

    trainer = Trainer(
        _trainer_cfg(
            num_classes=3,
            run_dir=Path("unused"),
            loss_type="cross_entropy",
            class_gate_evidence_auxiliary_enabled=True,
            class_gate_evidence_auxiliary_weight=0.2,
            class_gate_diversity_regularization_enabled=True,
            class_gate_diversity_regularization_weight=0.3,
        )
    )
    batch = ClipBatch(
        input_values=torch.zeros(2, 1),
        labels=torch.tensor([1, 0], dtype=torch.long),
        audio_paths=("a.wav", "b.wav"),
        label_names=("wheeze", "normal"),
    )

    result = trainer._epoch(
        ClassGateModel(),
        [batch],
        optimizer=None,
        scheduler=None,
        device=torch.device("cpu"),
        epoch=1,
    )

    assert "evidence_auxiliary_loss" in result.loss_components
    assert "class_gate_diversity" in result.loss_components
    assert "class_gate_diversity_regularization" in result.loss_components


def test_configured_checkpoint_monitors_keep_top_three(tmp_path: Path) -> None:
    manager = CheckpointManager(
        tmp_path,
        top_k=1,
        checkpointing=CheckpointingConfig(
            monitors=(
                CheckpointMonitorConfig(
                    name="val_macro_f1",
                    mode="max",
                    top_k=3,
                ),
                CheckpointMonitorConfig(
                    name="val_macro_recall",
                    mode="max",
                    top_k=3,
                ),
                CheckpointMonitorConfig(
                    name="val_loss_total_monitor",
                    mode="min",
                    top_k=3,
                    filename_prefix="best_loss",
                ),
                CheckpointMonitorConfig(
                    name="last",
                    mode="last",
                    top_k=3,
                ),
            )
        ),
    )

    for epoch, (macro_f1, macro_recall, loss) in enumerate(
        [
            (0.10, 0.20, 0.90),
            (0.40, 0.30, 0.80),
            (0.20, 0.50, 0.70),
            (0.30, 0.10, 0.60),
        ],
        start=1,
    ):
        state = {"epoch": epoch}
        manager.save_last(state)
        manager.save_configured_monitors(
            {
                "val_macro_f1": macro_f1,
                "val_macro_recall": macro_recall,
                "val_loss_total_monitor": loss,
            },
            state,
            epoch=epoch,
        )

    assert len(sorted(tmp_path.glob("best_macro_f1_rank*.pt"))) == 3
    assert len(sorted(tmp_path.glob("best_macro_recall_rank*.pt"))) == 3
    assert len(sorted(tmp_path.glob("best_loss_rank*.pt"))) == 3
    assert len(sorted(tmp_path.glob("last_epoch*.pt"))) == 3
    assert (tmp_path / "last.pt").exists()
    assert sorted(path.name for path in tmp_path.glob("last_epoch*.pt")) == [
        "last_epoch002.pt",
        "last_epoch003.pt",
        "last_epoch004.pt",
    ]


def test_legacy_checkpoint_behavior_is_preserved(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, top_k=1)
    state = {"epoch": 1}

    manager.save_last(state)
    manager.maybe_save_best("loss", 0.5, state, maximize=False)
    manager.maybe_save_best("loss", 0.4, {"epoch": 2}, maximize=False)
    manager.maybe_save_best("f1", 0.1, state, maximize=True)

    assert (tmp_path / "last.pt").exists()
    assert len(sorted(tmp_path.glob("best_loss_*.pt"))) == 1
    assert len(sorted(tmp_path.glob("best_f1_*.pt"))) == 1


def test_binary_trainer_and_evaluator_smoke_b3(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "clips")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(rdt_enabled=True, rdt_steps=3),
        loss_type="focal",
        pos_weight=2.0,
        branch_auxiliary_enabled=True,
    )
    run_dir = checkpoint_path.parent

    assert checkpoint_path.exists()
    assert sorted(run_dir.glob("best_loss_*.pt"))

    checkpoint = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    assert checkpoint["dims"] == {"num_mel_bins": 32, "max_length": 32}

    result = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    metrics, rows, diagnostics = result

    assert metrics["threshold_optimization"]["enabled"] is True
    assert len(rows) == len(build_dataset(data_cfg, split="eval"))
    assert len(diagnostics) == len(rows)
    assert "branch_logits" in diagnostics[0]
    assert "selected_evidence_indices" in diagnostics[0]
    assert "selected_evidence_scores" in diagnostics[0]
    assert "selected_evidence_branch_ids" in diagnostics[0]
    assert diagnostics[0]["evidence_score_source"] == "attention_weight"
    assert "selected_evidence_tokens" in diagnostics[0]


def test_multiclass_trainer_and_evaluator_smoke_b3(tmp_path: Path) -> None:
    data_cfg = _prepare_multiclass_dataset(tmp_path / "multiclass")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(
            classifier_type="mlp",
            rdt_enabled=True,
            rdt_steps=2,
        ),
        loss_type="cross_entropy",
        branch_auxiliary_enabled=True,
    )

    result = evaluate_checkpoint(
        _eval_cfg(tmp_path, checkpoint_path, data_cfg),
        checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    metrics, rows, diagnostics = result

    assert metrics["optimized_metrics"]["decision_threshold"] is None
    assert len(rows) == len(build_dataset(data_cfg, split="eval"))
    assert len(diagnostics) == len(rows)
    assert "branch_logits" in diagnostics[0]
    assert "selected_evidence_indices" in diagnostics[0]
    assert "selected_evidence_scores" in diagnostics[0]
    assert "selected_evidence_branch_ids" in diagnostics[0]
    assert diagnostics[0]["evidence_score_source"] == "attention_weight"


def test_evaluator_rejects_frontend_dim_mismatch(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "clips")
    checkpoint_path, data_cfg = _train_smoke_run(
        tmp_path,
        data_cfg=data_cfg,
        model_cfg=_model_cfg(rdt_enabled=False, rdt_steps=3),
        loss_type="bce",
        branch_auxiliary_enabled=False,
    )
    mismatched_cfg = DataConfig(
        train_dirs=data_cfg.train_dirs,
        val_dirs=data_cfg.val_dirs,
        eval_dirs=data_cfg.eval_dirs,
        label_to_index=data_cfg.label_to_index,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        audio=data_cfg.audio,
        preprocessing=PreprocessingConfig(
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=40,
                do_normalize=True,
                mean=-4.2677393,
                std=4.5689974,
            ),
        ),
    )

    with pytest.raises(ValueError, match="Evaluation frontend dims do not match"):
        evaluate_checkpoint(
            _eval_cfg(tmp_path, checkpoint_path, mismatched_cfg),
            checkpoint_path,
        )


def test_cv_cli_smoke_b3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = _prepare_cv_dataset(tmp_path / "cv")
    config_path = _write_json(
        tmp_path / "cv.json",
        _cv_payload(dataset_root, tmp_path / "cv_outputs"),
    )

    monkeypatch.setattr(sys, "argv", ["cv", "--config", str(config_path)])
    cv_main()

    fold_dir = tmp_path / "cv_outputs" / "cv_smoke" / "fold_0"
    assert (fold_dir / "last.pt").exists()
