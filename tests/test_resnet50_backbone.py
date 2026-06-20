from __future__ import annotations

import builtins
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from src.cli.evaluate import evaluate_checkpoint
from src.data.audio import ResNetSpectrogramFeatureConfig, ResNetSpectrogramImage
from src.data.loaders import build_dataset
from src.models.resnet50_model import (
    ResNet50AdaptationConfig,
    ResNet50ClassifierConfig,
    ResNet50EncoderRuntimeConfig,
    ResNet50ModelConfig,
    RespiratoryResNet50Model,
)
from src.training.model_setup import (
    apply_resnet50_encoder_adaptation,
    build_model_bundle,
)
from src.utils.config import (
    AnalysisConfig,
    AnalysisOutputConfig,
    AudioConfig,
    BandPassConfig,
    ClassifierConfig,
    CvRunConfig,
    DataConfig,
    EarlyStoppingConfig,
    EncoderAdaptationConfig,
    EvalConfig,
    ExperimentConfig,
    JsonConfigLoader,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    ResNet50EncoderConfig,
    ResNetSpectrogramConfig,
    SamplerConfig,
    SchedulerConfig,
    TrainConfig,
)


def _write_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(path, tone.astype(np.float32), sample_rate)


def _analysis_cfg() -> AnalysisConfig:
    return AnalysisConfig(
        outputs=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=False,
            save_clip_metadata=True,
        )
    )


def _train_cfg() -> TrainConfig:
    return TrainConfig(
        epochs=1,
        top_k=1,
        max_grad_norm=1.0,
        optimizer=OptimizerConfig(encoder_lr=1e-4, head_lr=1e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(warmup_ratio=0.0),
        loss=LossConfig(type="focal", pos_weight=1.0),
        sampler=SamplerConfig(weighted_random=False),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            monitor="val_loss",
            patience=3,
            min_delta=1e-4,
        ),
    )


def _resnet_data_cfg(
    root: Path,
    *,
    clip_duration_sec: float = 0.2,
    use_hpss: bool = True,
    image_size: int = 64,
) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=clip_duration_sec),
        preprocessing=PreprocessingConfig(
            feature_type="resnet_spectrogram",
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            resnet_spectrogram=ResNetSpectrogramConfig(
                n_fft=400,
                hop_length=160,
                win_length=400,
                n_mels=16,
                use_hpss=use_hpss,
                image_size=image_size,
            ),
        ),
    )


def _prepare_binary_dataset(root: Path) -> DataConfig:
    for split in ("train", "val"):
        for label in ("normal", "wheeze"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    _write_wav(root / "train" / "normal" / "normal_a.wav", 0.2)
    _write_wav(root / "train" / "wheeze" / "wheeze_a.wav", 0.2)
    _write_wav(root / "val" / "normal" / "normal_b.wav", 0.2)
    _write_wav(root / "val" / "wheeze" / "wheeze_b.wav", 0.2)
    return _resnet_data_cfg(root)


def _small_runtime_cfg(
    *,
    num_classes: int,
    image_size: int = 64,
    adaptation: ResNet50AdaptationConfig | None = None,
) -> ResNet50ModelConfig:
    return ResNet50ModelConfig(
        encoder=ResNet50EncoderRuntimeConfig(
            weights="none",
            adaptation=adaptation or ResNet50AdaptationConfig(mode="frozen"),
            image_size=image_size,
        ),
        classifier=ResNet50ClassifierConfig(type="mlp", hidden_dim=32, dropout=0.0),
        num_classes=num_classes,
    )


def _small_run_model_cfg(*, image_size: int = 64) -> ModelConfig:
    return ModelConfig(
        encoder=ResNet50EncoderConfig(
            weights="none",
            adaptation=EncoderAdaptationConfig(mode="frozen", num_layers=1),
            image_size=image_size,
        ),
        classifier=ClassifierConfig(
            type="mlp",
            hidden_dim=32,
            dropout=0.0,
            pooling="mean",
        ),
    )


def test_resnet50_disease_cv_configs_load() -> None:
    for path in [
        "configs/resnet50/cv_run_direct_3class.json",
        "configs/resnet50/cv_run_cascade_stage1_normal_vs_abnormal.json",
        "configs/resnet50/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json",
    ]:
        cfg = JsonConfigLoader.load_cv(path)
        assert isinstance(cfg, CvRunConfig)
        assert cfg.model.encoder.type == "resnet50"
        assert cfg.data.audio.clip_duration_sec == 15.0
        assert cfg.data.preprocessing.feature_type == "resnet_spectrogram"
        assert cfg.data.preprocessing.resnet_spectrogram.use_hpss is True
        assert cfg.experiment.output_dir == "Disease_Group_Results/ResNet50"


def test_resnet50_partial_l1_cv_configs_load() -> None:
    for path in [
        "configs/resnet50_partial_l1/cv_run_direct_3class.json",
        "configs/resnet50_partial_l1/cv_run_cascade_stage1_normal_vs_abnormal.json",
        "configs/resnet50_partial_l1/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json",
    ]:
        cfg = JsonConfigLoader.load_cv(path)
        assert isinstance(cfg, CvRunConfig)
        assert cfg.model.encoder.type == "resnet50"
        assert cfg.model.encoder.adaptation.mode == "partial"
        assert cfg.model.encoder.adaptation.num_layers == 1
        assert cfg.experiment.output_dir == "Disease_Group_Results/ResNet50_Partial_L1"


def test_resnet50_spectrogram_frontend_shape() -> None:
    extractor = ResNetSpectrogramImage(
        ResNetSpectrogramFeatureConfig(
            clip_seconds=0.2,
            n_mels=16,
            use_hpss=True,
            image_size=32,
        )
    )
    audio = torch.zeros(3200)

    features = extractor(audio)

    assert features.shape == (3, 32, 32)
    assert torch.isfinite(features).all()
    assert extractor.n_frames == 20


def test_resnet50_hpss_missing_librosa_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def raising_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "librosa":
            raise ImportError("mocked missing librosa")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with pytest.raises(RuntimeError, match="requires librosa"):
        ResNetSpectrogramImage(ResNetSpectrogramFeatureConfig(use_hpss=True))


def test_resnet50_model_forward_binary_and_multiclass() -> None:
    binary_model = RespiratoryResNet50Model(_small_runtime_cfg(num_classes=2))
    multiclass_model = RespiratoryResNet50Model(_small_runtime_cfg(num_classes=3))
    x = torch.randn(2, 3, 64, 64)

    binary_output = binary_model(x)
    multiclass_output = multiclass_model(x)

    assert binary_output.logits.shape == (2,)
    assert binary_output.pooled_embedding is not None
    assert binary_output.pooled_embedding.shape == (2, 2048)
    assert multiclass_output.logits.shape == (2, 3)


def test_resnet50_adaptation_scopes() -> None:
    frozen = RespiratoryResNet50Model(
        _small_runtime_cfg(
            num_classes=2,
            adaptation=ResNet50AdaptationConfig(mode="frozen"),
        )
    )
    frozen_summary = apply_resnet50_encoder_adaptation(
        frozen,
        frozen.cfg.encoder.adaptation,
    )
    assert frozen_summary.trainable_parameters == 0

    partial = RespiratoryResNet50Model(
        _small_runtime_cfg(
            num_classes=2,
            adaptation=ResNet50AdaptationConfig(mode="partial", num_layers=1),
        )
    )
    apply_resnet50_encoder_adaptation(partial, partial.cfg.encoder.adaptation)
    assert any(
        parameter.requires_grad for parameter in partial.encoder.layer4.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in partial.encoder.layer3.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in partial.encoder.conv1.parameters()
    )

    full = RespiratoryResNet50Model(
        _small_runtime_cfg(
            num_classes=2,
            adaptation=ResNet50AdaptationConfig(mode="full"),
        )
    )
    apply_resnet50_encoder_adaptation(full, full.cfg.encoder.adaptation)
    assert all(parameter.requires_grad for parameter in full.encoder.parameters())


def test_resnet50_model_bundle_and_evaluate_checkpoint_smoke(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "bundle")
    train_dataset = build_dataset(data_cfg, split="train")
    model_cfg = _small_run_model_cfg()
    bundle = build_model_bundle(
        model_cfg,
        _train_cfg(),
        train_dataset,
        num_classes=len(data_cfg.label_to_index),
    )

    checkpoint_path = tmp_path / "resnet50_last.pt"
    torch.save(
        {
            "model_family": bundle.model_family,
            "model_cfg": asdict(bundle.model_cfg),
            "feature_cfg": bundle.feature_cfg,
            "label_to_index": dict(data_cfg.label_to_index),
            "num_classes": len(data_cfg.label_to_index),
            "model_state_dict": bundle.model.state_dict(),
        },
        checkpoint_path,
    )
    eval_cfg = EvalConfig(
        experiment=ExperimentConfig(
            name="eval",
            task="normal_vs_wheeze",
            mode="clip",
            seed=0,
            device="cpu",
            output_dir=str(tmp_path / "eval"),
        ),
        checkpoint_path=str(checkpoint_path),
        data=data_cfg,
        analysis=_analysis_cfg(),
    )

    metrics, rows = evaluate_checkpoint(
        eval_cfg,
        checkpoint_path,
        return_predictions=True,
    )

    assert bundle.model_family == "resnet50"
    assert bundle.feature_cfg["feature_type"] == "resnet_spectrogram"
    assert bundle.feature_cfg["input_channels"] == 3
    assert bundle.feature_cfg["image_size"] == 64
    assert metrics["accuracy"] >= 0.0
    assert len(rows) == 2
    assert rows[0].class_probabilities is None
