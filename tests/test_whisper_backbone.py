from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from src.cli.evaluate import evaluate_checkpoint
from src.data.loaders import build_dataset
from src.models.model import EncoderAdaptationConfig as RuntimeEncoderAdaptationConfig
from src.models.whisper_encoder import WhisperEncoderDims
from src.models.whisper_model import RespiratoryWhisperModel, WhisperModelConfig
from src.pretrained.whisper import (
    OpenAIWhisperCheckpointLoader,
    WhisperPretrainedConfig,
)
from src.training.model_setup import (
    apply_whisper_encoder_adaptation,
    build_model_bundle,
)
from src.utils.checkpoint import parse_model_cfg
from src.utils.config import (
    AnalysisConfig,
    AnalysisOutputConfig,
    AudioConfig,
    BandPassConfig,
    ClassifierConfig,
    CvRunConfig,
    DataConfig,
    EarlyStoppingConfig,
    EvalConfig,
    ExperimentConfig,
    JsonConfigLoader,
    LogMelConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    SamplerConfig,
    SchedulerConfig,
    TrainConfig,
    WhisperEncoderConfig,
)
from src.utils.config import (
    EncoderAdaptationConfig as ConfigEncoderAdaptationConfig,
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


def _whisper_data_cfg(root: Path, *, clip_duration_sec: float = 0.2) -> DataConfig:
    return DataConfig(
        train_dirs=[str(root / "train")],
        val_dirs=[str(root / "val")],
        eval_dirs=[str(root / "val")],
        label_to_index={"normal": 0, "wheeze": 1},
        batch_size=2,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=clip_duration_sec),
        preprocessing=PreprocessingConfig(
            feature_type="log_mel",
            source_type="original",
            bandpass=BandPassConfig(enabled=False),
            log_mel=LogMelConfig(
                n_fft=400,
                hop_length=160,
                win_length=400,
                n_mels=16,
            ),
        ),
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


def _small_whisper_model_cfg(
    *,
    num_classes: int,
    adaptation: RuntimeEncoderAdaptationConfig | None = None,
) -> WhisperModelConfig:
    return WhisperModelConfig(
        encoder=WhisperEncoderDims(
            n_mels=16,
            n_audio_ctx=10,
            n_audio_state=32,
            n_audio_head=4,
            n_audio_layer=2,
        ),
        num_classes=num_classes,
        adaptation=adaptation
        or RuntimeEncoderAdaptationConfig(mode="frozen", num_layers=1),
        head_type="hf",
        pooling="mean",
        use_weighted_layer_sum=True,
        classifier_proj_size=16,
    )


def _small_run_model_cfg(
    *,
    checkpoint_path: str | None = None,
    adaptation: ConfigEncoderAdaptationConfig | None = None,
) -> ModelConfig:
    pretrained = (
        WhisperPretrainedConfig(
            name_or_path=checkpoint_path,
            load_encoder_only=True,
            strict=True,
            freeze_encoder=True,
        )
        if checkpoint_path is not None
        else None
    )
    return ModelConfig(
        encoder=WhisperEncoderConfig(
            n_mels=16,
            n_audio_ctx=10,
            n_audio_state=32,
            n_audio_head=4,
            n_audio_layer=2,
            adaptation=adaptation
            or ConfigEncoderAdaptationConfig(mode="frozen", num_layers=1),
            pretrained=pretrained,
        ),
        classifier=ClassifierConfig(
            type="hf",
            hidden_dim=16,
            dropout=0.0,
            pooling="mean",
            use_weighted_layer_sum=True,
            classifier_proj_size=16,
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
    return _whisper_data_cfg(root)


def _write_pretrained_checkpoint(path: Path, model: RespiratoryWhisperModel) -> None:
    encoder_state = {
        f"encoder.{key}": value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }
    torch.save(
        {
            "dims": asdict(model.cfg.encoder),
            "model_state_dict": encoder_state,
        },
        path,
    )


def test_whisper_disease_cv_configs_load() -> None:
    for path in [
        "configs/whisper/cv_run_direct_3class.json",
        "configs/whisper/cv_run_cascade_stage1_normal_vs_abnormal.json",
        "configs/whisper/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json",
    ]:
        cfg = JsonConfigLoader.load_cv(path)
        assert isinstance(cfg, CvRunConfig)
        assert cfg.model.encoder.type == "whisper"
        assert cfg.model.encoder.adaptation.mode == "frozen"
        assert cfg.data.preprocessing.feature_type == "log_mel"
        assert cfg.experiment.output_dir == "Disease_Group_Results/Whisper"


def test_whisper_partial_l1_cv_configs_load() -> None:
    for path in [
        "configs/whisper_partial_l1/cv_run_direct_3class.json",
        "configs/whisper_partial_l1/cv_run_cascade_stage1_normal_vs_abnormal.json",
        "configs/whisper_partial_l1/cv_run_cascade_stage2_airway_vs_lung_parenchymal.json",
    ]:
        cfg = JsonConfigLoader.load_cv(path)
        assert isinstance(cfg, CvRunConfig)
        assert cfg.model.encoder.type == "whisper"
        assert cfg.model.encoder.adaptation.mode == "partial"
        assert cfg.model.encoder.adaptation.num_layers == 1
        assert cfg.experiment.output_dir == "Disease_Group_Results/Whisper_Partial_L1"


def test_whisper_log_mel_dataset_shape(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "clips")
    dataset = build_dataset(data_cfg, split="train")

    sample = dataset[0]

    assert dataset.feature_type == "log_mel"
    assert sample.input_values.shape == (16, 20)
    assert dataset.num_mel_bins == 16
    assert dataset.max_length == 20
    assert dataset.n_audio_ctx == 10


def test_whisper_model_forward_binary_and_multiclass() -> None:
    binary_model = RespiratoryWhisperModel(_small_whisper_model_cfg(num_classes=2))
    multiclass_model = RespiratoryWhisperModel(_small_whisper_model_cfg(num_classes=3))
    x = torch.randn(2, 16, 20)

    binary_output = binary_model(x)
    multiclass_output = multiclass_model(x)

    assert binary_output.logits.shape == (2,)
    assert binary_output.pooled_embedding is not None
    assert binary_output.pooled_embedding.shape == (2, 16)
    assert multiclass_output.logits.shape == (2, 3)


def test_whisper_encoder_optional_attentions_preserve_default_behavior() -> None:
    model = RespiratoryWhisperModel(_small_whisper_model_cfg(num_classes=2))
    x = torch.randn(2, 16, 20)

    default_output = model.encoder(x, output_hidden_states=True)
    attention_output = model.encoder(
        x,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert default_output.attentions is None
    assert default_output.last_hidden_state.shape == (2, 10, 32)
    assert attention_output.hidden_states is not None
    assert attention_output.attentions is not None
    assert len(attention_output.attentions) == 2
    assert attention_output.attentions[-1].shape == (2, 4, 10, 10)


def test_whisper_encoder_attentions_include_prefix_tokens() -> None:
    model = RespiratoryWhisperModel(
        WhisperModelConfig(
            encoder=WhisperEncoderDims(
                n_mels=16,
                n_audio_ctx=10,
                n_audio_state=32,
                n_audio_head=4,
                n_audio_layer=2,
            ),
            num_classes=2,
            head_type="hf",
            pooling="cls",
            use_weighted_layer_sum=False,
            classifier_proj_size=16,
        )
    )
    x = torch.randn(2, 16, 20)
    assert hasattr(model.classifier, "build_prefix_tokens")
    prefix_tokens = model.classifier.build_prefix_tokens(
        2,
        device=x.device,
        dtype=x.dtype,
    )

    output = model.encoder(
        x,
        output_attentions=True,
        prefix_tokens=prefix_tokens,
    )

    assert output.attentions is not None
    assert output.last_hidden_state.shape == (2, 11, 32)
    assert output.attentions[-1].shape == (2, 4, 11, 11)


def test_whisper_adaptation_scopes() -> None:
    frozen = RespiratoryWhisperModel(_small_whisper_model_cfg(num_classes=2))
    frozen_summary = apply_whisper_encoder_adaptation(
        frozen,
        frozen.cfg.adaptation,
    )
    assert frozen_summary.mode == "frozen"
    assert frozen_summary.trainable_parameters == 0

    partial = RespiratoryWhisperModel(
        _small_whisper_model_cfg(
            num_classes=2,
            adaptation=RuntimeEncoderAdaptationConfig(
                mode="partial",
                num_layers=1,
            ),
        )
    )
    partial_summary = apply_whisper_encoder_adaptation(
        partial,
        partial.cfg.adaptation,
    )
    assert partial_summary.mode == "partial"
    assert partial_summary.num_layers == 1
    assert partial_summary.trainable_parameters > 0
    assert any(
        parameter.requires_grad for parameter in partial.encoder.blocks[-1].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in partial.encoder.blocks[0].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in partial.encoder.conv1.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in partial.encoder.conv2.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in partial.encoder.ln_post.parameters()
    )

    full = RespiratoryWhisperModel(
        _small_whisper_model_cfg(
            num_classes=2,
            adaptation=RuntimeEncoderAdaptationConfig(mode="full", num_layers=1),
        )
    )
    full_summary = apply_whisper_encoder_adaptation(full, full.cfg.adaptation)
    assert full_summary.mode == "full"
    assert all(parameter.requires_grad for parameter in full.encoder.parameters())


def test_whisper_local_pretrained_loader_freezes_encoder(tmp_path: Path) -> None:
    source_model = RespiratoryWhisperModel(_small_whisper_model_cfg(num_classes=2))
    checkpoint_path = tmp_path / "tiny_local.pt"
    _write_pretrained_checkpoint(checkpoint_path, source_model)
    target_model = RespiratoryWhisperModel(_small_whisper_model_cfg(num_classes=2))

    info = OpenAIWhisperCheckpointLoader().load_encoder_into(
        target_model,
        WhisperPretrainedConfig(
            name_or_path=str(checkpoint_path),
            load_encoder_only=True,
            strict=True,
            freeze_encoder=True,
        ),
    )

    assert info.source == "local_path"
    assert info.loaded_keys > 0
    assert all(
        not parameter.requires_grad for parameter in target_model.encoder.parameters()
    )


def test_whisper_checkpoint_parser_defaults_legacy_adaptation() -> None:
    raw = asdict(_small_whisper_model_cfg(num_classes=2))
    raw.pop("adaptation")

    parsed = parse_model_cfg(raw)

    assert isinstance(parsed, WhisperModelConfig)
    assert parsed.adaptation.mode == "frozen"
    assert parsed.adaptation.num_layers == 1


def test_whisper_checkpoint_parser_accepts_encoder_adaptation() -> None:
    raw = asdict(_small_whisper_model_cfg(num_classes=2))
    raw.pop("adaptation")
    raw["encoder"]["adaptation"] = {
        "mode": "partial",
        "num_layers": 1,
    }

    parsed = parse_model_cfg(raw)

    assert isinstance(parsed, WhisperModelConfig)
    assert parsed.adaptation.mode == "partial"
    assert parsed.adaptation.num_layers == 1


def test_whisper_partial_model_bundle_uses_encoder_group(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "partial_bundle")
    train_dataset = build_dataset(data_cfg, split="train")
    model_cfg = _small_run_model_cfg(
        adaptation=ConfigEncoderAdaptationConfig(mode="partial", num_layers=1),
    )

    bundle = build_model_bundle(
        model_cfg,
        _train_cfg(),
        train_dataset,
        num_classes=len(data_cfg.label_to_index),
    )

    assert bundle.model_family == "whisper"
    assert bundle.adaptation_summary.mode == "partial"
    assert bundle.adaptation_summary.num_layers == 1
    assert bundle.adaptation_summary.trainable_parameters > 0
    assert bundle.optimizer_summary.encoder_trainable_parameters > 0


def test_whisper_model_bundle_and_evaluate_checkpoint_smoke(tmp_path: Path) -> None:
    data_cfg = _prepare_binary_dataset(tmp_path / "bundle")
    train_dataset = build_dataset(data_cfg, split="train")
    model_cfg = _small_run_model_cfg()
    bundle = build_model_bundle(
        model_cfg,
        _train_cfg(),
        train_dataset,
        num_classes=len(data_cfg.label_to_index),
    )

    checkpoint_path = tmp_path / "whisper_last.pt"
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

    assert bundle.model_family == "whisper"
    assert metrics["accuracy"] >= 0.0
    assert len(rows) == 2
    assert rows[0].class_probabilities is None
