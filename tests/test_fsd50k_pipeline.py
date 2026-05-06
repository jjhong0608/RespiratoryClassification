from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from src.data.fsd50k_dataset import build_fsd50k_dataset
from src.evaluation.metrics import MultiLabelMetricsComputer
from src.models.model import (
    AstFeatureDims,
    ClassifierConfig,
    EncoderAdaptationConfig,
    EvidencePoolingConfig,
    MultiScaleRdtArchitectureConfig,
    MultiScaleRdtAstModel,
    MultiScaleRdtAstModelConfig,
    MultiScaleRdtEncoderConfig,
    RdtConfig,
)
from src.models.ssl import (
    BranchConv1dFbankDecoder,
    MaskedFbankSSLConfig,
    MaskedFbankSSLWrapper,
    SslDecoderConfig,
    SslMaskingConfig,
    compute_patch_grid_size,
    masked_mse,
    sample_token_mask,
    token_grid_to_fbank_mask,
)
from src.training.multilabel_trainer import compute_sqrt_neg_pos_multilabel_pos_weight
from src.training.transfer import (
    apply_cnuh_transfer_freeze,
    build_cnuh_transfer_optimizer,
    build_fsd50k_multilabel_optimizer,
    load_compatible_model_state,
)
from src.utils.config import (
    AstFbankConfig,
    AudioConfig,
    DataAugmentationConfig,
    Fsd50kDataConfig,
    JsonConfigLoader,
    PreprocessingConfig,
)

from conftest import small_patch_branches


def _write_wav(
    path: Path, *, duration_sec: float = 1.5, sample_rate: int = 16000
) -> None:
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    audio = (0.2 * np.sin(2.0 * np.pi * (220.0 + 80.0 * t) * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fsd_fixture(tmp_path: Path, *, unknown_label: bool = False) -> Fsd50kDataConfig:
    dev_audio = tmp_path / "dev_audio"
    eval_audio = tmp_path / "eval_audio"
    gt = tmp_path / "ground_truth"
    dev_audio.mkdir()
    eval_audio.mkdir()
    gt.mkdir()
    for clip_id in ["clip_train", "clip_val", "clip_eval"]:
        _write_wav(
            (dev_audio if clip_id != "clip_eval" else eval_audio) / f"{clip_id}.wav"
        )
    vocab_rows = [
        {
            "index": str(index),
            "mid": f"/m/{index:03d}",
            "display_name": f"label_{index:03d}",
        }
        for index in range(200)
    ]
    _write_csv(gt / "vocabulary.csv", ["index", "mid", "display_name"], vocab_rows)
    dev_rows = [
        {"fname": "clip_train", "mids": "/m/000,/m/001", "split": "train"},
        {
            "fname": "clip_val",
            "mids": "/m/199" if not unknown_label else "/m/unknown",
            "split": "val",
        },
    ]
    eval_rows = [{"fname": "clip_eval", "mids": "/m/002"}]
    _write_csv(gt / "dev.csv", ["fname", "mids", "split"], dev_rows)
    _write_csv(gt / "eval.csv", ["fname", "mids"], eval_rows)
    return Fsd50kDataConfig(
        dataset="fsd50k",
        mode="supervised",
        root=str(tmp_path),
        dev_audio_dir=str(dev_audio),
        eval_audio_dir=str(eval_audio),
        ground_truth_dir=str(gt),
        vocabulary_csv=str(gt / "vocabulary.csv"),
        dev_csv=str(gt / "dev.csv"),
        eval_csv=str(gt / "eval.csv"),
        val_ratio=0.5,
        split_seed=7,
        num_workers=0,
        audio=AudioConfig(sample_rate=16000, clip_duration_sec=1.0),
        preprocessing=PreprocessingConfig(
            ast_fbank=AstFbankConfig(
                num_mel_bins=32,
                max_length=48,
                do_normalize=True,
            )
        ),
        augmentation=DataAugmentationConfig(),
    )


def _small_model(num_classes: int = 2) -> MultiScaleRdtAstModel:
    return MultiScaleRdtAstModel(
        MultiScaleRdtAstModelConfig(
            encoder=MultiScaleRdtEncoderConfig(
                feature_dims=AstFeatureDims(num_mel_bins=32, max_length=32),
                adaptation=EncoderAdaptationConfig(mode="full", num_layers=0),
                architecture=MultiScaleRdtArchitectureConfig(
                    hidden_size=16,
                    num_attention_heads=4,
                    mlp_ratio=2.0,
                    shared_stem_depth=1,
                    adapter_depth=1,
                    patch_branches=small_patch_branches(),
                    rdt=RdtConfig(enabled=True, steps=1, top_tokens_per_branch=2),
                    evidence_pooling=EvidencePoolingConfig(
                        type="branch_gated", dropout=0.0
                    ),
                ),
            ),
            classifier=ClassifierConfig(type="linear", hidden_dim=16, dropout=0.0),
            num_classes=num_classes,
        )
    )


def test_fsd50k_ssl_and_supervised_dataset_outputs(tmp_path: Path) -> None:
    cfg = _fsd_fixture(tmp_path)
    supervised = build_fsd50k_dataset(cfg, split="train")
    supervised_sample = supervised[0]
    assert supervised_sample.input_values.shape == (48, 32)
    assert supervised_sample.labels is not None
    assert supervised_sample.labels.shape == (200,)
    assert supervised_sample.labels[0] == 1
    assert supervised_sample.labels[1] == 1

    ssl_cfg = Fsd50kDataConfig(**{**cfg.__dict__, "mode": "ssl"})
    ssl_dataset = build_fsd50k_dataset(ssl_cfg, split="train")
    ssl_sample = ssl_dataset[0]
    assert ssl_sample.input_values.shape == (48, 32)
    assert ssl_sample.labels is None
    assert ssl_sample.clip_id == "clip_train"


def test_fsd50k_unknown_label_raises(tmp_path: Path) -> None:
    cfg = _fsd_fixture(tmp_path, unknown_label=True)
    with pytest.raises(ValueError, match="missing from vocabulary"):
        build_fsd50k_dataset(cfg, split="val")


def test_fsd50k_val_crop_is_deterministic(tmp_path: Path) -> None:
    cfg = _fsd_fixture(tmp_path)
    dataset = build_fsd50k_dataset(cfg, split="val")
    assert torch.allclose(dataset[0].input_values, dataset[0].input_values)


def test_ssl_patch_grid_and_mask_helpers() -> None:
    branches = [
        ((16, 16), (8, 16), (127, 8)),
        ((8, 32), (4, 32), (255, 4)),
        ((4, 64), (2, 64), (511, 2)),
        ((2, 128), (1, 128), (1023, 1)),
    ]
    for patch_size, stride, expected in branches:
        assert compute_patch_grid_size(1024, 128, patch_size, stride) == expected
    token_mask = sample_token_mask(8, 10, 10, 0.4, torch.device("cpu"))
    assert token_mask.shape == (8, 10, 10)
    assert torch.isclose(token_mask.float().mean(), torch.tensor(0.4))
    fbank_mask = token_grid_to_fbank_mask(
        token_mask[:, :2, :2],
        T=32,
        F_bins=32,
        patch_size=(8, 8),
        stride=(4, 8),
    )
    assert fbank_mask.shape == (8, 32, 32)


def test_masked_mse_uses_only_masked_bins() -> None:
    recon = torch.tensor([[[1.0, 10.0], [3.0, 4.0]]])
    target = torch.zeros_like(recon)
    mask = torch.tensor([[[True, False], [True, False]]])
    assert torch.isclose(masked_mse(recon, target, mask), torch.tensor(5.0))


def test_ssl_wrapper_forward_returns_branch_losses_and_reconstructions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _small_model(num_classes=2)
    wrapper = MaskedFbankSSLWrapper(
        model,
        MaskedFbankSSLConfig(
            masking=SslMaskingConfig(
                token_mask_ratio=1.0,
                actual_mask_ratio_warning_threshold=0.01,
            ),
            decoder=SslDecoderConfig(
                channels=8, kernel_size=3, num_layers=2, dropout=0.0
            ),
            return_reconstructions=True,
        ),
    )
    output = wrapper(torch.randn(2, 32, 32))
    assert output.loss.ndim == 0
    assert output.branch_losses.shape == (4,)
    assert output.actual_mask_ratios.shape == (4,)
    assert output.token_mask_ratios.shape == (4,)
    assert output.reconstructions is not None
    assert output.reconstructions[0].shape == (2, 32, 32)
    assert "actual_fbank_mask_ratio" in caplog.text


def test_branch_conv1d_decoder_shape() -> None:
    decoder = BranchConv1dFbankDecoder(
        hidden_size=16,
        freq_bins=32,
        channels=8,
        kernel_size=3,
        num_layers=2,
        dropout=0.0,
    )
    assert decoder(torch.randn(2, 7, 16), output_frames=32).shape == (2, 32, 32)


def test_multilabel_model_and_pos_weight() -> None:
    model = _small_model(num_classes=200)
    output = model(torch.randn(2, 32, 32))
    assert output.logits.shape == (2, 200)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    pos_weight = compute_sqrt_neg_pos_multilabel_pos_weight(targets, cap=10.0)
    expected = torch.sqrt(torch.tensor([2.0 / 1.0, 2.0 / 1.0, 0.0 / 3.0]))
    assert torch.allclose(pos_weight, expected)


def test_multilabel_metrics_exclude_zero_positive_classes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    y_true = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    y_prob = np.asarray([[0.8, 0.1, 0.9], [0.2, 0.7, 0.8]], dtype=np.float64)
    metrics = MultiLabelMetricsComputer.compute(y_true, y_prob)
    assert len(metrics.per_class_AP) == 3
    assert np.isnan(metrics.per_class_AP[2])
    assert metrics.micro_AP > 0
    assert "zero positives" in caplog.text


def test_ssl_checkpoint_loads_into_supervised_model_and_transfer_freezes(
    tmp_path: Path,
) -> None:
    ssl_model = _small_model(num_classes=2)
    wrapper = MaskedFbankSSLWrapper(
        ssl_model,
        MaskedFbankSSLConfig(decoder=SslDecoderConfig(channels=8, kernel_size=3)),
    )
    checkpoint_path = tmp_path / "ssl.pt"
    torch.save(
        {
            "model_state_dict": wrapper.state_dict(),
            "base_model_state_dict": ssl_model.state_dict(),
        },
        checkpoint_path,
    )
    supervised_model = _small_model(num_classes=200)
    summary = load_compatible_model_state(
        model=supervised_model,
        checkpoint_path=str(checkpoint_path),
        reset_classifier=True,
        strict=False,
    )
    assert any(key.startswith("encoder.") for key in summary.loaded_keys)
    assert any("classifier" in key for key in summary.skipped_keys)

    freeze_summary = apply_cnuh_transfer_freeze(
        supervised_model,
        freeze_encoder=True,
        freeze_modules=(
            "patch_tokenizers",
            "position_embeddings",
            "scale_embeddings",
            "shared_stem",
            "scale_specific_adapters",
            "frequency_attention_poolers",
        ),
    )
    assert any(
        name.startswith("encoder.patch_tokenizers")
        for name in freeze_summary.frozen_parameter_names
    )
    assert any(
        name.startswith("branch_mil_heads")
        for name in freeze_summary.trainable_parameter_names
    )
    transfer_optimizer, transfer_optimizer_summary = build_cnuh_transfer_optimizer(
        supervised_model,
        frozen_encoder_lr=0.0,
        body_lr=1e-5,
        head_lr=3e-4,
    )
    assert transfer_optimizer_summary.param_group_count == len(
        transfer_optimizer.param_groups
    )
    ml_optimizer, ml_summary = build_fsd50k_multilabel_optimizer(
        _small_model(num_classes=200),
        encoder_lr=1e-5,
        body_lr=3e-5,
        head_lr=3e-4,
        weight_decay=0.01,
    )
    assert ml_summary.param_group_count == len(ml_optimizer.param_groups) == 3


def test_new_fsd50k_configs_parse() -> None:
    ssl_cfg = JsonConfigLoader.load_fsd50k_ssl(
        "configs/fsd50k_ssl_masked_fbank_pretrain.json"
    )
    supervised_cfg = JsonConfigLoader.load_fsd50k_supervised(
        "configs/fsd50k_supervised_multilabel_finetune.json"
    )
    transfer_cfg = JsonConfigLoader.load_cnuh_transfer_template(
        "configs/cnuh_4class_from_fsd50k_transfer_template.json"
    )
    assert ssl_cfg.ssl.masking.token_mask_ratio == 0.4
    assert supervised_cfg.train.loss.branch_auxiliary.enabled is False
    assert supervised_cfg.train.loss.branch_binary_auxiliary.enabled is False
    assert transfer_cfg.transfer.reset_classifier is True
