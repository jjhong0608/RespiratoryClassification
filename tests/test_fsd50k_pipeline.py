from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile as sf
import torch
from src.cli.cnuh_transfer_train import prepare_cnuh_transfer_model_and_optimizer
from src.data.fsd50k_dataset import (
    Fsd50kBatch,
    build_fsd50k_dataset,
    parse_fsd50k_vocabulary,
)
from src.evaluation.metrics import (
    MultiLabelMetricsComputer,
    multilabel_topk_metrics,
    topk_label_frequency,
    true_label_frequency,
)
from src.models.classifier import ClassifierDims, LinearClassifier, MlpClassifier
from src.models.model import (
    AstFeatureDims,
    AstModelOutput,
    ClassifierBiasInitConfig,
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
from src.training.ast_setup import build_ast_model
from src.training.classifier_bias_init import (
    apply_classifier_bias_init,
    compute_classifier_bias_init,
    find_final_classifier_linear,
)
from src.training.multilabel_trainer import (
    MultiLabelTrainer,
    MultiLabelTrainerConfig,
    compute_multilabel_probability_stats,
    compute_multilabel_target_stats,
    compute_multilabel_topk_summary,
    compute_pos_weight_stats,
    compute_sqrt_neg_pos_multilabel_pos_weight,
)
from src.training.transfer import (
    apply_cnuh_transfer_freeze,
    build_cnuh_transfer_optimizer,
    build_fsd50k_multilabel_optimizer,
    load_compatible_model_state,
)
from src.utils.config import (
    AnalysisConfig,
    AnalysisOutputConfig,
    AstFbankConfig,
    AudioConfig,
    CnuhTransferConfig,
    CnuhTransferOptimizerConfig,
    DataAugmentationConfig,
    Fsd50kDataConfig,
    Fsd50kTopKMetricsConfig,
    JsonConfigLoader,
    ModelConfig,
    ModelEncoderConfig,
    PreprocessingConfig,
)
from torch import nn
from torch.utils.data import DataLoader

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


def _write_headerless_vocabulary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row["index"], row["display_name"], row["mid"]])


def _fsd_fixture(
    tmp_path: Path,
    *,
    unknown_label: bool = False,
    headerless_vocabulary: bool = False,
) -> Fsd50kDataConfig:
    dev_audio = tmp_path / "dev_audio"
    eval_audio = tmp_path / "eval_audio"
    gt = tmp_path / "ground_truth"
    dev_audio.mkdir(parents=True)
    eval_audio.mkdir(parents=True)
    gt.mkdir(parents=True)
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
    if headerless_vocabulary:
        _write_headerless_vocabulary(gt / "vocabulary.csv", vocab_rows)
    else:
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


def test_fsd50k_vocabulary_parses_headered_and_headerless_formats(
    tmp_path: Path,
) -> None:
    headered_cfg = _fsd_fixture(tmp_path / "headered")
    headered = parse_fsd50k_vocabulary(Path(headered_cfg.vocabulary_csv))
    assert len(headered.index_to_label) == 200
    assert headered.label_to_index["label_000"] == 0
    assert headered.mid_to_index["/m/199"] == 199

    headerless_cfg = _fsd_fixture(
        tmp_path / "headerless",
        headerless_vocabulary=True,
    )
    headerless = parse_fsd50k_vocabulary(Path(headerless_cfg.vocabulary_csv))
    assert len(headerless.index_to_label) == 200
    assert headerless.index_to_label[1] == "label_001"
    assert headerless.index_to_mid[1] == "/m/001"
    assert headerless.mid_to_index["/m/199"] == 199


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


def _transfer_model_cfg() -> ModelConfig:
    return ModelConfig(
        encoder=ModelEncoderConfig(
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
                    type="branch_gated",
                    dropout=0.0,
                ),
            ),
        ),
        classifier=ClassifierConfig(type="linear", hidden_dim=16, dropout=0.0),
    )


class _TinyMultiLabelModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, num_classes)

    def forward(self, input_values: torch.Tensor) -> AstModelOutput:
        features = input_values.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        logits = self.linear(features)
        return AstModelOutput(logits=logits, pooled_embedding=logits)


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


def test_fsd50k_headerless_vocabulary_dataset_outputs(tmp_path: Path) -> None:
    cfg = _fsd_fixture(tmp_path, headerless_vocabulary=True)
    supervised = build_fsd50k_dataset(cfg, split="train")
    supervised_sample = supervised[0]
    assert supervised_sample.labels is not None
    assert supervised_sample.labels[0] == 1
    assert supervised_sample.labels[1] == 1

    ssl_cfg = Fsd50kDataConfig(**{**cfg.__dict__, "mode": "ssl"})
    ssl_dataset = build_fsd50k_dataset(ssl_cfg, split="train")
    ssl_sample = ssl_dataset[0]
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


def _targets_from_counts(
    positive_counts: list[int],
    *,
    total_samples: int,
) -> torch.Tensor:
    targets = torch.zeros(total_samples, len(positive_counts), dtype=torch.float32)
    for class_index, count in enumerate(positive_counts):
        targets[:count, class_index] = 1.0
    return targets


def test_classifier_bias_init_formulas_and_clamping() -> None:
    train_targets = _targets_from_counts([100, 1000], total_samples=10000)
    pos_weight = torch.tensor([10.0, 3.0])
    cfg = ClassifierBiasInitConfig(
        enabled=True,
        type="weighted_prior",
        eps=1e-6,
        clamp_min=-100.0,
        clamp_max=100.0,
    )

    bias, metadata = compute_classifier_bias_init(
        train_targets=train_targets,
        pos_weight=pos_weight,
        cfg=cfg,
        class_names=("rare", "common"),
    )

    expected = torch.log(
        (pos_weight * torch.tensor([100.0, 1000.0]) + cfg.eps)
        / (torch.tensor([9900.0, 9000.0]) + cfg.eps)
    )
    assert torch.allclose(bias, expected)
    assert metadata["type"] == "weighted_prior"
    assert metadata["class_examples"][0]["class_name"] == "rare"

    prior_cfg = ClassifierBiasInitConfig(
        enabled=True,
        type="prior",
        eps=1e-6,
        clamp_min=-100.0,
        clamp_max=100.0,
    )
    prior_bias, _ = compute_classifier_bias_init(
        train_targets=train_targets,
        pos_weight=pos_weight,
        cfg=prior_cfg,
    )
    prior_expected = torch.log(
        (torch.tensor([100.0, 1000.0]) + prior_cfg.eps)
        / (torch.tensor([9900.0, 9000.0]) + prior_cfg.eps)
    )
    assert torch.allclose(prior_bias, prior_expected)

    clamped_cfg = ClassifierBiasInitConfig(
        enabled=True,
        type="prior",
        eps=1e-6,
        clamp_min=-1.0,
        clamp_max=1.0,
    )
    clamped_bias, clamped_metadata = compute_classifier_bias_init(
        train_targets=_targets_from_counts([1, 9999], total_samples=10000),
        pos_weight=pos_weight,
        cfg=clamped_cfg,
    )
    assert torch.all(clamped_bias >= -1.0)
    assert torch.all(clamped_bias <= 1.0)
    assert clamped_metadata["num_clamped_min"] == 1
    assert clamped_metadata["num_clamped_max"] == 1


def test_classifier_bias_init_applies_only_to_final_classifier_layer() -> None:
    bias = torch.linspace(-2.0, 2.0, steps=200)
    linear_classifier = LinearClassifier(
        ClassifierDims(in_dim=16, num_classes=200, hidden_dim=16)
    )
    assert linear_classifier.proj.bias is not None
    linear_weight_before = linear_classifier.proj.weight.detach().clone()

    target_name = apply_classifier_bias_init(
        linear_classifier,
        bias=bias,
        num_classes=200,
    )

    assert target_name == "proj"
    assert torch.allclose(linear_classifier.proj.bias, bias)
    assert torch.allclose(linear_classifier.proj.weight, linear_weight_before)

    mlp_classifier = MlpClassifier(
        ClassifierDims(in_dim=16, num_classes=200, hidden_dim=32)
    )
    first_linear = mlp_classifier.net[0]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.bias is not None
    first_weight_before = first_linear.weight.detach().clone()
    first_bias_before = first_linear.bias.detach().clone()

    target_name = apply_classifier_bias_init(
        mlp_classifier,
        bias=bias,
        num_classes=200,
    )

    assert target_name == "net.3"
    final_linear = mlp_classifier.net[3]
    assert isinstance(final_linear, nn.Linear)
    assert final_linear.bias is not None
    assert torch.allclose(final_linear.bias, bias)
    assert torch.allclose(first_linear.weight, first_weight_before)
    assert torch.allclose(first_linear.bias, first_bias_before)


def test_classifier_bias_init_rejects_ambiguous_final_classifier() -> None:
    classifier = nn.Sequential(
        nn.Linear(8, 200),
        nn.Linear(200, 200),
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        find_final_classifier_linear(classifier, num_classes=200)


def test_multilabel_monitoring_stats_helpers() -> None:
    train_targets = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    val_targets = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    target_stats = compute_multilabel_target_stats(
        train_targets,
        val_targets,
        class_names=("a", "b", "c", "d"),
    )
    assert target_stats["train_samples"] == 3
    assert target_stats["val_samples"] == 2
    assert target_stats["num_classes"] == 4
    assert target_stats["train_mean_labels_per_sample"] == pytest.approx(5 / 3)
    assert target_stats["train_min_labels_per_sample"] == pytest.approx(1.0)
    assert target_stats["train_max_labels_per_sample"] == pytest.approx(2.0)
    assert target_stats["val_mean_labels_per_sample"] == pytest.approx(1.5)
    assert target_stats["train_num_classes_with_positive_count"] == 3
    assert target_stats["val_num_classes_with_positive_count"] == 3
    assert target_stats["top10_positive_class_counts"][0] == {
        "class_index": 0,
        "count": 3,
        "class_name": "a",
    }
    assert target_stats["bottom10_nonzero_positive_class_counts"][0]["count"] == 1

    probabilities = np.asarray(
        [
            [0.9, 0.2, 0.4],
            [0.1, 0.8, 0.6],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    prob_stats = compute_multilabel_probability_stats(probabilities, targets)
    assert prob_stats["prob_mean"] == pytest.approx(float(probabilities.mean()))
    assert prob_stats["prob_std"] == pytest.approx(float(probabilities.std()))
    assert prob_stats["prob_max_mean"] == pytest.approx(0.85)
    assert prob_stats["prob_max_p95"] == pytest.approx(
        float(np.percentile([0.9, 0.8], 95))
    )
    assert prob_stats["mean_predicted_positives_at_0_5"] == pytest.approx(1.5)
    assert prob_stats["mean_predicted_positives_at_0_3"] == pytest.approx(2.0)
    assert prob_stats["mean_predicted_positives_at_0_1"] == pytest.approx(3.0)
    assert prob_stats["mean_true_positives"] == pytest.approx(1.5)

    pos_weight_stats = compute_pos_weight_stats(
        torch.tensor([1.0, 2.0, 10.0, 10.0]),
        cap=10.0,
    )
    assert pos_weight_stats["pos_weight_min"] == pytest.approx(1.0)
    assert pos_weight_stats["pos_weight_mean"] == pytest.approx(5.75)
    assert pos_weight_stats["pos_weight_median"] == pytest.approx(6.0)
    assert pos_weight_stats["pos_weight_max"] == pytest.approx(10.0)
    assert pos_weight_stats["num_capped_classes"] == 2


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


def test_multilabel_topk_metrics_correctness_and_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    y_true = torch.tensor(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    y_score = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.1, 0.0],
            [0.6, 0.5, 0.4, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )

    metrics = multilabel_topk_metrics(y_true, y_score, ks=(1, 3, 10))

    assert metrics["hit_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.25)
    assert metrics["hit_at_3"] == pytest.approx(1.0)
    assert metrics["recall_at_3"] == pytest.approx(0.75)
    assert metrics["hit_at_10"] == pytest.approx(1.0)
    assert metrics["recall_at_10"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="shape"):
        multilabel_topk_metrics(y_true[0], y_score[0])
    with pytest.raises(ValueError, match="Shape mismatch"):
        multilabel_topk_metrics(y_true, y_score[:, :4])

    with_empty = torch.cat([y_true, torch.zeros(1, 5)], dim=0)
    scores_with_empty = torch.cat([y_score, torch.ones(1, 5)], dim=0)
    empty_metrics = multilabel_topk_metrics(with_empty, scores_with_empty, ks=(1,))
    assert empty_metrics["hit_at_1"] == pytest.approx(0.5)
    assert "empty-label samples" in caplog.text


def test_topk_label_frequency_helpers_attach_class_names() -> None:
    y_score = torch.tensor(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.7, 0.2, 0.6, 0.1],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    class_names = ("a", "b", "c", "d")

    top1 = topk_label_frequency(y_score, class_names, k=1, top_n=3)
    assert top1 == [{"class_index": 0, "class_name": "a", "count": 2}]

    true_top = true_label_frequency(y_true, class_names, top_n=2)
    assert true_top[0] == {"class_index": 2, "class_name": "c", "count": 2}
    assert true_top[1]["count"] == 1

    summary = compute_multilabel_topk_summary(
        y_score.numpy(),
        y_true.numpy(),
        cfg=Fsd50kTopKMetricsConfig(ks=(1, 3), top_n_frequency=2),
        class_names=class_names,
    )
    assert summary["topk_metrics"]["hit_at_1"] == pytest.approx(0.5)
    assert "top5_label_frequency" in summary
    assert "true_label_frequency_top20" in summary


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


def test_cnuh_transfer_template_parses_as_training_and_transfer() -> None:
    training_cfg = JsonConfigLoader.load_training(
        "configs/cnuh_4class_from_fsd50k_transfer_template.json"
    )
    transfer_cfg = JsonConfigLoader.load_cnuh_transfer_template(
        "configs/cnuh_4class_from_fsd50k_transfer_template.json"
    )

    assert training_cfg.experiment.name == "respiratory_classification_with_CNUH_data"
    assert training_cfg.data.label_to_index == {
        "normal": 0,
        "crackle": 1,
        "wheeze": 2,
        "rhonchi": 3,
    }
    assert transfer_cfg.transfer.reset_classifier is True
    assert transfer_cfg.transfer.strict is False
    assert transfer_cfg.optimizer.frozen_encoder_lr == 0.0
    assert transfer_cfg.optimizer.body_lr == pytest.approx(1e-5)
    assert transfer_cfg.optimizer.head_lr == pytest.approx(3e-4)


def test_cnuh_transfer_setup_loads_freezes_and_builds_optimizer(
    tmp_path: Path,
) -> None:
    source_model = build_ast_model(
        _transfer_model_cfg(),
        num_mel_bins=32,
        max_length=32,
        num_classes=200,
    )
    checkpoint_path = tmp_path / "fsd50k_supervised.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    setup = prepare_cnuh_transfer_model_and_optimizer(
        _transfer_model_cfg(),
        CnuhTransferConfig(
            checkpoint_path=str(checkpoint_path),
            reset_classifier=True,
            freeze_encoder=True,
            strict=False,
            freeze_modules=(
                "patch_tokenizers",
                "position_embeddings",
                "scale_embeddings",
                "shared_stem",
                "scale_specific_adapters",
                "frequency_attention_poolers",
            ),
        ),
        CnuhTransferOptimizerConfig(
            frozen_encoder_lr=0.0,
            body_lr=1e-5,
            head_lr=3e-4,
        ),
        num_mel_bins=32,
        max_length=32,
        num_classes=4,
        weight_decay=0.01,
    )

    assert any(
        key.startswith("encoder.") for key in setup.transfer_load_summary.loaded_keys
    )
    assert any("classifier" in key for key in setup.transfer_load_summary.skipped_keys)
    assert any(
        name.startswith("encoder.patch_tokenizers")
        for name in setup.freeze_summary.frozen_parameter_names
    )
    assert any(
        name.startswith("branch_mil_heads")
        for name in setup.freeze_summary.trainable_parameter_names
    )
    assert [group["name"] for group in setup.optimizer.param_groups] == [
        "body",
        "head",
    ]
    assert setup.optimizer_summary.param_group_count == 2


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
    assert not hasattr(ssl_cfg, "analysis")
    assert ssl_cfg.terminal.width is None
    assert supervised_cfg.train.loss.branch_auxiliary.enabled is False
    assert supervised_cfg.train.loss.branch_binary_auxiliary.enabled is False
    assert supervised_cfg.model.classifier.bias_init.enabled is True
    assert supervised_cfg.model.classifier.bias_init.type == "weighted_prior"
    assert supervised_cfg.metrics.topk.enabled is True
    assert supervised_cfg.metrics.topk.ks == (5, 10)
    assert supervised_cfg.metrics.topk.save_label_frequency is True
    assert supervised_cfg.metrics.topk.top_n_frequency == 20
    assert supervised_cfg.terminal.width == 250
    assert supervised_cfg.analysis.outputs.save_logits is True
    assert supervised_cfg.analysis.outputs.save_probabilities is True
    assert supervised_cfg.analysis.outputs.save_embeddings is False
    assert supervised_cfg.analysis.outputs.save_clip_metadata is True
    assert transfer_cfg.transfer.reset_classifier is True


@pytest.mark.parametrize(
    ("topk_payload", "match"),
    [
        ({"enabled": "yes"}, "metrics.topk.enabled"),
        ({"ks": []}, "metrics.topk.ks"),
        ({"ks": [5, 0]}, "metrics.topk.ks"),
        ({"ks": [5, True]}, "metrics.topk.ks"),
        ({"save_label_frequency": 1}, "metrics.topk.save_label_frequency"),
        ({"top_n_frequency": 0}, "metrics.topk.top_n_frequency"),
    ],
)
def test_invalid_fsd50k_topk_metrics_config_is_rejected(
    tmp_path: Path,
    topk_payload: dict,
    match: str,
) -> None:
    payload = json.loads(
        Path("configs/fsd50k_supervised_multilabel_finetune.json").read_text(
            encoding="utf-8"
        )
    )
    payload["metrics"]["topk"] = topk_payload
    config_path = tmp_path / "bad_fsd50k_topk.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        JsonConfigLoader.load_fsd50k_supervised(config_path)


def test_fsd50k_topk_metrics_config_defaults_when_omitted(tmp_path: Path) -> None:
    payload = json.loads(
        Path("configs/fsd50k_supervised_multilabel_finetune.json").read_text(
            encoding="utf-8"
        )
    )
    del payload["metrics"]["topk"]
    config_path = tmp_path / "fsd50k_without_topk.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = JsonConfigLoader.load_fsd50k_supervised(config_path)

    assert cfg.metrics.topk.enabled is True
    assert cfg.metrics.topk.ks == (5, 10)
    assert cfg.metrics.topk.save_label_frequency is True
    assert cfg.metrics.topk.top_n_frequency == 20


def _multilabel_batch() -> Fsd50kBatch:
    return Fsd50kBatch(
        input_values=torch.randn(2, 8, 8),
        labels=torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        clip_ids=("clip_train", "clip_val"),
        audio_paths=("clip_train.wav", "clip_val.wav"),
    )


def _single_batch_loader() -> DataLoader[Fsd50kBatch]:
    return cast(DataLoader[Fsd50kBatch], [_multilabel_batch()])


def test_multilabel_trainer_writes_validation_diagnostics(tmp_path: Path) -> None:
    model = _TinyMultiLabelModel(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = MultiLabelTrainer(
        MultiLabelTrainerConfig(
            device="cpu",
            epochs=1,
            warmup_ratio=0.0,
            max_grad_norm=None,
            run_dir=tmp_path,
            threshold=0.5,
            analysis=AnalysisConfig(
                outputs=AnalysisOutputConfig(
                    save_logits=False,
                    save_probabilities=False,
                    save_embeddings=False,
                    save_clip_metadata=True,
                )
            ),
            class_names=("label_a", "label_b", "label_c"),
        ),
        pos_weight=torch.ones(3),
    )

    trainer.fit(
        model,
        _single_batch_loader(),
        _single_batch_loader(),
        optimizer,
        extra_state={
            "classifier_bias_init": {
                "enabled": True,
                "type": "weighted_prior",
                "values": [0.0, 0.1, 0.2],
            }
        },
    )

    diagnostics_path = tmp_path / "diagnostics" / "val_epoch_001.jsonl"
    diagnostics_summary_path = tmp_path / "diagnostics" / "val_epoch_001_summary.json"
    assert diagnostics_path.exists()
    assert diagnostics_summary_path.exists()
    rows = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["clip_id"] == "clip_train"
    assert "true_label_names" in rows[0]
    assert "top5_label_names" in rows[0]
    assert "logits" not in rows[0]
    assert "probabilities" not in rows[0]
    summary = json.loads(diagnostics_summary_path.read_text(encoding="utf-8"))
    assert summary["record_type"] == "summary"
    assert summary["epoch"] == 1
    assert "hit_at_5" in summary["topk_metrics"]
    assert "top5_label_frequency" in summary
    assert "top10_label_frequency" in summary
    assert "true_label_frequency_top20" in summary
    checkpoint = torch.load(
        tmp_path / "last.pt", map_location="cpu", weights_only=False
    )
    assert len(checkpoint["val_probability_stats"]) == 1
    assert "prob_mean" in checkpoint["val_probability_stats"][0]
    assert (
        "mean_predicted_positives_at_0_5" in checkpoint["current_val_probability_stats"]
    )
    assert "hit_at_5" in checkpoint["val_metrics"][0]
    assert "val_hit_at_5" in checkpoint
    assert len(checkpoint["val_topk_summaries"]) == 1
    assert checkpoint["current_val_topk_summary"]["topk_metrics"]["hit_at_5"] == 1.0
    assert checkpoint["diagnostics_summary_path"] == str(diagnostics_summary_path)
    assert checkpoint["classifier_bias_init"]["type"] == "weighted_prior"


def test_multilabel_trainer_skips_diagnostics_when_outputs_disabled(
    tmp_path: Path,
) -> None:
    model = _TinyMultiLabelModel(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = MultiLabelTrainer(
        MultiLabelTrainerConfig(
            device="cpu",
            epochs=1,
            warmup_ratio=0.0,
            max_grad_norm=None,
            run_dir=tmp_path,
            threshold=0.5,
            analysis=AnalysisConfig(
                outputs=AnalysisOutputConfig(
                    save_logits=False,
                    save_probabilities=False,
                    save_embeddings=False,
                    save_clip_metadata=False,
                )
            ),
            class_names=("label_a", "label_b", "label_c"),
        ),
        pos_weight=torch.ones(3),
    )

    trainer.fit(
        model,
        _single_batch_loader(),
        _single_batch_loader(),
        optimizer,
    )

    assert not (tmp_path / "diagnostics").exists()
    checkpoint = torch.load(
        tmp_path / "last.pt", map_location="cpu", weights_only=False
    )
    assert "hit_at_5" in checkpoint["current_val_topk_summary"]["topk_metrics"]
    assert checkpoint["diagnostics_summary_path"] is None
