from __future__ import annotations

import pytest
import torch
from src.data.augmentation import (
    AugmentationPipeline,
    AugmentationPolicy,
    FbankAugmenter,
    WaveformAugmenter,
)
from src.utils.config import (
    AdditiveNoiseConfig,
    AugmentationPolicyChoiceConfig,
    AugmentationPolicyConfig,
    DataAugmentationConfig,
    FbankAugmentationConfig,
    MaskConfig,
    RandomGainConfig,
    TimeShiftConfig,
    WaveformAugmentationConfig,
)


def _waveform_cfg(
    *,
    gain: RandomGainConfig | None = None,
    noise: AdditiveNoiseConfig | None = None,
    time_shift: TimeShiftConfig | None = None,
) -> DataAugmentationConfig:
    return DataAugmentationConfig(
        enabled=True,
        waveform=WaveformAugmentationConfig(
            enabled=True,
            probability=1.0,
            gain=gain or RandomGainConfig(enabled=False),
            noise=noise or AdditiveNoiseConfig(enabled=False),
            time_shift=time_shift or TimeShiftConfig(enabled=False),
        ),
        fbank=FbankAugmentationConfig(enabled=False),
    )


def _fbank_cfg(
    *,
    time_mask: MaskConfig | None = None,
    freq_mask: MaskConfig | None = None,
    mask_value: float = 0.0,
) -> DataAugmentationConfig:
    return DataAugmentationConfig(
        enabled=True,
        waveform=WaveformAugmentationConfig(enabled=False),
        fbank=FbankAugmentationConfig(
            enabled=True,
            probability=1.0,
            time_mask=time_mask or MaskConfig(enabled=False),
            freq_mask=freq_mask or MaskConfig(enabled=False),
            mask_value=mask_value,
        ),
    )


def _oneof_cfg(choice_name: str) -> DataAugmentationConfig:
    return DataAugmentationConfig(
        enabled=True,
        policy=AugmentationPolicyConfig(
            type="one_of",
            choices=(
                AugmentationPolicyChoiceConfig(
                    name=choice_name,  # type: ignore[arg-type]
                    probability=1.0,
                ),
            ),
        ),
        waveform=WaveformAugmentationConfig(
            enabled=True,
            probability=1.0,
            gain=RandomGainConfig(
                enabled=True,
                probability=1.0,
                min_db=6.0,
                max_db=6.0,
            ),
        ),
        fbank=FbankAugmentationConfig(
            enabled=True,
            probability=1.0,
            time_mask=MaskConfig(enabled=True, num_masks=1, max_width=4),
            mask_value=-9.0,
        ),
    )


def test_disabled_waveform_augmenter_returns_identical_waveform() -> None:
    waveform = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32)
    augmenter = WaveformAugmenter(DataAugmentationConfig(), sample_rate=16000)

    output = augmenter(waveform)

    assert output is waveform
    assert torch.equal(output, waveform)
    assert output.shape == waveform.shape
    assert output.dtype == waveform.dtype
    assert output.device == waveform.device


def test_forced_random_gain_changes_amplitude_and_preserves_shape() -> None:
    waveform = torch.ones(8, dtype=torch.float32)
    cfg = _waveform_cfg(
        gain=RandomGainConfig(
            enabled=True,
            probability=1.0,
            min_db=6.0,
            max_db=6.0,
        )
    )
    augmenter = WaveformAugmenter(cfg, sample_rate=16000)

    output = augmenter(waveform)

    assert output.shape == waveform.shape
    assert output.dtype == waveform.dtype
    assert output.device == waveform.device
    assert torch.all(output > waveform)
    assert torch.isfinite(output).all()


def test_forced_additive_noise_adds_finite_values() -> None:
    torch.manual_seed(0)
    waveform = torch.ones(64, dtype=torch.float32)
    cfg = _waveform_cfg(
        noise=AdditiveNoiseConfig(
            enabled=True,
            probability=1.0,
            snr_db_min=20.0,
            snr_db_max=20.0,
        )
    )
    augmenter = WaveformAugmenter(cfg, sample_rate=16000)

    output = augmenter(waveform)

    assert output.shape == waveform.shape
    assert output.dtype == waveform.dtype
    assert output.device == waveform.device
    assert not torch.allclose(output, waveform)
    assert torch.isfinite(output).all()


def test_forced_zero_pad_time_shift_preserves_length_and_pads() -> None:
    torch.manual_seed(0)
    waveform = torch.arange(4, dtype=torch.float32)
    cfg = _waveform_cfg(
        time_shift=TimeShiftConfig(
            enabled=True,
            probability=1.0,
            max_shift_fraction=0.5,
            mode="zero_pad",
        )
    )
    augmenter = WaveformAugmenter(cfg, sample_rate=16000)

    output = augmenter(waveform)

    assert output.shape == waveform.shape
    assert output.dtype == waveform.dtype
    assert output.device == waveform.device
    assert torch.equal(output, torch.tensor([0.0, 0.0, 1.0, 2.0]))
    assert torch.isfinite(output).all()


def test_disabled_fbank_augmenter_returns_identical_feature() -> None:
    feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    augmenter = FbankAugmenter(DataAugmentationConfig())

    output = augmenter(feature)

    assert output is feature
    assert torch.equal(output, feature)
    assert output.shape == feature.shape
    assert output.dtype == feature.dtype
    assert output.device == feature.device


def test_forced_time_mask_masks_time_rows_and_respects_mask_value() -> None:
    torch.manual_seed(0)
    feature = torch.ones(4, 4, dtype=torch.float32)
    cfg = _fbank_cfg(
        time_mask=MaskConfig(enabled=True, num_masks=1, max_width=4),
        mask_value=-9.0,
    )
    augmenter = FbankAugmenter(cfg)

    output = augmenter(feature)

    assert output.shape == feature.shape
    assert output.dtype == feature.dtype
    assert output.device == feature.device
    assert torch.any(torch.all(output == -9.0, dim=1))
    assert torch.isfinite(output).all()
    assert torch.equal(feature, torch.ones_like(feature))


def test_forced_frequency_mask_masks_frequency_columns() -> None:
    torch.manual_seed(0)
    feature = torch.ones(4, 4, dtype=torch.float32)
    cfg = _fbank_cfg(
        freq_mask=MaskConfig(enabled=True, num_masks=1, max_width=4),
        mask_value=-5.0,
    )
    augmenter = FbankAugmenter(cfg)

    output = augmenter(feature)

    assert output.shape == feature.shape
    assert output.dtype == feature.dtype
    assert output.device == feature.device
    assert torch.any(torch.all(output == -5.0, dim=0))
    assert torch.isfinite(output).all()


def test_oneof_policy_samples_only_allowed_choices() -> None:
    cfg = DataAugmentationConfig(
        enabled=True,
        policy=AugmentationPolicyConfig(
            type="one_of",
            choices=(
                AugmentationPolicyChoiceConfig(name="none", probability=0.25),
                AugmentationPolicyChoiceConfig(name="waveform", probability=0.25),
                AugmentationPolicyChoiceConfig(name="fbank", probability=0.25),
                AugmentationPolicyChoiceConfig(name="both_light", probability=0.25),
            ),
        ),
    )
    policy = AugmentationPolicy(cfg)

    choices = {policy.sample() for _ in range(100)}

    assert choices <= {"none", "waveform", "fbank", "both_light"}
    assert choices


@pytest.mark.parametrize(
    ("choice_name", "expect_waveform_change", "expect_fbank_change"),
    [
        ("none", False, False),
        ("waveform", True, False),
        ("fbank", False, True),
        ("both_light", True, True),
    ],
)
def test_oneof_pipeline_applies_only_selected_stages(
    choice_name: str,
    expect_waveform_change: bool,
    expect_fbank_change: bool,
) -> None:
    torch.manual_seed(1)
    pipeline = AugmentationPipeline(_oneof_cfg(choice_name), sample_rate=16000)
    waveform = torch.ones(4, dtype=torch.float32)
    feature = torch.ones(4, 4, dtype=torch.float32)

    choice = pipeline.sample_choice()
    waveform_output = pipeline.apply_waveform(waveform, choice)
    feature_output = pipeline.apply_fbank(feature, choice)

    assert choice == choice_name
    assert (not torch.equal(waveform_output, waveform)) is expect_waveform_change
    assert (not torch.equal(feature_output, feature)) is expect_fbank_change


def test_independent_pipeline_preserves_existing_stacked_behavior() -> None:
    torch.manual_seed(0)
    cfg = DataAugmentationConfig(
        enabled=True,
        waveform=WaveformAugmentationConfig(
            enabled=True,
            probability=1.0,
            gain=RandomGainConfig(
                enabled=True,
                probability=1.0,
                min_db=6.0,
                max_db=6.0,
            ),
        ),
        fbank=FbankAugmentationConfig(
            enabled=True,
            probability=1.0,
            time_mask=MaskConfig(enabled=True, num_masks=1, max_width=4),
            mask_value=-9.0,
        ),
    )
    pipeline = AugmentationPipeline(cfg, sample_rate=16000)
    waveform = torch.ones(4, dtype=torch.float32)
    feature = torch.ones(4, 4, dtype=torch.float32)

    choice = pipeline.sample_choice()
    waveform_output = pipeline.apply_waveform(waveform, choice)
    feature_output = pipeline.apply_fbank(feature, choice)

    assert choice == "independent"
    assert not torch.equal(waveform_output, waveform)
    assert not torch.equal(feature_output, feature)
