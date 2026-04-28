from __future__ import annotations

import torch
from torch import Tensor

from src.utils.config import DataAugmentationConfig

AugmentationChoice = str


class WaveformAugmenter:
    def __init__(self, cfg: DataAugmentationConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate

    @staticmethod
    def _passes(probability: float, *, device: torch.device) -> bool:
        return bool(torch.rand((), device=device).item() <= probability)

    @staticmethod
    def _uniform(
        low: float,
        high: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.empty((), device=device, dtype=dtype).uniform_(low, high)

    @staticmethod
    def _ensure_finite(waveform: Tensor) -> Tensor:
        if not torch.isfinite(waveform).all():
            raise ValueError("Waveform augmentation produced non-finite values")
        return waveform

    def _apply_gain(self, waveform: Tensor) -> Tensor:
        cfg = self.cfg.waveform.gain
        gain_db = self._uniform(
            cfg.min_db,
            cfg.max_db,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        gain = torch.pow(
            torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype),
            gain_db / 20.0,
        )
        return waveform * gain

    def _apply_noise(self, waveform: Tensor) -> Tensor:
        cfg = self.cfg.waveform.noise
        signal_power = waveform.pow(2).mean().clamp_min(1e-12)
        snr_db = self._uniform(
            cfg.snr_db_min,
            cfg.snr_db_max,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        noise_power = signal_power / torch.pow(
            torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype),
            snr_db / 10.0,
        )
        return waveform + torch.randn_like(waveform) * noise_power.sqrt()

    def _apply_time_shift(self, waveform: Tensor) -> Tensor:
        cfg = self.cfg.waveform.time_shift
        max_shift = int(round(cfg.max_shift_fraction * waveform.numel()))
        if max_shift == 0:
            return waveform
        shift = int(
            torch.randint(
                low=-max_shift,
                high=max_shift + 1,
                size=(1,),
                device=waveform.device,
            ).item()
        )
        if shift == 0:
            return waveform
        if cfg.mode == "roll":
            return torch.roll(waveform, shifts=shift, dims=0)

        shifted = torch.zeros_like(waveform)
        if shift > 0:
            shifted[shift:] = waveform[:-shift]
        else:
            shifted[:shift] = waveform[-shift:]
        return shifted

    def __call__(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 1:
            raise ValueError(
                f"waveform must have shape (T,), got {tuple(waveform.shape)}"
            )
        if not self.cfg.enabled or not self.cfg.waveform.enabled:
            return waveform
        if not self._passes(self.cfg.waveform.probability, device=waveform.device):
            return waveform

        output = waveform
        gain = self.cfg.waveform.gain
        if gain.enabled and self._passes(gain.probability, device=waveform.device):
            output = self._apply_gain(output)
        noise = self.cfg.waveform.noise
        if noise.enabled and self._passes(noise.probability, device=waveform.device):
            output = self._apply_noise(output)
        time_shift = self.cfg.waveform.time_shift
        if time_shift.enabled and self._passes(
            time_shift.probability,
            device=waveform.device,
        ):
            output = self._apply_time_shift(output)
        return self._ensure_finite(output)


class FbankAugmenter:
    def __init__(self, cfg: DataAugmentationConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _passes(probability: float, *, device: torch.device) -> bool:
        return bool(torch.rand((), device=device).item() <= probability)

    @staticmethod
    def _sample_width(max_width: int, limit: int, *, device: torch.device) -> int:
        if max_width <= 0 or limit <= 0:
            return 0
        width = int(
            torch.randint(
                low=0,
                high=max_width + 1,
                size=(1,),
                device=device,
            ).item()
        )
        return min(width, limit)

    @staticmethod
    def _sample_start(limit: int, width: int, *, device: torch.device) -> int:
        if width <= 0:
            return 0
        return int(
            torch.randint(
                low=0,
                high=(limit - width) + 1,
                size=(1,),
                device=device,
            ).item()
        )

    @staticmethod
    def _ensure_finite(feature: Tensor) -> Tensor:
        if not torch.isfinite(feature).all():
            raise ValueError("Fbank augmentation produced non-finite values")
        return feature

    def _apply_time_masks(self, feature: Tensor) -> Tensor:
        cfg = self.cfg.fbank.time_mask
        time_steps = int(feature.shape[0])
        for _ in range(cfg.num_masks):
            width = self._sample_width(cfg.max_width, time_steps, device=feature.device)
            if width == 0:
                continue
            start = self._sample_start(time_steps, width, device=feature.device)
            feature[start : start + width, :] = self.cfg.fbank.mask_value
        return feature

    def _apply_frequency_masks(self, feature: Tensor) -> Tensor:
        cfg = self.cfg.fbank.freq_mask
        freq_steps = int(feature.shape[1])
        for _ in range(cfg.num_masks):
            width = self._sample_width(cfg.max_width, freq_steps, device=feature.device)
            if width == 0:
                continue
            start = self._sample_start(freq_steps, width, device=feature.device)
            feature[:, start : start + width] = self.cfg.fbank.mask_value
        return feature

    def __call__(self, feature: Tensor) -> Tensor:
        if feature.ndim != 2:
            raise ValueError(
                f"feature must have shape (T, F), got {tuple(feature.shape)}"
            )
        if not self.cfg.enabled or not self.cfg.fbank.enabled:
            return feature
        if not self._passes(self.cfg.fbank.probability, device=feature.device):
            return feature

        output = feature.clone()
        if self.cfg.fbank.time_mask.enabled:
            output = self._apply_time_masks(output)
        if self.cfg.fbank.freq_mask.enabled:
            output = self._apply_frequency_masks(output)
        return self._ensure_finite(output)


class AugmentationPolicy:
    def __init__(self, cfg: DataAugmentationConfig) -> None:
        self.cfg = cfg
        self.policy = cfg.policy

    @property
    def is_one_of(self) -> bool:
        return self.policy.type == "one_of"

    def sample(self) -> AugmentationChoice:
        if not self.cfg.enabled or not self.is_one_of:
            return "independent"
        names = [choice.name for choice in self.policy.choices]
        probabilities = torch.tensor(
            [choice.probability for choice in self.policy.choices],
            dtype=torch.float32,
        )
        selected_index = int(torch.multinomial(probabilities, num_samples=1).item())
        return names[selected_index]


class AugmentationPipeline:
    def __init__(self, cfg: DataAugmentationConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.policy = AugmentationPolicy(cfg)
        self.waveform_augmenter = (
            WaveformAugmenter(cfg, sample_rate=sample_rate)
            if cfg.enabled and cfg.waveform.enabled
            else None
        )
        self.fbank_augmenter = (
            FbankAugmenter(cfg) if cfg.enabled and cfg.fbank.enabled else None
        )

    @property
    def waveform_enabled(self) -> bool:
        return self.waveform_augmenter is not None

    @property
    def fbank_enabled(self) -> bool:
        return self.fbank_augmenter is not None

    def sample_choice(self) -> AugmentationChoice:
        return self.policy.sample()

    def apply_waveform(self, waveform: Tensor, choice: AugmentationChoice) -> Tensor:
        if self.waveform_augmenter is None:
            return waveform
        if self.policy.is_one_of and choice not in {"waveform", "both_light"}:
            return waveform
        return self.waveform_augmenter(waveform)

    def apply_fbank(self, feature: Tensor, choice: AugmentationChoice) -> Tensor:
        if self.fbank_augmenter is None:
            return feature
        if self.policy.is_one_of and choice not in {"fbank", "both_light"}:
            return feature
        return self.fbank_augmenter(feature)
