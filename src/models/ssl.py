from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.multiscale_rdt_ast import MultiScaleRdtAstModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SslMaskingConfig:
    type: Literal["branch_specific_token_grid_input_patch"] = (
        "branch_specific_token_grid_input_patch"
    )
    sampling: Literal["random"] = "random"
    token_mask_ratio: float = 0.4
    mask_value: float = 0.0
    actual_mask_ratio_warning_threshold: float = 0.85


@dataclass(frozen=True)
class SslDecoderConfig:
    type: Literal["branch_conv1d"] = "branch_conv1d"
    sharing: Literal["separate"] = "separate"
    channels: int = 256
    kernel_size: int = 5
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True


@dataclass(frozen=True)
class SslLossConfig:
    type: Literal["mean_branch_masked_mse"] = "mean_branch_masked_mse"


@dataclass(frozen=True)
class SslVisualizationConfig:
    enabled: bool = False
    num_examples: int = 8
    save_every_n_epochs: int = 10


@dataclass(frozen=True)
class MaskedFbankSSLConfig:
    target: Literal["normalized_fbank"] = "normalized_fbank"
    masking: SslMaskingConfig = field(default_factory=SslMaskingConfig)
    decoder: SslDecoderConfig = field(default_factory=SslDecoderConfig)
    loss: SslLossConfig = field(default_factory=SslLossConfig)
    visualization: SslVisualizationConfig = field(
        default_factory=SslVisualizationConfig
    )
    return_reconstructions: bool = False


@dataclass(frozen=True)
class MaskedFbankSSLOutput:
    loss: Tensor
    branch_losses: Tensor
    actual_mask_ratios: Tensor
    token_mask_ratios: Tensor
    reconstructions: list[Tensor] | None = None
    masks: list[Tensor] | None = None
    token_masks: list[Tensor] | None = None
    reconstructed_average: Tensor | None = None


def compute_patch_grid_size(
    T: int,
    F_bins: int,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, int]:
    patch_t, patch_f = patch_size
    stride_t, stride_f = stride
    if patch_t <= 0 or patch_f <= 0 or stride_t <= 0 or stride_f <= 0:
        raise ValueError("patch_size and stride values must be positive")
    if patch_t > T or patch_f > F_bins:
        raise ValueError("patch_size must fit inside the fbank shape")
    return ((T - patch_t) // stride_t) + 1, ((F_bins - patch_f) // stride_f) + 1


def sample_token_mask(
    batch_size: int,
    nt: int,
    nf: int,
    token_mask_ratio: float,
    device: torch.device,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if nt <= 0 or nf <= 0:
        raise ValueError("token grid dimensions must be greater than zero")
    if not (0.0 <= token_mask_ratio <= 1.0):
        raise ValueError("token_mask_ratio must be within [0, 1]")
    num_tokens = nt * nf
    num_mask = int(round(float(token_mask_ratio) * float(num_tokens)))
    token_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        if num_mask <= 0:
            continue
        order = torch.randperm(num_tokens, device=device)[:num_mask]
        token_mask[batch_index, order] = True
    return token_mask.view(batch_size, nt, nf)


def token_grid_to_fbank_mask(
    token_mask: Tensor,
    *,
    T: int,
    F_bins: int,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> Tensor:
    if token_mask.ndim != 3:
        raise ValueError(
            f"token_mask must have shape (B, Nt, Nf), got {tuple(token_mask.shape)}"
        )
    batch_size, nt, nf = token_mask.shape
    patch_t, patch_f = patch_size
    stride_t, stride_f = stride
    fbank_mask = torch.zeros(
        batch_size,
        T,
        F_bins,
        dtype=torch.bool,
        device=token_mask.device,
    )
    for time_index in range(nt):
        t0 = time_index * stride_t
        for freq_index in range(nf):
            f0 = freq_index * stride_f
            selected = token_mask[:, time_index, freq_index]
            if selected.any():
                fbank_mask[selected, t0 : t0 + patch_t, f0 : f0 + patch_f] = True
    return fbank_mask


def masked_mse(
    recon: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8
) -> Tensor:
    if recon.shape != target.shape or mask.shape != target.shape:
        raise ValueError(
            "recon, target, and mask must share shape; "
            f"got recon={tuple(recon.shape)} target={tuple(target.shape)} "
            f"mask={tuple(mask.shape)}"
        )
    diff2 = (recon - target).pow(2)
    mask_f = mask.to(dtype=diff2.dtype)
    return (diff2 * mask_f).sum() / mask_f.sum().clamp_min(eps)


class BranchConv1dFbankDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        freq_bins: int = 128,
        channels: int = 256,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm: nn.Module = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )
        self.proj = nn.Conv1d(hidden_size, channels, kernel_size=1)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.conv = nn.Sequential(*layers)
        self.out = nn.Conv1d(channels, freq_bins, kernel_size=1)

    def forward(self, tokens: Tensor, *, output_frames: int = 1024) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (B, T, D), got {tokens.shape}")
        x = tokens.transpose(1, 2)
        x = F.interpolate(x, size=output_frames, mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.conv(x)
        x = self.out(x)
        return x.transpose(1, 2)


class MaskedFbankSSLWrapper(nn.Module):
    def __init__(self, base_model: MultiScaleRdtAstModel, cfg: MaskedFbankSSLConfig):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        hidden_size = base_model.cfg.encoder.architecture.hidden_size
        freq_bins = base_model.cfg.encoder.feature_dims.num_mel_bins
        decoder_cfg = cfg.decoder
        if decoder_cfg.type != "branch_conv1d" or decoder_cfg.sharing != "separate":
            raise ValueError("SSL decoder must use separate branch_conv1d decoders")
        self.decoders = nn.ModuleList(
            BranchConv1dFbankDecoder(
                hidden_size=hidden_size,
                freq_bins=freq_bins,
                channels=decoder_cfg.channels,
                kernel_size=decoder_cfg.kernel_size,
                num_layers=decoder_cfg.num_layers,
                dropout=decoder_cfg.dropout,
                use_layer_norm=decoder_cfg.use_layer_norm,
            )
            for _ in base_model.cfg.encoder.architecture.patch_branches
        )

    def _sample_masks(
        self,
        input_values: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        batch_size, time_steps, freq_bins = input_values.shape
        token_masks: list[Tensor] = []
        fbank_masks: list[Tensor] = []
        token_mask_ratios: list[Tensor] = []
        actual_mask_ratios: list[Tensor] = []
        for branch in self.base_model.cfg.encoder.architecture.patch_branches:
            nt, nf = compute_patch_grid_size(
                time_steps,
                freq_bins,
                branch.patch_size,
                branch.stride,
            )
            token_mask = sample_token_mask(
                batch_size,
                nt,
                nf,
                self.cfg.masking.token_mask_ratio,
                input_values.device,
            )
            fbank_mask = token_grid_to_fbank_mask(
                token_mask,
                T=time_steps,
                F_bins=freq_bins,
                patch_size=branch.patch_size,
                stride=branch.stride,
            )
            token_masks.append(token_mask)
            fbank_masks.append(fbank_mask)
            token_mask_ratios.append(token_mask.float().mean())
            actual_mask_ratios.append(fbank_mask.float().mean())

        token_ratio_tensor = torch.stack(token_mask_ratios)
        actual_ratio_tensor = torch.stack(actual_mask_ratios)
        threshold = float(self.cfg.masking.actual_mask_ratio_warning_threshold)
        for branch_index, ratio in enumerate(actual_ratio_tensor.detach().cpu()):
            if float(ratio.item()) > threshold:
                logger.warning(
                    "SSL branch %d actual_fbank_mask_ratio %.6f exceeds threshold %.6f",
                    branch_index,
                    float(ratio.item()),
                    threshold,
                )
        return token_masks, fbank_masks, token_ratio_tensor, actual_ratio_tensor

    def forward(
        self,
        input_values: Tensor,
        *,
        return_reconstructions: bool | None = None,
    ) -> MaskedFbankSSLOutput:
        if input_values.ndim != 3:
            raise ValueError(
                f"input_values must have shape (B, T, F), got {tuple(input_values.shape)}"
            )
        expected_shape = (
            self.base_model.cfg.encoder.feature_dims.max_length,
            self.base_model.cfg.encoder.feature_dims.num_mel_bins,
        )
        actual_shape = (int(input_values.shape[1]), int(input_values.shape[2]))
        if actual_shape != expected_shape:
            raise ValueError(
                f"input_values shape {actual_shape} does not match {expected_shape}"
            )
        keep_reconstructions = (
            self.cfg.return_reconstructions
            if return_reconstructions is None
            else bool(return_reconstructions)
        )
        token_masks, fbank_masks, token_ratios, actual_ratios = self._sample_masks(
            input_values
        )
        masked_branch_inputs = [
            input_values.masked_fill(mask, float(self.cfg.masking.mask_value))
            for mask in fbank_masks
        ]
        branch_tokens = self.base_model.forward_branches_for_ssl(masked_branch_inputs)
        reconstructions: list[Tensor] = []
        branch_losses: list[Tensor] = []
        for decoder, tokens, mask in zip(
            self.decoders,
            branch_tokens,
            fbank_masks,
            strict=True,
        ):
            recon = decoder(tokens, output_frames=input_values.shape[1])
            branch_losses.append(masked_mse(recon, input_values, mask))
            if keep_reconstructions:
                reconstructions.append(recon)
        branch_loss_tensor = torch.stack(branch_losses)
        reconstructed_average = (
            torch.stack(reconstructions, dim=0).mean(dim=0)
            if keep_reconstructions and reconstructions
            else None
        )
        return MaskedFbankSSLOutput(
            loss=branch_loss_tensor.mean(),
            branch_losses=branch_loss_tensor,
            actual_mask_ratios=actual_ratios,
            token_mask_ratios=token_ratios,
            reconstructions=reconstructions if keep_reconstructions else None,
            masks=fbank_masks if keep_reconstructions else None,
            token_masks=token_masks if keep_reconstructions else None,
            reconstructed_average=reconstructed_average,
        )
