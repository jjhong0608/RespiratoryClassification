from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class WhisperEncoderDims:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int


@dataclass(frozen=True)
class AudioEncoderOutput:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> Tensor:
    if channels % 2 != 0:
        raise ValueError("channels must be even for sinusoidal positions")
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (
        channels // 2 - 1
    )
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.out(self._qkv_attention(q, k, v))

    def _qkv_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        n_batch, n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head
        scale = head_dim**-0.5

        q = q.view(n_batch, n_ctx, self.n_head, head_dim).transpose(1, 2)
        k = k.view(n_batch, n_ctx, self.n_head, head_dim).transpose(1, 2)
        v = v.view(n_batch, n_ctx, self.n_head, head_dim).transpose(1, 2)

        attn = (q * scale) @ (k * scale).transpose(-1, -2)
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = (attn @ v).transpose(1, 2).contiguous().view(n_batch, n_ctx, n_state)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    positional_embedding: Tensor

    def __init__(self, dims: WhisperEncoderDims):
        super().__init__()
        self.dims = dims
        self.conv1 = Conv1d(dims.n_mels, dims.n_audio_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(
            dims.n_audio_state,
            dims.n_audio_state,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.register_buffer(
            "positional_embedding",
            sinusoids(dims.n_audio_ctx, dims.n_audio_state),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(dims.n_audio_state, dims.n_audio_head)
                for _ in range(dims.n_audio_layer)
            ]
        )
        self.ln_post = LayerNorm(dims.n_audio_state)

    def forward(
        self,
        x: Tensor,
        *,
        output_hidden_states: bool = False,
        prefix_tokens: Tensor | None = None,
    ) -> AudioEncoderOutput:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        if x.shape[1:] != self.positional_embedding.shape:
            raise ValueError(
                f"Expected audio ctx/state {tuple(self.positional_embedding.shape)}, "
                f"got {tuple(x.shape[1:])}"
            )
        x = (x + self.positional_embedding).to(x.dtype)
        if prefix_tokens is not None:
            if prefix_tokens.ndim != 3:
                raise ValueError("prefix_tokens must have shape (B, P, C)")
            if prefix_tokens.shape[0] != x.shape[0]:
                raise ValueError("prefix_tokens batch size must match audio batch size")
            if prefix_tokens.shape[2] != x.shape[2]:
                raise ValueError(
                    "prefix_tokens hidden size must match encoder hidden size"
                )
            x = torch.cat([prefix_tokens.to(device=x.device, dtype=x.dtype), x], dim=1)
        hidden_states: list[Tensor] | None = None
        if output_hidden_states:
            hidden_states = [x]
        for block in self.blocks:
            x = block(x)
            if hidden_states is not None:
                hidden_states.append(x)
        x = self.ln_post(x)
        return AudioEncoderOutput(
            last_hidden_state=x,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
