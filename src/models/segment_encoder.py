from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.models.whisper_encoder import AudioEncoder, WhisperEncoderDims


@dataclass(frozen=True)
class SegmentEncoderOutput:
    embeddings: Tensor
    token_states: Tensor


class WhisperSegmentEncoder(nn.Module):
    def __init__(self, dims: WhisperEncoderDims):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(dims)

    def forward(self, segments: Tensor) -> SegmentEncoderOutput:
        encoded = self.encoder(segments)
        token_states = encoded.last_hidden_state
        return SegmentEncoderOutput(
            embeddings=token_states.mean(dim=1),
            token_states=token_states,
        )
