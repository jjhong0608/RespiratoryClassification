from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor, nn

from src.models.whisper_encoder import AudioEncoder, WhisperEncoderDims


@dataclass(frozen=True)
class SupervisedContrastiveEncoderConfig:
    encoder: WhisperEncoderDims
    projection_hidden_dim: int = 384
    projection_output_dim: int = 128


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SupervisedContrastiveEncoder(nn.Module):
    def __init__(self, cfg: SupervisedContrastiveEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AudioEncoder(cfg.encoder)
        self.projector = ProjectionHead(
            cfg.encoder.n_audio_state,
            cfg.projection_hidden_dim,
            cfg.projection_output_dim,
        )

    def encode(self, x: Tensor) -> Tensor:
        features = self.encoder(x).last_hidden_state
        return features.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(self.encode(x))

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {
            f"encoder.{key}": value for key, value in self.encoder.state_dict().items()
        }

    def export_pretrained_checkpoint(
        self,
        *,
        epoch: int,
        train_losses: list[float],
        val_losses: list[float],
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "epoch": epoch,
            "dims": asdict(self.cfg.encoder),
            "model_state_dict": self.export_encoder_state_dict(),
            "train_losses": list(train_losses),
            "val_losses": list(val_losses),
            "train_loss": float(train_losses[-1]) if train_losses else None,
            "val_loss": float(val_losses[-1]) if val_losses else None,
            "checkpoint_type": "supervised_contrastive_encoder",
        }
        if extra_state:
            state.update(extra_state)
        return state
