from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.fsd50k_dataset import Fsd50kBatch
from src.models.ssl import MaskedFbankSSLOutput, MaskedFbankSSLWrapper
from src.training.scheduler import WarmupCosineScheduler
from src.training.trainer import CheckpointManager
from src.utils.config import CheckpointingConfig
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class SslTrainerConfig:
    device: str
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    run_dir: Path
    checkpointing: CheckpointingConfig | None = None


@dataclass(frozen=True)
class SslEpochResult:
    loss: float
    branch_losses: tuple[float, ...]
    actual_mask_ratios: tuple[float, ...]
    token_mask_ratios: tuple[float, ...]


class SslTrainer(LoggingMixin):
    def __init__(self, cfg: SslTrainerConfig) -> None:
        self.cfg = cfg
        Fs.ensure_dir(cfg.run_dir)
        self.ckpt = CheckpointManager(
            cfg.run_dir, top_k=3, checkpointing=cfg.checkpointing
        )

    def _epoch(
        self,
        model: MaskedFbankSSLWrapper,
        loader: DataLoader[Fsd50kBatch],
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        device: torch.device,
    ) -> SslEpochResult:
        total_loss = 0.0
        total_examples = 0
        branch_totals: torch.Tensor | None = None
        actual_ratio_totals: torch.Tensor | None = None
        token_ratio_totals: torch.Tensor | None = None
        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            output = model(batch.input_values)
            if not isinstance(output, MaskedFbankSSLOutput):
                raise TypeError("SSL wrapper must return MaskedFbankSSLOutput")
            loss = output.loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = int(batch.input_values.shape[0])
            total_examples += batch_size
            total_loss += float(loss.item()) * float(batch_size)
            branch_values = output.branch_losses.detach().cpu()
            actual_values = output.actual_mask_ratios.detach().cpu()
            token_values = output.token_mask_ratios.detach().cpu()
            if branch_totals is None:
                branch_totals = torch.zeros_like(branch_values)
                actual_ratio_totals = torch.zeros_like(actual_values)
                token_ratio_totals = torch.zeros_like(token_values)
            assert actual_ratio_totals is not None
            assert token_ratio_totals is not None
            branch_totals += branch_values * float(batch_size)
            actual_ratio_totals = actual_ratio_totals + actual_values * float(
                batch_size
            )
            token_ratio_totals = token_ratio_totals + token_values * float(batch_size)

        denom = float(max(1, total_examples))
        if (
            branch_totals is None
            or actual_ratio_totals is None
            or token_ratio_totals is None
        ):
            raise ValueError("SSL epoch received no batches")
        return SslEpochResult(
            loss=total_loss / denom,
            branch_losses=tuple(float(value.item() / denom) for value in branch_totals),
            actual_mask_ratios=tuple(
                float(value.item() / denom) for value in actual_ratio_totals
            ),
            token_mask_ratios=tuple(
                float(value.item() / denom) for value in token_ratio_totals
            ),
        )

    def fit(
        self,
        model: MaskedFbankSSLWrapper,
        train_loader: DataLoader[Fsd50kBatch],
        val_loader: DataLoader[Fsd50kBatch],
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        device = torch.device(self.cfg.device)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        total_steps = self.cfg.epochs * max(1, len(train_loader))
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_history: list[dict[str, Any]] = []
        val_history: list[dict[str, Any]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            train_result = self._epoch(
                model,
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            model.eval()
            with torch.no_grad():
                val_result = self._epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    scheduler=None,
                    device=device,
                )
            train_losses.append(train_result.loss)
            val_losses.append(val_result.loss)
            train_history.append(asdict(train_result))
            val_history.append(asdict(val_result))
            self.logger.info(
                "Epoch %d/%d | Train SSL Loss: %.6f | Val SSL Loss: %.6f | "
                "Train Branch Losses: %s | Val Branch Losses: %s",
                epoch,
                self.cfg.epochs,
                train_result.loss,
                val_result.loss,
                list(train_result.branch_losses),
                list(val_result.branch_losses),
            )
            state: dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "base_model_state_dict": model.base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_ssl_history": train_history,
                "val_ssl_history": val_history,
                "val_ssl_loss": val_result.loss,
                "val_branch_losses": val_result.branch_losses,
                "val_actual_mask_ratios": val_result.actual_mask_ratios,
                "val_token_mask_ratios": val_result.token_mask_ratios,
            }
            if extra_state:
                state.update(extra_state)
            self.ckpt.save_last(state)
            if self.cfg.checkpointing is None:
                self.ckpt.maybe_save_best(
                    "ssl_loss",
                    val_result.loss,
                    state,
                    maximize=False,
                )
            else:
                self.ckpt.save_configured_monitors(
                    {"val_ssl_loss": val_result.loss},
                    state,
                    epoch=epoch,
                )
