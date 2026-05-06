from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.fsd50k_dataset import Fsd50kBatch
from src.evaluation.metrics import MultiLabelMetrics, MultiLabelMetricsComputer
from src.models.model import AstModelOutput
from src.training.scheduler import WarmupCosineScheduler
from src.training.trainer import CheckpointManager
from src.utils.config import CheckpointingConfig
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class MultiLabelTrainerConfig:
    device: str
    epochs: int
    warmup_ratio: float
    max_grad_norm: float | None
    run_dir: Path
    threshold: float = 0.5
    checkpointing: CheckpointingConfig | None = None


@dataclass(frozen=True)
class MultiLabelEpochResult:
    loss: float
    metrics: MultiLabelMetrics
    probabilities: np.ndarray
    targets: np.ndarray


def compute_sqrt_neg_pos_multilabel_pos_weight(
    targets: Tensor,
    *,
    cap: float = 10.0,
) -> Tensor:
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (N, C), got {tuple(targets.shape)}")
    positives = targets.to(dtype=torch.float32).sum(dim=0)
    negatives = float(targets.shape[0]) - positives
    pos_weight = torch.sqrt(negatives / positives.clamp_min(1.0))
    return torch.clamp(pos_weight, max=float(cap))


class MultiLabelTrainer(LoggingMixin):
    def __init__(self, cfg: MultiLabelTrainerConfig, *, pos_weight: Tensor) -> None:
        self.cfg = cfg
        self.pos_weight = pos_weight.detach().to(dtype=torch.float32)
        Fs.ensure_dir(cfg.run_dir)
        self.ckpt = CheckpointManager(
            cfg.run_dir, top_k=3, checkpointing=cfg.checkpointing
        )

    def _epoch(
        self,
        model: nn.Module,
        loader: DataLoader[Fsd50kBatch],
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        device: torch.device,
    ) -> MultiLabelEpochResult:
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(device))
        total_loss = 0.0
        total_examples = 0
        probabilities: list[list[float]] = []
        targets: list[list[float]] = []
        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            if batch.labels is None:
                raise ValueError("FSD50K supervised batches must include labels")
            output = model(batch.input_values)
            if not isinstance(output, AstModelOutput):
                raise TypeError("FSD50K supervised model must return AstModelOutput")
            labels = batch.labels.to(device=device, dtype=output.logits.dtype)
            loss = criterion(output.logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            probs = torch.sigmoid(output.logits.detach())
            probabilities.extend(probs.cpu().tolist())
            targets.extend(batch.labels.detach().cpu().tolist())
            batch_size = int(batch.labels.shape[0])
            total_examples += batch_size
            total_loss += float(loss.item()) * float(batch_size)

        if total_examples == 0:
            raise ValueError("FSD50K supervised epoch received no examples")
        target_arr = np.asarray(targets, dtype=np.float64)
        prob_arr = np.asarray(probabilities, dtype=np.float64)
        metrics = MultiLabelMetricsComputer.compute(
            target_arr,
            prob_arr,
            threshold=self.cfg.threshold,
        )
        return MultiLabelEpochResult(
            loss=total_loss / float(total_examples),
            metrics=metrics,
            probabilities=prob_arr,
            targets=target_arr,
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[Fsd50kBatch],
        val_loader: DataLoader[Fsd50kBatch],
        optimizer: torch.optim.Optimizer,
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        device = torch.device(self.cfg.device)
        model.to(device)
        total_steps = self.cfg.epochs * max(1, len(train_loader))
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_metrics: list[dict[str, Any]] = []
        val_metrics: list[dict[str, Any]] = []
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
            train_metrics.append(train_result.metrics.to_dict())
            val_metrics.append(val_result.metrics.to_dict())
            self.logger.info(
                "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | "
                "Val macro AP: %.6f | Val micro AP: %.6f | Val macro F1@0.5: %.6f",
                epoch,
                self.cfg.epochs,
                train_result.loss,
                val_result.loss,
                val_result.metrics.macro_AP,
                val_result.metrics.micro_AP,
                val_result.metrics.macro_f1_at_0_5,
            )
            state: dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_loss": val_result.loss,
                "val_macro_AP": val_result.metrics.macro_AP,
                "val_micro_AP": val_result.metrics.micro_AP,
                "val_per_class_AP": val_result.metrics.per_class_AP,
                "pos_weight": self.pos_weight.cpu().tolist(),
            }
            if extra_state:
                state.update(extra_state)
            self.ckpt.save_last(state)
            if self.cfg.checkpointing is None:
                self.ckpt.maybe_save_best(
                    "loss", val_result.loss, state, maximize=False
                )
                self.ckpt.maybe_save_best(
                    "macro_AP",
                    val_result.metrics.macro_AP,
                    state,
                    maximize=True,
                )
            else:
                self.ckpt.save_configured_monitors(
                    {
                        "val_macro_AP": val_result.metrics.macro_AP,
                        "val_micro_AP": val_result.metrics.micro_AP,
                        "val_loss": val_result.loss,
                    },
                    state,
                    epoch=epoch,
                )
