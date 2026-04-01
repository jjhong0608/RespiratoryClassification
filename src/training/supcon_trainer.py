from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.supcon import SupervisedContrastiveEncoder
from src.training.scheduler import WarmupCosineScheduler
from src.training.supcon import SupervisedContrastiveLoss
from src.training.trainer import CheckpointManager, Trainer
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class SupConTrainerConfig:
    device: str
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    top_k: int
    run_dir: Path
    temperature: float
    normalize: bool


class SupervisedContrastiveTrainer(LoggingMixin):
    def __init__(self, cfg: SupConTrainerConfig):
        self.cfg = cfg
        Fs.ensure_dir(cfg.run_dir)
        self.ckpt = CheckpointManager(cfg.run_dir, cfg.top_k)

    def _epoch(
        self,
        model: SupervisedContrastiveEncoder,
        loader: DataLoader,
        criterion: SupervisedContrastiveLoss,
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        device: torch.device,
    ) -> float:
        total_loss = 0.0
        total_examples = 0
        for view_1, view_2, labels in tqdm(loader, leave=False):
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)
            labels = labels.to(device)

            batch_size = int(labels.shape[0])
            embeddings = model(torch.cat([view_1, view_2], dim=0))
            embeddings = embeddings.view(2, batch_size, -1).permute(1, 0, 2)
            loss = criterion(embeddings, labels)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.cfg.max_grad_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += float(loss.item()) * float(batch_size)
            total_examples += batch_size
        return total_loss / float(max(1, total_examples))

    def fit(
        self,
        model: SupervisedContrastiveEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        device = torch.device(self.cfg.device)
        model.to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        total_steps = self.cfg.epochs * max(1, len(train_loader))
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        criterion = SupervisedContrastiveLoss(
            temperature=self.cfg.temperature,
            normalize=self.cfg.normalize,
        )

        best_val_loss = float("inf")
        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            train_loss = self._epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
            )
            train_losses.append(train_loss)
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            lr_str = (
                f"{lrs[0]:.8f}"
                if len(lrs) == 1
                else "[" + ", ".join(f"{lr:.8f}" for lr in lrs) + "]"
            )

            model.eval()
            with torch.no_grad():
                val_loss = self._epoch(
                    model,
                    val_loader,
                    criterion,
                    None,
                    None,
                    device,
                )
            val_losses.append(val_loss)

            self.logger.info(
                f"Epoch {epoch}/{self.cfg.epochs} | "
                f"LR: {lr_str} | "
                f"Train SupCon Loss: {train_loss:.4f} | "
                f"Val SupCon Loss: {val_loss:.4f}"
            )

            state = model.export_pretrained_checkpoint(
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                extra_state=(
                    Trainer._sanitize_extra_state(extra_state) if extra_state else None
                ),
            )
            self.ckpt.save_last(state)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            self.ckpt.maybe_save_best(val_loss, state)
