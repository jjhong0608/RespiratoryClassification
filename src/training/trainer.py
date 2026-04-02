from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loaders import BagBatch
from src.evaluation.diagnostics import build_diagnostic_rows, write_diagnostics_jsonl
from src.evaluation.metrics import EvalMetrics, MetricsComputer
from src.models.model import MILModelOutput
from src.training.scheduler import WarmupCosineScheduler
from src.utils.config import AnalysisConfig
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class TrainerConfig:
    device: str
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    top_k: int
    run_dir: Path
    pos_weight: float | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: EvalMetrics
    probabilities: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    diagnostics: list[dict[str, Any]]


class CheckpointManager(LoggingMixin):
    def __init__(self, run_dir: Path, top_k: int):
        self.run_dir = run_dir
        self.top_k = top_k
        self._best: list[tuple[float, Path]] = []

    def save_last(self, state: dict[str, Any]) -> Path:
        path = self.run_dir / "last.pt"
        torch.save(state, path)
        self.logger.info("Saved last checkpoint: %s", path)
        return path

    def maybe_save_best(self, val_loss: float, state: dict[str, Any]) -> None:
        path = self.run_dir / f"best_loss_{val_loss:.6f}.pt"
        torch.save(state, path)
        self._best.append((val_loss, path))
        self._best.sort(key=lambda item: item[0])
        rank = next(
            index
            for index, (_, item_path) in enumerate(self._best, start=1)
            if item_path == path
        )
        self.logger.info(
            "Saved best checkpoint (rank %d/%d): %s (val_loss=%.6f)",
            rank,
            self.top_k,
            path,
            val_loss,
        )
        while len(self._best) > self.top_k:
            _, to_remove = self._best.pop(-1)
            with suppress(FileNotFoundError):
                to_remove.unlink()
            self.logger.info(
                "Removed checkpoint (exceeds top_k=%d): %s",
                self.top_k,
                to_remove,
            )


class Trainer(LoggingMixin):
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        Fs.ensure_dir(cfg.run_dir)
        self.ckpt = CheckpointManager(cfg.run_dir, cfg.top_k)
        self._criterion = nn.BCEWithLogitsLoss(
            pos_weight=None
            if cfg.pos_weight is None
            else torch.tensor(cfg.pos_weight, dtype=torch.float32)
        )

    def _criterion_on(self, device: torch.device) -> nn.BCEWithLogitsLoss:
        if self.cfg.pos_weight is None:
            return self._criterion
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.cfg.pos_weight, device=device, dtype=torch.float32
            )
        )

    def _epoch(
        self,
        model: nn.Module,
        loader: DataLoader[BagBatch],
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        device: torch.device,
    ) -> EpochResult:
        criterion = self._criterion_on(device)
        total_loss = 0.0
        total_examples = 0
        targets: list[int] = []
        probabilities: list[float] = []
        predictions: list[int] = []
        diagnostics: list[dict[str, Any]] = []

        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            output = model(batch.segments, batch.instance_mask)
            if not isinstance(output, MILModelOutput):
                raise TypeError("MIL model must return MILModelOutput")
            bag_logits = output.bag_logits
            loss = criterion(bag_logits, batch.labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_probs = torch.sigmoid(bag_logits.detach())
            batch_preds = (batch_probs >= 0.5).to(torch.long)
            targets.extend(batch.labels.detach().cpu().to(torch.long).tolist())
            probabilities.extend(batch_probs.cpu().tolist())
            predictions.extend(batch_preds.cpu().tolist())

            total_examples += int(batch.labels.numel())
            total_loss += float(loss.item()) * float(batch.labels.numel())

            if any(asdict(self.cfg.analysis.outputs).values()):
                diagnostics.extend(
                    build_diagnostic_rows(
                        batch,
                        output,
                        probabilities=batch_probs.cpu(),
                        predicted_labels=batch_preds.cpu(),
                        analysis=self.cfg.analysis.outputs,
                    )
                )

        target_arr = np.asarray(targets, dtype=np.int64)
        prob_arr = np.asarray(probabilities, dtype=np.float64)
        pred_arr = np.asarray(predictions, dtype=np.int64)
        metrics = MetricsComputer.compute(target_arr, pred_arr, prob_arr)
        return EpochResult(
            loss=total_loss / float(max(1, total_examples)),
            metrics=metrics,
            probabilities=prob_arr,
            predictions=pred_arr,
            targets=target_arr,
            diagnostics=diagnostics,
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[BagBatch],
        val_loader: DataLoader[BagBatch],
        optimizer: torch.optim.Optimizer | None = None,
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        device = torch.device(self.cfg.device)
        model.to(device)
        if optimizer is None:
            optimizer = AdamW(
                model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        total_steps = self.cfg.epochs * max(1, len(train_loader))
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

        train_losses: list[float] = []
        val_losses: list[float] = []
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
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            lr_str = (
                f"{lrs[0]:.8f}"
                if len(lrs) == 1
                else "[" + ", ".join(f"{lr:.8f}" for lr in lrs) + "]"
            )
            self.logger.info(
                "Epoch %d/%d | LR: %s | Train Loss: %.4f | Train Acc: %.4f | "
                "Train Recall: %.4f | Train Precision: %.4f | Train F1: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f | Val Recall: %.4f | "
                "Val Precision: %.4f | Val F1: %.4f | Val Balanced Acc: %.4f",
                epoch,
                self.cfg.epochs,
                lr_str,
                train_result.loss,
                train_result.metrics.accuracy,
                train_result.metrics.recall,
                train_result.metrics.precision,
                train_result.metrics.f1_score,
                val_result.loss,
                val_result.metrics.accuracy,
                val_result.metrics.recall,
                val_result.metrics.precision,
                val_result.metrics.f1_score,
                val_result.metrics.balanced_accuracy,
            )

            diagnostics_path = None
            if val_result.diagnostics:
                diagnostics_path = write_diagnostics_jsonl(
                    val_result.diagnostics,
                    self.cfg.run_dir / "diagnostics" / f"val_epoch_{epoch:03d}.jsonl",
                )
                self.logger.info("Saved validation diagnostics: %s", diagnostics_path)

            model_cfg = getattr(model, "cfg", None)
            dims = None
            if model_cfg is not None:
                encoder_dims = getattr(model_cfg, "encoder", None)
                if encoder_dims is not None and is_dataclass(encoder_dims):
                    dims = asdict(encoder_dims)

            state: dict[str, Any] = {
                "epoch": epoch,
                "dims": dims,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_metrics": [train_result.metrics.to_dict()],
                "val_metrics": [result.to_dict() for result in [val_result.metrics]],
                "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
            }
            if extra_state:
                for key, value in extra_state.items():
                    if is_dataclass(value) and not isinstance(value, type):
                        state[key] = asdict(value)
                    else:
                        state[key] = value
            self.ckpt.save_last(state)
            self.ckpt.maybe_save_best(val_result.loss, state)
