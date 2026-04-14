from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loaders import BagBatch
from src.evaluation.diagnostics import build_diagnostic_rows, write_diagnostics_jsonl
from src.evaluation.metrics import EvalMetrics, MetricsComputer
from src.evaluation.thresholds import compute_threshold_optimized_metrics
from src.models.model import MILModelOutput
from src.training.losses import FocalLoss
from src.training.scheduler import WarmupCosineScheduler
from src.utils.config import AnalysisConfig, EarlyStoppingConfig
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin


@dataclass(frozen=True)
class TrainerConfig:
    device: str
    epochs: int
    encoder_lr: float
    head_lr: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    top_k: int
    run_dir: Path
    loss_type: str = "bce"
    gamma: float = 2.0
    pos_weight: float | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


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
        self._best: dict[str, list[tuple[float, Path]]] = {}

    def save_last(self, state: dict[str, Any]) -> Path:
        path = self.run_dir / "last.pt"
        torch.save(state, path)
        self.logger.info("Saved last checkpoint: %s", path)
        return path

    def maybe_save_best(
        self,
        name: str,
        score: float,
        state: dict[str, Any],
        *,
        maximize: bool = False,
    ) -> None:
        path = self.run_dir / f"best_{name}_{score:.6f}.pt"
        torch.save(state, path)
        monitor = self._best.setdefault(name, [])
        monitor[:] = [
            (existing_score, existing_path)
            for existing_score, existing_path in monitor
            if existing_path != path
        ]
        monitor.append((score, path))
        monitor.sort(key=lambda item: item[0], reverse=maximize)
        rank = next(
            index
            for index, (_, item_path) in enumerate(monitor, start=1)
            if item_path == path
        )
        self.logger.info(
            "Saved best checkpoint [%s] (rank %d/%d): %s (%s=%.6f)",
            name,
            rank,
            self.top_k,
            path,
            name,
            score,
        )
        while len(monitor) > self.top_k:
            _, to_remove = monitor.pop(-1)
            with suppress(FileNotFoundError):
                to_remove.unlink()
            self.logger.info(
                "Removed checkpoint [%s] (exceeds top_k=%d): %s",
                name,
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

    def _criterion_on(self, device: torch.device) -> nn.Module:
        if self.cfg.loss_type == "bce":
            if self.cfg.pos_weight is None:
                return self._criterion
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    self.cfg.pos_weight,
                    device=device,
                    dtype=torch.float32,
                )
            )
        if self.cfg.loss_type == "focal":
            return FocalLoss(
                gamma=self.cfg.gamma,
                pos_weight=self.cfg.pos_weight,
            ).to(device)
        raise ValueError(f"Unknown loss type: {self.cfg.loss_type}")

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
        train_metrics_history: list[dict[str, Any]] = []
        val_metrics_history: list[dict[str, Any]] = []
        val_metrics_optimized_history: list[dict[str, Any]] = []
        best_monitor_value: float | None = None
        epochs_without_improvement = 0

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
            val_threshold_optimization, val_metrics_optimized = (
                compute_threshold_optimized_metrics(
                    val_result.targets,
                    val_result.probabilities,
                    "f1",
                )
            )

            train_losses.append(train_result.loss)
            val_losses.append(val_result.loss)
            train_metrics_history.append(train_result.metrics.to_dict())
            val_metrics_history.append(val_result.metrics.to_dict())
            val_metrics_optimized_history.append(val_metrics_optimized.to_dict())

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
                "Val Precision: %.4f | Val F1@0.5: %.4f | Val Balanced Acc@0.5: %.4f | "
                "Val F1@opt: %.4f | Val Balanced Acc@opt: %.4f | Val Opt Threshold: %.4f",
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
                val_metrics_optimized.f1_score,
                val_metrics_optimized.balanced_accuracy,
                val_threshold_optimization.selected_threshold,
            )

            current_monitor_value = val_result.loss
            stopped_early = False
            stop_reason: str | None = None
            if self.cfg.early_stopping.enabled:
                improved = best_monitor_value is None or current_monitor_value < (
                    best_monitor_value - self.cfg.early_stopping.min_delta
                )
                if improved:
                    best_monitor_value = current_monitor_value
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.cfg.early_stopping.patience:
                        stopped_early = True
                        stop_reason = (
                            "Early stopping triggered on val_loss after "
                            f"{epochs_without_improvement} epochs without improvement"
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
                segment_encoder_cfg = getattr(model_cfg, "segment_encoder", None)
                encoder_dims = getattr(segment_encoder_cfg, "dims", None)
                if encoder_dims is not None and is_dataclass(encoder_dims):
                    dims = asdict(encoder_dims)

            early_stopping_state = {
                "enabled": self.cfg.early_stopping.enabled,
                "monitor": self.cfg.early_stopping.monitor,
                "patience": self.cfg.early_stopping.patience,
                "min_delta": self.cfg.early_stopping.min_delta,
                "best_value": best_monitor_value,
                "current_value": current_monitor_value,
                "epochs_without_improvement": epochs_without_improvement,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
            }
            state: dict[str, Any] = {
                "epoch": epoch,
                "dims": dims,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_metrics": train_metrics_history,
                "val_metrics": val_metrics_history,
                "val_threshold_optimization": val_threshold_optimization.to_dict(),
                "val_metrics_optimized": val_metrics_optimized_history,
                "early_stopping": early_stopping_state,
                "optimizer_group_lrs": lrs,
                "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
            }
            if extra_state:
                for key, value in extra_state.items():
                    if is_dataclass(value) and not isinstance(value, type):
                        state[key] = asdict(value)
                    else:
                        state[key] = value
            self.ckpt.save_last(state)
            self.ckpt.maybe_save_best("loss", val_result.loss, state, maximize=False)
            self.ckpt.maybe_save_best(
                "f1",
                val_metrics_optimized.f1_score,
                state,
                maximize=True,
            )

            if stopped_early:
                self.logger.info("%s", stop_reason)
                break
