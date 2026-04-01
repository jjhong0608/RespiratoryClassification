from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.thresholds import (
    ThresholdOptimizationConfig,
    ThresholdOptimizationResult,
    ThresholdOptimizer,
)
from src.training.metrics import RunningConfusionMatrix
from src.training.scheduler import WarmupCosineScheduler
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
    num_classes: int
    run_dir: Path
    pos_weight: float | None = None
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )


@dataclass(frozen=True)
class BinaryEpochOutputs:
    targets: np.ndarray
    positive_scores: np.ndarray


@dataclass(frozen=True)
class EpochResult:
    loss: float
    stats: RunningConfusionMatrix
    binary_outputs: BinaryEpochOutputs | None = None


class CheckpointManager(LoggingMixin):
    def __init__(self, run_dir: Path, top_k: int):
        self.run_dir = run_dir
        self.top_k = top_k
        self._best: list[tuple[float, Path]] = []

    def save_last(self, state: dict[str, Any]) -> Path:
        path = self.run_dir / "last.pt"
        torch.save(state, path)
        self.logger.info(f"Saved last checkpoint: {path}")
        return path

    def maybe_save_best(self, val_loss: float, state: dict[str, Any]) -> None:
        path = self.run_dir / f"best_loss_{val_loss:.6f}.pt"
        torch.save(state, path)
        self._best.append((val_loss, path))
        self._best.sort(key=lambda x: x[0])
        rank = next(i for i, (loss, p) in enumerate(self._best, start=1) if p == path)
        self.logger.info(
            f"Saved best checkpoint (rank {rank}/{self.top_k}): {path} (val_loss={val_loss:.6f})"
        )
        while len(self._best) > self.top_k:
            _, to_remove = self._best.pop(-1)
            with suppress(FileNotFoundError):
                to_remove.unlink()
            self.logger.info(
                f"Removed checkpoint (exceeds top_k={self.top_k}): {to_remove}"
            )


class Trainer(LoggingMixin):
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        Fs.ensure_dir(cfg.run_dir)
        self.ckpt = CheckpointManager(cfg.run_dir, cfg.top_k)

    def _epoch(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        device: torch.device,
    ) -> EpochResult:
        use_bce = self.cfg.num_classes == 2
        ce_loss = nn.CrossEntropyLoss()
        pos_weight = None
        if self.cfg.pos_weight is not None:
            pos_weight = torch.tensor(self.cfg.pos_weight, device=device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        total_loss = 0.0
        cm = RunningConfusionMatrix(num_classes=self.cfg.num_classes)
        binary_targets: list[int] = []
        binary_scores: list[float] = []
        for x, y in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if use_bce:
                binary_logit = (logits[:, 1] - logits[:, 0]).view(-1)
                loss = bce_loss(binary_logit, y.to(dtype=binary_logit.dtype))
                positive_scores = torch.sigmoid(binary_logit)
                preds = (positive_scores >= 0.5).to(torch.long)
                binary_targets.extend(y.detach().cpu().tolist())
                binary_scores.extend(positive_scores.detach().cpu().tolist())
            else:
                loss = ce_loss(logits, y)
                preds = logits.argmax(dim=-1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += float(loss.item()) * float(y.numel())
            cm.update(preds.detach().cpu(), y.detach().cpu())
        avg_loss = total_loss / float(max(1, cm.total))
        binary_outputs = None
        if use_bce:
            binary_outputs = BinaryEpochOutputs(
                targets=np.asarray(binary_targets, dtype=np.int64),
                positive_scores=np.asarray(binary_scores, dtype=np.float64),
            )
        return EpochResult(loss=avg_loss, stats=cm, binary_outputs=binary_outputs)

    def _stats_from_binary_predictions(
        self,
        targets: np.ndarray,
        positive_scores: np.ndarray,
        threshold: float,
    ) -> RunningConfusionMatrix:
        preds = ThresholdOptimizer.predict(positive_scores, threshold)
        cm = RunningConfusionMatrix(num_classes=2)
        cm.update(torch.from_numpy(preds), torch.from_numpy(targets))
        return cm

    @staticmethod
    def _optimize_threshold(
        cfg: ThresholdOptimizationConfig,
        outputs: BinaryEpochOutputs | None,
    ) -> ThresholdOptimizationResult | None:
        if not cfg.enabled or outputs is None:
            return None
        return ThresholdOptimizer(outputs.targets, outputs.positive_scores).optimize(
            cfg.metric
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        if self.cfg.pos_weight is not None and self.cfg.num_classes == 2:
            self.logger.info(f"Using BCE pos_weight={self.cfg.pos_weight:.6f}")

        best_val_loss = float("inf")
        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            train_result = self._epoch(
                model, train_loader, optimizer, scheduler, device
            )
            train_loss = train_result.loss
            train_stats = train_result.stats
            train_losses.append(train_loss)
            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            if len(lrs) == 1:
                lr_str = f"{lrs[0]:.8f}"
            else:
                lr_str = "[" + ", ".join(f"{lr:.8f}" for lr in lrs) + "]"

            model.eval()
            with torch.no_grad():
                val_result = self._epoch(model, val_loader, None, None, device)
            val_loss = val_result.loss
            val_stats = val_result.stats
            val_losses.append(val_loss)

            threshold_result = self._optimize_threshold(
                self.cfg.threshold_optimization,
                val_result.binary_outputs,
            )
            val_display_stats = val_stats
            threshold_log = ""
            if threshold_result is not None:
                val_binary_outputs = val_result.binary_outputs
                if val_binary_outputs is None:
                    raise RuntimeError(
                        "Binary threshold optimization requires validation scores"
                    )
                val_display_stats = self._stats_from_binary_predictions(
                    val_binary_outputs.targets,
                    val_binary_outputs.positive_scores,
                    threshold_result.selected_threshold,
                )
                if threshold_result.applied:
                    threshold_log = (
                        f" | Val Threshold[{threshold_result.selected_metric}]: "
                        f"{threshold_result.selected_threshold:.4f} "
                        f"(score={threshold_result.selected_score:.4f})"
                    )
                else:
                    threshold_log = (
                        f" | Val Threshold[{threshold_result.selected_metric}]: "
                        f"{threshold_result.selected_threshold:.4f} "
                        f"(fallback: {threshold_result.reason})"
                    )

            if self.cfg.num_classes == 2:
                train_tp, train_fp, train_fn, _ = train_stats.binary_counts()
                val_tp, val_fp, val_fn, _ = val_display_stats.binary_counts()

                train_precision = train_stats.binary_precision()
                train_recall = train_stats.binary_recall()
                val_precision = val_display_stats.binary_precision()
                val_recall = val_display_stats.binary_recall()

                train_precision_str = (
                    f"{train_precision:.4f} [{train_tp}/{train_tp + train_fp}]"
                )
                train_recall_str = (
                    f"{train_recall:.4f} [{train_tp}/{train_tp + train_fn}]"
                )
                val_precision_str = f"{val_precision:.4f} [{val_tp}/{val_tp + val_fp}]"
                val_recall_str = f"{val_recall:.4f} [{val_tp}/{val_tp + val_fn}]"
            else:
                train_precision = train_stats.macro_precision()
                train_recall = train_stats.macro_recall()
                val_precision = val_display_stats.macro_precision()
                val_recall = val_display_stats.macro_recall()
                train_precision_str = f"{train_precision:.4f}"
                train_recall_str = f"{train_recall:.4f}"
                val_precision_str = f"{val_precision:.4f}"
                val_recall_str = f"{val_recall:.4f}"

            self.logger.info(
                f"Epoch {epoch}/{self.cfg.epochs} | "
                f"LR: {lr_str} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_stats.accuracy:.4f} [{train_stats.correct}/{train_stats.total}] | "
                f"Train Recall: {train_recall_str} | "
                f"Train Precision: {train_precision_str} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_display_stats.accuracy:.4f} [{val_display_stats.correct}/{val_display_stats.total}] | "
                f"Val Recall: {val_recall_str} | "
                f"Val Precision: {val_precision_str}"
                f"{threshold_log}"
            )

            state: dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "num_classes": self.cfg.num_classes,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "threshold_optimization_cfg": asdict(self.cfg.threshold_optimization),
                "threshold_optimization_result": (
                    threshold_result.to_dict() if threshold_result is not None else None
                ),
            }
            if extra_state:
                state.update(self._sanitize_extra_state(extra_state))
            self.ckpt.save_last(state)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            self.ckpt.maybe_save_best(val_loss, state)

    @staticmethod
    def _sanitize_extra_state(extra_state: dict[str, Any]) -> dict[str, Any]:
        def sanitize(value: Any) -> Any:
            if is_dataclass(value) and not isinstance(value, type):
                return sanitize(asdict(value))
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): sanitize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(v) for v in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        return {str(k): sanitize(v) for k, v in extra_state.items()}
