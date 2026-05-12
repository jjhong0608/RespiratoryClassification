from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data.fsd50k_dataset import Fsd50kBatch
from src.evaluation.diagnostics import (
    build_multilabel_diagnostic_rows,
    write_diagnostics_jsonl,
)
from src.evaluation.metrics import MultiLabelMetrics, MultiLabelMetricsComputer
from src.models.model import AstModelOutput
from src.training.scheduler import WarmupCosineScheduler
from src.training.trainer import CheckpointManager
from src.utils.config import AnalysisConfig, CheckpointingConfig
from src.utils.fs import Fs
from src.utils.logging import LoggingMixin
from src.utils.progress import iter_progress


@dataclass(frozen=True)
class MultiLabelTrainerConfig:
    device: str
    epochs: int
    warmup_ratio: float
    max_grad_norm: float | None
    run_dir: Path
    threshold: float = 0.5
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    class_names: tuple[str, ...] = ()
    checkpointing: CheckpointingConfig | None = None
    terminal_width: int | None = None


@dataclass(frozen=True)
class MultiLabelEpochResult:
    loss: float
    metrics: MultiLabelMetrics
    probabilities: np.ndarray
    targets: np.ndarray
    diagnostics: list[dict[str, Any]]
    probability_stats: dict[str, float]


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


def _validate_2d_targets(name: str, targets: Tensor) -> Tensor:
    if targets.ndim != 2:
        raise ValueError(f"{name} must have shape (N, C), got {tuple(targets.shape)}")
    return targets.detach().cpu().to(dtype=torch.float32)


def _label_density_stats(prefix: str, targets: Tensor) -> dict[str, float]:
    labels_per_sample = targets.sum(dim=1)
    return {
        f"{prefix}_mean_labels_per_sample": float(labels_per_sample.mean().item()),
        f"{prefix}_min_labels_per_sample": float(labels_per_sample.min().item()),
        f"{prefix}_max_labels_per_sample": float(labels_per_sample.max().item()),
    }


def _class_count_entries(
    counts: Tensor,
    indices: Tensor,
    class_names: Sequence[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_tensor in indices:
        class_index = int(index_tensor.item())
        row: dict[str, Any] = {
            "class_index": class_index,
            "count": int(counts[class_index].item()),
        }
        if class_names is not None and class_index < len(class_names):
            row["class_name"] = class_names[class_index]
        rows.append(row)
    return rows


def compute_multilabel_target_stats(
    train_targets: Tensor,
    val_targets: Tensor,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    train = _validate_2d_targets("train_targets", train_targets)
    val = _validate_2d_targets("val_targets", val_targets)
    if train.shape[1] != val.shape[1]:
        raise ValueError(
            "train_targets and val_targets must have the same number of classes"
        )
    train_positive_counts = train.sum(dim=0).to(dtype=torch.int64)
    val_positive_counts = val.sum(dim=0).to(dtype=torch.int64)
    nonzero_indices = torch.nonzero(
        train_positive_counts > 0,
        as_tuple=False,
    ).flatten()
    sorted_nonzero = nonzero_indices[
        torch.argsort(train_positive_counts[nonzero_indices], descending=True)
    ]
    bottom_nonzero = nonzero_indices[
        torch.argsort(train_positive_counts[nonzero_indices], descending=False)
    ]
    stats: dict[str, Any] = {
        "train_samples": int(train.shape[0]),
        "val_samples": int(val.shape[0]),
        "num_classes": int(train.shape[1]),
        "train_num_classes_with_positive_count": int(
            (train_positive_counts > 0).sum().item()
        ),
        "val_num_classes_with_positive_count": int(
            (val_positive_counts > 0).sum().item()
        ),
        "top10_positive_class_counts": _class_count_entries(
            train_positive_counts,
            sorted_nonzero[:10],
            class_names,
        ),
        "bottom10_nonzero_positive_class_counts": _class_count_entries(
            train_positive_counts,
            bottom_nonzero[:10],
            class_names,
        ),
    }
    stats.update(_label_density_stats("train", train))
    stats.update(_label_density_stats("val", val))
    return stats


def compute_multilabel_probability_stats(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must have shape (N, C), got {probabilities.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (N, C), got {targets.shape}")
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have matching shapes")
    max_probabilities = probabilities.max(axis=1)
    return {
        "prob_mean": float(np.mean(probabilities)),
        "prob_std": float(np.std(probabilities)),
        "prob_max_mean": float(np.mean(max_probabilities)),
        "prob_max_p95": float(np.percentile(max_probabilities, 95)),
        "mean_predicted_positives_at_0_5": float(
            (probabilities >= 0.5).sum(axis=1).mean()
        ),
        "mean_predicted_positives_at_0_3": float(
            (probabilities >= 0.3).sum(axis=1).mean()
        ),
        "mean_predicted_positives_at_0_1": float(
            (probabilities >= 0.1).sum(axis=1).mean()
        ),
        "mean_true_positives": float(targets.sum(axis=1).mean()),
    }


def compute_pos_weight_stats(
    pos_weight: Tensor, *, cap: float
) -> dict[str, float | int]:
    values = pos_weight.detach().cpu().to(dtype=torch.float32)
    if values.ndim != 1:
        raise ValueError(f"pos_weight must have shape (C,), got {tuple(values.shape)}")
    if cap <= 0:
        raise ValueError("cap must be greater than zero")
    return {
        "pos_weight_min": float(values.min().item()),
        "pos_weight_mean": float(values.mean().item()),
        "pos_weight_median": float(torch.quantile(values, 0.5).item()),
        "pos_weight_max": float(values.max().item()),
        "num_capped_classes": int((values >= float(cap)).sum().item()),
    }


def _analysis_outputs_enabled(cfg: AnalysisConfig) -> bool:
    outputs = cfg.outputs
    return (
        outputs.save_logits
        or outputs.save_probabilities
        or outputs.save_embeddings
        or outputs.save_clip_metadata
    )


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
        collect_diagnostics: bool = False,
    ) -> MultiLabelEpochResult:
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(device))
        total_loss = 0.0
        total_examples = 0
        probabilities: list[list[float]] = []
        targets: list[list[float]] = []
        diagnostics: list[dict[str, Any]] = []
        for batch in iter_progress(
            loader,
            terminal_width=self.cfg.terminal_width,
            leave=False,
        ):
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
            if collect_diagnostics:
                diagnostics.extend(
                    build_multilabel_diagnostic_rows(
                        batch,
                        output,
                        probabilities=probs.cpu(),
                        class_names=self.cfg.class_names,
                        threshold=self.cfg.threshold,
                        analysis=self.cfg.analysis.outputs,
                    )
                )
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
        probability_stats = compute_multilabel_probability_stats(prob_arr, target_arr)
        return MultiLabelEpochResult(
            loss=total_loss / float(total_examples),
            metrics=metrics,
            probabilities=prob_arr,
            targets=target_arr,
            diagnostics=diagnostics,
            probability_stats=probability_stats,
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
        val_probability_stats_history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            train_result = self._epoch(
                model,
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                collect_diagnostics=False,
            )
            model.eval()
            with torch.no_grad():
                val_result = self._epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    scheduler=None,
                    device=device,
                    collect_diagnostics=_analysis_outputs_enabled(self.cfg.analysis),
                )
            train_losses.append(train_result.loss)
            val_losses.append(val_result.loss)
            train_metrics.append(train_result.metrics.to_dict())
            val_metrics.append(val_result.metrics.to_dict())
            val_probability_stats_history.append(val_result.probability_stats)
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
            self.logger.info(
                "Validation probability stats | epoch=%d | prob_mean=%.6f | "
                "prob_std=%.6f | prob_max_mean=%.6f | prob_max_p95=%.6f | "
                "mean_predicted_positives_at_0_5=%.6f | "
                "mean_predicted_positives_at_0_3=%.6f | "
                "mean_predicted_positives_at_0_1=%.6f | mean_true_positives=%.6f",
                epoch,
                val_result.probability_stats["prob_mean"],
                val_result.probability_stats["prob_std"],
                val_result.probability_stats["prob_max_mean"],
                val_result.probability_stats["prob_max_p95"],
                val_result.probability_stats["mean_predicted_positives_at_0_5"],
                val_result.probability_stats["mean_predicted_positives_at_0_3"],
                val_result.probability_stats["mean_predicted_positives_at_0_1"],
                val_result.probability_stats["mean_true_positives"],
            )
            diagnostics_path = None
            if val_result.diagnostics:
                diagnostics_path = write_diagnostics_jsonl(
                    val_result.diagnostics,
                    self.cfg.run_dir / "diagnostics" / f"val_epoch_{epoch:03d}.jsonl",
                )
                self.logger.info("Saved validation diagnostics: %s", diagnostics_path)

            state: dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_probability_stats": val_probability_stats_history,
                "current_val_probability_stats": val_result.probability_stats,
                "val_loss": val_result.loss,
                "val_macro_AP": val_result.metrics.macro_AP,
                "val_micro_AP": val_result.metrics.micro_AP,
                "val_per_class_AP": val_result.metrics.per_class_AP,
                "pos_weight": self.pos_weight.cpu().tolist(),
                "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
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
