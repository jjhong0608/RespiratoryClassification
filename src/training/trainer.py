from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loaders import ClipBatch
from src.evaluation.diagnostics import build_diagnostic_rows, write_diagnostics_jsonl
from src.evaluation.metrics import EvalMetrics, MetricsComputer
from src.evaluation.thresholds import (
    ThresholdOptimizationResult,
    compute_threshold_optimized_metrics,
)
from src.models.model import AstModelOutput, ClassGateEvidenceAuxiliaryConfig
from src.training.losses import FocalLoss
from src.training.scheduler import WarmupCosineScheduler
from src.utils.config import (
    AnalysisConfig,
    AttentionEntropyLossConfig,
    BranchAuxiliaryLossConfig,
    BranchBinaryAuxiliaryLossConfig,
    CheckpointingConfig,
    ClassGateDiversityRegularizationConfig,
    EarlyStoppingConfig,
    GateEntropyRegularizationConfig,
    LabelSmoothingConfig,
)
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
    num_classes: int
    loss_type: str = "bce"
    gamma: float = 2.0
    pos_weight: float | None = None
    class_weights: tuple[float, ...] | None = None
    label_smoothing: LabelSmoothingConfig = field(default_factory=LabelSmoothingConfig)
    branch_auxiliary: BranchAuxiliaryLossConfig = field(
        default_factory=BranchAuxiliaryLossConfig
    )
    branch_binary_auxiliary: BranchBinaryAuxiliaryLossConfig = field(
        default_factory=BranchBinaryAuxiliaryLossConfig
    )
    branch_binary_pos_weight: float | None = None
    main_index_to_binary_target: tuple[int, ...] | None = None
    attention_entropy: AttentionEntropyLossConfig = field(
        default_factory=AttentionEntropyLossConfig
    )
    gate_entropy_regularization: GateEntropyRegularizationConfig = field(
        default_factory=GateEntropyRegularizationConfig
    )
    class_gate_evidence_auxiliary: ClassGateEvidenceAuxiliaryConfig = field(
        default_factory=ClassGateEvidenceAuxiliaryConfig
    )
    class_gate_diversity_regularization: ClassGateDiversityRegularizationConfig = field(
        default_factory=ClassGateDiversityRegularizationConfig
    )
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpointing: CheckpointingConfig | None = None


@dataclass(frozen=True)
class LossComponents:
    total: Tensor
    total_scheduled: Tensor
    total_monitor: Tensor
    main: Tensor
    branch_auxiliary: Tensor | None = None
    branch_binary_auxiliary: Tensor | None = None
    evidence_auxiliary_loss: Tensor | None = None
    attention_entropy: Tensor | None = None
    gate_entropy: Tensor | None = None
    gate_entropy_regularization: Tensor | None = None
    class_gate_diversity: Tensor | None = None
    class_gate_diversity_regularization: Tensor | None = None
    branch_binary_aux_weight: float = 0.0
    branch_binary_aux_monitor_weight: float = 0.0


@dataclass(frozen=True)
class EpochResult:
    loss: float
    loss_components: dict[str, float]
    metrics: EvalMetrics
    probabilities: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    diagnostics: list[dict[str, Any]]


@dataclass(frozen=True)
class MonitorRecord:
    score: float
    epoch: int
    path: Path | None


def resolve_branch_binary_aux_weight(
    cfg: BranchBinaryAuxiliaryLossConfig,
    *,
    epoch: int,
    total_epochs: int,
) -> float:
    if not cfg.enabled:
        return 0.0
    if not cfg.schedule.enabled:
        return float(cfg.weight)
    horizon = int(cfg.schedule.total_epochs or total_epochs)
    if horizon <= 1:
        return float(cfg.schedule.min_weight)
    progress = (float(epoch) - 1.0) / (float(horizon) - 1.0)
    progress = min(max(progress, 0.0), 1.0)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        cfg.schedule.min_weight
        + ((cfg.schedule.max_weight - cfg.schedule.min_weight) * cosine_scale)
    )


class CheckpointManager(LoggingMixin):
    def __init__(
        self,
        run_dir: Path,
        top_k: int,
        checkpointing: CheckpointingConfig | None = None,
    ):
        self.run_dir = run_dir
        self.top_k = top_k
        self.checkpointing = checkpointing
        self._best: dict[str, list[tuple[float, Path]]] = {}
        self._monitor_records: dict[str, list[MonitorRecord]] = {}

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

    @staticmethod
    def _short_monitor_name(name: str) -> str:
        if name.startswith("val_"):
            return name.removeprefix("val_")
        return name

    def _monitor_path(
        self,
        name: str,
        *,
        rank: int,
        record: MonitorRecord,
        filename_prefix: str | None = None,
    ) -> Path:
        short_name = filename_prefix or f"best_{self._short_monitor_name(name)}"
        return (
            self.run_dir
            / f"{short_name}_rank{rank}_{record.score:.6f}_epoch{record.epoch:03d}.pt"
        )

    def save_configured_monitors(
        self,
        scores: Mapping[str, float],
        state: dict[str, Any],
        *,
        epoch: int,
    ) -> None:
        if self.checkpointing is None:
            return
        for monitor in self.checkpointing.monitors:
            if monitor.name == "last":
                self._save_latest_epoch_checkpoint(
                    state,
                    epoch=epoch,
                    top_k=monitor.top_k,
                )
                continue
            score = scores.get(monitor.name)
            if score is None:
                self.logger.warning(
                    "Skipping unavailable checkpoint monitor: %s",
                    monitor.name,
                )
                continue
            self._save_ranked_monitor(
                monitor.name,
                float(score),
                state,
                epoch=epoch,
                mode=monitor.mode,
                top_k=monitor.top_k,
                filename_prefix=monitor.filename_prefix,
            )

    def _save_latest_epoch_checkpoint(
        self,
        state: dict[str, Any],
        *,
        epoch: int,
        top_k: int,
    ) -> None:
        records = self._monitor_records.setdefault("last", [])
        path = self.run_dir / f"last_epoch{epoch:03d}.pt"
        torch.save(state, path)
        records.append(MonitorRecord(score=float(epoch), epoch=epoch, path=path))
        records.sort(key=lambda record: record.epoch, reverse=True)
        while len(records) > top_k:
            record = records.pop(-1)
            if record.path is not None:
                with suppress(FileNotFoundError):
                    record.path.unlink()
                self.logger.info("Removed old last checkpoint: %s", record.path)
        self.logger.info("Saved latest epoch checkpoint: %s", path)

    def _save_ranked_monitor(
        self,
        name: str,
        score: float,
        state: dict[str, Any],
        *,
        epoch: int,
        mode: str,
        top_k: int,
        filename_prefix: str | None,
    ) -> None:
        records = self._monitor_records.setdefault(name, [])
        records.append(MonitorRecord(score=score, epoch=epoch, path=None))
        records.sort(key=lambda record: record.score, reverse=(mode == "max"))
        removed = records[top_k:]
        del records[top_k:]
        for record in removed:
            if record.path is not None:
                with suppress(FileNotFoundError):
                    record.path.unlink()
                self.logger.info(
                    "Removed checkpoint [%s] outside top_k: %s",
                    name,
                    record.path,
                )
        for index, record in enumerate(list(records)):
            if record.path is None or not record.path.exists():
                continue
            tmp_path = self.run_dir / (
                f".tmp_{name}_{index}_{record.epoch:03d}_{record.score:.6f}.pt"
            )
            with suppress(FileNotFoundError):
                tmp_path.unlink()
            record.path.rename(tmp_path)
            records[index] = MonitorRecord(
                score=record.score,
                epoch=record.epoch,
                path=tmp_path,
            )
        for rank, record in enumerate(records, start=1):
            path = self._monitor_path(
                name,
                rank=rank,
                record=record,
                filename_prefix=filename_prefix,
            )
            if record.path is None:
                torch.save(state, path)
            elif record.path != path:
                with suppress(FileNotFoundError):
                    path.unlink()
                record.path.rename(path)
            records[rank - 1] = MonitorRecord(
                score=record.score,
                epoch=record.epoch,
                path=path,
            )
        if any(record.epoch == epoch and record.score == score for record in records):
            self.logger.info(
                "Saved configured checkpoint [%s]: score=%.6f epoch=%03d",
                name,
                score,
                epoch,
            )


class Trainer(LoggingMixin):
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        Fs.ensure_dir(cfg.run_dir)
        self.ckpt = CheckpointManager(
            cfg.run_dir,
            cfg.top_k,
            checkpointing=cfg.checkpointing,
        )

    def _criterion_on(self, device: torch.device) -> nn.Module:
        if self.cfg.loss_type == "bce":
            if self.cfg.pos_weight is None:
                return nn.BCEWithLogitsLoss()
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
        if self.cfg.loss_type == "cross_entropy":
            label_smoothing = (
                float(self.cfg.label_smoothing.value)
                if self.cfg.label_smoothing.enabled
                else 0.0
            )
            if self.cfg.class_weights is not None:
                return nn.CrossEntropyLoss(
                    weight=torch.tensor(
                        self.cfg.class_weights,
                        device=device,
                        dtype=torch.float32,
                    ),
                    label_smoothing=label_smoothing,
                ).to(device)
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        raise ValueError(f"Unknown loss type: {self.cfg.loss_type}")

    def _branch_binary_criterion_on(self, device: torch.device) -> nn.Module:
        if self.cfg.branch_binary_pos_weight is None:
            return nn.BCEWithLogitsLoss()
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.cfg.branch_binary_pos_weight,
                device=device,
                dtype=torch.float32,
            )
        )

    def _compute_main_loss(
        self,
        criterion: nn.Module,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        if self.cfg.num_classes == 2:
            return criterion(
                logits, labels.to(device=logits.device, dtype=logits.dtype)
            )
        return criterion(logits, labels.to(device=logits.device, dtype=torch.long))

    def _predict(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        if self.cfg.num_classes == 2:
            probabilities = torch.sigmoid(logits.detach())
            predictions = (probabilities >= 0.5).to(torch.long)
            return probabilities, predictions
        probabilities = torch.softmax(logits.detach(), dim=-1)
        predictions = probabilities.argmax(dim=-1)
        return probabilities, predictions

    def _compute_branch_auxiliary_losses(
        self,
        criterion: nn.Module,
        branch_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        if self.cfg.branch_auxiliary.aggregation != "mean":
            raise ValueError(
                "Unsupported branch auxiliary aggregation: "
                f"{self.cfg.branch_auxiliary.aggregation}"
            )
        if self.cfg.num_classes == 2:
            if branch_logits.ndim != 2:
                raise ValueError(
                    "Binary branch logits must have shape (B, num_branches), "
                    f"got {tuple(branch_logits.shape)}"
                )
            branch_losses = []
            branch_labels = labels.to(
                device=branch_logits.device,
                dtype=branch_logits.dtype,
            )
            for branch_index in range(branch_logits.shape[1]):
                branch_losses.append(
                    criterion(branch_logits[:, branch_index], branch_labels)
                )
            return torch.stack(branch_losses)
        if branch_logits.ndim != 3:
            raise ValueError(
                "Multiclass branch logits must have shape (B, num_branches, C), "
                f"got {tuple(branch_logits.shape)}"
            )
        branch_losses = []
        branch_labels = labels.to(device=branch_logits.device, dtype=torch.long)
        for branch_index in range(branch_logits.shape[1]):
            branch_losses.append(
                criterion(branch_logits[:, branch_index, :], branch_labels)
            )
        return torch.stack(branch_losses)

    def _compute_branch_auxiliary_loss(
        self,
        criterion: nn.Module,
        branch_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        branch_losses = self._compute_branch_auxiliary_losses(
            criterion,
            branch_logits,
            labels,
        )
        weights = self.cfg.branch_auxiliary.weights
        if weights is None:
            return branch_losses.mean()
        if len(weights) != int(branch_losses.shape[0]):
            raise ValueError(
                "train.loss.branch_auxiliary.weights length must match branch logits"
            )
        weight_tensor = torch.tensor(
            weights,
            device=branch_losses.device,
            dtype=branch_losses.dtype,
        )
        return (branch_losses * weight_tensor).sum() / weight_tensor.sum()

    def _compute_attention_entropy_loss(
        self,
        output: AstModelOutput,
    ) -> Tensor:
        if output.branch_attention_weights is None:
            raise ValueError(
                "attention entropy loss enabled but model did not return branch_attention_weights"
            )
        if not output.branch_attention_weights:
            raise ValueError("attention entropy loss requires at least one branch")
        entropies = []
        for attention_weights in output.branch_attention_weights:
            entropy = (
                -(attention_weights * (attention_weights + 1e-8).log())
                .sum(dim=1)
                .mean()
            )
            entropies.append(entropy)
        return torch.stack(entropies).mean()

    def _compute_evidence_auxiliary_loss(
        self,
        criterion: nn.Module,
        output: AstModelOutput,
        labels: Tensor,
    ) -> Tensor:
        if output.class_evidence_logits is None:
            raise ValueError(
                "class gate evidence auxiliary loss enabled but model did not return "
                "class_evidence_logits"
            )
        return self._compute_main_loss(criterion, output.class_evidence_logits, labels)

    def _compute_gate_entropy_regularization(
        self,
        output: AstModelOutput,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        target = self.cfg.gate_entropy_regularization.target
        if target == "evidence_gate":
            if output.evidence_gate_entropy is None:
                raise ValueError(
                    "gate entropy regularization enabled but model did not return "
                    "evidence_gate_entropy"
                )
            entropy_values = output.evidence_gate_entropy
        elif target == "class_evidence_gate":
            if output.class_evidence_gate_entropy is None:
                raise ValueError(
                    "gate entropy regularization enabled but model did not return "
                    "class_evidence_gate_entropy"
                )
            entropy_values = output.class_evidence_gate_entropy
        elif target == "true_class_evidence_gate":
            if output.class_evidence_gate_entropy is None:
                raise ValueError(
                    "gate entropy regularization enabled but model did not return "
                    "class_evidence_gate_entropy"
                )
            if labels is None:
                raise ValueError(
                    "target='true_class_evidence_gate' requires labels for gate "
                    "entropy regularization"
                )
            if output.class_evidence_gate_entropy.ndim != 2:
                raise ValueError(
                    "class_evidence_gate_entropy must have shape (B, C), "
                    f"got {tuple(output.class_evidence_gate_entropy.shape)}"
                )
            label_indices = labels.to(
                device=output.class_evidence_gate_entropy.device,
                dtype=torch.long,
            )
            if label_indices.numel() > 0 and int(label_indices.max().item()) >= int(
                output.class_evidence_gate_entropy.shape[1]
            ):
                raise ValueError(
                    "labels contain a class index outside class_evidence_gate_entropy"
                )
            entropy_values = output.class_evidence_gate_entropy.gather(
                1,
                label_indices.unsqueeze(1),
            ).squeeze(1)
        else:
            raise ValueError(
                "gate entropy regularization target must be one of "
                "'evidence_gate', 'class_evidence_gate', "
                "'true_class_evidence_gate'"
            )
        gate_entropy = entropy_values.mean()
        regularization = (
            -float(self.cfg.gate_entropy_regularization.weight) * gate_entropy
        )
        return gate_entropy, regularization

    def _compute_class_gate_diversity_regularization(
        self,
        output: AstModelOutput,
    ) -> tuple[Tensor, Tensor]:
        if self.cfg.class_gate_diversity_regularization.target != "class_evidence_gate":
            raise ValueError(
                "class gate diversity regularization supports only "
                "target='class_evidence_gate'"
            )
        if self.cfg.class_gate_diversity_regularization.metric != "js_divergence":
            raise ValueError(
                "class gate diversity regularization supports only "
                "metric='js_divergence'"
            )
        if output.class_evidence_gate_weights is None:
            raise ValueError(
                "class gate diversity regularization enabled but model did not return "
                "class_evidence_gate_weights"
            )
        gate_weights = output.class_evidence_gate_weights
        if gate_weights.ndim != 3:
            raise ValueError(
                "class_evidence_gate_weights must have shape (B, C, R), "
                f"got {tuple(gate_weights.shape)}"
            )
        if gate_weights.shape[1] <= 1:
            raise ValueError(
                "class gate diversity regularization requires at least two classes"
            )
        eps = torch.finfo(gate_weights.dtype).eps
        distributions = gate_weights.clamp_min(eps)
        distributions = distributions / distributions.sum(dim=-1, keepdim=True)
        first = distributions.unsqueeze(2)
        second = distributions.unsqueeze(1)
        midpoint = 0.5 * (first + second)
        first_kl = (first * (first / midpoint).log()).sum(dim=-1)
        second_kl = (second * (second / midpoint).log()).sum(dim=-1)
        js_divergence = 0.5 * (first_kl + second_kl)
        num_classes = int(gate_weights.shape[1])
        pair_mask = torch.triu(
            torch.ones(
                num_classes,
                num_classes,
                device=gate_weights.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        diversity = js_divergence[:, pair_mask].mean()
        regularization = (
            -float(self.cfg.class_gate_diversity_regularization.weight) * diversity
        )
        return diversity, regularization

    def _branch_binary_targets(
        self,
        labels: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.cfg.main_index_to_binary_target is None:
            raise ValueError(
                "branch binary auxiliary is enabled but main_index_to_binary_target "
                "was not provided"
            )
        mapping = torch.tensor(
            self.cfg.main_index_to_binary_target,
            device=device,
            dtype=torch.long,
        )
        labels_long = labels.to(device=device, dtype=torch.long)
        if labels_long.numel() > 0 and int(labels_long.max().item()) >= mapping.numel():
            raise ValueError(
                "labels contain a class index outside main_index_to_binary_target"
            )
        return mapping[labels_long].to(dtype=dtype)

    def _compute_branch_binary_auxiliary_loss(
        self,
        criterion: nn.Module,
        output: AstModelOutput,
        labels: Tensor,
    ) -> Tensor:
        if output.branch_binary_logits is None:
            raise ValueError(
                "branch binary auxiliary loss enabled but model did not return "
                "branch_binary_logits"
            )
        if output.branch_binary_logits.ndim != 2:
            raise ValueError(
                "branch_binary_logits must have shape (B, num_branches), "
                f"got {tuple(output.branch_binary_logits.shape)}"
            )
        binary_targets = self._branch_binary_targets(
            labels,
            device=output.branch_binary_logits.device,
            dtype=output.branch_binary_logits.dtype,
        )
        expanded_targets = binary_targets.unsqueeze(1).expand_as(
            output.branch_binary_logits
        )
        return criterion(output.branch_binary_logits, expanded_targets)

    def _compute_total_loss(
        self,
        criterion: nn.Module,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int = 1,
        use_monitor_total: bool = False,
    ) -> LossComponents:
        final_loss = self._compute_main_loss(criterion, output.logits, labels)
        scheduled_total = final_loss
        monitor_total = final_loss
        auxiliary_loss: Tensor | None = None
        if self.cfg.branch_auxiliary.enabled:
            if output.branch_logits is None:
                raise ValueError(
                    "branch auxiliary loss enabled but model did not return branch_logits"
                )
            auxiliary_loss = self._compute_branch_auxiliary_loss(
                criterion,
                output.branch_logits,
                labels,
            )
            if self.cfg.branch_auxiliary.weights is None:
                auxiliary_term = self.cfg.branch_auxiliary.weight * auxiliary_loss
                scheduled_total = scheduled_total + auxiliary_term
                monitor_total = monitor_total + auxiliary_term
            else:
                scheduled_total = scheduled_total + auxiliary_loss
                monitor_total = monitor_total + auxiliary_loss
        branch_binary_loss: Tensor | None = None
        branch_binary_weight = resolve_branch_binary_aux_weight(
            self.cfg.branch_binary_auxiliary,
            epoch=epoch,
            total_epochs=self.cfg.epochs,
        )
        branch_binary_monitor_weight = (
            float(
                self.cfg.branch_binary_auxiliary.monitor.loss_weight
                if self.cfg.branch_binary_auxiliary.schedule.enabled
                else self.cfg.branch_binary_auxiliary.weight
            )
            if self.cfg.branch_binary_auxiliary.enabled
            else 0.0
        )
        if self.cfg.branch_binary_auxiliary.enabled:
            branch_binary_criterion = self._branch_binary_criterion_on(
                output.logits.device
            )
            branch_binary_loss = self._compute_branch_binary_auxiliary_loss(
                branch_binary_criterion,
                output,
                labels,
            )
            scheduled_total = scheduled_total + (
                branch_binary_weight * branch_binary_loss
            )
            monitor_total = monitor_total + (
                branch_binary_monitor_weight * branch_binary_loss
            )
        evidence_auxiliary_loss: Tensor | None = None
        if self.cfg.class_gate_evidence_auxiliary.enabled:
            evidence_auxiliary_loss = self._compute_evidence_auxiliary_loss(
                criterion,
                output,
                labels,
            )
            evidence_auxiliary_term = (
                float(self.cfg.class_gate_evidence_auxiliary.weight)
                * evidence_auxiliary_loss
            )
            scheduled_total = scheduled_total + evidence_auxiliary_term
            monitor_total = monitor_total + evidence_auxiliary_term
        entropy_loss: Tensor | None = None
        if self.cfg.attention_entropy.enabled:
            entropy_loss = self._compute_attention_entropy_loss(output)
            entropy_term = self.cfg.attention_entropy.weight * entropy_loss
            scheduled_total = scheduled_total + entropy_term
            monitor_total = monitor_total + entropy_term
        gate_entropy: Tensor | None = None
        gate_entropy_regularization: Tensor | None = None
        if self.cfg.gate_entropy_regularization.enabled:
            gate_entropy, gate_entropy_regularization = (
                self._compute_gate_entropy_regularization(output, labels)
            )
            scheduled_total = scheduled_total + gate_entropy_regularization
            monitor_total = monitor_total + gate_entropy_regularization
        class_gate_diversity: Tensor | None = None
        class_gate_diversity_regularization: Tensor | None = None
        if self.cfg.class_gate_diversity_regularization.enabled:
            class_gate_diversity, class_gate_diversity_regularization = (
                self._compute_class_gate_diversity_regularization(output)
            )
            scheduled_total = scheduled_total + class_gate_diversity_regularization
            monitor_total = monitor_total + class_gate_diversity_regularization
        return LossComponents(
            total=monitor_total if use_monitor_total else scheduled_total,
            total_scheduled=scheduled_total,
            total_monitor=monitor_total,
            main=final_loss,
            branch_auxiliary=auxiliary_loss,
            branch_binary_auxiliary=branch_binary_loss,
            evidence_auxiliary_loss=evidence_auxiliary_loss,
            attention_entropy=entropy_loss,
            gate_entropy=gate_entropy,
            gate_entropy_regularization=gate_entropy_regularization,
            class_gate_diversity=class_gate_diversity,
            class_gate_diversity_regularization=class_gate_diversity_regularization,
            branch_binary_aux_weight=branch_binary_weight,
            branch_binary_aux_monitor_weight=branch_binary_monitor_weight,
        )

    def _epoch(
        self,
        model: nn.Module,
        loader: DataLoader[ClipBatch],
        *,
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        device: torch.device,
        epoch: int,
        use_monitor_total: bool = False,
    ) -> EpochResult:
        criterion = self._criterion_on(device)
        total_loss = 0.0
        total_examples = 0
        component_totals: dict[str, float] = {
            "total": 0.0,
            "total_scheduled": 0.0,
            "total_monitor": 0.0,
            "main": 0.0,
            "branch_auxiliary": 0.0,
            "branch_binary_auxiliary": 0.0,
            "evidence_auxiliary_loss": 0.0,
            "attention_entropy": 0.0,
            "gate_entropy": 0.0,
            "gate_entropy_regularization": 0.0,
            "class_gate_diversity": 0.0,
            "class_gate_diversity_regularization": 0.0,
            "branch_binary_aux_weight": 0.0,
            "branch_binary_aux_monitor_weight": 0.0,
        }
        component_counts: dict[str, int] = {key: 0 for key in component_totals}
        targets: list[int] = []
        probabilities: list[float] | list[list[float]] = []
        predictions: list[int] = []
        branch_binary_probabilities: list[list[float]] = []
        branch_binary_targets: list[int] = []
        diagnostics: list[dict[str, Any]] = []

        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            output = model(batch.input_values)
            if not isinstance(output, AstModelOutput):
                raise TypeError("AST model must return AstModelOutput")

            logits = output.logits
            loss_components = self._compute_total_loss(
                criterion,
                output,
                batch.labels,
                epoch=epoch,
                use_monitor_total=use_monitor_total,
            )
            loss = loss_components.total

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_probs, batch_preds = self._predict(logits)
            targets.extend(batch.labels.detach().cpu().to(torch.long).tolist())
            predictions.extend(batch_preds.cpu().tolist())
            probabilities.extend(batch_probs.cpu().tolist())

            total_examples += int(batch.labels.numel())
            total_loss += float(loss.item()) * float(batch.labels.numel())
            batch_size = int(batch.labels.numel())
            for name, value in {
                "total": loss_components.total,
                "total_scheduled": loss_components.total_scheduled,
                "total_monitor": loss_components.total_monitor,
                "main": loss_components.main,
                "branch_auxiliary": loss_components.branch_auxiliary,
                "branch_binary_auxiliary": loss_components.branch_binary_auxiliary,
                "evidence_auxiliary_loss": loss_components.evidence_auxiliary_loss,
                "attention_entropy": loss_components.attention_entropy,
                "gate_entropy": loss_components.gate_entropy,
                "gate_entropy_regularization": (
                    loss_components.gate_entropy_regularization
                ),
                "class_gate_diversity": loss_components.class_gate_diversity,
                "class_gate_diversity_regularization": (
                    loss_components.class_gate_diversity_regularization
                ),
                "branch_binary_aux_weight": (loss_components.branch_binary_aux_weight),
                "branch_binary_aux_monitor_weight": (
                    loss_components.branch_binary_aux_monitor_weight
                ),
            }.items():
                if value is None:
                    continue
                if isinstance(value, Tensor):
                    scalar_value = float(value.item())
                else:
                    scalar_value = float(value)
                component_totals[name] += scalar_value * float(batch_size)
                component_counts[name] += batch_size

            binary_targets_for_batch: Tensor | None = None
            if (
                output.branch_binary_logits is not None
                and self.cfg.main_index_to_binary_target is not None
            ):
                binary_targets_for_batch = self._branch_binary_targets(
                    batch.labels,
                    device=output.branch_binary_logits.device,
                    dtype=output.branch_binary_logits.dtype,
                )
                branch_binary_targets.extend(
                    binary_targets_for_batch.detach().cpu().to(torch.long).tolist()
                )
                branch_binary_probabilities.extend(
                    torch.sigmoid(output.branch_binary_logits.detach()).cpu().tolist()
                )

            if any(asdict(self.cfg.analysis.outputs).values()):
                diagnostics.extend(
                    build_diagnostic_rows(
                        batch,
                        output,
                        probabilities=batch_probs.cpu(),
                        predicted_labels=batch_preds.cpu(),
                        analysis=self.cfg.analysis.outputs,
                        binary_auxiliary_targets=(
                            binary_targets_for_batch.detach().cpu()
                            if binary_targets_for_batch is not None
                            else None
                        ),
                    )
                )

        target_arr = np.asarray(targets, dtype=np.int64)
        prob_arr = np.asarray(probabilities, dtype=np.float64)
        pred_arr = np.asarray(predictions, dtype=np.int64)
        branch_binary_prob_arr = (
            np.asarray(branch_binary_probabilities, dtype=np.float64)
            if branch_binary_probabilities
            else None
        )
        branch_binary_target_arr = (
            np.asarray(branch_binary_targets, dtype=np.int64)
            if branch_binary_targets
            else None
        )
        metrics = MetricsComputer.compute(
            target_arr,
            pred_arr,
            prob_arr,
            branch_binary_probabilities=branch_binary_prob_arr,
            branch_binary_targets=branch_binary_target_arr,
        )
        averaged_components = {
            name: value / float(max(1, component_counts[name]))
            for name, value in component_totals.items()
            if component_counts[name] > 0
        }
        return EpochResult(
            loss=total_loss / float(max(1, total_examples)),
            loss_components=averaged_components,
            metrics=metrics,
            probabilities=prob_arr,
            predictions=pred_arr,
            targets=target_arr,
            diagnostics=diagnostics,
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[ClipBatch],
        val_loader: DataLoader[ClipBatch],
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
        train_loss_component_history: list[dict[str, float]] = []
        val_loss_component_history: list[dict[str, float]] = []
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
                epoch=epoch,
                use_monitor_total=False,
            )
            model.eval()
            with torch.no_grad():
                val_result = self._epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    scheduler=None,
                    device=device,
                    epoch=epoch,
                    use_monitor_total=True,
                )

            if self.cfg.num_classes == 2:
                val_threshold_optimization, val_metrics_optimized = (
                    compute_threshold_optimized_metrics(
                        val_result.targets,
                        val_result.probabilities,
                        "f1",
                    )
                )
                best_f1 = val_metrics_optimized.f1_score
            else:
                val_threshold_optimization = ThresholdOptimizationResult.disabled(
                    "f1",
                    reason="threshold optimization is only supported for binary classification",
                )
                val_metrics_optimized = val_result.metrics
                best_f1 = val_result.metrics.f1_score

            train_losses.append(train_result.loss)
            val_losses.append(val_result.loss)
            train_metrics_history.append(train_result.metrics.to_dict())
            val_metrics_history.append(val_result.metrics.to_dict())
            val_metrics_optimized_history.append(val_metrics_optimized.to_dict())
            train_components = dict(train_result.loss_components)
            train_components.update(
                {
                    "train_loss_total_scheduled": train_components.get(
                        "total_scheduled",
                        train_result.loss,
                    ),
                    "train_loss_4cls": train_components.get(
                        "main",
                        train_result.loss,
                    ),
                    "train_loss_branch_binary": train_components.get(
                        "branch_binary_auxiliary",
                        0.0,
                    ),
                    "train_loss_evidence_auxiliary": train_components.get(
                        "evidence_auxiliary_loss",
                        0.0,
                    ),
                    "train_branch_binary_aux_weight": train_components.get(
                        "branch_binary_aux_weight",
                        0.0,
                    ),
                    "train_gate_entropy": train_components.get("gate_entropy", 0.0),
                    "train_loss_gate_entropy_regularization": train_components.get(
                        "gate_entropy_regularization",
                        0.0,
                    ),
                    "train_class_gate_diversity": train_components.get(
                        "class_gate_diversity",
                        0.0,
                    ),
                    "train_loss_class_gate_diversity_regularization": (
                        train_components.get(
                            "class_gate_diversity_regularization",
                            0.0,
                        )
                    ),
                }
            )
            val_components = dict(val_result.loss_components)
            val_components.update(
                {
                    "val_loss_total_scheduled": val_components.get(
                        "total_scheduled",
                        val_result.loss,
                    ),
                    "val_loss_total_monitor": val_components.get(
                        "total_monitor",
                        val_result.loss,
                    ),
                    "val_loss_4cls": val_components.get(
                        "main",
                        val_result.loss,
                    ),
                    "val_loss_branch_binary": val_components.get(
                        "branch_binary_auxiliary",
                        0.0,
                    ),
                    "val_loss_evidence_auxiliary": val_components.get(
                        "evidence_auxiliary_loss",
                        0.0,
                    ),
                    "val_branch_binary_aux_weight": val_components.get(
                        "branch_binary_aux_weight",
                        0.0,
                    ),
                    "val_branch_binary_aux_monitor_weight": val_components.get(
                        "branch_binary_aux_monitor_weight",
                        0.0,
                    ),
                    "val_gate_entropy": val_components.get("gate_entropy", 0.0),
                    "val_loss_gate_entropy_regularization": val_components.get(
                        "gate_entropy_regularization",
                        0.0,
                    ),
                    "val_class_gate_diversity": val_components.get(
                        "class_gate_diversity",
                        0.0,
                    ),
                    "val_loss_class_gate_diversity_regularization": (
                        val_components.get(
                            "class_gate_diversity_regularization",
                            0.0,
                        )
                    ),
                }
            )
            train_loss_component_history.append(train_components)
            val_loss_component_history.append(val_components)

            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            lr_str = (
                f"{lrs[0]:.8f}"
                if len(lrs) == 1
                else "[" + ", ".join(f"{lr:.8f}" for lr in lrs) + "]"
            )
            if self.cfg.num_classes == 2:
                self.logger.info(
                    "Epoch %d/%d | LR: %s | Train Loss: %.4f | Train Acc: %.4f | "
                    "Train Recall: %.4f | Train Precision: %.4f | Train F1: %.4f | "
                    "Val Loss: %.4f | Val Acc: %.4f | Val Recall: %.4f | "
                    "Val Precision: %.4f | Val F1@0.5: %.4f | Val Balanced Acc@0.5: %.4f | "
                    "Val F1@opt: %.4f | Val Balanced Acc@opt: %.4f | Val Opt Threshold: %.4f | "
                    "Train Main Loss: %.4f | Val Main Loss: %.4f | "
                    "Train Branch Binary Loss: %.4f | Val Branch Binary Loss: %.4f | "
                    "Branch Binary Weight: %.4f | Train Gate Entropy: %.4f | "
                    "Val Gate Entropy: %.4f | Train Evidence Aux Loss: %.4f | "
                    "Val Evidence Aux Loss: %.4f | Train Class Gate Diversity: %.4f | "
                    "Val Class Gate Diversity: %.4f | Val Scheduled Loss: %.4f",
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
                    train_components.get("main", train_result.loss),
                    val_components.get("main", val_result.loss),
                    train_components.get("branch_binary_auxiliary", 0.0),
                    val_components.get("branch_binary_auxiliary", 0.0),
                    train_components.get("branch_binary_aux_weight", 0.0),
                    train_components.get("gate_entropy", 0.0),
                    val_components.get("gate_entropy", 0.0),
                    train_components.get("evidence_auxiliary_loss", 0.0),
                    val_components.get("evidence_auxiliary_loss", 0.0),
                    train_components.get("class_gate_diversity", 0.0),
                    val_components.get("class_gate_diversity", 0.0),
                    val_components.get("total_scheduled", val_result.loss),
                )
            else:
                self.logger.info(
                    "Epoch %d/%d | LR: %s | Train Loss: %.4f | Train Acc: %.4f | "
                    "Train Recall: %.4f | Train Precision: %.4f | Train F1: %.4f | "
                    "Val Loss: %.4f | Val Acc: %.4f | Val Recall: %.4f | "
                    "Val Precision: %.4f | Val F1: %.4f | Val Balanced Acc: %.4f | "
                    "Train Main Loss: %.4f | Val Main Loss: %.4f | "
                    "Train Branch Binary Loss: %.4f | Val Branch Binary Loss: %.4f | "
                    "Branch Binary Weight: %.4f | Train Gate Entropy: %.4f | "
                    "Val Gate Entropy: %.4f | Train Evidence Aux Loss: %.4f | "
                    "Val Evidence Aux Loss: %.4f | Train Class Gate Diversity: %.4f | "
                    "Val Class Gate Diversity: %.4f | Val Scheduled Loss: %.4f",
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
                    train_components.get("main", train_result.loss),
                    val_components.get("main", val_result.loss),
                    train_components.get("branch_binary_auxiliary", 0.0),
                    val_components.get("branch_binary_auxiliary", 0.0),
                    train_components.get("branch_binary_aux_weight", 0.0),
                    train_components.get("gate_entropy", 0.0),
                    val_components.get("gate_entropy", 0.0),
                    train_components.get("evidence_auxiliary_loss", 0.0),
                    val_components.get("evidence_auxiliary_loss", 0.0),
                    train_components.get("class_gate_diversity", 0.0),
                    val_components.get("class_gate_diversity", 0.0),
                    val_components.get("total_scheduled", val_result.loss),
                )

            current_monitor_value = val_components.get(
                "val_loss_total_monitor",
                val_result.loss,
            )
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
                encoder_cfg = getattr(model_cfg, "encoder", None)
                feature_dims = getattr(encoder_cfg, "feature_dims", None)
                if feature_dims is not None and is_dataclass(feature_dims):
                    dims = asdict(feature_dims)

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
                "train_loss_components": train_loss_component_history,
                "val_loss_components": val_loss_component_history,
                "early_stopping": early_stopping_state,
                "optimizer_group_lrs": lrs,
                "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
                "branch_binary_auxiliary_schedule": asdict(
                    self.cfg.branch_binary_auxiliary.schedule
                ),
                "branch_binary_aux_weight": train_components.get(
                    "branch_binary_aux_weight",
                    0.0,
                ),
                "branch_binary_aux_monitor_weight": val_components.get(
                    "branch_binary_aux_monitor_weight",
                    0.0,
                ),
                "current_epoch_loss_components": {
                    "train": train_components,
                    "val": val_components,
                },
            }
            if extra_state:
                for key, value in extra_state.items():
                    if is_dataclass(value) and not isinstance(value, type):
                        state[key] = asdict(value)
                    else:
                        state[key] = value
            self.ckpt.save_last(state)
            if self.cfg.checkpointing is None:
                self.ckpt.maybe_save_best(
                    "loss", val_result.loss, state, maximize=False
                )
                self.ckpt.maybe_save_best("f1", best_f1, state, maximize=True)
            else:
                self.ckpt.save_configured_monitors(
                    {
                        "val_macro_f1": val_result.metrics.macro_f1
                        if val_result.metrics.macro_f1 is not None
                        else val_result.metrics.f1_score,
                        "val_macro_recall": val_result.metrics.macro_recall
                        if val_result.metrics.macro_recall is not None
                        else val_result.metrics.recall,
                        "val_loss": val_result.loss,
                        "val_loss_total_monitor": val_components.get(
                            "val_loss_total_monitor",
                            val_result.loss,
                        ),
                    },
                    state,
                    epoch=epoch,
                )

            if stopped_early:
                self.logger.info("%s", stop_reason)
                break
