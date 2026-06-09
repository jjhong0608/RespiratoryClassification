from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
from src.training.epoch_logging import (
    EpochLogContext,
    append_epoch_jsonl_logs,
    format_epoch_log_block,
)
from src.training.losses import FocalLoss
from src.training.scheduler import WarmupCosineScheduler
from src.utils.config import (
    AnalysisConfig,
    AttentionEntropyLossConfig,
    BranchAuxiliaryLossConfig,
    BranchBinaryAuxiliaryLossConfig,
    BranchDirectScoreMarginConfig,
    BranchPathDominanceConstraintConfig,
    BranchSupportDisagreementCapRegularizationConfig,
    BranchSupportScoreMarginConfig,
    BranchToEvidenceRankingConsistencyConfig,
    CheckpointingConfig,
    ClassEvidenceGapCapRegularizationConfig,
    ClassEvidenceMarginConfig,
    ClassEvidencePositiveGapCapRegularizationConfig,
    ClassGatedBranchLogitMarginConfig,
    ClassGateDiversityRegularizationConfig,
    ClassTopBranchRelativeMarginConfig,
    EarlyStoppingConfig,
    GateBadBranchSuppressionConfig,
    GateBranchRegretConfig,
    GateBranchRegretWeightScheduleConfig,
    GateEntropyRegularizationConfig,
    GateWeightedBranchMarginConfig,
    GlobalResidualAntiVetoConfig,
    InteractionGapCapRegularizationConfig,
    LabelSmoothingConfig,
    ResidualContradictionRegularizationConfig,
    TopBranchMarginConfig,
    TopSupportGapMinConstraintConfig,
    TopSupportScoreMarginConfig,
    TopTeacherGapMinConstraintConfig,
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
    class_names: tuple[str, ...] = ()
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
    class_evidence_margin: ClassEvidenceMarginConfig = field(
        default_factory=ClassEvidenceMarginConfig
    )
    class_evidence_margin_major_index: int | None = None
    class_evidence_gap_cap_regularization: ClassEvidenceGapCapRegularizationConfig = (
        field(default_factory=ClassEvidenceGapCapRegularizationConfig)
    )
    class_evidence_gap_cap_negative_cap_by_class: tuple[float, ...] | None = None
    class_evidence_gap_cap_label_weight_by_class: tuple[float, ...] | None = None
    class_evidence_positive_gap_cap_regularization: ClassEvidencePositiveGapCapRegularizationConfig = field(
        default_factory=ClassEvidencePositiveGapCapRegularizationConfig
    )
    interaction_gap_cap_regularization: InteractionGapCapRegularizationConfig = field(
        default_factory=InteractionGapCapRegularizationConfig
    )
    top_support_score_margin: TopSupportScoreMarginConfig = field(
        default_factory=TopSupportScoreMarginConfig
    )
    top_support_score_margin_label_weight_by_class: tuple[float, ...] | None = None
    top_support_gap_min_constraint: TopSupportGapMinConstraintConfig = field(
        default_factory=TopSupportGapMinConstraintConfig
    )
    top_support_gap_min_base_by_class: tuple[float, ...] | None = None
    class_top_branch_relative_margin: ClassTopBranchRelativeMarginConfig = field(
        default_factory=ClassTopBranchRelativeMarginConfig
    )
    top_teacher_gap_min_constraint: TopTeacherGapMinConstraintConfig = field(
        default_factory=TopTeacherGapMinConstraintConfig
    )
    branch_support_score_margin: BranchSupportScoreMarginConfig = field(
        default_factory=BranchSupportScoreMarginConfig
    )
    branch_direct_score_margin: BranchDirectScoreMarginConfig = field(
        default_factory=BranchDirectScoreMarginConfig
    )
    branch_path_dominance_constraint: BranchPathDominanceConstraintConfig = field(
        default_factory=BranchPathDominanceConstraintConfig
    )
    branch_path_dominance_allowed_drop_by_class: tuple[float, ...] | None = None
    branch_path_dominance_label_weight_by_class: tuple[float, ...] | None = None
    branch_support_disagreement_cap_regularization: BranchSupportDisagreementCapRegularizationConfig = field(
        default_factory=BranchSupportDisagreementCapRegularizationConfig
    )
    class_gated_branch_logit_margin: ClassGatedBranchLogitMarginConfig = field(
        default_factory=ClassGatedBranchLogitMarginConfig
    )
    branch_to_evidence_ranking_consistency: BranchToEvidenceRankingConsistencyConfig = (
        field(default_factory=BranchToEvidenceRankingConsistencyConfig)
    )
    branch_to_evidence_teacher_floor_by_class: tuple[float, ...] | None = None
    global_residual_anti_veto: GlobalResidualAntiVetoConfig = field(
        default_factory=GlobalResidualAntiVetoConfig
    )
    residual_contradiction_regularization: ResidualContradictionRegularizationConfig = (
        field(default_factory=ResidualContradictionRegularizationConfig)
    )
    gate_weighted_branch_margin: GateWeightedBranchMarginConfig = field(
        default_factory=GateWeightedBranchMarginConfig
    )
    gate_branch_regret: GateBranchRegretConfig = field(
        default_factory=GateBranchRegretConfig
    )
    gate_branch_regret_positive_threshold_by_class: tuple[float, ...] | None = None
    gate_bad_branch_suppression: GateBadBranchSuppressionConfig = field(
        default_factory=GateBadBranchSuppressionConfig
    )
    gate_bad_branch_suppression_threshold_by_class: tuple[float, ...] | None = None
    top_branch_margin: TopBranchMarginConfig = field(
        default_factory=TopBranchMarginConfig
    )
    top_branch_margin_by_class: tuple[float, ...] | None = None
    top_branch_margin_phase_start_multiplier_by_class: tuple[float, ...] | None = None
    top_branch_margin_phase_label_multiplier_by_class: tuple[float, ...] | None = None
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
    class_evidence_margin: Tensor | None = None
    class_evidence_margin_loss: Tensor | None = None
    class_evidence_gap_cap_regularization: Tensor | None = None
    class_evidence_gap_cap_regularization_loss: Tensor | None = None
    class_evidence_gap_cap_label_multiplier: Tensor | None = None
    class_evidence_gap_cap_effective_cap: Tensor | None = None
    class_evidence_positive_gap_cap_regularization: Tensor | None = None
    class_evidence_positive_gap_cap_regularization_loss: Tensor | None = None
    interaction_gap_cap_regularization: Tensor | None = None
    interaction_gap_cap_regularization_loss: Tensor | None = None
    top_support_score_margin: Tensor | None = None
    top_support_score_margin_loss: Tensor | None = None
    top_support_score_margin_label_multiplier: Tensor | None = None
    top_support_score_margin_support_multiplier: Tensor | None = None
    top_support_score_margin_hardness_multiplier: Tensor | None = None
    top_support_gap_min_constraint: Tensor | None = None
    top_support_gap_min_constraint_loss: Tensor | None = None
    top_support_gap_min_target: Tensor | None = None
    class_top_branch_relative_margin: Tensor | None = None
    class_top_branch_relative_margin_loss: Tensor | None = None
    class_top_branch_relative_margin_support_multiplier: Tensor | None = None
    class_top_branch_relative_margin_hardness_multiplier: Tensor | None = None
    top_teacher_gap_min_constraint: Tensor | None = None
    top_teacher_gap_min_constraint_loss: Tensor | None = None
    top_teacher_gap_min_target: Tensor | None = None
    branch_direct_score_margin: Tensor | None = None
    branch_direct_score_margin_loss: Tensor | None = None
    branch_path_dominance_constraint: Tensor | None = None
    branch_path_dominance_constraint_loss: Tensor | None = None
    branch_path_dominance_support_multiplier: Tensor | None = None
    branch_path_dominance_label_multiplier: Tensor | None = None
    branch_path_dominance_allowed_drop: Tensor | None = None
    branch_support_disagreement_cap_regularization: Tensor | None = None
    branch_support_disagreement_cap_regularization_loss: Tensor | None = None
    branch_support_disagreement_weight: Tensor | None = None
    branch_support_score_margin: Tensor | None = None
    branch_support_score_margin_loss: Tensor | None = None
    class_gated_branch_logit_margin: Tensor | None = None
    class_gated_branch_logit_margin_loss: Tensor | None = None
    branch_to_evidence_ranking_consistency: Tensor | None = None
    branch_to_evidence_ranking_consistency_loss: Tensor | None = None
    global_residual_anti_veto: Tensor | None = None
    global_residual_anti_veto_loss: Tensor | None = None
    global_residual_anti_veto_eligible_fraction: Tensor | None = None
    residual_contradiction_regularization: Tensor | None = None
    residual_contradiction_regularization_loss: Tensor | None = None
    residual_contradiction_regularization_eligible_fraction: Tensor | None = None
    gate_weighted_branch_margin: Tensor | None = None
    gate_weighted_branch_margin_loss: Tensor | None = None
    gate_branch_regret: Tensor | None = None
    gate_branch_regret_loss: Tensor | None = None
    gate_branch_regret_eligible_fraction: Tensor | None = None
    gate_branch_regret_weight_multiplier: Tensor | None = None
    gate_branch_regret_effective_weight: Tensor | None = None
    gate_bad_branch_suppression: Tensor | None = None
    gate_bad_branch_suppression_loss: Tensor | None = None
    gate_bad_branch_suppression_bad_gate_mass: Tensor | None = None
    gate_bad_branch_suppression_weight_multiplier: Tensor | None = None
    gate_bad_branch_suppression_effective_weight: Tensor | None = None
    top_branch_margin: Tensor | None = None
    top_branch_margin_loss: Tensor | None = None
    top_branch_margin_effective_weight: Tensor | None = None
    top_branch_margin_hardness_multiplier: Tensor | None = None
    branch_binary_aux_weight: float = 0.0
    branch_binary_aux_monitor_weight: float = 0.0


@dataclass(frozen=True)
class EpochResult:
    loss: float
    loss_components: dict[str, Any]
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


@dataclass
class AdaptiveBranchObjectiveState:
    top_branch_margin_by_class: list[float] | None
    gate_branch_regret_positive_threshold_by_class: list[float] | None
    gate_bad_branch_suppression_threshold_by_class: list[float] | None
    top_branch_violation_rate_ema_by_class: list[float | None]
    gate_branch_regret_eligible_rate_ema_by_class: list[float | None]
    gate_bad_branch_suppression_bad_gate_mass_ema_by_class: list[float | None]
    last_update_epoch: int = 0


@dataclass
class BranchObjectiveEpochStats:
    top_counts: list[int]
    top_violation_counts: list[int]
    regret_counts: list[int]
    regret_eligible_counts: list[int]
    bad_suppression_counts: list[int]
    bad_suppression_gate_mass_sums: list[float]


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
        self._adaptive_branch_objective_state = (
            self._init_adaptive_branch_objective_state()
        )

    def _init_adaptive_branch_objective_state(self) -> AdaptiveBranchObjectiveState:
        num_classes = int(self.cfg.num_classes)
        top_margin_by_class: list[float] | None = None
        if self.cfg.top_branch_margin.enabled:
            top_margin_by_class = list(
                self.cfg.top_branch_margin_by_class
                or (float(self.cfg.top_branch_margin.margin),) * num_classes
            )
            if len(top_margin_by_class) != num_classes:
                raise ValueError(
                    "top_branch_margin_by_class length must match num_classes"
                )
        regret_threshold_by_class: list[float] | None = None
        if self.cfg.gate_branch_regret.enabled:
            regret_threshold_by_class = list(
                self.cfg.gate_branch_regret_positive_threshold_by_class
                or (float(self.cfg.gate_branch_regret.positive_threshold),)
                * num_classes
            )
            if len(regret_threshold_by_class) != num_classes:
                raise ValueError(
                    "gate_branch_regret_positive_threshold_by_class length must "
                    "match num_classes"
                )
        bad_threshold_by_class: list[float] | None = None
        if self.cfg.gate_bad_branch_suppression.enabled:
            bad_threshold_by_class = list(
                self.cfg.gate_bad_branch_suppression_threshold_by_class
                or (float(self.cfg.gate_bad_branch_suppression.bad_margin_threshold),)
                * num_classes
            )
            if len(bad_threshold_by_class) != num_classes:
                raise ValueError(
                    "gate_bad_branch_suppression_threshold_by_class length must "
                    "match num_classes"
                )
        return AdaptiveBranchObjectiveState(
            top_branch_margin_by_class=top_margin_by_class,
            gate_branch_regret_positive_threshold_by_class=regret_threshold_by_class,
            gate_bad_branch_suppression_threshold_by_class=bad_threshold_by_class,
            top_branch_violation_rate_ema_by_class=[None] * num_classes,
            gate_branch_regret_eligible_rate_ema_by_class=[None] * num_classes,
            gate_bad_branch_suppression_bad_gate_mass_ema_by_class=[None] * num_classes,
        )

    @staticmethod
    def _epoch_window_active(
        *,
        enabled: bool,
        start_epoch: int,
        end_epoch: int | None,
        epoch: int,
    ) -> bool:
        if not enabled:
            return False
        if int(epoch) < int(start_epoch):
            return False
        return not (end_epoch is not None and int(epoch) > int(end_epoch))

    def _class_names(self) -> tuple[str, ...]:
        if self.cfg.class_names:
            return self.cfg.class_names
        return tuple(f"class_{index}" for index in range(int(self.cfg.num_classes)))

    def _label_value_dict(
        self, values: Sequence[float | None] | None
    ) -> dict[str, float | None]:
        if values is None:
            return {}
        return {
            label_name: values[index]
            for index, label_name in enumerate(self._class_names())
        }

    def _branch_objective_state_dict(self) -> dict[str, Any]:
        state = self._adaptive_branch_objective_state
        return {
            "top_branch_margin_by_class": (
                list(state.top_branch_margin_by_class)
                if state.top_branch_margin_by_class is not None
                else None
            ),
            "top_branch_margin_by_label": self._label_value_dict(
                state.top_branch_margin_by_class
            ),
            "gate_branch_regret_positive_threshold_by_class": (
                list(state.gate_branch_regret_positive_threshold_by_class)
                if state.gate_branch_regret_positive_threshold_by_class is not None
                else None
            ),
            "gate_branch_regret_positive_threshold_by_label": self._label_value_dict(
                state.gate_branch_regret_positive_threshold_by_class
            ),
            "gate_bad_branch_suppression_threshold_by_class": (
                list(state.gate_bad_branch_suppression_threshold_by_class)
                if state.gate_bad_branch_suppression_threshold_by_class is not None
                else None
            ),
            "gate_bad_branch_suppression_threshold_by_label": self._label_value_dict(
                state.gate_bad_branch_suppression_threshold_by_class
            ),
            "top_branch_violation_rate_ema_by_class": list(
                state.top_branch_violation_rate_ema_by_class
            ),
            "top_branch_violation_rate_ema_by_label": self._label_value_dict(
                state.top_branch_violation_rate_ema_by_class
            ),
            "gate_branch_regret_eligible_rate_ema_by_class": list(
                state.gate_branch_regret_eligible_rate_ema_by_class
            ),
            "gate_branch_regret_eligible_rate_ema_by_label": self._label_value_dict(
                state.gate_branch_regret_eligible_rate_ema_by_class
            ),
            "gate_bad_branch_suppression_bad_gate_mass_ema_by_class": list(
                state.gate_bad_branch_suppression_bad_gate_mass_ema_by_class
            ),
            "gate_bad_branch_suppression_bad_gate_mass_ema_by_label": (
                self._label_value_dict(
                    state.gate_bad_branch_suppression_bad_gate_mass_ema_by_class
                )
            ),
            "last_update_epoch": state.last_update_epoch,
        }

    def _format_label_values(self, values: Mapping[str, float | None]) -> str:
        if not values:
            return "{}"
        return (
            "{"
            + ", ".join(
                f"{label}=" + ("None" if value is None else f"{float(value):.4f}")
                for label, value in values.items()
            )
            + "}"
        )

    def _class_values_tensor(
        self,
        values_by_class: Sequence[float] | None,
        label_indices: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        fallback: float,
    ) -> Tensor:
        if values_by_class is None:
            return torch.full_like(label_indices, float(fallback), dtype=dtype).to(
                device=device
            )
        values = torch.tensor(values_by_class, device=device, dtype=dtype)
        if label_indices.numel() > 0 and int(label_indices.max().item()) >= len(
            values_by_class
        ):
            raise ValueError("labels contain a class index outside adaptive values")
        return values[label_indices.to(device=device, dtype=torch.long)]

    def _branch_to_evidence_teacher_floors(
        self,
        label_indices: Tensor,
        reference: Tensor,
    ) -> Tensor:
        return self._class_values_tensor(
            self.cfg.branch_to_evidence_teacher_floor_by_class,
            label_indices,
            device=reference.device,
            dtype=reference.dtype,
            fallback=0.0,
        )

    def _top_branch_support_gap(
        self,
        output: AstModelOutput,
        labels: Tensor,
        label_indices: Tensor,
        *,
        loss_name: str,
    ) -> Tensor:
        branch_logits, branch_label_indices = self._branch_margin_inputs(
            output,
            labels,
            loss_name=loss_name,
        )
        if not torch.equal(branch_label_indices, label_indices):
            raise ValueError(
                f"branch_logits labels must match class logits labels for {loss_name}"
            )
        support_gap, _, _ = self._top_true_class_branch_margin(
            branch_logits,
            label_indices,
        )
        return support_gap

    def _margin_support_weights(
        self,
        output: AstModelOutput,
        labels: Tensor,
        label_indices: Tensor,
        reference: Tensor,
        cfg: Any,
        *,
        loss_name: str,
    ) -> tuple[Tensor, Tensor]:
        if not cfg.enabled:
            ones = torch.ones_like(reference)
            return ones, reference.detach() * 0.0
        if cfg.source != "top_branch_margin" or cfg.mode != "linear":
            raise ValueError(
                f"{loss_name}.support_weighting supports only "
                "source='top_branch_margin' and mode='linear'"
            )
        support_gap = self._top_branch_support_gap(
            output,
            labels,
            label_indices,
            loss_name=loss_name,
        ).detach()
        support = torch.clamp(support_gap, min=0.0, max=float(cfg.cap))
        support_weight = 1.0 + (float(cfg.gain) * support)
        return support_weight, support_gap

    @staticmethod
    def _margin_hardness_weights(
        margin_gap: Tensor,
        cfg: Any,
        *,
        loss_name: str,
        expected_source: str = "evidence_gap",
    ) -> tuple[Tensor, Tensor]:
        if not cfg.enabled:
            ones = torch.ones_like(margin_gap)
            return ones, margin_gap.detach() * 0.0
        if cfg.source != expected_source or cfg.mode != "negative_gap":
            raise ValueError(
                f"{loss_name}.hardness_weighting supports only "
                f"source='{expected_source}' and mode='negative_gap'"
            )
        hardness = torch.relu(-margin_gap)
        hard_weight = 1.0 + (
            float(cfg.gain) * torch.clamp(hardness, min=0.0, max=float(cfg.cap))
        )
        return hard_weight, hardness

    @staticmethod
    def _top_branch_margin_hardness_weights(
        margin_deficit: Tensor,
        cfg: Any,
        *,
        loss_name: str,
    ) -> tuple[Tensor, Tensor]:
        if not cfg.enabled:
            ones = torch.ones_like(margin_deficit)
            return ones, margin_deficit.detach() * 0.0
        if cfg.source != "margin_deficit" or cfg.mode != "linear":
            raise ValueError(
                f"{loss_name}.hardness_weighting supports only "
                "source='margin_deficit' and mode='linear'"
            )
        hardness = torch.clamp(margin_deficit.detach(), min=0.0)
        hard_weight = 1.0 + (
            float(cfg.gain) * torch.clamp(hardness, min=0.0, max=float(cfg.cap))
        )
        return hard_weight, hardness

    def _top_branch_margin_targets(
        self, label_indices: Tensor, reference: Tensor
    ) -> Tensor:
        return self._class_values_tensor(
            self._adaptive_branch_objective_state.top_branch_margin_by_class,
            label_indices,
            device=reference.device,
            dtype=reference.dtype,
            fallback=float(self.cfg.top_branch_margin.margin),
        )

    def _top_branch_margin_phase_multipliers(
        self,
        label_indices: Tensor,
        reference: Tensor,
        *,
        epoch: int,
    ) -> Tensor:
        schedule = self.cfg.top_branch_margin.phase_weight_schedule
        if not schedule.enabled or int(epoch) < int(schedule.start_epoch):
            return torch.ones_like(reference)
        end_multipliers = self._class_values_tensor(
            self.cfg.top_branch_margin_phase_label_multiplier_by_class,
            label_indices,
            device=reference.device,
            dtype=reference.dtype,
            fallback=1.0,
        )
        if schedule.end_epoch is None:
            return end_multipliers
        start_multipliers = self._class_values_tensor(
            self.cfg.top_branch_margin_phase_start_multiplier_by_class,
            label_indices,
            device=reference.device,
            dtype=reference.dtype,
            fallback=1.0,
        )
        if int(epoch) >= int(schedule.end_epoch):
            return end_multipliers
        epoch_span = max(1, int(schedule.end_epoch) - int(schedule.start_epoch))
        alpha = float(int(epoch) - int(schedule.start_epoch)) / float(epoch_span)
        return start_multipliers + (alpha * (end_multipliers - start_multipliers))

    def _gate_branch_regret_thresholds(
        self,
        label_indices: Tensor,
        reference: Tensor,
    ) -> Tensor:
        return self._class_values_tensor(
            self._adaptive_branch_objective_state.gate_branch_regret_positive_threshold_by_class,
            label_indices,
            device=reference.device,
            dtype=reference.dtype,
            fallback=float(self.cfg.gate_branch_regret.positive_threshold),
        )

    def _gate_bad_branch_suppression_thresholds(
        self,
        label_indices: Tensor,
        reference: Tensor,
    ) -> Tensor:
        return self._class_values_tensor(
            self._adaptive_branch_objective_state.gate_bad_branch_suppression_threshold_by_class,
            label_indices,
            device=reference.device,
            dtype=reference.dtype,
            fallback=float(self.cfg.gate_bad_branch_suppression.bad_margin_threshold),
        )

    @staticmethod
    def _scheduled_weight_multiplier(
        *,
        epoch: int,
        warmup_epochs: int,
        schedule: GateBranchRegretWeightScheduleConfig,
    ) -> float:
        if int(epoch) <= int(warmup_epochs):
            return 0.0
        if not schedule.enabled:
            return 1.0
        if int(epoch) < int(schedule.start_epoch):
            return float(schedule.start_multiplier)
        if int(epoch) > int(schedule.end_epoch):
            return float(schedule.end_multiplier)
        if int(schedule.start_epoch) == int(schedule.end_epoch):
            return float(schedule.end_multiplier)
        progress = (float(epoch) - float(schedule.start_epoch)) / (
            float(schedule.end_epoch) - float(schedule.start_epoch)
        )
        return float(schedule.start_multiplier) + (
            progress
            * (float(schedule.end_multiplier) - float(schedule.start_multiplier))
        )

    def _gate_branch_regret_weight_multiplier(self, epoch: int) -> float:
        return self._scheduled_weight_multiplier(
            epoch=epoch,
            warmup_epochs=self.cfg.gate_branch_regret.warmup_epochs,
            schedule=self.cfg.gate_branch_regret.weight_schedule,
        )

    def _gate_bad_branch_suppression_weight_multiplier(self, epoch: int) -> float:
        return self._scheduled_weight_multiplier(
            epoch=epoch,
            warmup_epochs=self.cfg.gate_bad_branch_suppression.warmup_epochs,
            schedule=self.cfg.gate_bad_branch_suppression.weight_schedule,
        )

    def _new_branch_objective_epoch_stats(self) -> BranchObjectiveEpochStats:
        num_classes = int(self.cfg.num_classes)
        return BranchObjectiveEpochStats(
            top_counts=[0] * num_classes,
            top_violation_counts=[0] * num_classes,
            regret_counts=[0] * num_classes,
            regret_eligible_counts=[0] * num_classes,
            bad_suppression_counts=[0] * num_classes,
            bad_suppression_gate_mass_sums=[0.0] * num_classes,
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

    def _uses_binary_sigmoid_output(self) -> bool:
        return self.cfg.loss_type in {"bce", "focal"}

    def _compute_main_loss(
        self,
        criterion: nn.Module,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        if self._uses_binary_sigmoid_output():
            return criterion(
                logits, labels.to(device=logits.device, dtype=logits.dtype)
            )
        return criterion(logits, labels.to(device=logits.device, dtype=torch.long))

    def _predict(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        if self._uses_binary_sigmoid_output():
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
        if self._uses_binary_sigmoid_output():
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
        elif target == "class_evidence_learned_gate":
            if output.class_evidence_learned_gate_entropy is None:
                raise ValueError(
                    "gate entropy regularization enabled but model did not return "
                    "class_evidence_learned_gate_entropy"
                )
            entropy_values = output.class_evidence_learned_gate_entropy
        else:
            raise ValueError(
                "gate entropy regularization target must be one of "
                "'evidence_gate', 'class_evidence_gate', "
                "'true_class_evidence_gate', 'class_evidence_learned_gate'"
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
        target = self.cfg.class_gate_diversity_regularization.target
        if target not in {"class_evidence_gate", "class_evidence_learned_gate"}:
            raise ValueError(
                "class gate diversity regularization supports only "
                "target='class_evidence_gate' or "
                "target='class_evidence_learned_gate'"
            )
        if self.cfg.class_gate_diversity_regularization.metric != "js_divergence":
            raise ValueError(
                "class gate diversity regularization supports only "
                "metric='js_divergence'"
            )
        if target == "class_evidence_learned_gate":
            gate_weights = output.class_evidence_learned_gate_weights
            gate_name = "class_evidence_learned_gate_weights"
        else:
            gate_weights = output.class_evidence_gate_weights
            gate_name = "class_evidence_gate_weights"
        if gate_weights is None:
            raise ValueError(
                "class gate diversity regularization enabled but model did not return "
                f"{gate_name}"
            )
        if gate_weights.ndim != 3:
            raise ValueError(
                f"{gate_name} must have shape (B, C, R), "
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

    def _validate_class_margin_inputs(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        logits_name: str,
        loss_name: str,
    ) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(
                f"{logits_name} must have shape (B, C), got {tuple(logits.shape)}"
            )
        if int(logits.shape[1]) < 2:
            raise ValueError(f"{loss_name} requires at least two classes")
        label_indices = labels.to(device=logits.device, dtype=torch.long)
        if label_indices.ndim != 1:
            raise ValueError(f"labels must have shape (B,) for {loss_name}")
        if int(label_indices.shape[0]) != int(logits.shape[0]):
            raise ValueError(f"labels batch size must match {logits_name} batch size")
        if label_indices.numel() == 0:
            return label_indices
        if int(label_indices.min().item()) < 0 or int(
            label_indices.max().item()
        ) >= int(logits.shape[1]):
            raise ValueError(f"labels contain a class index outside {logits_name}")
        return label_indices

    def _class_margin_weights(
        self,
        label_indices: Tensor,
        logits: Tensor,
        *,
        enabled: bool,
        loss_name: str,
    ) -> Tensor:
        if not enabled:
            return torch.ones_like(label_indices, dtype=logits.dtype)
        if self.cfg.class_weights is None:
            raise ValueError(f"{loss_name}.class_weighted=true requires class_weights")
        class_weights = torch.tensor(
            self.cfg.class_weights,
            device=logits.device,
            dtype=logits.dtype,
        )
        if int(class_weights.numel()) != int(logits.shape[1]):
            raise ValueError(
                f"class_weights length must match {loss_name} class dimension"
            )
        return class_weights.gather(0, label_indices)

    def _true_vs_hardest_negative_penalties(
        self,
        logits: Tensor,
        label_indices: Tensor,
        *,
        margin: float,
    ) -> Tensor:
        margin_gap, _ = self._true_vs_hardest_negative_gap(logits, label_indices)
        return torch.relu(margin - margin_gap)

    def _true_vs_hardest_negative_gap(
        self,
        logits: Tensor,
        label_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        true_logits = logits.gather(1, label_indices.unsqueeze(1)).squeeze(1)
        negative_logits = logits.masked_fill(
            F.one_hot(
                label_indices,
                num_classes=int(logits.shape[1]),
            ).to(dtype=torch.bool, device=logits.device),
            -torch.inf,
        )
        hardest_negative_values, hardest_negative_indices = negative_logits.max(dim=1)
        return true_logits - hardest_negative_values, hardest_negative_indices

    def _reduce_class_margin_penalties(
        self,
        penalties: Tensor,
        label_indices: Tensor,
        class_weights: Tensor,
        *,
        reduction: str,
    ) -> Tensor:
        weighted_penalties = penalties * class_weights
        if reduction == "mean":
            return weighted_penalties.mean()
        if reduction == "class_balanced_violating_mean":
            violating = penalties > 0
            if not bool(violating.any()):
                return weighted_penalties.sum() * 0.0
            class_losses: list[Tensor] = []
            for class_index in torch.unique(label_indices[violating]):
                class_mask = (label_indices == class_index) & violating
                class_losses.append(weighted_penalties[class_mask].mean())
            return torch.stack(class_losses).mean()
        raise ValueError(
            "margin reduction must be 'mean' or 'class_balanced_violating_mean'"
        )

    def _compute_class_evidence_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.cfg.class_evidence_margin.target != "class_evidence_logits":
            raise ValueError(
                "class evidence margin supports only target='class_evidence_logits'"
            )
        if output.class_evidence_logits is None:
            raise ValueError(
                "class evidence margin enabled but model did not return "
                "class_evidence_logits"
            )
        logits = output.class_evidence_logits
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="class evidence margin",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss

        true_logits = logits.gather(1, label_indices.unsqueeze(1)).squeeze(1)
        margin = float(self.cfg.class_evidence_margin.margin)
        mode = self.cfg.class_evidence_margin.mode
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=self.cfg.class_evidence_margin.class_weighted,
            loss_name="class_evidence_margin",
        )
        if mode == "minority_vs_major":
            major_index = self.cfg.class_evidence_margin_major_index
            if major_index is None:
                raise ValueError(
                    "class evidence margin mode='minority_vs_major' requires "
                    "class_evidence_margin_major_index"
                )
            if major_index < 0 or major_index >= int(logits.shape[1]):
                raise ValueError(
                    "class_evidence_margin_major_index is outside class_evidence_logits"
                )
            eligible = label_indices != int(major_index)
            if not bool(eligible.any()):
                raw_loss = logits.sum() * 0.0
            else:
                major_logits = logits[:, int(major_index)]
                margin_gap = true_logits - major_logits
                support_weight, _ = self._margin_support_weights(
                    output,
                    labels,
                    label_indices,
                    margin_gap,
                    self.cfg.class_evidence_margin.support_weighting,
                    loss_name="class_evidence_margin",
                )
                penalties = torch.relu(margin - margin_gap[eligible])
                penalties = penalties * support_weight[eligible]
                raw_loss = self._reduce_class_margin_penalties(
                    penalties,
                    label_indices[eligible],
                    class_weights[eligible],
                    reduction=self.cfg.class_evidence_margin.reduction,
                )
        elif mode == "true_vs_hardest_negative":
            margin_gap, _ = self._true_vs_hardest_negative_gap(
                logits,
                label_indices,
            )
            support_weight, _ = self._margin_support_weights(
                output,
                labels,
                label_indices,
                margin_gap,
                self.cfg.class_evidence_margin.support_weighting,
                loss_name="class_evidence_margin",
            )
            penalties = torch.relu(margin - margin_gap) * support_weight
            raw_loss = self._reduce_class_margin_penalties(
                penalties,
                label_indices,
                class_weights,
                reduction=self.cfg.class_evidence_margin.reduction,
            )
        elif mode == "softplus_true_vs_hardest_negative":
            margin_gap, _ = self._true_vs_hardest_negative_gap(
                logits,
                label_indices,
            )
            temperature = float(self.cfg.class_evidence_margin.temperature)
            penalties = temperature * F.softplus(-margin_gap / temperature)
            raw_loss = self._reduce_class_margin_penalties(
                penalties,
                label_indices,
                class_weights,
                reduction=self.cfg.class_evidence_margin.reduction,
            )
        else:
            raise ValueError(
                "class evidence margin mode must be 'minority_vs_major', "
                "'true_vs_hardest_negative', or "
                "'softplus_true_vs_hardest_negative'"
            )
        weighted_loss = float(self.cfg.class_evidence_margin.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_class_evidence_gap_cap_regularization_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.class_evidence_gap_cap_regularization
        if cfg.target != "class_evidence_logits":
            raise ValueError(
                "class evidence gap cap regularization supports only "
                "target='class_evidence_logits'"
            )
        if cfg.mode != "negative_gap_hinge":
            raise ValueError(
                "class evidence gap cap regularization supports only "
                "mode='negative_gap_hinge'"
            )
        if output.class_evidence_logits is None:
            raise ValueError(
                "class evidence gap cap regularization enabled but model did not "
                "return class_evidence_logits"
            )
        logits = output.class_evidence_logits
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="class evidence gap cap regularization",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss
        evidence_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        negative_gap_caps = self._class_values_tensor(
            self.cfg.class_evidence_gap_cap_negative_cap_by_class,
            label_indices,
            device=logits.device,
            dtype=logits.dtype,
            fallback=float(cfg.negative_gap_cap),
        )
        label_multipliers = self._class_values_tensor(
            self.cfg.class_evidence_gap_cap_label_weight_by_class,
            label_indices,
            device=logits.device,
            dtype=logits.dtype,
            fallback=1.0,
        )
        penalties = torch.relu(-evidence_gap - negative_gap_caps)
        penalties = penalties * label_multipliers
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="class_evidence_gap_cap_regularization",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return (
            raw_loss,
            weighted_loss,
            label_multipliers.detach().mean(),
            negative_gap_caps.detach().mean(),
        )

    def _compute_class_evidence_positive_gap_cap_regularization_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.class_evidence_positive_gap_cap_regularization
        if cfg.target != "class_evidence_logits":
            raise ValueError(
                "class evidence positive gap cap regularization supports only "
                "target='class_evidence_logits'"
            )
        if cfg.mode != "positive_gap_hinge":
            raise ValueError(
                "class evidence positive gap cap regularization supports only "
                "mode='positive_gap_hinge'"
            )
        if output.class_evidence_logits is None:
            raise ValueError(
                "class evidence positive gap cap regularization enabled but model "
                "did not return class_evidence_logits"
            )
        logits = output.class_evidence_logits
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="class evidence positive gap cap regularization",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        evidence_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        penalties = torch.relu(evidence_gap - float(cfg.positive_gap_cap))
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="class_evidence_positive_gap_cap_regularization",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_interaction_gap_cap_regularization_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.interaction_gap_cap_regularization
        if cfg.target != "class_evidence_interaction_scores":
            raise ValueError(
                "interaction gap cap regularization supports only "
                "target='class_evidence_interaction_scores'"
            )
        if cfg.mode != "absolute_gap_hinge":
            raise ValueError(
                "interaction gap cap regularization supports only "
                "mode='absolute_gap_hinge'"
            )
        if output.class_evidence_interaction_scores is None:
            raise ValueError(
                "interaction gap cap regularization enabled but model did not "
                "return class_evidence_interaction_scores"
            )
        logits = output.class_evidence_interaction_scores
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_interaction_scores",
            loss_name="interaction gap cap regularization",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        interaction_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        penalties = torch.relu(interaction_gap.abs() - float(cfg.gap_cap))
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="interaction_gap_cap_regularization",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_branch_support_score_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.branch_support_score_margin
        if cfg.target != "class_evidence_branch_support_scores":
            raise ValueError(
                "branch support score margin supports only "
                "target='class_evidence_branch_support_scores'"
            )
        if cfg.mode != "softplus_true_vs_hardest_negative":
            raise ValueError(
                "branch support score margin supports only "
                "mode='softplus_true_vs_hardest_negative'"
            )
        if output.class_evidence_branch_support_scores is None:
            raise ValueError(
                "branch support score margin enabled but model did not return "
                "class_evidence_branch_support_scores"
            )
        logits = output.class_evidence_branch_support_scores
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_branch_support_scores",
            loss_name="branch support score margin",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        margin_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        temperature = float(cfg.temperature)
        penalties = temperature * F.softplus(-margin_gap / temperature)
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="branch_support_score_margin",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_top_support_score_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.top_support_score_margin
        if cfg.target != "class_evidence_top_support_scores":
            raise ValueError(
                "top support score margin supports only "
                "target='class_evidence_top_support_scores'"
            )
        if cfg.mode != "softplus_true_vs_hardest_negative":
            raise ValueError(
                "top support score margin supports only "
                "mode='softplus_true_vs_hardest_negative'"
            )
        if output.class_evidence_top_support_scores is None:
            raise ValueError(
                "top support score margin enabled but model did not return "
                "class_evidence_top_support_scores"
            )
        logits = output.class_evidence_top_support_scores
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_top_support_scores",
            loss_name="top support score margin",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss, raw_loss
        margin_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        temperature = float(cfg.temperature)
        base_penalties = temperature * F.softplus(-margin_gap / temperature)
        label_multipliers = self._class_values_tensor(
            self.cfg.top_support_score_margin_label_weight_by_class,
            label_indices,
            device=logits.device,
            dtype=logits.dtype,
            fallback=1.0,
        )
        support_multipliers, _ = self._margin_support_weights(
            output,
            labels,
            label_indices,
            margin_gap,
            cfg.support_conditioned_multiplier,
            loss_name="top_support_score_margin",
        )
        hardness_multipliers, _ = self._margin_hardness_weights(
            margin_gap,
            cfg.hardness_weighting,
            loss_name="top_support_score_margin",
            expected_source="top_support_gap",
        )
        penalties = (
            base_penalties
            * label_multipliers
            * support_multipliers
            * hardness_multipliers
        )
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="top_support_score_margin",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return (
            raw_loss,
            weighted_loss,
            label_multipliers.detach().mean(),
            support_multipliers.detach().mean(),
            hardness_multipliers.detach().mean(),
        )

    def _compute_top_support_gap_min_constraint_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg.top_support_gap_min_constraint
        if cfg.target != "class_evidence_top_support_scores":
            raise ValueError(
                "top support gap min constraint supports only "
                "target='class_evidence_top_support_scores'"
            )
        if cfg.mode != "support_conditioned_min_gap":
            raise ValueError(
                "top support gap min constraint supports only "
                "mode='support_conditioned_min_gap'"
            )
        if cfg.support_source != "top_branch_margin":
            raise ValueError(
                "top support gap min constraint supports only "
                "support_source='top_branch_margin'"
            )
        if output.class_evidence_top_support_scores is None:
            raise ValueError(
                "top support gap min constraint enabled but model did not return "
                "class_evidence_top_support_scores"
            )
        if output.class_top_branch_margin_features is None:
            raise ValueError(
                "top support gap min constraint enabled but model did not return "
                "class_top_branch_margin_features"
            )
        logits = output.class_evidence_top_support_scores
        top_branch_margin = output.class_top_branch_margin_features
        if tuple(logits.shape) != tuple(top_branch_margin.shape):
            raise ValueError(
                "top support gap min constraint requires top support scores and "
                "top branch margin features to have the same shape"
            )
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_top_support_scores",
            loss_name="top support gap min constraint",
        )
        self._validate_class_margin_inputs(
            top_branch_margin,
            labels,
            logits_name="class_top_branch_margin_features",
            loss_name="top support gap min constraint",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        top_support_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        base_min_gap = self._class_values_tensor(
            self.cfg.top_support_gap_min_base_by_class,
            label_indices,
            device=logits.device,
            dtype=logits.dtype,
            fallback=float(cfg.base_min_gap),
        )
        true_top_margin = top_branch_margin.gather(
            1,
            label_indices.unsqueeze(1),
        ).squeeze(1)
        support = torch.clamp(
            torch.relu(true_top_margin),
            max=float(cfg.support_cap),
        )
        target_min_gap = base_min_gap + float(cfg.support_gain) * support
        penalties = torch.relu(target_min_gap - top_support_gap)
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="top_support_gap_min_constraint",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss, target_min_gap.detach().mean()

    def _teacher_gap_deficit_hardness_weights(
        self,
        margin_deficit: Tensor,
        cfg: Any,
        *,
        loss_name: str,
    ) -> tuple[Tensor, Tensor]:
        if not cfg.enabled:
            return torch.ones_like(margin_deficit), torch.zeros_like(margin_deficit)
        if cfg.source != "teacher_gap_deficit":
            raise ValueError(
                f"{loss_name} hardness weighting supports only "
                "source='teacher_gap_deficit'"
            )
        if cfg.mode != "linear":
            raise ValueError(
                f"{loss_name} hardness weighting supports only mode='linear'"
            )
        hardness = torch.relu(margin_deficit.detach())
        weights = 1.0 + float(cfg.gain) * torch.clamp(
            hardness,
            max=float(cfg.cap),
        )
        return weights, hardness

    def _compute_class_top_branch_relative_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.class_top_branch_relative_margin
        if cfg.target != "class_top_branch_margin_features":
            raise ValueError(
                "class top branch relative margin supports only "
                "target='class_top_branch_margin_features'"
            )
        if cfg.mode != "true_vs_hardest_negative_hinge":
            raise ValueError(
                "class top branch relative margin supports only "
                "mode='true_vs_hardest_negative_hinge'"
            )
        if output.class_top_branch_margin_features is None:
            raise ValueError(
                "class top branch relative margin enabled but model did not return "
                "class_top_branch_margin_features"
            )
        teacher_logits = output.class_top_branch_margin_features
        label_indices = self._validate_class_margin_inputs(
            teacher_logits,
            labels,
            logits_name="class_top_branch_margin_features",
            loss_name="class top branch relative margin",
        )
        if label_indices.numel() == 0:
            raw_loss = teacher_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = teacher_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss
        teacher_gap, _ = self._true_vs_hardest_negative_gap(
            teacher_logits,
            label_indices,
        )
        margin_deficit = torch.relu(float(cfg.margin) - teacher_gap)
        support_weight, _ = self._margin_support_weights(
            output,
            labels,
            label_indices,
            teacher_gap,
            cfg.support_weighting,
            loss_name="class_top_branch_relative_margin",
        )
        hardness_weight, _ = self._teacher_gap_deficit_hardness_weights(
            margin_deficit,
            cfg.hardness_weighting,
            loss_name="class_top_branch_relative_margin",
        )
        penalties = margin_deficit * support_weight * hardness_weight
        class_weights = self._class_margin_weights(
            label_indices,
            teacher_logits,
            enabled=cfg.class_weighted,
            loss_name="class_top_branch_relative_margin",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return (
            raw_loss,
            weighted_loss,
            support_weight.detach().mean(),
            hardness_weight.detach().mean(),
        )

    def _compute_top_teacher_gap_min_constraint_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg.top_teacher_gap_min_constraint
        if cfg.target != "class_top_branch_margin_features":
            raise ValueError(
                "top teacher gap min constraint supports only "
                "target='class_top_branch_margin_features'"
            )
        if cfg.mode != "support_conditioned_min_gap":
            raise ValueError(
                "top teacher gap min constraint supports only "
                "mode='support_conditioned_min_gap'"
            )
        if cfg.support_source != "top_branch_margin":
            raise ValueError(
                "top teacher gap min constraint supports only "
                "support_source='top_branch_margin'"
            )
        if output.class_top_branch_margin_features is None:
            raise ValueError(
                "top teacher gap min constraint enabled but model did not return "
                "class_top_branch_margin_features"
            )
        teacher_logits = output.class_top_branch_margin_features
        label_indices = self._validate_class_margin_inputs(
            teacher_logits,
            labels,
            logits_name="class_top_branch_margin_features",
            loss_name="top teacher gap min constraint",
        )
        if label_indices.numel() == 0:
            raw_loss = teacher_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = teacher_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        teacher_gap, _ = self._true_vs_hardest_negative_gap(
            teacher_logits,
            label_indices,
        )
        support_gap = self._top_branch_support_gap(
            output,
            labels,
            label_indices,
            loss_name="top_teacher_gap_min_constraint",
        ).detach()
        support = torch.clamp(
            torch.relu(support_gap),
            max=float(cfg.support_cap),
        )
        target_min_gap = float(cfg.base_min_gap) + float(cfg.support_gain) * support
        penalties = torch.relu(target_min_gap - teacher_gap)
        class_weights = self._class_margin_weights(
            label_indices,
            teacher_logits,
            enabled=cfg.class_weighted,
            loss_name="top_teacher_gap_min_constraint",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss, target_min_gap.detach().mean()

    def _compute_branch_direct_score_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.branch_direct_score_margin
        if cfg.target != "class_evidence_branch_direct_scores":
            raise ValueError(
                "branch direct score margin supports only "
                "target='class_evidence_branch_direct_scores'"
            )
        if cfg.mode != "softplus_true_vs_hardest_negative":
            raise ValueError(
                "branch direct score margin supports only "
                "mode='softplus_true_vs_hardest_negative'"
            )
        if output.class_evidence_branch_direct_scores is None:
            raise ValueError(
                "branch direct score margin enabled but model did not return "
                "class_evidence_branch_direct_scores"
            )
        logits = output.class_evidence_branch_direct_scores
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_evidence_branch_direct_scores",
            loss_name="branch direct score margin",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        margin_gap, _ = self._true_vs_hardest_negative_gap(
            logits,
            label_indices,
        )
        temperature = float(cfg.temperature)
        penalties = temperature * F.softplus(-margin_gap / temperature)
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="branch_direct_score_margin",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_branch_path_dominance_constraint_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.branch_path_dominance_constraint
        if cfg.branch_source != "class_evidence_branch_support_scores":
            raise ValueError(
                "branch path dominance supports only "
                "branch_source='class_evidence_branch_support_scores'"
            )
        if cfg.target != "class_evidence_logits":
            raise ValueError(
                "branch path dominance supports only target='class_evidence_logits'"
            )
        if cfg.mode != "branch_gap_preservation":
            raise ValueError(
                "branch path dominance supports only mode='branch_gap_preservation'"
            )
        if cfg.support_source != "top_branch_margin":
            raise ValueError(
                "branch path dominance supports only support_source='top_branch_margin'"
            )
        if output.class_evidence_branch_support_scores is None:
            raise ValueError(
                "branch path dominance enabled but model did not return "
                "class_evidence_branch_support_scores"
            )
        if output.class_evidence_logits is None:
            raise ValueError(
                "branch path dominance enabled but model did not return "
                "class_evidence_logits"
            )
        branch_scores = output.class_evidence_branch_support_scores
        evidence_logits = output.class_evidence_logits
        if tuple(branch_scores.shape) != tuple(evidence_logits.shape):
            raise ValueError(
                "branch path dominance requires branch support scores and "
                "class evidence logits to have the same shape"
            )
        label_indices = self._validate_class_margin_inputs(
            evidence_logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="branch path dominance",
        )
        self._validate_class_margin_inputs(
            branch_scores,
            labels,
            logits_name="class_evidence_branch_support_scores",
            loss_name="branch path dominance",
        )
        if label_indices.numel() == 0:
            raw_loss = evidence_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = evidence_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss, raw_loss

        branch_gap, _ = self._true_vs_hardest_negative_gap(
            branch_scores,
            label_indices,
        )
        evidence_gap, _ = self._true_vs_hardest_negative_gap(
            evidence_logits,
            label_indices,
        )
        support_weight, _ = self._margin_support_weights(
            output,
            labels,
            label_indices,
            branch_gap,
            cfg.support_weighting,
            loss_name="branch_path_dominance_constraint",
        )
        allowed_drops = self._class_values_tensor(
            self.cfg.branch_path_dominance_allowed_drop_by_class,
            label_indices,
            device=evidence_logits.device,
            dtype=evidence_logits.dtype,
            fallback=float(cfg.allowed_drop),
        )
        label_multipliers = self._class_values_tensor(
            self.cfg.branch_path_dominance_label_weight_by_class,
            label_indices,
            device=evidence_logits.device,
            dtype=evidence_logits.dtype,
            fallback=1.0,
        )
        penalties = (
            torch.relu(branch_gap - evidence_gap - allowed_drops)
            * support_weight
            * label_multipliers
        )
        class_weights = self._class_margin_weights(
            label_indices,
            evidence_logits,
            enabled=cfg.class_weighted,
            loss_name="branch_path_dominance_constraint",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return (
            raw_loss,
            weighted_loss,
            support_weight.detach().mean(),
            label_multipliers.detach().mean(),
            allowed_drops.detach().mean(),
        )

    def _compute_branch_support_disagreement_cap_regularization_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg.branch_support_disagreement_cap_regularization
        if cfg.branch_source != "class_evidence_branch_support_scores":
            raise ValueError(
                "branch support disagreement cap supports only "
                "branch_source='class_evidence_branch_support_scores'"
            )
        if cfg.embedding_source != "class_evidence_embedding_scores":
            raise ValueError(
                "branch support disagreement cap supports only "
                "embedding_source='class_evidence_embedding_scores'"
            )
        if cfg.interaction_source != "class_evidence_interaction_scores":
            raise ValueError(
                "branch support disagreement cap supports only "
                "interaction_source='class_evidence_interaction_scores'"
            )
        if cfg.mode != "branch_gap_conditioned_abs_gap_cap":
            raise ValueError(
                "branch support disagreement cap supports only "
                "mode='branch_gap_conditioned_abs_gap_cap'"
            )
        if output.class_evidence_branch_support_scores is None:
            raise ValueError(
                "branch support disagreement cap enabled but model did not return "
                "class_evidence_branch_support_scores"
            )
        if output.class_evidence_embedding_scores is None:
            raise ValueError(
                "branch support disagreement cap enabled but model did not return "
                "class_evidence_embedding_scores"
            )
        if output.class_evidence_interaction_scores is None:
            raise ValueError(
                "branch support disagreement cap enabled but model did not return "
                "class_evidence_interaction_scores"
            )
        branch_scores = output.class_evidence_branch_support_scores
        embedding_scores = output.class_evidence_embedding_scores
        interaction_scores = output.class_evidence_interaction_scores
        if tuple(branch_scores.shape) != tuple(embedding_scores.shape) or tuple(
            branch_scores.shape
        ) != tuple(interaction_scores.shape):
            raise ValueError(
                "branch support disagreement cap requires branch, embedding, "
                "and interaction scores to have the same shape"
            )
        label_indices = self._validate_class_margin_inputs(
            branch_scores,
            labels,
            logits_name="class_evidence_branch_support_scores",
            loss_name="branch support disagreement cap",
        )
        self._validate_class_margin_inputs(
            embedding_scores,
            labels,
            logits_name="class_evidence_embedding_scores",
            loss_name="branch support disagreement cap",
        )
        self._validate_class_margin_inputs(
            interaction_scores,
            labels,
            logits_name="class_evidence_interaction_scores",
            loss_name="branch support disagreement cap",
        )
        if label_indices.numel() == 0:
            raw_loss = branch_scores.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = branch_scores.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        branch_gap, _ = self._true_vs_hardest_negative_gap(
            branch_scores,
            label_indices,
        )
        embedding_gap, _ = self._true_vs_hardest_negative_gap(
            embedding_scores,
            label_indices,
        )
        interaction_gap, _ = self._true_vs_hardest_negative_gap(
            interaction_scores,
            label_indices,
        )
        disagreement = 1.0 + float(cfg.condition_gain) * torch.clamp(
            torch.relu(float(cfg.disagreement_threshold) - branch_gap),
            max=float(cfg.condition_cap),
        )
        embedding_penalties = disagreement * torch.relu(
            embedding_gap.abs() - float(cfg.embedding_gap_cap)
        )
        interaction_penalties = disagreement * torch.relu(
            interaction_gap.abs() - float(cfg.interaction_gap_cap)
        )
        penalties = embedding_penalties + interaction_penalties
        class_weights = self._class_margin_weights(
            label_indices,
            branch_scores,
            enabled=cfg.class_weighted,
            loss_name="branch_support_disagreement_cap_regularization",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss, disagreement.detach().mean()

    def _compute_class_gated_branch_logit_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.class_gated_branch_logit_margin
        if cfg.target != "class_gated_branch_logits":
            raise ValueError(
                "class-gated branch logit margin supports only "
                "target='class_gated_branch_logits'"
            )
        if cfg.mode != "true_vs_hardest_negative":
            raise ValueError(
                "class-gated branch logit margin supports only "
                "mode='true_vs_hardest_negative'"
            )
        if output.class_gated_branch_logits is None:
            raise ValueError(
                "class-gated branch logit margin enabled but model did not "
                "return class_gated_branch_logits"
            )
        logits = output.class_gated_branch_logits
        label_indices = self._validate_class_margin_inputs(
            logits,
            labels,
            logits_name="class_gated_branch_logits",
            loss_name="class-gated branch logit margin",
        )
        if label_indices.numel() == 0:
            raw_loss = logits.sum() * 0.0
            return raw_loss, raw_loss
        class_weights = self._class_margin_weights(
            label_indices,
            logits,
            enabled=cfg.class_weighted,
            loss_name="class_gated_branch_logit_margin",
        )
        penalties = self._true_vs_hardest_negative_penalties(
            logits,
            label_indices,
            margin=float(cfg.margin),
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_branch_to_evidence_teacher_distribution_kl_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        evidence_logits: Tensor,
        label_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.branch_to_evidence_ranking_consistency
        if cfg.source != "class_top_branch_margin_features":
            raise ValueError(
                "branch-to-evidence teacher distribution KL supports only "
                "source='class_top_branch_margin_features'"
            )
        if output.class_top_branch_margin_features is None:
            raise ValueError(
                "branch-to-evidence teacher distribution KL enabled but model "
                "did not return class_top_branch_margin_features"
            )
        teacher_logits = output.class_top_branch_margin_features
        if tuple(teacher_logits.shape) != tuple(evidence_logits.shape):
            raise ValueError(
                "class_top_branch_margin_features must match "
                "class_evidence_logits shape for branch-to-evidence teacher "
                "distribution KL"
            )
        self._validate_class_margin_inputs(
            teacher_logits,
            labels,
            logits_name="class_top_branch_margin_features",
            loss_name="branch-to-evidence teacher distribution KL",
        )
        if cfg.teacher_detach:
            teacher_logits = teacher_logits.detach()
        teacher_probs = F.softmax(
            teacher_logits / float(cfg.teacher_temperature),
            dim=-1,
        )
        student_log_probs = F.log_softmax(
            evidence_logits / float(cfg.student_temperature),
            dim=-1,
        )
        per_sample_kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)
        class_weights = self._class_margin_weights(
            label_indices,
            evidence_logits,
            enabled=cfg.class_weighted,
            loss_name="branch_to_evidence_ranking_consistency",
        )
        raw_loss = (per_sample_kl * class_weights).mean()
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_branch_to_evidence_true_label_anchored_softplus_loss(
        self,
        output: AstModelOutput,
        evidence_logits: Tensor,
        label_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.branch_to_evidence_ranking_consistency
        if cfg.source != "class_top_branch_margin_relative_features":
            raise ValueError(
                "branch-to-evidence true-label anchored softplus supports only "
                "source='class_top_branch_margin_relative_features'"
            )
        if output.class_top_branch_margin_relative_features is None:
            raise ValueError(
                "branch-to-evidence true-label anchored softplus enabled but "
                "model did not return class_top_branch_margin_relative_features"
            )
        support_logits = output.class_top_branch_margin_relative_features
        if tuple(support_logits.shape) != tuple(evidence_logits.shape):
            raise ValueError(
                "class_top_branch_margin_relative_features must match "
                "class_evidence_logits shape for branch-to-evidence true-label "
                "anchored softplus"
            )
        support_logits = support_logits.detach()
        evidence_gap, _ = self._true_vs_hardest_negative_gap(
            evidence_logits,
            label_indices,
        )
        true_support = support_logits.gather(1, label_indices.unsqueeze(1)).squeeze(1)
        if cfg.support_weighting.enabled:
            if (
                cfg.support_weighting.source != "same_as_source"
                or cfg.support_weighting.mode != "positive_linear"
            ):
                raise ValueError(
                    "branch_to_evidence_ranking_consistency.support_weighting "
                    "must use source='same_as_source' and mode='positive_linear' "
                    "for true_label_anchored_softplus"
                )
            support = torch.clamp(
                torch.relu(true_support),
                min=0.0,
                max=float(cfg.support_weighting.cap),
            )
            support_weight = 1.0 + (float(cfg.support_weighting.gain) * support)
        else:
            support_weight = torch.ones_like(evidence_gap)
        temperature = float(cfg.temperature)
        penalties = (
            temperature * F.softplus(-evidence_gap / temperature) * support_weight
        )
        class_weights = self._class_margin_weights(
            label_indices,
            evidence_logits,
            enabled=cfg.class_weighted,
            loss_name="branch_to_evidence_ranking_consistency",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_branch_to_evidence_ranking_consistency_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.branch_to_evidence_ranking_consistency
        if cfg.target != "class_evidence_logits":
            raise ValueError(
                "branch-to-evidence ranking consistency supports only "
                "target='class_evidence_logits'"
            )
        if cfg.source not in {
            "class_gated_branch_logits",
            "top_branch_margin",
            "class_top_branch_margin_features",
            "class_top_branch_margin_relative_features",
        }:
            raise ValueError(
                "branch-to-evidence ranking consistency supports only "
                "source='class_gated_branch_logits', source='top_branch_margin', "
                "source='class_top_branch_margin_features', or "
                "source='class_top_branch_margin_relative_features'"
            )
        if cfg.mode not in {
            "true_vs_hardest_negative",
            "teacher_distribution_kl",
            "true_label_anchored_softplus",
        }:
            raise ValueError(
                "branch-to-evidence ranking consistency supports only "
                "mode='true_vs_hardest_negative', "
                "mode='teacher_distribution_kl', or "
                "mode='true_label_anchored_softplus'"
            )
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss
        if output.class_evidence_logits is None:
            raise ValueError(
                "branch-to-evidence ranking consistency enabled but model did "
                "not return class_evidence_logits"
            )
        evidence_logits = output.class_evidence_logits
        label_indices = self._validate_class_margin_inputs(
            evidence_logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="branch-to-evidence ranking consistency",
        )
        if label_indices.numel() == 0:
            raw_loss = evidence_logits.sum() * 0.0
            return raw_loss, raw_loss

        if cfg.mode == "teacher_distribution_kl":
            return self._compute_branch_to_evidence_teacher_distribution_kl_loss(
                output,
                labels,
                evidence_logits,
                label_indices,
            )
        if cfg.mode == "true_label_anchored_softplus":
            return self._compute_branch_to_evidence_true_label_anchored_softplus_loss(
                output,
                evidence_logits,
                label_indices,
            )

        if cfg.source == "class_gated_branch_logits":
            if output.class_gated_branch_logits is None:
                raise ValueError(
                    "branch-to-evidence ranking consistency enabled but model did "
                    "not return class_gated_branch_logits"
                )
            source_logits = output.class_gated_branch_logits
            if tuple(source_logits.shape) != tuple(evidence_logits.shape):
                raise ValueError(
                    "class_gated_branch_logits must match class_evidence_logits "
                    "shape for branch-to-evidence ranking consistency"
                )
            self._validate_class_margin_inputs(
                source_logits,
                labels,
                logits_name="class_gated_branch_logits",
                loss_name="branch-to-evidence ranking consistency",
            )
            teacher_gap, _ = self._true_vs_hardest_negative_gap(
                source_logits,
                label_indices,
            )
        elif cfg.source == "top_branch_margin":
            branch_logits, branch_label_indices = self._branch_margin_inputs(
                output,
                labels,
                loss_name="branch-to-evidence ranking consistency",
            )
            if not torch.equal(branch_label_indices, label_indices):
                raise ValueError(
                    "branch_logits labels must match class_evidence_logits labels "
                    "for branch-to-evidence ranking consistency"
                )
            teacher_gap, _, _ = self._top_true_class_branch_margin(
                branch_logits,
                label_indices,
            )
        else:
            raise ValueError(
                "branch-to-evidence hard-margin consistency supports only "
                "source='class_gated_branch_logits' or source='top_branch_margin'"
            )
        teacher_floor = self._branch_to_evidence_teacher_floors(
            label_indices,
            teacher_gap,
        )
        teacher_gap = torch.maximum(teacher_gap, teacher_floor)
        if cfg.teacher_gap_cap is not None:
            teacher_gap = torch.clamp(teacher_gap, max=float(cfg.teacher_gap_cap))
        if cfg.teacher_detach:
            teacher_gap = teacher_gap.detach()
        evidence_gap, _ = self._true_vs_hardest_negative_gap(
            evidence_logits,
            label_indices,
        )
        support_weight, _ = self._margin_support_weights(
            output,
            labels,
            label_indices,
            evidence_gap,
            cfg.support_weighting,
            loss_name="branch_to_evidence_ranking_consistency",
        )
        hard_weight, _ = self._margin_hardness_weights(
            evidence_gap,
            cfg.hardness_weighting,
            loss_name="branch_to_evidence_ranking_consistency",
        )
        penalties = (
            torch.relu(teacher_gap - evidence_gap + float(cfg.tolerance))
            * support_weight
            * hard_weight
        )
        class_weights = self._class_margin_weights(
            label_indices,
            evidence_logits,
            enabled=cfg.class_weighted,
            loss_name="branch_to_evidence_ranking_consistency",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_global_residual_anti_veto_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg.global_residual_anti_veto
        if cfg.target not in {"global_residual_logits", "final_logits"}:
            raise ValueError(
                "global residual anti-veto supports only "
                "target='global_residual_logits' or target='final_logits'"
            )
        if cfg.reference != "class_evidence_logits":
            raise ValueError(
                "global residual anti-veto supports only "
                "reference='class_evidence_logits'"
            )
        if cfg.mode not in {"true_vs_hardest_negative", "final_gap_preservation"}:
            raise ValueError(
                "global residual anti-veto supports only "
                "mode='true_vs_hardest_negative' or "
                "mode='final_gap_preservation'"
            )
        if cfg.margin_mode != "true_vs_hardest_negative":
            raise ValueError(
                "global residual anti-veto supports only "
                "margin_mode='true_vs_hardest_negative'"
            )
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        if output.class_evidence_logits is None:
            raise ValueError(
                "global residual anti-veto enabled but model did not return "
                "class_evidence_logits"
            )
        evidence_logits = output.class_evidence_logits
        target_logits: Tensor
        if cfg.mode == "true_vs_hardest_negative":
            if cfg.target != "global_residual_logits":
                raise ValueError(
                    "global residual anti-veto target must be "
                    "'global_residual_logits' when mode='true_vs_hardest_negative'"
                )
            if output.global_residual_logits is None:
                raise ValueError(
                    "global residual anti-veto enabled but model did not return "
                    "global_residual_logits"
                )
            target_logits = output.global_residual_logits
            target_name = "global_residual_logits"
        else:
            if cfg.target != "final_logits":
                raise ValueError(
                    "global residual anti-veto target must be 'final_logits' "
                    "when mode='final_gap_preservation'"
                )
            if cfg.support_source != "top_branch_margin":
                raise ValueError(
                    "global residual anti-veto mode='final_gap_preservation' "
                    "requires support_source='top_branch_margin'"
                )
            target_logits = output.logits
            target_name = "final_logits"
        label_indices = self._validate_class_margin_inputs(
            target_logits,
            labels,
            logits_name=target_name,
            loss_name="global residual anti-veto",
        )
        if tuple(evidence_logits.shape) != tuple(target_logits.shape):
            raise ValueError(
                "class_evidence_logits must match anti-veto target logits shape "
                "for global residual anti-veto"
            )
        self._validate_class_margin_inputs(
            evidence_logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="global residual anti-veto",
        )
        if label_indices.numel() == 0:
            raw_loss = target_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss

        evidence_gap, negative_indices = self._true_vs_hardest_negative_gap(
            evidence_logits,
            label_indices,
        )
        target_true = target_logits.gather(1, label_indices.unsqueeze(1)).squeeze(1)
        target_negative = target_logits.gather(
            1,
            negative_indices.unsqueeze(1),
        ).squeeze(1)
        target_gap = target_true - target_negative
        if cfg.mode == "true_vs_hardest_negative":
            eligible = evidence_gap.detach() > float(cfg.evidence_confidence_threshold)
            penalties = torch.relu(float(cfg.min_residual_gap) - target_gap)
        else:
            branch_logits, branch_label_indices = self._branch_margin_inputs(
                output,
                labels,
                loss_name="global residual anti-veto",
            )
            if not torch.equal(branch_label_indices, label_indices):
                raise ValueError(
                    "branch_logits labels must match final logits labels for "
                    "global residual anti-veto"
                )
            support_gap, _, _ = self._top_true_class_branch_margin(
                branch_logits,
                label_indices,
            )
            eligible = support_gap.detach() > float(cfg.support_threshold)
            penalties = torch.relu(
                evidence_gap.detach() - float(cfg.allowed_gap_drop) - target_gap
            )
        eligible_fraction = eligible.to(dtype=target_logits.dtype).mean()
        if not bool(eligible.any().item()):
            raw_loss = target_logits.sum() * 0.0
        else:
            class_weights = self._class_margin_weights(
                label_indices,
                target_logits,
                enabled=cfg.class_weighted,
                loss_name="global_residual_anti_veto",
            )
            raw_loss = self._reduce_class_margin_penalties(
                penalties[eligible],
                label_indices[eligible],
                class_weights[eligible],
                reduction=cfg.reduction,
            )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss, eligible_fraction

    def _compute_residual_contradiction_regularization_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg.residual_contradiction_regularization
        if cfg.target != "global_residual_logits":
            raise ValueError(
                "residual contradiction regularization supports only "
                "target='global_residual_logits'"
            )
        if cfg.reference != "class_evidence_logits":
            raise ValueError(
                "residual contradiction regularization supports only "
                "reference='class_evidence_logits'"
            )
        if cfg.mode != "opposite_gap_penalty":
            raise ValueError(
                "residual contradiction regularization supports only "
                "mode='opposite_gap_penalty'"
            )
        if cfg.margin_mode != "true_vs_hardest_negative":
            raise ValueError(
                "residual contradiction regularization supports only "
                "margin_mode='true_vs_hardest_negative'"
            )
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        if output.class_evidence_logits is None:
            raise ValueError(
                "residual contradiction regularization enabled but model did "
                "not return class_evidence_logits"
            )
        if output.global_residual_logits is None:
            raise ValueError(
                "residual contradiction regularization enabled but model did "
                "not return global_residual_logits"
            )
        evidence_logits = output.class_evidence_logits
        residual_logits = output.global_residual_logits
        if tuple(evidence_logits.shape) != tuple(residual_logits.shape):
            raise ValueError(
                "class_evidence_logits must match global_residual_logits shape "
                "for residual contradiction regularization"
            )
        label_indices = self._validate_class_margin_inputs(
            residual_logits,
            labels,
            logits_name="global_residual_logits",
            loss_name="residual contradiction regularization",
        )
        self._validate_class_margin_inputs(
            evidence_logits,
            labels,
            logits_name="class_evidence_logits",
            loss_name="residual contradiction regularization",
        )
        if label_indices.numel() == 0:
            raw_loss = residual_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss
        evidence_gap, negative_indices = self._true_vs_hardest_negative_gap(
            evidence_logits,
            label_indices,
        )
        residual_true = residual_logits.gather(
            1,
            label_indices.unsqueeze(1),
        ).squeeze(1)
        residual_negative = residual_logits.gather(
            1,
            negative_indices.unsqueeze(1),
        ).squeeze(1)
        residual_gap = residual_true - residual_negative
        eligible = evidence_gap.detach() > float(cfg.evidence_gap_threshold)
        eligible_fraction = eligible.to(dtype=residual_logits.dtype).mean()
        penalties = torch.relu(float(cfg.min_residual_gap) - residual_gap)
        if not bool(eligible.any().item()):
            raw_loss = residual_logits.sum() * 0.0
        else:
            class_weights = self._class_margin_weights(
                label_indices,
                residual_logits,
                enabled=cfg.class_weighted,
                loss_name="residual_contradiction_regularization",
            )
            raw_loss = self._reduce_class_margin_penalties(
                penalties[eligible],
                label_indices[eligible],
                class_weights[eligible],
                reduction=cfg.reduction,
            )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss, eligible_fraction

    def _class_aware_branch_margin_inputs(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        loss_name: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if output.branch_logits is None:
            raise ValueError(
                f"{loss_name} enabled but model did not return branch_logits"
            )
        if output.class_evidence_gate_weights is None:
            raise ValueError(
                f"{loss_name} enabled but model did not return "
                "class_evidence_gate_weights"
            )
        branch_logits = output.branch_logits
        gate_weights = output.class_evidence_gate_weights
        if branch_logits.ndim != 3:
            raise ValueError(
                "branch_logits must have shape (B, R, C) for "
                f"{loss_name}, got {tuple(branch_logits.shape)}"
            )
        if gate_weights.ndim != 3:
            raise ValueError(
                "class_evidence_gate_weights must have shape (B, C, R) for "
                f"{loss_name}, got {tuple(gate_weights.shape)}"
            )
        batch_size, num_branches, num_classes = branch_logits.shape
        if int(num_classes) < 2:
            raise ValueError(f"{loss_name} requires at least two classes")
        if tuple(gate_weights.shape) != (batch_size, num_classes, num_branches):
            raise ValueError(
                "class_evidence_gate_weights shape must match branch_logits as "
                "(B, C, R)"
            )
        label_indices = labels.to(device=branch_logits.device, dtype=torch.long)
        if label_indices.ndim != 1:
            raise ValueError(f"labels must have shape (B,) for {loss_name}")
        if int(label_indices.shape[0]) != int(batch_size):
            raise ValueError(
                f"labels batch size must match branch_logits batch size for {loss_name}"
            )
        if label_indices.numel() > 0 and (
            int(label_indices.min().item()) < 0
            or int(label_indices.max().item()) >= int(num_classes)
        ):
            raise ValueError("labels contain a class index outside branch_logits")
        return branch_logits, gate_weights, label_indices

    def _branch_margin_inputs(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        loss_name: str,
    ) -> tuple[Tensor, Tensor]:
        if output.branch_logits is None:
            raise ValueError(
                f"{loss_name} enabled but model did not return branch_logits"
            )
        branch_logits = output.branch_logits
        if branch_logits.ndim != 3:
            raise ValueError(
                "branch_logits must have shape (B, R, C) for "
                f"{loss_name}, got {tuple(branch_logits.shape)}"
            )
        batch_size, _, num_classes = branch_logits.shape
        if int(num_classes) < 2:
            raise ValueError(f"{loss_name} requires at least two classes")
        label_indices = labels.to(device=branch_logits.device, dtype=torch.long)
        if label_indices.ndim != 1:
            raise ValueError(f"labels must have shape (B,) for {loss_name}")
        if int(label_indices.shape[0]) != int(batch_size):
            raise ValueError(
                f"labels batch size must match branch_logits batch size for {loss_name}"
            )
        if label_indices.numel() > 0 and (
            int(label_indices.min().item()) < 0
            or int(label_indices.max().item()) >= int(num_classes)
        ):
            raise ValueError("labels contain a class index outside branch_logits")
        return branch_logits, label_indices

    def _true_class_branch_margins(
        self,
        branch_logits: Tensor,
        label_indices: Tensor,
    ) -> Tensor:
        batch_size, num_branches, num_classes = branch_logits.shape
        class_gather = label_indices.view(batch_size, 1, 1).expand(
            -1,
            num_branches,
            1,
        )
        true_branch_logits = branch_logits.gather(2, class_gather).squeeze(2)
        true_class_mask = F.one_hot(
            label_indices,
            num_classes=int(num_classes),
        ).to(dtype=torch.bool, device=branch_logits.device)
        hardest_negative = (
            branch_logits.masked_fill(
                true_class_mask.unsqueeze(1),
                -torch.inf,
            )
            .max(dim=2)
            .values
        )
        return true_branch_logits - hardest_negative

    def _top_true_class_branch_margin(
        self,
        branch_logits: Tensor,
        label_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        branch_margin = self._true_class_branch_margins(
            branch_logits,
            label_indices,
        )
        top_margin, top_branch_indices = branch_margin.max(dim=1)
        true_class_mask = F.one_hot(
            label_indices,
            num_classes=int(branch_logits.shape[2]),
        ).to(dtype=torch.bool, device=branch_logits.device)
        negative_logits = branch_logits.masked_fill(
            true_class_mask.unsqueeze(1),
            -torch.inf,
        )
        batch_indices = torch.arange(
            int(branch_logits.shape[0]),
            device=branch_logits.device,
        )
        top_negative_indices = (
            negative_logits[
                batch_indices,
                top_branch_indices,
                :,
            ]
            .max(dim=1)
            .indices
        )
        return top_margin, top_branch_indices, top_negative_indices

    def _true_class_gate_weights(
        self,
        gate_weights: Tensor,
        label_indices: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size, num_classes, num_branches = gate_weights.shape
        del num_classes
        gate_gather = label_indices.view(batch_size, 1, 1).expand(
            -1,
            1,
            num_branches,
        )
        return (
            gate_weights.to(device=label_indices.device, dtype=dtype)
            .gather(1, gate_gather)
            .squeeze(1)
        )

    def _compute_gate_weighted_branch_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg.gate_weighted_branch_margin
        if cfg.target != "true_class_gate":
            raise ValueError(
                "gate-weighted branch margin supports only target='true_class_gate'"
            )
        if cfg.source != "branch_logits":
            raise ValueError(
                "gate-weighted branch margin supports only source='branch_logits'"
            )
        if cfg.mode != "true_vs_hardest_negative":
            raise ValueError(
                "gate-weighted branch margin supports only "
                "mode='true_vs_hardest_negative'"
            )
        if cfg.branch_selection != "gate_weighted":
            raise ValueError(
                "gate-weighted branch margin branch_selection must be 'gate_weighted'"
            )
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss
        branch_logits, gate_weights, label_indices = (
            self._class_aware_branch_margin_inputs(
                output,
                labels,
                loss_name="gate-weighted branch margin",
            )
        )
        if label_indices.numel() == 0:
            raw_loss = branch_logits.sum() * 0.0
            return raw_loss, raw_loss

        branch_margin = self._true_class_branch_margins(branch_logits, label_indices)
        branch_penalty = torch.relu(float(cfg.margin) - branch_margin)
        true_class_gate = self._true_class_gate_weights(
            gate_weights,
            label_indices,
            dtype=branch_logits.dtype,
        ).detach()
        sample_penalty = (true_class_gate * branch_penalty).sum(dim=1)
        class_weights = self._class_margin_weights(
            label_indices,
            branch_logits[:, 0, :],
            enabled=cfg.class_weighted,
            loss_name="gate_weighted_branch_margin",
        )
        raw_loss = self._reduce_class_margin_penalties(
            sample_penalty,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * raw_loss
        return raw_loss, weighted_loss

    def _compute_top_branch_margin_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.top_branch_margin
        if cfg.target != "branch_logits":
            raise ValueError("top branch margin supports only target='branch_logits'")
        if cfg.mode != "true_vs_hardest_negative":
            raise ValueError(
                "top branch margin supports only mode='true_vs_hardest_negative'"
            )
        if cfg.branch_reduction != "max":
            raise ValueError("top branch margin supports only branch_reduction='max'")
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss
        branch_logits, label_indices = self._branch_margin_inputs(
            output,
            labels,
            loss_name="top branch margin",
        )
        if label_indices.numel() == 0:
            raw_loss = branch_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, raw_loss

        branch_margin = self._true_class_branch_margins(branch_logits, label_indices)
        top_margin = branch_margin.max(dim=1).values
        margin_targets = self._top_branch_margin_targets(label_indices, top_margin)
        penalties = torch.relu(margin_targets - top_margin)
        hardness_multipliers, _ = self._top_branch_margin_hardness_weights(
            penalties,
            cfg.hardness_weighting,
            loss_name="top_branch_margin",
        )
        class_weights = self._class_margin_weights(
            label_indices,
            branch_logits[:, 0, :],
            enabled=cfg.class_weighted,
            loss_name="top_branch_margin",
        )
        raw_loss = self._reduce_class_margin_penalties(
            penalties,
            label_indices,
            class_weights,
            reduction=cfg.reduction,
        )
        phase_multipliers = self._top_branch_margin_phase_multipliers(
            label_indices,
            top_margin,
            epoch=epoch,
        )
        weighted_raw_loss = self._reduce_class_margin_penalties(
            penalties * hardness_multipliers,
            label_indices,
            class_weights * phase_multipliers,
            reduction=cfg.reduction,
        )
        weighted_loss = float(cfg.weight) * weighted_raw_loss
        effective_weight = (float(cfg.weight) * phase_multipliers).detach().mean()
        return (
            raw_loss,
            weighted_loss,
            effective_weight,
            hardness_multipliers.detach().mean(),
        )

    def _compute_gate_branch_regret_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.gate_branch_regret
        weight_multiplier_value = self._gate_branch_regret_weight_multiplier(epoch)
        weight_multiplier = output.logits.new_tensor(weight_multiplier_value)
        effective_weight = output.logits.new_tensor(
            float(cfg.weight) * weight_multiplier_value
        )
        if cfg.target != "true_class_gate":
            raise ValueError(
                "gate-branch regret supports only target='true_class_gate'"
            )
        if cfg.source != "branch_logits":
            raise ValueError("gate-branch regret supports only source='branch_logits'")
        if cfg.mode != "best_margin_regret":
            raise ValueError(
                "gate-branch regret supports only mode='best_margin_regret'"
            )
        if cfg.margin_mode != "true_vs_hardest_negative":
            raise ValueError(
                "gate-branch regret supports only "
                "margin_mode='true_vs_hardest_negative'"
            )
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, weight_multiplier, effective_weight
        branch_logits, gate_weights, label_indices = (
            self._class_aware_branch_margin_inputs(
                output,
                labels,
                loss_name="gate-branch regret",
            )
        )
        if label_indices.numel() == 0:
            raw_loss = branch_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, weight_multiplier, effective_weight

        branch_margin = self._true_class_branch_margins(
            branch_logits,
            label_indices,
        ).detach()
        true_class_gate = self._true_class_gate_weights(
            gate_weights,
            label_indices,
            dtype=branch_logits.dtype,
        )
        best_margin = branch_margin.max(dim=1).values
        gate_expected_margin = (true_class_gate * branch_margin).sum(dim=1)
        positive_thresholds = self._gate_branch_regret_thresholds(
            label_indices,
            best_margin,
        )
        eligible = best_margin > positive_thresholds
        eligible_fraction = eligible.to(dtype=branch_logits.dtype).mean()
        regret_penalty = torch.relu(
            best_margin - gate_expected_margin - float(cfg.tolerance)
        )
        if bool(eligible.any().item()):
            raw_loss = regret_penalty[eligible].mean()
        else:
            raw_loss = branch_logits.sum() * 0.0
        weighted_loss = effective_weight * raw_loss
        return (
            raw_loss,
            weighted_loss,
            eligible_fraction,
            weight_multiplier,
            effective_weight,
        )

    def _compute_gate_bad_branch_suppression_loss(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        cfg = self.cfg.gate_bad_branch_suppression
        weight_multiplier_value = self._gate_bad_branch_suppression_weight_multiplier(
            epoch
        )
        weight_multiplier = output.logits.new_tensor(weight_multiplier_value)
        effective_weight = output.logits.new_tensor(
            float(cfg.weight) * weight_multiplier_value
        )
        if cfg.target != "true_class_gate":
            raise ValueError(
                "gate bad branch suppression supports only target='true_class_gate'"
            )
        if cfg.source != "branch_logits":
            raise ValueError(
                "gate bad branch suppression supports only source='branch_logits'"
            )
        if cfg.mode != "margin_below_threshold":
            raise ValueError(
                "gate bad branch suppression supports only "
                "mode='margin_below_threshold'"
            )
        if cfg.margin_mode != "true_vs_hardest_negative":
            raise ValueError(
                "gate bad branch suppression supports only "
                "margin_mode='true_vs_hardest_negative'"
            )
        if int(epoch) <= int(cfg.warmup_epochs):
            raw_loss = output.logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, weight_multiplier, effective_weight
        branch_logits, gate_weights, label_indices = (
            self._class_aware_branch_margin_inputs(
                output,
                labels,
                loss_name="gate bad branch suppression",
            )
        )
        if label_indices.numel() == 0:
            raw_loss = branch_logits.sum() * 0.0
            return raw_loss, raw_loss, raw_loss, weight_multiplier, effective_weight

        branch_margin = self._true_class_branch_margins(
            branch_logits,
            label_indices,
        ).detach()
        true_class_gate = self._true_class_gate_weights(
            gate_weights,
            label_indices,
            dtype=branch_logits.dtype,
        )
        thresholds = self._gate_bad_branch_suppression_thresholds(
            label_indices,
            branch_margin[:, 0],
        )
        bad_penalty = torch.relu(thresholds.unsqueeze(1) - branch_margin)
        sample_penalty = (true_class_gate * bad_penalty).sum(dim=1)
        raw_loss = sample_penalty.mean()
        bad_gate_mass = (
            true_class_gate
            * (branch_margin < thresholds.unsqueeze(1)).to(dtype=branch_logits.dtype)
        ).sum(dim=1)
        bad_gate_mass_mean = bad_gate_mass.mean()
        weighted_loss = effective_weight * raw_loss
        return (
            raw_loss,
            weighted_loss,
            bad_gate_mass_mean,
            weight_multiplier,
            effective_weight,
        )

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
        if self._epoch_window_active(
            enabled=self.cfg.gate_entropy_regularization.enabled,
            start_epoch=self.cfg.gate_entropy_regularization.start_epoch,
            end_epoch=self.cfg.gate_entropy_regularization.end_epoch,
            epoch=epoch,
        ):
            gate_entropy, gate_entropy_regularization = (
                self._compute_gate_entropy_regularization(output, labels)
            )
            scheduled_total = scheduled_total + gate_entropy_regularization
            monitor_total = monitor_total + gate_entropy_regularization
        class_gate_diversity: Tensor | None = None
        class_gate_diversity_regularization: Tensor | None = None
        if self._epoch_window_active(
            enabled=self.cfg.class_gate_diversity_regularization.enabled,
            start_epoch=self.cfg.class_gate_diversity_regularization.start_epoch,
            end_epoch=self.cfg.class_gate_diversity_regularization.end_epoch,
            epoch=epoch,
        ):
            class_gate_diversity, class_gate_diversity_regularization = (
                self._compute_class_gate_diversity_regularization(output)
            )
            scheduled_total = scheduled_total + class_gate_diversity_regularization
            monitor_total = monitor_total + class_gate_diversity_regularization
        class_evidence_margin: Tensor | None = None
        class_evidence_margin_loss: Tensor | None = None
        if self.cfg.class_evidence_margin.enabled:
            class_evidence_margin, class_evidence_margin_loss = (
                self._compute_class_evidence_margin_loss(output, labels)
            )
            scheduled_total = scheduled_total + class_evidence_margin_loss
            monitor_total = monitor_total + class_evidence_margin_loss
        class_evidence_gap_cap_regularization: Tensor | None = None
        class_evidence_gap_cap_regularization_loss: Tensor | None = None
        class_evidence_gap_cap_label_multiplier: Tensor | None = None
        class_evidence_gap_cap_effective_cap: Tensor | None = None
        if self.cfg.class_evidence_gap_cap_regularization.enabled:
            (
                class_evidence_gap_cap_regularization,
                class_evidence_gap_cap_regularization_loss,
                class_evidence_gap_cap_label_multiplier,
                class_evidence_gap_cap_effective_cap,
            ) = self._compute_class_evidence_gap_cap_regularization_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = (
                scheduled_total + class_evidence_gap_cap_regularization_loss
            )
            monitor_total = monitor_total + class_evidence_gap_cap_regularization_loss
        class_evidence_positive_gap_cap_regularization: Tensor | None = None
        class_evidence_positive_gap_cap_regularization_loss: Tensor | None = None
        if self.cfg.class_evidence_positive_gap_cap_regularization.enabled:
            (
                class_evidence_positive_gap_cap_regularization,
                class_evidence_positive_gap_cap_regularization_loss,
            ) = self._compute_class_evidence_positive_gap_cap_regularization_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = (
                scheduled_total + class_evidence_positive_gap_cap_regularization_loss
            )
            monitor_total = (
                monitor_total + class_evidence_positive_gap_cap_regularization_loss
            )
        interaction_gap_cap_regularization: Tensor | None = None
        interaction_gap_cap_regularization_loss: Tensor | None = None
        if self.cfg.interaction_gap_cap_regularization.enabled:
            (
                interaction_gap_cap_regularization,
                interaction_gap_cap_regularization_loss,
            ) = self._compute_interaction_gap_cap_regularization_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + interaction_gap_cap_regularization_loss
            monitor_total = monitor_total + interaction_gap_cap_regularization_loss
        class_top_branch_relative_margin: Tensor | None = None
        class_top_branch_relative_margin_loss: Tensor | None = None
        class_top_branch_relative_margin_support_multiplier: Tensor | None = None
        class_top_branch_relative_margin_hardness_multiplier: Tensor | None = None
        if self.cfg.class_top_branch_relative_margin.enabled:
            (
                class_top_branch_relative_margin,
                class_top_branch_relative_margin_loss,
                class_top_branch_relative_margin_support_multiplier,
                class_top_branch_relative_margin_hardness_multiplier,
            ) = self._compute_class_top_branch_relative_margin_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + class_top_branch_relative_margin_loss
            monitor_total = monitor_total + class_top_branch_relative_margin_loss
        top_teacher_gap_min_constraint: Tensor | None = None
        top_teacher_gap_min_constraint_loss: Tensor | None = None
        top_teacher_gap_min_target: Tensor | None = None
        if self.cfg.top_teacher_gap_min_constraint.enabled:
            (
                top_teacher_gap_min_constraint,
                top_teacher_gap_min_constraint_loss,
                top_teacher_gap_min_target,
            ) = self._compute_top_teacher_gap_min_constraint_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + top_teacher_gap_min_constraint_loss
            monitor_total = monitor_total + top_teacher_gap_min_constraint_loss
        top_support_score_margin: Tensor | None = None
        top_support_score_margin_loss: Tensor | None = None
        top_support_score_margin_label_multiplier: Tensor | None = None
        top_support_score_margin_support_multiplier: Tensor | None = None
        top_support_score_margin_hardness_multiplier: Tensor | None = None
        if self.cfg.top_support_score_margin.enabled:
            (
                top_support_score_margin,
                top_support_score_margin_loss,
                top_support_score_margin_label_multiplier,
                top_support_score_margin_support_multiplier,
                top_support_score_margin_hardness_multiplier,
            ) = self._compute_top_support_score_margin_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + top_support_score_margin_loss
            monitor_total = monitor_total + top_support_score_margin_loss
        top_support_gap_min_constraint: Tensor | None = None
        top_support_gap_min_constraint_loss: Tensor | None = None
        top_support_gap_min_target: Tensor | None = None
        if self.cfg.top_support_gap_min_constraint.enabled:
            (
                top_support_gap_min_constraint,
                top_support_gap_min_constraint_loss,
                top_support_gap_min_target,
            ) = self._compute_top_support_gap_min_constraint_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + top_support_gap_min_constraint_loss
            monitor_total = monitor_total + top_support_gap_min_constraint_loss
        branch_direct_score_margin: Tensor | None = None
        branch_direct_score_margin_loss: Tensor | None = None
        if self.cfg.branch_direct_score_margin.enabled:
            branch_direct_score_margin, branch_direct_score_margin_loss = (
                self._compute_branch_direct_score_margin_loss(
                    output,
                    labels,
                    epoch=epoch,
                )
            )
            scheduled_total = scheduled_total + branch_direct_score_margin_loss
            monitor_total = monitor_total + branch_direct_score_margin_loss
        branch_support_score_margin: Tensor | None = None
        branch_support_score_margin_loss: Tensor | None = None
        if self.cfg.branch_support_score_margin.enabled:
            branch_support_score_margin, branch_support_score_margin_loss = (
                self._compute_branch_support_score_margin_loss(
                    output,
                    labels,
                    epoch=epoch,
                )
            )
            scheduled_total = scheduled_total + branch_support_score_margin_loss
            monitor_total = monitor_total + branch_support_score_margin_loss
        branch_path_dominance_constraint: Tensor | None = None
        branch_path_dominance_constraint_loss: Tensor | None = None
        branch_path_dominance_support_multiplier: Tensor | None = None
        branch_path_dominance_label_multiplier: Tensor | None = None
        branch_path_dominance_allowed_drop: Tensor | None = None
        if self.cfg.branch_path_dominance_constraint.enabled:
            (
                branch_path_dominance_constraint,
                branch_path_dominance_constraint_loss,
                branch_path_dominance_support_multiplier,
                branch_path_dominance_label_multiplier,
                branch_path_dominance_allowed_drop,
            ) = self._compute_branch_path_dominance_constraint_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + branch_path_dominance_constraint_loss
            monitor_total = monitor_total + branch_path_dominance_constraint_loss
        branch_support_disagreement_cap_regularization: Tensor | None = None
        branch_support_disagreement_cap_regularization_loss: Tensor | None = None
        branch_support_disagreement_weight: Tensor | None = None
        if self.cfg.branch_support_disagreement_cap_regularization.enabled:
            (
                branch_support_disagreement_cap_regularization,
                branch_support_disagreement_cap_regularization_loss,
                branch_support_disagreement_weight,
            ) = self._compute_branch_support_disagreement_cap_regularization_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = (
                scheduled_total + branch_support_disagreement_cap_regularization_loss
            )
            monitor_total = (
                monitor_total + branch_support_disagreement_cap_regularization_loss
            )
        class_gated_branch_logit_margin: Tensor | None = None
        class_gated_branch_logit_margin_loss: Tensor | None = None
        if self.cfg.class_gated_branch_logit_margin.enabled:
            (
                class_gated_branch_logit_margin,
                class_gated_branch_logit_margin_loss,
            ) = self._compute_class_gated_branch_logit_margin_loss(output, labels)
            scheduled_total = scheduled_total + class_gated_branch_logit_margin_loss
            monitor_total = monitor_total + class_gated_branch_logit_margin_loss
        branch_to_evidence_ranking_consistency: Tensor | None = None
        branch_to_evidence_ranking_consistency_loss: Tensor | None = None
        if self.cfg.branch_to_evidence_ranking_consistency.enabled:
            (
                branch_to_evidence_ranking_consistency,
                branch_to_evidence_ranking_consistency_loss,
            ) = self._compute_branch_to_evidence_ranking_consistency_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = (
                scheduled_total + branch_to_evidence_ranking_consistency_loss
            )
            monitor_total = monitor_total + branch_to_evidence_ranking_consistency_loss
        global_residual_anti_veto: Tensor | None = None
        global_residual_anti_veto_loss: Tensor | None = None
        global_residual_anti_veto_eligible_fraction: Tensor | None = None
        if self.cfg.global_residual_anti_veto.enabled:
            (
                global_residual_anti_veto,
                global_residual_anti_veto_loss,
                global_residual_anti_veto_eligible_fraction,
            ) = self._compute_global_residual_anti_veto_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + global_residual_anti_veto_loss
            monitor_total = monitor_total + global_residual_anti_veto_loss
        residual_contradiction_regularization: Tensor | None = None
        residual_contradiction_regularization_loss: Tensor | None = None
        residual_contradiction_regularization_eligible_fraction: Tensor | None = None
        if self.cfg.residual_contradiction_regularization.enabled:
            (
                residual_contradiction_regularization,
                residual_contradiction_regularization_loss,
                residual_contradiction_regularization_eligible_fraction,
            ) = self._compute_residual_contradiction_regularization_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = (
                scheduled_total + residual_contradiction_regularization_loss
            )
            monitor_total = monitor_total + residual_contradiction_regularization_loss
        gate_weighted_branch_margin: Tensor | None = None
        gate_weighted_branch_margin_loss: Tensor | None = None
        if self.cfg.gate_weighted_branch_margin.enabled:
            gate_weighted_branch_margin, gate_weighted_branch_margin_loss = (
                self._compute_gate_weighted_branch_margin_loss(
                    output,
                    labels,
                    epoch=epoch,
                )
            )
            scheduled_total = scheduled_total + gate_weighted_branch_margin_loss
            monitor_total = monitor_total + gate_weighted_branch_margin_loss
        top_branch_margin: Tensor | None = None
        top_branch_margin_loss: Tensor | None = None
        top_branch_margin_effective_weight: Tensor | None = None
        top_branch_margin_hardness_multiplier: Tensor | None = None
        if self.cfg.top_branch_margin.enabled:
            (
                top_branch_margin,
                top_branch_margin_loss,
                top_branch_margin_effective_weight,
                top_branch_margin_hardness_multiplier,
            ) = self._compute_top_branch_margin_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + top_branch_margin_loss
            monitor_total = monitor_total + top_branch_margin_loss
        gate_branch_regret: Tensor | None = None
        gate_branch_regret_loss: Tensor | None = None
        gate_branch_regret_eligible_fraction: Tensor | None = None
        gate_branch_regret_weight_multiplier: Tensor | None = None
        gate_branch_regret_effective_weight: Tensor | None = None
        if self.cfg.gate_branch_regret.enabled:
            (
                gate_branch_regret,
                gate_branch_regret_loss,
                gate_branch_regret_eligible_fraction,
                gate_branch_regret_weight_multiplier,
                gate_branch_regret_effective_weight,
            ) = self._compute_gate_branch_regret_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + gate_branch_regret_loss
            monitor_total = monitor_total + gate_branch_regret_loss
        gate_bad_branch_suppression: Tensor | None = None
        gate_bad_branch_suppression_loss: Tensor | None = None
        gate_bad_branch_suppression_bad_gate_mass: Tensor | None = None
        gate_bad_branch_suppression_weight_multiplier: Tensor | None = None
        gate_bad_branch_suppression_effective_weight: Tensor | None = None
        if self.cfg.gate_bad_branch_suppression.enabled:
            (
                gate_bad_branch_suppression,
                gate_bad_branch_suppression_loss,
                gate_bad_branch_suppression_bad_gate_mass,
                gate_bad_branch_suppression_weight_multiplier,
                gate_bad_branch_suppression_effective_weight,
            ) = self._compute_gate_bad_branch_suppression_loss(
                output,
                labels,
                epoch=epoch,
            )
            scheduled_total = scheduled_total + gate_bad_branch_suppression_loss
            monitor_total = monitor_total + gate_bad_branch_suppression_loss
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
            class_evidence_margin=class_evidence_margin,
            class_evidence_margin_loss=class_evidence_margin_loss,
            class_evidence_gap_cap_regularization=(
                class_evidence_gap_cap_regularization
            ),
            class_evidence_gap_cap_regularization_loss=(
                class_evidence_gap_cap_regularization_loss
            ),
            class_evidence_gap_cap_label_multiplier=(
                class_evidence_gap_cap_label_multiplier
            ),
            class_evidence_gap_cap_effective_cap=(class_evidence_gap_cap_effective_cap),
            class_evidence_positive_gap_cap_regularization=(
                class_evidence_positive_gap_cap_regularization
            ),
            class_evidence_positive_gap_cap_regularization_loss=(
                class_evidence_positive_gap_cap_regularization_loss
            ),
            interaction_gap_cap_regularization=interaction_gap_cap_regularization,
            interaction_gap_cap_regularization_loss=(
                interaction_gap_cap_regularization_loss
            ),
            top_support_score_margin=top_support_score_margin,
            top_support_score_margin_loss=top_support_score_margin_loss,
            top_support_score_margin_label_multiplier=(
                top_support_score_margin_label_multiplier
            ),
            top_support_score_margin_support_multiplier=(
                top_support_score_margin_support_multiplier
            ),
            top_support_score_margin_hardness_multiplier=(
                top_support_score_margin_hardness_multiplier
            ),
            top_support_gap_min_constraint=top_support_gap_min_constraint,
            top_support_gap_min_constraint_loss=top_support_gap_min_constraint_loss,
            top_support_gap_min_target=top_support_gap_min_target,
            class_top_branch_relative_margin=class_top_branch_relative_margin,
            class_top_branch_relative_margin_loss=(
                class_top_branch_relative_margin_loss
            ),
            class_top_branch_relative_margin_support_multiplier=(
                class_top_branch_relative_margin_support_multiplier
            ),
            class_top_branch_relative_margin_hardness_multiplier=(
                class_top_branch_relative_margin_hardness_multiplier
            ),
            top_teacher_gap_min_constraint=top_teacher_gap_min_constraint,
            top_teacher_gap_min_constraint_loss=top_teacher_gap_min_constraint_loss,
            top_teacher_gap_min_target=top_teacher_gap_min_target,
            branch_direct_score_margin=branch_direct_score_margin,
            branch_direct_score_margin_loss=branch_direct_score_margin_loss,
            branch_path_dominance_constraint=branch_path_dominance_constraint,
            branch_path_dominance_constraint_loss=(
                branch_path_dominance_constraint_loss
            ),
            branch_path_dominance_support_multiplier=(
                branch_path_dominance_support_multiplier
            ),
            branch_path_dominance_label_multiplier=(
                branch_path_dominance_label_multiplier
            ),
            branch_path_dominance_allowed_drop=branch_path_dominance_allowed_drop,
            branch_support_disagreement_cap_regularization=(
                branch_support_disagreement_cap_regularization
            ),
            branch_support_disagreement_cap_regularization_loss=(
                branch_support_disagreement_cap_regularization_loss
            ),
            branch_support_disagreement_weight=branch_support_disagreement_weight,
            branch_support_score_margin=branch_support_score_margin,
            branch_support_score_margin_loss=branch_support_score_margin_loss,
            class_gated_branch_logit_margin=class_gated_branch_logit_margin,
            class_gated_branch_logit_margin_loss=(class_gated_branch_logit_margin_loss),
            branch_to_evidence_ranking_consistency=(
                branch_to_evidence_ranking_consistency
            ),
            branch_to_evidence_ranking_consistency_loss=(
                branch_to_evidence_ranking_consistency_loss
            ),
            global_residual_anti_veto=global_residual_anti_veto,
            global_residual_anti_veto_loss=global_residual_anti_veto_loss,
            global_residual_anti_veto_eligible_fraction=(
                global_residual_anti_veto_eligible_fraction
            ),
            residual_contradiction_regularization=(
                residual_contradiction_regularization
            ),
            residual_contradiction_regularization_loss=(
                residual_contradiction_regularization_loss
            ),
            residual_contradiction_regularization_eligible_fraction=(
                residual_contradiction_regularization_eligible_fraction
            ),
            gate_weighted_branch_margin=gate_weighted_branch_margin,
            gate_weighted_branch_margin_loss=gate_weighted_branch_margin_loss,
            gate_branch_regret=gate_branch_regret,
            gate_branch_regret_loss=gate_branch_regret_loss,
            gate_branch_regret_eligible_fraction=(gate_branch_regret_eligible_fraction),
            gate_branch_regret_weight_multiplier=(gate_branch_regret_weight_multiplier),
            gate_branch_regret_effective_weight=gate_branch_regret_effective_weight,
            gate_bad_branch_suppression=gate_bad_branch_suppression,
            gate_bad_branch_suppression_loss=gate_bad_branch_suppression_loss,
            gate_bad_branch_suppression_bad_gate_mass=(
                gate_bad_branch_suppression_bad_gate_mass
            ),
            gate_bad_branch_suppression_weight_multiplier=(
                gate_bad_branch_suppression_weight_multiplier
            ),
            gate_bad_branch_suppression_effective_weight=(
                gate_bad_branch_suppression_effective_weight
            ),
            top_branch_margin=top_branch_margin,
            top_branch_margin_loss=top_branch_margin_loss,
            top_branch_margin_effective_weight=top_branch_margin_effective_weight,
            top_branch_margin_hardness_multiplier=(
                top_branch_margin_hardness_multiplier
            ),
            branch_binary_aux_weight=branch_binary_weight,
            branch_binary_aux_monitor_weight=branch_binary_monitor_weight,
        )

    def _add_counts_by_label(
        self,
        label_indices: Tensor,
        flags: Tensor,
        *,
        counts: list[int],
        flag_counts: list[int],
    ) -> None:
        labels_cpu = label_indices.detach().cpu()
        flags_cpu = flags.detach().cpu().to(torch.bool)
        for class_index in range(int(self.cfg.num_classes)):
            class_mask = labels_cpu == class_index
            count = int(class_mask.sum().item())
            if count == 0:
                continue
            counts[class_index] += count
            flag_counts[class_index] += int(flags_cpu[class_mask].sum().item())

    def _add_sums_by_label(
        self,
        label_indices: Tensor,
        values: Tensor,
        *,
        counts: list[int],
        value_sums: list[float],
    ) -> None:
        labels_cpu = label_indices.detach().cpu()
        values_cpu = values.detach().cpu()
        for class_index in range(int(self.cfg.num_classes)):
            class_mask = labels_cpu == class_index
            count = int(class_mask.sum().item())
            if count == 0:
                continue
            counts[class_index] += count
            value_sums[class_index] += float(values_cpu[class_mask].sum().item())

    def _observe_branch_objective_stats(
        self,
        output: AstModelOutput,
        labels: Tensor,
        stats: BranchObjectiveEpochStats,
    ) -> None:
        if self.cfg.top_branch_margin.auto_margin_by_train_stats.enabled:
            branch_logits, label_indices = self._branch_margin_inputs(
                output,
                labels,
                loss_name="top branch margin adaptive stats",
            )
            if label_indices.numel() > 0:
                branch_margin = self._true_class_branch_margins(
                    branch_logits,
                    label_indices,
                ).detach()
                top_margin = branch_margin.max(dim=1).values
                margin_targets = self._top_branch_margin_targets(
                    label_indices,
                    top_margin,
                )
                violations = top_margin < margin_targets
                self._add_counts_by_label(
                    label_indices,
                    violations,
                    counts=stats.top_counts,
                    flag_counts=stats.top_violation_counts,
                )
        if self.cfg.gate_branch_regret.auto_positive_threshold_by_train_stats.enabled:
            branch_logits, _, label_indices = self._class_aware_branch_margin_inputs(
                output,
                labels,
                loss_name="gate-branch regret adaptive stats",
            )
            if label_indices.numel() > 0:
                branch_margin = self._true_class_branch_margins(
                    branch_logits,
                    label_indices,
                ).detach()
                best_margin = branch_margin.max(dim=1).values
                positive_thresholds = self._gate_branch_regret_thresholds(
                    label_indices,
                    best_margin,
                )
                eligible = best_margin > positive_thresholds
                self._add_counts_by_label(
                    label_indices,
                    eligible,
                    counts=stats.regret_counts,
                    flag_counts=stats.regret_eligible_counts,
                )
        bad_cfg = self.cfg.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats
        if bad_cfg.enabled:
            branch_logits, gate_weights, label_indices = (
                self._class_aware_branch_margin_inputs(
                    output,
                    labels,
                    loss_name="gate bad branch suppression adaptive stats",
                )
            )
            if label_indices.numel() > 0:
                branch_margin = self._true_class_branch_margins(
                    branch_logits,
                    label_indices,
                ).detach()
                true_class_gate = self._true_class_gate_weights(
                    gate_weights,
                    label_indices,
                    dtype=branch_logits.dtype,
                ).detach()
                thresholds = self._gate_bad_branch_suppression_thresholds(
                    label_indices,
                    branch_margin[:, 0],
                )
                bad_gate_mass = (
                    true_class_gate
                    * (branch_margin < thresholds.unsqueeze(1)).to(
                        dtype=branch_logits.dtype
                    )
                ).sum(dim=1)
                self._add_sums_by_label(
                    label_indices,
                    bad_gate_mass,
                    counts=stats.bad_suppression_counts,
                    value_sums=stats.bad_suppression_gate_mass_sums,
                )

    def _update_adaptive_rate(
        self,
        previous: float | None,
        observed: float,
        *,
        ema: float,
    ) -> float:
        if previous is None:
            return observed
        return (float(ema) * previous) + ((1.0 - float(ema)) * observed)

    def _update_adaptive_branch_objective_state(
        self,
        *,
        epoch: int,
        stats: BranchObjectiveEpochStats,
    ) -> None:
        state = self._adaptive_branch_objective_state
        top_cfg = self.cfg.top_branch_margin.auto_margin_by_train_stats
        if (
            top_cfg.enabled
            and state.top_branch_margin_by_class is not None
            and epoch >= int(top_cfg.start_epoch)
            and (epoch - int(top_cfg.start_epoch)) % int(top_cfg.update_interval_epochs)
            == 0
        ):
            for class_index, count in enumerate(stats.top_counts):
                if count == 0:
                    continue
                observed = stats.top_violation_counts[class_index] / float(count)
                previous = state.top_branch_violation_rate_ema_by_class[class_index]
                ema_rate = self._update_adaptive_rate(
                    previous,
                    observed,
                    ema=float(top_cfg.ema),
                )
                state.top_branch_violation_rate_ema_by_class[class_index] = ema_rate
                label_name = self._class_names()[class_index]
                target_rate = float(top_cfg.target_violation_rate_by_label[label_name])
                current = state.top_branch_margin_by_class[class_index]
                if ema_rate < target_rate:
                    current += float(top_cfg.step)
                else:
                    current -= float(top_cfg.step)
                current = min(
                    max(current, float(top_cfg.min_margin_by_label[label_name])),
                    float(top_cfg.max_margin_by_label[label_name]),
                )
                state.top_branch_margin_by_class[class_index] = current
        regret_cfg = self.cfg.gate_branch_regret.auto_positive_threshold_by_train_stats
        if (
            regret_cfg.enabled
            and state.gate_branch_regret_positive_threshold_by_class is not None
            and epoch >= int(regret_cfg.start_epoch)
            and (epoch - int(regret_cfg.start_epoch))
            % int(regret_cfg.update_interval_epochs)
            == 0
        ):
            for class_index, count in enumerate(stats.regret_counts):
                if count == 0:
                    continue
                observed = stats.regret_eligible_counts[class_index] / float(count)
                previous = state.gate_branch_regret_eligible_rate_ema_by_class[
                    class_index
                ]
                ema_rate = self._update_adaptive_rate(
                    previous,
                    observed,
                    ema=float(regret_cfg.ema),
                )
                state.gate_branch_regret_eligible_rate_ema_by_class[class_index] = (
                    ema_rate
                )
                label_name = self._class_names()[class_index]
                target_rate = float(
                    regret_cfg.target_eligible_rate_by_label[label_name]
                )
                current = state.gate_branch_regret_positive_threshold_by_class[
                    class_index
                ]
                if ema_rate < target_rate:
                    current -= float(regret_cfg.step)
                else:
                    current += float(regret_cfg.step)
                current = min(
                    max(current, float(regret_cfg.min_threshold_by_label[label_name])),
                    float(regret_cfg.max_threshold_by_label[label_name]),
                )
                state.gate_branch_regret_positive_threshold_by_class[class_index] = (
                    current
                )
        bad_cfg = self.cfg.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats
        if (
            bad_cfg.enabled
            and state.gate_bad_branch_suppression_threshold_by_class is not None
            and epoch >= int(bad_cfg.start_epoch)
            and (epoch - int(bad_cfg.start_epoch)) % int(bad_cfg.update_interval_epochs)
            == 0
        ):
            for class_index, count in enumerate(stats.bad_suppression_counts):
                if count == 0:
                    continue
                observed = stats.bad_suppression_gate_mass_sums[class_index] / float(
                    count
                )
                previous = state.gate_bad_branch_suppression_bad_gate_mass_ema_by_class[
                    class_index
                ]
                ema_rate = self._update_adaptive_rate(
                    previous,
                    observed,
                    ema=float(bad_cfg.ema),
                )
                state.gate_bad_branch_suppression_bad_gate_mass_ema_by_class[
                    class_index
                ] = ema_rate
                label_name = self._class_names()[class_index]
                target_mass = float(bad_cfg.target_bad_gate_mass_by_label[label_name])
                current = state.gate_bad_branch_suppression_threshold_by_class[
                    class_index
                ]
                if ema_rate < target_mass:
                    current += float(bad_cfg.step)
                else:
                    current -= float(bad_cfg.step)
                current = min(
                    max(
                        current,
                        float(bad_cfg.min_threshold_by_label[label_name]),
                    ),
                    float(bad_cfg.max_threshold_by_label[label_name]),
                )
                state.gate_bad_branch_suppression_threshold_by_class[class_index] = (
                    current
                )
        state.last_update_epoch = int(epoch)

    def _branch_objective_stats_summary(
        self,
        stats: BranchObjectiveEpochStats,
    ) -> dict[str, dict[str, float | None]]:
        labels = self._class_names()

        def rates(counts: list[int], flag_counts: list[int]) -> dict[str, float | None]:
            return {
                label_name: (
                    flag_counts[index] / float(counts[index])
                    if counts[index] > 0
                    else None
                )
                for index, label_name in enumerate(labels)
            }

        return {
            "top_branch_violation_rate_by_label": rates(
                stats.top_counts,
                stats.top_violation_counts,
            ),
            "gate_branch_regret_eligible_rate_by_label": rates(
                stats.regret_counts,
                stats.regret_eligible_counts,
            ),
            "gate_bad_branch_suppression_bad_gate_mass_by_label": {
                label_name: (
                    stats.bad_suppression_gate_mass_sums[index]
                    / float(stats.bad_suppression_counts[index])
                    if stats.bad_suppression_counts[index] > 0
                    else None
                )
                for index, label_name in enumerate(labels)
            },
        }

    def _branch_objective_diagnostic_rows(
        self,
        output: AstModelOutput,
        labels: Tensor,
        *,
        epoch: int,
    ) -> list[dict[str, Any]]:
        if output.branch_logits is None:
            return [{} for _ in range(int(labels.numel()))]
        rows: list[dict[str, Any]] = [{} for _ in range(int(labels.numel()))]
        branch_logits = output.branch_logits.detach()
        if branch_logits.ndim != 3:
            return rows
        label_indices = labels.to(device=branch_logits.device, dtype=torch.long)
        if label_indices.ndim != 1 or int(label_indices.shape[0]) != int(
            branch_logits.shape[0]
        ):
            return rows
        branch_margin = self._true_class_branch_margins(
            branch_logits,
            label_indices,
        ).detach()
        top_margin, top_branch_indices, top_negative_indices = (
            self._top_true_class_branch_margin(branch_logits, label_indices)
        )
        top_margin = top_margin.detach()
        top_branch_indices = top_branch_indices.detach()
        top_negative_indices = top_negative_indices.detach()
        if self.cfg.top_branch_margin.enabled:
            margin_targets = self._top_branch_margin_targets(label_indices, top_margin)
            top_violations = top_margin < margin_targets
            phase_multipliers = self._top_branch_margin_phase_multipliers(
                label_indices,
                top_margin,
                epoch=epoch,
            )
            hardness_multipliers, hardness = self._top_branch_margin_hardness_weights(
                torch.relu(margin_targets - top_margin),
                self.cfg.top_branch_margin.hardness_weighting,
                loss_name="top_branch_margin",
            )
            effective_weights = float(self.cfg.top_branch_margin.weight) * (
                phase_multipliers * hardness_multipliers
            )
            for index, row in enumerate(rows):
                row["top_branch_margin_target"] = float(
                    margin_targets[index].detach().cpu().item()
                )
                row["top_branch_margin_value"] = float(
                    top_margin[index].detach().cpu().item()
                )
                row["top_branch_margin_violation"] = bool(
                    top_violations[index].detach().cpu().item()
                )
                row["top_branch_margin_effective_label_multiplier"] = float(
                    phase_multipliers[index].detach().cpu().item()
                )
                row["top_branch_margin_hardness"] = float(
                    hardness[index].detach().cpu().item()
                )
                row["top_branch_margin_hardness_weight"] = float(
                    hardness_multipliers[index].detach().cpu().item()
                )
                row["top_branch_margin_total_multiplier"] = float(
                    (phase_multipliers[index] * hardness_multipliers[index])
                    .detach()
                    .cpu()
                    .item()
                )
                row["top_branch_margin_effective_weight"] = float(
                    effective_weights[index].detach().cpu().item()
                )
        if (
            self.cfg.class_top_branch_relative_margin.enabled
            and output.class_top_branch_margin_features is not None
        ):
            rel_cfg = self.cfg.class_top_branch_relative_margin
            teacher_logits = output.class_top_branch_margin_features.detach()
            if tuple(teacher_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                teacher_gap, teacher_negative = self._true_vs_hardest_negative_gap(
                    teacher_logits,
                    label_indices,
                )
                active = int(epoch) > int(rel_cfg.warmup_epochs)
                margin_deficit = torch.relu(float(rel_cfg.margin) - teacher_gap)
                support_weight, _ = self._margin_support_weights(
                    output,
                    labels,
                    label_indices,
                    teacher_gap,
                    rel_cfg.support_weighting,
                    loss_name="class_top_branch_relative_margin",
                )
                hardness_weight, hardness = self._teacher_gap_deficit_hardness_weights(
                    margin_deficit,
                    rel_cfg.hardness_weighting,
                    loss_name="class_top_branch_relative_margin",
                )
                penalties = margin_deficit * support_weight * hardness_weight
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["class_top_branch_relative_gap"] = float(
                        teacher_gap[index].detach().cpu().item()
                    )
                    row["class_top_branch_relative_negative_class"] = int(
                        teacher_negative[index].detach().cpu().item()
                    )
                    row["class_top_branch_relative_margin_target"] = float(
                        rel_cfg.margin
                    )
                    row["class_top_branch_relative_margin_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["class_top_branch_relative_margin_support_weight"] = float(
                        support_weight[index].detach().cpu().item()
                    )
                    row["class_top_branch_relative_margin_hardness"] = float(
                        hardness[index].detach().cpu().item()
                    )
                    row["class_top_branch_relative_margin_hardness_weight"] = float(
                        hardness_weight[index].detach().cpu().item()
                    )
                    row["class_top_branch_relative_margin_eligible"] = bool(active)
        if (
            self.cfg.top_teacher_gap_min_constraint.enabled
            and output.class_top_branch_margin_features is not None
        ):
            teacher_min_cfg = self.cfg.top_teacher_gap_min_constraint
            teacher_logits = output.class_top_branch_margin_features.detach()
            if tuple(teacher_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                teacher_gap, teacher_negative = self._true_vs_hardest_negative_gap(
                    teacher_logits,
                    label_indices,
                )
                support_gap = self._top_branch_support_gap(
                    output,
                    labels,
                    label_indices,
                    loss_name="top_teacher_gap_min_constraint",
                ).detach()
                support = torch.clamp(
                    torch.relu(support_gap),
                    max=float(teacher_min_cfg.support_cap),
                )
                target_min_gap = float(teacher_min_cfg.base_min_gap) + (
                    float(teacher_min_cfg.support_gain) * support
                )
                active = int(epoch) > int(teacher_min_cfg.warmup_epochs)
                penalties = torch.relu(target_min_gap - teacher_gap)
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["top_teacher_gap_min_gap"] = float(
                        teacher_gap[index].detach().cpu().item()
                    )
                    row["top_teacher_gap_min_negative_class"] = int(
                        teacher_negative[index].detach().cpu().item()
                    )
                    row["top_teacher_gap_min_target"] = float(
                        target_min_gap[index].detach().cpu().item()
                    )
                    row["top_teacher_gap_min_support_value"] = float(
                        support_gap[index].detach().cpu().item()
                    )
                    row["top_teacher_gap_min_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["top_teacher_gap_min_eligible"] = bool(active)
        if (
            self.cfg.class_evidence_margin.enabled
            and output.class_evidence_logits is not None
        ):
            margin_cfg = self.cfg.class_evidence_margin
            evidence_logits = output.class_evidence_logits.detach()
            if tuple(evidence_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                margin_gap: Tensor | None = None
                margin_negative: Tensor | None = None
                eligible = torch.ones_like(label_indices, dtype=torch.bool)
                if margin_cfg.mode in {
                    "true_vs_hardest_negative",
                    "softplus_true_vs_hardest_negative",
                }:
                    margin_gap, margin_negative = self._true_vs_hardest_negative_gap(
                        evidence_logits,
                        label_indices,
                    )
                elif (
                    margin_cfg.mode == "minority_vs_major"
                    and self.cfg.class_evidence_margin_major_index is not None
                ):
                    major_index = int(self.cfg.class_evidence_margin_major_index)
                    if 0 <= major_index < int(evidence_logits.shape[1]):
                        true_logits = evidence_logits.gather(
                            1,
                            label_indices.unsqueeze(1),
                        ).squeeze(1)
                        margin_gap = true_logits - evidence_logits[:, major_index]
                        margin_negative = torch.full_like(label_indices, major_index)
                        eligible = label_indices != major_index
                if margin_gap is not None and margin_negative is not None:
                    if margin_cfg.mode == "softplus_true_vs_hardest_negative":
                        support_weight = torch.ones_like(margin_gap)
                        support_gap = torch.zeros_like(margin_gap)
                        temperature = float(margin_cfg.temperature)
                        penalties = temperature * F.softplus(-margin_gap / temperature)
                    else:
                        support_weight, support_gap = self._margin_support_weights(
                            output,
                            labels,
                            label_indices,
                            margin_gap,
                            margin_cfg.support_weighting,
                            loss_name="class_evidence_margin",
                        )
                        penalties = (
                            torch.relu(float(margin_cfg.margin) - margin_gap)
                            * support_weight
                        )
                    penalties = torch.where(
                        eligible,
                        penalties,
                        torch.zeros_like(penalties),
                    )
                    for index, row in enumerate(rows):
                        row["class_evidence_margin_gap"] = float(
                            margin_gap[index].detach().cpu().item()
                        )
                        row["class_evidence_margin_negative_class"] = int(
                            margin_negative[index].detach().cpu().item()
                        )
                        row["class_evidence_margin_support_gap"] = float(
                            support_gap[index].detach().cpu().item()
                        )
                        row["class_evidence_margin_support_weight"] = float(
                            support_weight[index].detach().cpu().item()
                        )
                        row["class_evidence_margin_penalty"] = float(
                            penalties[index].detach().cpu().item()
                        )
                        row["class_evidence_margin_mode"] = str(margin_cfg.mode)
                        row["class_evidence_margin_temperature"] = float(
                            margin_cfg.temperature
                        )
                        row["class_evidence_margin_eligible"] = bool(
                            eligible[index].detach().cpu().item()
                        )
        if (
            self.cfg.class_evidence_gap_cap_regularization.enabled
            and output.class_evidence_logits is not None
        ):
            gap_cap_cfg = self.cfg.class_evidence_gap_cap_regularization
            evidence_logits = output.class_evidence_logits.detach()
            if tuple(evidence_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                gap_cap_gap, gap_cap_negative = self._true_vs_hardest_negative_gap(
                    evidence_logits,
                    label_indices,
                )
                active = int(epoch) > int(gap_cap_cfg.warmup_epochs)
                effective_caps = self._class_values_tensor(
                    self.cfg.class_evidence_gap_cap_negative_cap_by_class,
                    label_indices,
                    device=evidence_logits.device,
                    dtype=evidence_logits.dtype,
                    fallback=float(gap_cap_cfg.negative_gap_cap),
                )
                label_multipliers = self._class_values_tensor(
                    self.cfg.class_evidence_gap_cap_label_weight_by_class,
                    label_indices,
                    device=evidence_logits.device,
                    dtype=evidence_logits.dtype,
                    fallback=1.0,
                )
                gap_cap_penalty = (
                    torch.relu(-gap_cap_gap - effective_caps) * label_multipliers
                )
                if not active:
                    gap_cap_penalty = torch.zeros_like(gap_cap_penalty)
                gap_cap_eligible = gap_cap_penalty > 0
                for index, row in enumerate(rows):
                    row["class_evidence_gap_cap_gap"] = float(
                        gap_cap_gap[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_negative_class"] = int(
                        gap_cap_negative[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_penalty"] = float(
                        gap_cap_penalty[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_cap"] = float(
                        effective_caps[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_effective_cap"] = float(
                        effective_caps[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_label_multiplier"] = float(
                        label_multipliers[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_eligible"] = bool(
                        gap_cap_eligible[index].detach().cpu().item()
                    )
                    row["class_evidence_gap_cap_mode"] = str(gap_cap_cfg.mode)
        if (
            self.cfg.class_evidence_positive_gap_cap_regularization.enabled
            and output.class_evidence_logits is not None
        ):
            positive_gap_cap_cfg = (
                self.cfg.class_evidence_positive_gap_cap_regularization
            )
            evidence_logits = output.class_evidence_logits.detach()
            if tuple(evidence_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                positive_gap, positive_negative = self._true_vs_hardest_negative_gap(
                    evidence_logits,
                    label_indices,
                )
                active = int(epoch) > int(positive_gap_cap_cfg.warmup_epochs)
                positive_penalty = torch.relu(
                    positive_gap - float(positive_gap_cap_cfg.positive_gap_cap)
                )
                if not active:
                    positive_penalty = torch.zeros_like(positive_penalty)
                positive_eligible = positive_penalty > 0
                for index, row in enumerate(rows):
                    row["class_evidence_positive_gap_cap_gap"] = float(
                        positive_gap[index].detach().cpu().item()
                    )
                    row["class_evidence_positive_gap_cap_negative_class"] = int(
                        positive_negative[index].detach().cpu().item()
                    )
                    row["class_evidence_positive_gap_cap_penalty"] = float(
                        positive_penalty[index].detach().cpu().item()
                    )
                    row["class_evidence_positive_gap_cap_cap"] = float(
                        positive_gap_cap_cfg.positive_gap_cap
                    )
                    row["class_evidence_positive_gap_cap_eligible"] = bool(
                        positive_eligible[index].detach().cpu().item()
                    )
                    row["class_evidence_positive_gap_cap_mode"] = str(
                        positive_gap_cap_cfg.mode
                    )
        if (
            self.cfg.interaction_gap_cap_regularization.enabled
            and output.class_evidence_interaction_scores is not None
        ):
            interaction_gap_cap_cfg = self.cfg.interaction_gap_cap_regularization
            interaction_scores = output.class_evidence_interaction_scores.detach()
            if tuple(interaction_scores.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                interaction_gap, interaction_negative = (
                    self._true_vs_hardest_negative_gap(
                        interaction_scores,
                        label_indices,
                    )
                )
                interaction_abs_gap = interaction_gap.abs()
                active = int(epoch) > int(interaction_gap_cap_cfg.warmup_epochs)
                interaction_penalty = torch.relu(
                    interaction_abs_gap - float(interaction_gap_cap_cfg.gap_cap)
                )
                if not active:
                    interaction_penalty = torch.zeros_like(interaction_penalty)
                interaction_eligible = interaction_penalty > 0
                for index, row in enumerate(rows):
                    row["interaction_gap_cap_gap"] = float(
                        interaction_gap[index].detach().cpu().item()
                    )
                    row["interaction_gap_cap_abs_gap"] = float(
                        interaction_abs_gap[index].detach().cpu().item()
                    )
                    row["interaction_gap_cap_negative_class"] = int(
                        interaction_negative[index].detach().cpu().item()
                    )
                    row["interaction_gap_cap_penalty"] = float(
                        interaction_penalty[index].detach().cpu().item()
                    )
                    row["interaction_gap_cap_cap"] = float(
                        interaction_gap_cap_cfg.gap_cap
                    )
                    row["interaction_gap_cap_eligible"] = bool(
                        interaction_eligible[index].detach().cpu().item()
                    )
                    row["interaction_gap_cap_mode"] = str(interaction_gap_cap_cfg.mode)
        if (
            self.cfg.top_support_score_margin.enabled
            and output.class_evidence_top_support_scores is not None
        ):
            top_support_cfg = self.cfg.top_support_score_margin
            top_support_scores = output.class_evidence_top_support_scores.detach()
            if tuple(top_support_scores.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                margin_gap, margin_negative = self._true_vs_hardest_negative_gap(
                    top_support_scores,
                    label_indices,
                )
                active = int(epoch) > int(top_support_cfg.warmup_epochs)
                temperature = float(top_support_cfg.temperature)
                base_penalties = temperature * F.softplus(-margin_gap / temperature)
                label_multipliers = self._class_values_tensor(
                    self.cfg.top_support_score_margin_label_weight_by_class,
                    label_indices,
                    device=top_support_scores.device,
                    dtype=top_support_scores.dtype,
                    fallback=1.0,
                )
                support_multipliers, support_gap = self._margin_support_weights(
                    output,
                    labels,
                    label_indices,
                    margin_gap,
                    top_support_cfg.support_conditioned_multiplier,
                    loss_name="top_support_score_margin",
                )
                hardness_multipliers, hardness = self._margin_hardness_weights(
                    margin_gap,
                    top_support_cfg.hardness_weighting,
                    loss_name="top_support_score_margin",
                    expected_source="top_support_gap",
                )
                total_multipliers = (
                    label_multipliers * support_multipliers * hardness_multipliers
                )
                penalties = base_penalties * total_multipliers
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["top_support_score_margin_gap"] = float(
                        margin_gap[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_negative_class"] = int(
                        margin_negative[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_base_penalty"] = float(
                        base_penalties[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_label_multiplier"] = float(
                        label_multipliers[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_support_gap"] = float(
                        support_gap[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_support_multiplier"] = float(
                        support_multipliers[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_hardness"] = float(
                        hardness[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_hardness_weight"] = float(
                        hardness_multipliers[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_total_multiplier"] = float(
                        total_multipliers[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_weighted_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["top_support_score_margin_mode"] = str(top_support_cfg.mode)
                    row["top_support_score_margin_temperature"] = float(
                        top_support_cfg.temperature
                    )
                    row["top_support_score_margin_eligible"] = bool(active)
        if (
            self.cfg.top_support_gap_min_constraint.enabled
            and output.class_evidence_top_support_scores is not None
            and output.class_top_branch_margin_features is not None
        ):
            top_min_cfg = self.cfg.top_support_gap_min_constraint
            top_support_scores = output.class_evidence_top_support_scores.detach()
            top_branch_margin_features = (
                output.class_top_branch_margin_features.detach()
            )
            if (
                tuple(top_support_scores.shape)
                == tuple(top_branch_margin_features.shape)
                == (
                    int(labels.numel()),
                    int(branch_logits.shape[2]),
                )
            ):
                margin_gap, margin_negative = self._true_vs_hardest_negative_gap(
                    top_support_scores,
                    label_indices,
                )
                base_min_gap = self._class_values_tensor(
                    self.cfg.top_support_gap_min_base_by_class,
                    label_indices,
                    device=top_support_scores.device,
                    dtype=top_support_scores.dtype,
                    fallback=float(top_min_cfg.base_min_gap),
                )
                true_top_margin = top_branch_margin_features.gather(
                    1,
                    label_indices.unsqueeze(1),
                ).squeeze(1)
                support = torch.clamp(
                    torch.relu(true_top_margin),
                    max=float(top_min_cfg.support_cap),
                )
                target_min_gap = (
                    base_min_gap + float(top_min_cfg.support_gain) * support
                )
                active = int(epoch) > int(top_min_cfg.warmup_epochs)
                penalties = torch.relu(target_min_gap - margin_gap)
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["top_support_gap_min_constraint_gap"] = float(
                        margin_gap[index].detach().cpu().item()
                    )
                    row["top_support_gap_min_constraint_negative_class"] = int(
                        margin_negative[index].detach().cpu().item()
                    )
                    row["top_support_gap_min_target"] = float(
                        target_min_gap[index].detach().cpu().item()
                    )
                    row["top_support_gap_min_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["top_support_gap_min_support_value"] = float(
                        support[index].detach().cpu().item()
                    )
                    row["top_support_gap_min_base_min_gap"] = float(
                        base_min_gap[index].detach().cpu().item()
                    )
                    row["top_support_gap_min_eligible"] = bool(active)
        if (
            self.cfg.branch_support_score_margin.enabled
            and output.class_evidence_branch_support_scores is not None
        ):
            support_cfg = self.cfg.branch_support_score_margin
            support_scores = output.class_evidence_branch_support_scores.detach()
            if tuple(support_scores.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                margin_gap, margin_negative = self._true_vs_hardest_negative_gap(
                    support_scores,
                    label_indices,
                )
                active = int(epoch) > int(support_cfg.warmup_epochs)
                temperature = float(support_cfg.temperature)
                penalties = temperature * F.softplus(-margin_gap / temperature)
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["branch_support_score_margin_gap"] = float(
                        margin_gap[index].detach().cpu().item()
                    )
                    row["branch_support_score_margin_negative_class"] = int(
                        margin_negative[index].detach().cpu().item()
                    )
                    row["branch_support_score_margin_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["branch_support_score_margin_mode"] = str(support_cfg.mode)
                    row["branch_support_score_margin_temperature"] = float(
                        support_cfg.temperature
                    )
                    row["branch_support_score_margin_eligible"] = bool(active)
        if (
            self.cfg.branch_path_dominance_constraint.enabled
            and output.class_evidence_branch_support_scores is not None
            and output.class_evidence_logits is not None
        ):
            dominance_cfg = self.cfg.branch_path_dominance_constraint
            branch_scores = output.class_evidence_branch_support_scores.detach()
            evidence_logits = output.class_evidence_logits.detach()
            if (
                tuple(branch_scores.shape)
                == tuple(evidence_logits.shape)
                == (
                    int(labels.numel()),
                    int(branch_logits.shape[2]),
                )
            ):
                branch_gap, branch_negative = self._true_vs_hardest_negative_gap(
                    branch_scores,
                    label_indices,
                )
                evidence_gap, evidence_negative = self._true_vs_hardest_negative_gap(
                    evidence_logits,
                    label_indices,
                )
                support_weight, support_gap = self._margin_support_weights(
                    output,
                    labels,
                    label_indices,
                    branch_gap,
                    dominance_cfg.support_weighting,
                    loss_name="branch_path_dominance_constraint",
                )
                allowed_drops = self._class_values_tensor(
                    self.cfg.branch_path_dominance_allowed_drop_by_class,
                    label_indices,
                    device=evidence_logits.device,
                    dtype=evidence_logits.dtype,
                    fallback=float(dominance_cfg.allowed_drop),
                )
                label_multipliers = self._class_values_tensor(
                    self.cfg.branch_path_dominance_label_weight_by_class,
                    label_indices,
                    device=evidence_logits.device,
                    dtype=evidence_logits.dtype,
                    fallback=1.0,
                )
                active = int(epoch) > int(dominance_cfg.warmup_epochs)
                penalties = (
                    torch.relu(branch_gap - evidence_gap - allowed_drops)
                    * support_weight
                    * label_multipliers
                )
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["branch_path_dominance_branch_gap"] = float(
                        branch_gap[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_branch_negative_class"] = int(
                        branch_negative[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_evidence_gap"] = float(
                        evidence_gap[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_evidence_negative_class"] = int(
                        evidence_negative[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_support_gap"] = float(
                        support_gap[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_support_weight"] = float(
                        support_weight[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_label_multiplier"] = float(
                        label_multipliers[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_allowed_drop"] = float(
                        allowed_drops[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_allowed_drop_effective"] = float(
                        allowed_drops[index].detach().cpu().item()
                    )
                    row["branch_path_dominance_eligible"] = bool(active)
        if (
            self.cfg.branch_support_disagreement_cap_regularization.enabled
            and output.class_evidence_branch_support_scores is not None
            and output.class_evidence_embedding_scores is not None
            and output.class_evidence_interaction_scores is not None
        ):
            disagree_cfg = self.cfg.branch_support_disagreement_cap_regularization
            branch_scores = output.class_evidence_branch_support_scores.detach()
            embedding_scores = output.class_evidence_embedding_scores.detach()
            interaction_scores = output.class_evidence_interaction_scores.detach()
            if (
                tuple(branch_scores.shape)
                == tuple(embedding_scores.shape)
                == tuple(interaction_scores.shape)
                == (
                    int(labels.numel()),
                    int(branch_logits.shape[2]),
                )
            ):
                branch_gap, branch_negative = self._true_vs_hardest_negative_gap(
                    branch_scores,
                    label_indices,
                )
                embedding_gap, embedding_negative = self._true_vs_hardest_negative_gap(
                    embedding_scores,
                    label_indices,
                )
                interaction_gap, interaction_negative = (
                    self._true_vs_hardest_negative_gap(
                        interaction_scores,
                        label_indices,
                    )
                )
                active = int(epoch) > int(disagree_cfg.warmup_epochs)
                disagreement_weight = 1.0 + float(
                    disagree_cfg.condition_gain
                ) * torch.clamp(
                    torch.relu(float(disagree_cfg.disagreement_threshold) - branch_gap),
                    max=float(disagree_cfg.condition_cap),
                )
                embedding_penalty = disagreement_weight * torch.relu(
                    embedding_gap.abs() - float(disagree_cfg.embedding_gap_cap)
                )
                interaction_penalty = disagreement_weight * torch.relu(
                    interaction_gap.abs() - float(disagree_cfg.interaction_gap_cap)
                )
                penalties = embedding_penalty + interaction_penalty
                if not active:
                    embedding_penalty = torch.zeros_like(embedding_penalty)
                    interaction_penalty = torch.zeros_like(interaction_penalty)
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["branch_support_disagreement_branch_gap"] = float(
                        branch_gap[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_branch_negative_class"] = int(
                        branch_negative[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_embedding_gap"] = float(
                        embedding_gap[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_embedding_negative_class"] = int(
                        embedding_negative[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_interaction_gap"] = float(
                        interaction_gap[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_interaction_negative_class"] = int(
                        interaction_negative[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement"] = float(
                        disagreement_weight[index].detach().cpu().item()
                    )
                    row["embedding_disagreement_cap_penalty"] = float(
                        embedding_penalty[index].detach().cpu().item()
                    )
                    row["interaction_disagreement_cap_penalty"] = float(
                        interaction_penalty[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_cap_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["branch_support_disagreement_embedding_gap_cap"] = float(
                        disagree_cfg.embedding_gap_cap
                    )
                    row["branch_support_disagreement_interaction_gap_cap"] = float(
                        disagree_cfg.interaction_gap_cap
                    )
                    row["branch_support_disagreement_cap_eligible"] = bool(active)
        if (
            self.cfg.branch_direct_score_margin.enabled
            and output.class_evidence_branch_direct_scores is not None
        ):
            direct_cfg = self.cfg.branch_direct_score_margin
            direct_scores = output.class_evidence_branch_direct_scores.detach()
            if tuple(direct_scores.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                margin_gap, margin_negative = self._true_vs_hardest_negative_gap(
                    direct_scores,
                    label_indices,
                )
                active = int(epoch) > int(direct_cfg.warmup_epochs)
                temperature = float(direct_cfg.temperature)
                penalties = temperature * F.softplus(-margin_gap / temperature)
                if not active:
                    penalties = torch.zeros_like(penalties)
                for index, row in enumerate(rows):
                    row["branch_direct_score_margin_gap"] = float(
                        margin_gap[index].detach().cpu().item()
                    )
                    row["branch_direct_score_margin_negative_class"] = int(
                        margin_negative[index].detach().cpu().item()
                    )
                    row["branch_direct_score_margin_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
                    row["branch_direct_score_margin_mode"] = str(direct_cfg.mode)
                    row["branch_direct_score_margin_temperature"] = float(
                        direct_cfg.temperature
                    )
                    row["branch_direct_score_margin_eligible"] = bool(active)
        if (
            self.cfg.gate_branch_regret.enabled
            and output.class_evidence_gate_weights is not None
        ):
            gate_weights = output.class_evidence_gate_weights.detach()
            if tuple(gate_weights.shape) == (
                int(branch_logits.shape[0]),
                int(branch_logits.shape[2]),
                int(branch_logits.shape[1]),
            ):
                true_class_gate = self._true_class_gate_weights(
                    gate_weights,
                    label_indices,
                    dtype=branch_logits.dtype,
                )
                best_margin = branch_margin.max(dim=1).values
                gate_expected_margin = (true_class_gate * branch_margin).sum(dim=1)
                thresholds = self._gate_branch_regret_thresholds(
                    label_indices,
                    best_margin,
                )
                eligible = best_margin > thresholds
                weight_multiplier = self._gate_branch_regret_weight_multiplier(epoch)
                effective_weight = float(self.cfg.gate_branch_regret.weight) * float(
                    weight_multiplier
                )
                for index, row in enumerate(rows):
                    row["gate_branch_regret_positive_threshold"] = float(
                        thresholds[index].detach().cpu().item()
                    )
                    row["gate_branch_regret_best_margin"] = float(
                        best_margin[index].detach().cpu().item()
                    )
                    row["gate_branch_regret_gate_expected_margin"] = float(
                        gate_expected_margin[index].detach().cpu().item()
                    )
                    row["gate_branch_regret_eligible"] = bool(
                        eligible[index].detach().cpu().item()
                    )
                    row["gate_branch_regret_weight_multiplier"] = float(
                        weight_multiplier
                    )
                    row["gate_branch_regret_effective_weight"] = float(effective_weight)
        if (
            self.cfg.gate_bad_branch_suppression.enabled
            and output.class_evidence_gate_weights is not None
        ):
            gate_weights = output.class_evidence_gate_weights.detach()
            if tuple(gate_weights.shape) == (
                int(branch_logits.shape[0]),
                int(branch_logits.shape[2]),
                int(branch_logits.shape[1]),
            ):
                true_class_gate = self._true_class_gate_weights(
                    gate_weights,
                    label_indices,
                    dtype=branch_logits.dtype,
                )
                thresholds = self._gate_bad_branch_suppression_thresholds(
                    label_indices,
                    branch_margin[:, 0],
                )
                bad_mask = branch_margin < thresholds.unsqueeze(1)
                bad_penalty = torch.relu(thresholds.unsqueeze(1) - branch_margin)
                sample_penalty = (true_class_gate * bad_penalty).sum(dim=1)
                bad_gate_mass = (
                    true_class_gate * bad_mask.to(dtype=branch_logits.dtype)
                ).sum(dim=1)
                weight_multiplier = self._gate_bad_branch_suppression_weight_multiplier(
                    epoch
                )
                effective_weight = float(
                    self.cfg.gate_bad_branch_suppression.weight
                ) * float(weight_multiplier)
                for index, row in enumerate(rows):
                    row["gate_bad_branch_suppression_threshold"] = float(
                        thresholds[index].detach().cpu().item()
                    )
                    row["gate_bad_branch_suppression_bad_gate_mass"] = float(
                        bad_gate_mass[index].detach().cpu().item()
                    )
                    row["gate_bad_branch_suppression_penalty"] = float(
                        sample_penalty[index].detach().cpu().item()
                    )
                    row["gate_bad_branch_suppression_weight_multiplier"] = float(
                        weight_multiplier
                    )
                    row["gate_bad_branch_suppression_effective_weight"] = float(
                        effective_weight
                    )
        if (
            self.cfg.branch_to_evidence_ranking_consistency.enabled
            and output.class_evidence_logits is not None
        ):
            cfg = self.cfg.branch_to_evidence_ranking_consistency
            evidence_logits = output.class_evidence_logits.detach()
            if tuple(evidence_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                if (
                    cfg.mode == "teacher_distribution_kl"
                    and output.class_top_branch_margin_features is not None
                ):
                    teacher_logits = output.class_top_branch_margin_features.detach()
                    if tuple(teacher_logits.shape) == tuple(evidence_logits.shape):
                        teacher_probs = F.softmax(
                            teacher_logits / float(cfg.teacher_temperature),
                            dim=-1,
                        )
                        student_probs = F.softmax(
                            evidence_logits / float(cfg.student_temperature),
                            dim=-1,
                        )
                        student_log_probs = F.log_softmax(
                            evidence_logits / float(cfg.student_temperature),
                            dim=-1,
                        )
                        kl_values = F.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="none",
                        ).sum(dim=-1)
                        teacher_top_class = teacher_probs.argmax(dim=-1)
                        student_top_class = student_probs.argmax(dim=-1)
                        true_teacher_prob = teacher_probs.gather(
                            1,
                            label_indices.unsqueeze(1),
                        ).squeeze(1)
                        true_student_prob = student_probs.gather(
                            1,
                            label_indices.unsqueeze(1),
                        ).squeeze(1)
                        for index, row in enumerate(rows):
                            row["branch_to_evidence_mode"] = cfg.mode
                            row["branch_to_evidence_source"] = cfg.source
                            row["branch_to_evidence_teacher_temperature"] = float(
                                cfg.teacher_temperature
                            )
                            row["branch_to_evidence_student_temperature"] = float(
                                cfg.student_temperature
                            )
                            row["branch_to_evidence_teacher_probs"] = [
                                float(value)
                                for value in teacher_probs[index]
                                .detach()
                                .cpu()
                                .tolist()
                            ]
                            row["branch_to_evidence_student_probs"] = [
                                float(value)
                                for value in student_probs[index]
                                .detach()
                                .cpu()
                                .tolist()
                            ]
                            row["branch_to_evidence_teacher_top_class"] = int(
                                teacher_top_class[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_student_top_class"] = int(
                                student_top_class[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_true_teacher_prob"] = float(
                                true_teacher_prob[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_true_student_prob"] = float(
                                true_student_prob[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_kl"] = float(
                                kl_values[index].detach().cpu().item()
                            )
                if (
                    cfg.mode == "true_label_anchored_softplus"
                    and output.class_top_branch_margin_relative_features is not None
                ):
                    support_logits = (
                        output.class_top_branch_margin_relative_features.detach()
                    )
                    if tuple(support_logits.shape) == tuple(evidence_logits.shape):
                        evidence_gap, evidence_negative = (
                            self._true_vs_hardest_negative_gap(
                                evidence_logits,
                                label_indices,
                            )
                        )
                        true_support = support_logits.gather(
                            1,
                            label_indices.unsqueeze(1),
                        ).squeeze(1)
                        if cfg.support_weighting.enabled:
                            support = torch.clamp(
                                torch.relu(true_support),
                                min=0.0,
                                max=float(cfg.support_weighting.cap),
                            )
                            support_weight = 1.0 + (
                                float(cfg.support_weighting.gain) * support
                            )
                        else:
                            support_weight = torch.ones_like(evidence_gap)
                        temperature = float(cfg.temperature)
                        penalties = temperature * F.softplus(
                            -evidence_gap / temperature
                        )
                        weighted_penalties = penalties * support_weight
                        for index, row in enumerate(rows):
                            row["branch_to_evidence_mode"] = cfg.mode
                            row["branch_to_evidence_source"] = cfg.source
                            row["branch_to_evidence_temperature"] = float(
                                cfg.temperature
                            )
                            row["branch_to_evidence_evidence_gap"] = float(
                                evidence_gap[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_evidence_negative_class"] = int(
                                evidence_negative[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_support_relative"] = float(
                                true_support[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_support_weight"] = float(
                                support_weight[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_penalty"] = float(
                                penalties[index].detach().cpu().item()
                            )
                            row["branch_to_evidence_weighted_penalty"] = float(
                                weighted_penalties[index].detach().cpu().item()
                            )
                teacher_floor = self._branch_to_evidence_teacher_floors(
                    label_indices,
                    top_margin,
                ).detach()
                b2e_teacher_gap: Tensor | None = None
                b2e_teacher_negative: Tensor | None = None
                teacher_branch_indices: Tensor | None = None
                if (
                    cfg.source == "class_gated_branch_logits"
                    and output.class_gated_branch_logits is not None
                ):
                    source_logits = output.class_gated_branch_logits.detach()
                    if tuple(source_logits.shape) == tuple(evidence_logits.shape):
                        b2e_teacher_gap, b2e_teacher_negative = (
                            self._true_vs_hardest_negative_gap(
                                source_logits,
                                label_indices,
                            )
                        )
                elif cfg.source == "top_branch_margin":
                    b2e_teacher_gap = top_margin
                    b2e_teacher_negative = top_negative_indices
                    teacher_branch_indices = top_branch_indices
                if b2e_teacher_gap is not None and b2e_teacher_negative is not None:
                    b2e_teacher_gap = torch.maximum(b2e_teacher_gap, teacher_floor)
                    if cfg.teacher_gap_cap is not None:
                        b2e_teacher_gap = torch.clamp(
                            b2e_teacher_gap,
                            max=float(cfg.teacher_gap_cap),
                        )
                    evidence_gap, evidence_negative = (
                        self._true_vs_hardest_negative_gap(
                            evidence_logits,
                            label_indices,
                        )
                    )
                    support_weight, support_gap = self._margin_support_weights(
                        output,
                        labels,
                        label_indices,
                        evidence_gap,
                        cfg.support_weighting,
                        loss_name="branch_to_evidence_ranking_consistency",
                    )
                    hard_weight, hardness = self._margin_hardness_weights(
                        evidence_gap,
                        cfg.hardness_weighting,
                        loss_name="branch_to_evidence_ranking_consistency",
                    )
                    base_penalties = torch.relu(
                        b2e_teacher_gap - evidence_gap + float(cfg.tolerance)
                    )
                    penalties = base_penalties * support_weight * hard_weight
                    for index, row in enumerate(rows):
                        row["branch_to_evidence_mode"] = cfg.mode
                        row["branch_to_evidence_source"] = cfg.source
                        row["branch_to_evidence_branch_gap"] = float(
                            b2e_teacher_gap[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_teacher_gap"] = float(
                            b2e_teacher_gap[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_teacher_floor"] = float(
                            teacher_floor[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_teacher_gap_cap"] = (
                            float(cfg.teacher_gap_cap)
                            if cfg.teacher_gap_cap is not None
                            else None
                        )
                        row["branch_to_evidence_evidence_gap"] = float(
                            evidence_gap[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_support_gap"] = float(
                            support_gap[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_support_weight"] = float(
                            support_weight[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_hardness"] = float(
                            hardness[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_hardness_weight"] = float(
                            hard_weight[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_branch_negative_class"] = int(
                            b2e_teacher_negative[index].detach().cpu().item()
                        )
                        if teacher_branch_indices is not None:
                            row["branch_to_evidence_teacher_branch_index"] = int(
                                teacher_branch_indices[index].detach().cpu().item()
                            )
                        row["branch_to_evidence_evidence_negative_class"] = int(
                            evidence_negative[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_penalty"] = float(
                            base_penalties[index].detach().cpu().item()
                        )
                        row["branch_to_evidence_weighted_penalty"] = float(
                            penalties[index].detach().cpu().item()
                        )
        if (
            self.cfg.global_residual_anti_veto.enabled
            and output.class_evidence_logits is not None
        ):
            anti_veto_cfg = self.cfg.global_residual_anti_veto
            evidence_logits = output.class_evidence_logits.detach()
            if tuple(evidence_logits.shape) == (
                int(labels.numel()),
                int(branch_logits.shape[2]),
            ):
                evidence_gap, negative_indices = self._true_vs_hardest_negative_gap(
                    evidence_logits,
                    label_indices,
                )
                if (
                    anti_veto_cfg.mode == "true_vs_hardest_negative"
                    and output.global_residual_logits is not None
                ):
                    residual_logits = output.global_residual_logits.detach()
                    if tuple(evidence_logits.shape) == tuple(residual_logits.shape):
                        residual_true = residual_logits.gather(
                            1,
                            label_indices.unsqueeze(1),
                        ).squeeze(1)
                        residual_negative = residual_logits.gather(
                            1,
                            negative_indices.unsqueeze(1),
                        ).squeeze(1)
                        residual_gap = residual_true - residual_negative
                        eligible = evidence_gap > float(
                            anti_veto_cfg.evidence_confidence_threshold
                        )
                        penalties = torch.relu(
                            float(anti_veto_cfg.min_residual_gap) - residual_gap
                        )
                        for index, row in enumerate(rows):
                            row["global_residual_anti_veto_mode"] = anti_veto_cfg.mode
                            row["global_residual_anti_veto_target"] = (
                                anti_veto_cfg.target
                            )
                            row["global_residual_anti_veto_evidence_gap"] = float(
                                evidence_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_reference_gap"] = float(
                                evidence_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_residual_gap"] = float(
                                residual_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_negative_class"] = int(
                                negative_indices[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_eligible"] = bool(
                                eligible[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_penalty"] = float(
                                penalties[index].detach().cpu().item()
                            )
                elif anti_veto_cfg.mode == "final_gap_preservation":
                    final_logits = output.logits.detach()
                    if tuple(evidence_logits.shape) == tuple(final_logits.shape):
                        final_true = final_logits.gather(
                            1,
                            label_indices.unsqueeze(1),
                        ).squeeze(1)
                        final_negative = final_logits.gather(
                            1,
                            negative_indices.unsqueeze(1),
                        ).squeeze(1)
                        final_gap = final_true - final_negative
                        support_gap = top_margin
                        eligible = support_gap > float(anti_veto_cfg.support_threshold)
                        penalties = torch.relu(
                            evidence_gap
                            - float(anti_veto_cfg.allowed_gap_drop)
                            - final_gap
                        )
                        for index, row in enumerate(rows):
                            row["global_residual_anti_veto_mode"] = anti_veto_cfg.mode
                            row["global_residual_anti_veto_target"] = (
                                anti_veto_cfg.target
                            )
                            row["global_residual_anti_veto_support_source"] = (
                                anti_veto_cfg.support_source
                            )
                            row["global_residual_anti_veto_evidence_gap"] = float(
                                evidence_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_reference_gap"] = float(
                                evidence_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_final_gap"] = float(
                                final_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_support_gap"] = float(
                                support_gap[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_support_threshold"] = float(
                                anti_veto_cfg.support_threshold
                            )
                            row["global_residual_anti_veto_allowed_gap_drop"] = float(
                                anti_veto_cfg.allowed_gap_drop
                            )
                            row["global_residual_anti_veto_negative_class"] = int(
                                negative_indices[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_eligible"] = bool(
                                eligible[index].detach().cpu().item()
                            )
                            row["global_residual_anti_veto_penalty"] = float(
                                penalties[index].detach().cpu().item()
                            )
        if (
            self.cfg.residual_contradiction_regularization.enabled
            and output.class_evidence_logits is not None
            and output.global_residual_logits is not None
        ):
            residual_cfg = self.cfg.residual_contradiction_regularization
            evidence_logits = output.class_evidence_logits.detach()
            residual_logits = output.global_residual_logits.detach()
            if (
                tuple(evidence_logits.shape)
                == tuple(residual_logits.shape)
                == (
                    int(labels.numel()),
                    int(branch_logits.shape[2]),
                )
            ):
                evidence_gap, negative_indices = self._true_vs_hardest_negative_gap(
                    evidence_logits,
                    label_indices,
                )
                residual_true = residual_logits.gather(
                    1,
                    label_indices.unsqueeze(1),
                ).squeeze(1)
                residual_negative = residual_logits.gather(
                    1,
                    negative_indices.unsqueeze(1),
                ).squeeze(1)
                residual_gap = residual_true - residual_negative
                eligible = evidence_gap > float(residual_cfg.evidence_gap_threshold)
                penalties = torch.relu(
                    float(residual_cfg.min_residual_gap) - residual_gap
                )
                for index, row in enumerate(rows):
                    row["residual_contradiction_evidence_gap"] = float(
                        evidence_gap[index].detach().cpu().item()
                    )
                    row["residual_contradiction_residual_gap"] = float(
                        residual_gap[index].detach().cpu().item()
                    )
                    row["residual_contradiction_negative_class"] = int(
                        negative_indices[index].detach().cpu().item()
                    )
                    row["residual_contradiction_evidence_gap_threshold"] = float(
                        residual_cfg.evidence_gap_threshold
                    )
                    row["residual_contradiction_min_residual_gap"] = float(
                        residual_cfg.min_residual_gap
                    )
                    row["residual_contradiction_eligible"] = bool(
                        eligible[index].detach().cpu().item()
                    )
                    row["residual_contradiction_penalty"] = float(
                        penalties[index].detach().cpu().item()
                    )
        return rows

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
            "class_evidence_margin": 0.0,
            "class_evidence_margin_loss": 0.0,
            "class_evidence_gap_cap_regularization": 0.0,
            "class_evidence_gap_cap_regularization_loss": 0.0,
            "class_evidence_gap_cap_label_multiplier": 0.0,
            "class_evidence_gap_cap_effective_cap": 0.0,
            "class_evidence_positive_gap_cap_regularization": 0.0,
            "class_evidence_positive_gap_cap_regularization_loss": 0.0,
            "interaction_gap_cap_regularization": 0.0,
            "interaction_gap_cap_regularization_loss": 0.0,
            "top_support_score_margin": 0.0,
            "top_support_score_margin_loss": 0.0,
            "top_support_score_margin_label_multiplier": 0.0,
            "top_support_score_margin_support_multiplier": 0.0,
            "top_support_score_margin_hardness_multiplier": 0.0,
            "top_support_gap_min_constraint": 0.0,
            "top_support_gap_min_constraint_loss": 0.0,
            "top_support_gap_min_target": 0.0,
            "class_top_branch_relative_margin": 0.0,
            "class_top_branch_relative_margin_loss": 0.0,
            "class_top_branch_relative_margin_support_multiplier": 0.0,
            "class_top_branch_relative_margin_hardness_multiplier": 0.0,
            "top_teacher_gap_min_constraint": 0.0,
            "top_teacher_gap_min_constraint_loss": 0.0,
            "top_teacher_gap_min_target": 0.0,
            "branch_direct_score_margin": 0.0,
            "branch_direct_score_margin_loss": 0.0,
            "branch_path_dominance_constraint": 0.0,
            "branch_path_dominance_constraint_loss": 0.0,
            "branch_path_dominance_support_multiplier": 0.0,
            "branch_path_dominance_label_multiplier": 0.0,
            "branch_path_dominance_allowed_drop": 0.0,
            "branch_support_disagreement_cap_regularization": 0.0,
            "branch_support_disagreement_cap_regularization_loss": 0.0,
            "branch_support_disagreement_weight": 0.0,
            "branch_support_score_margin": 0.0,
            "branch_support_score_margin_loss": 0.0,
            "class_gated_branch_logit_margin": 0.0,
            "class_gated_branch_logit_margin_loss": 0.0,
            "branch_to_evidence_ranking_consistency": 0.0,
            "branch_to_evidence_ranking_consistency_loss": 0.0,
            "global_residual_anti_veto": 0.0,
            "global_residual_anti_veto_loss": 0.0,
            "global_residual_anti_veto_eligible_fraction": 0.0,
            "residual_contradiction_regularization": 0.0,
            "residual_contradiction_regularization_loss": 0.0,
            "residual_contradiction_regularization_eligible_fraction": 0.0,
            "gate_weighted_branch_margin": 0.0,
            "gate_weighted_branch_margin_loss": 0.0,
            "gate_branch_regret": 0.0,
            "gate_branch_regret_loss": 0.0,
            "gate_branch_regret_eligible_fraction": 0.0,
            "gate_branch_regret_weight_multiplier": 0.0,
            "gate_branch_regret_effective_weight": 0.0,
            "gate_bad_branch_suppression": 0.0,
            "gate_bad_branch_suppression_loss": 0.0,
            "gate_bad_branch_suppression_bad_gate_mass": 0.0,
            "gate_bad_branch_suppression_weight_multiplier": 0.0,
            "gate_bad_branch_suppression_effective_weight": 0.0,
            "top_branch_margin": 0.0,
            "top_branch_margin_loss": 0.0,
            "top_branch_margin_effective_weight": 0.0,
            "top_branch_margin_hardness_multiplier": 0.0,
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
        branch_objective_stats = (
            self._new_branch_objective_epoch_stats()
            if optimizer is not None
            and (
                self.cfg.top_branch_margin.auto_margin_by_train_stats.enabled
                or self.cfg.gate_branch_regret.auto_positive_threshold_by_train_stats.enabled
                or self.cfg.gate_bad_branch_suppression.auto_bad_margin_threshold_by_train_stats.enabled
            )
            else None
        )
        set_runtime_epoch = getattr(model, "set_runtime_epoch", None)
        if callable(set_runtime_epoch):
            set_runtime_epoch(epoch)

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
            if branch_objective_stats is not None:
                with torch.no_grad():
                    self._observe_branch_objective_stats(
                        output,
                        batch.labels,
                        branch_objective_stats,
                    )

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
                "class_evidence_margin": loss_components.class_evidence_margin,
                "class_evidence_margin_loss": (
                    loss_components.class_evidence_margin_loss
                ),
                "class_evidence_gap_cap_regularization": (
                    loss_components.class_evidence_gap_cap_regularization
                ),
                "class_evidence_gap_cap_regularization_loss": (
                    loss_components.class_evidence_gap_cap_regularization_loss
                ),
                "class_evidence_gap_cap_label_multiplier": (
                    loss_components.class_evidence_gap_cap_label_multiplier
                ),
                "class_evidence_gap_cap_effective_cap": (
                    loss_components.class_evidence_gap_cap_effective_cap
                ),
                "class_evidence_positive_gap_cap_regularization": (
                    loss_components.class_evidence_positive_gap_cap_regularization
                ),
                "class_evidence_positive_gap_cap_regularization_loss": (
                    loss_components.class_evidence_positive_gap_cap_regularization_loss
                ),
                "interaction_gap_cap_regularization": (
                    loss_components.interaction_gap_cap_regularization
                ),
                "interaction_gap_cap_regularization_loss": (
                    loss_components.interaction_gap_cap_regularization_loss
                ),
                "top_support_score_margin": loss_components.top_support_score_margin,
                "top_support_score_margin_loss": (
                    loss_components.top_support_score_margin_loss
                ),
                "top_support_score_margin_label_multiplier": (
                    loss_components.top_support_score_margin_label_multiplier
                ),
                "top_support_score_margin_support_multiplier": (
                    loss_components.top_support_score_margin_support_multiplier
                ),
                "top_support_score_margin_hardness_multiplier": (
                    loss_components.top_support_score_margin_hardness_multiplier
                ),
                "top_support_gap_min_constraint": (
                    loss_components.top_support_gap_min_constraint
                ),
                "top_support_gap_min_constraint_loss": (
                    loss_components.top_support_gap_min_constraint_loss
                ),
                "top_support_gap_min_target": (
                    loss_components.top_support_gap_min_target
                ),
                "class_top_branch_relative_margin": (
                    loss_components.class_top_branch_relative_margin
                ),
                "class_top_branch_relative_margin_loss": (
                    loss_components.class_top_branch_relative_margin_loss
                ),
                "class_top_branch_relative_margin_support_multiplier": (
                    loss_components.class_top_branch_relative_margin_support_multiplier
                ),
                "class_top_branch_relative_margin_hardness_multiplier": (
                    loss_components.class_top_branch_relative_margin_hardness_multiplier
                ),
                "top_teacher_gap_min_constraint": (
                    loss_components.top_teacher_gap_min_constraint
                ),
                "top_teacher_gap_min_constraint_loss": (
                    loss_components.top_teacher_gap_min_constraint_loss
                ),
                "top_teacher_gap_min_target": (
                    loss_components.top_teacher_gap_min_target
                ),
                "branch_direct_score_margin": (
                    loss_components.branch_direct_score_margin
                ),
                "branch_direct_score_margin_loss": (
                    loss_components.branch_direct_score_margin_loss
                ),
                "branch_path_dominance_constraint": (
                    loss_components.branch_path_dominance_constraint
                ),
                "branch_path_dominance_constraint_loss": (
                    loss_components.branch_path_dominance_constraint_loss
                ),
                "branch_path_dominance_support_multiplier": (
                    loss_components.branch_path_dominance_support_multiplier
                ),
                "branch_path_dominance_label_multiplier": (
                    loss_components.branch_path_dominance_label_multiplier
                ),
                "branch_path_dominance_allowed_drop": (
                    loss_components.branch_path_dominance_allowed_drop
                ),
                "branch_support_disagreement_cap_regularization": (
                    loss_components.branch_support_disagreement_cap_regularization
                ),
                "branch_support_disagreement_cap_regularization_loss": (
                    loss_components.branch_support_disagreement_cap_regularization_loss
                ),
                "branch_support_disagreement_weight": (
                    loss_components.branch_support_disagreement_weight
                ),
                "branch_support_score_margin": (
                    loss_components.branch_support_score_margin
                ),
                "branch_support_score_margin_loss": (
                    loss_components.branch_support_score_margin_loss
                ),
                "class_gated_branch_logit_margin": (
                    loss_components.class_gated_branch_logit_margin
                ),
                "class_gated_branch_logit_margin_loss": (
                    loss_components.class_gated_branch_logit_margin_loss
                ),
                "branch_to_evidence_ranking_consistency": (
                    loss_components.branch_to_evidence_ranking_consistency
                ),
                "branch_to_evidence_ranking_consistency_loss": (
                    loss_components.branch_to_evidence_ranking_consistency_loss
                ),
                "global_residual_anti_veto": (
                    loss_components.global_residual_anti_veto
                ),
                "global_residual_anti_veto_loss": (
                    loss_components.global_residual_anti_veto_loss
                ),
                "global_residual_anti_veto_eligible_fraction": (
                    loss_components.global_residual_anti_veto_eligible_fraction
                ),
                "residual_contradiction_regularization": (
                    loss_components.residual_contradiction_regularization
                ),
                "residual_contradiction_regularization_loss": (
                    loss_components.residual_contradiction_regularization_loss
                ),
                "residual_contradiction_regularization_eligible_fraction": (
                    loss_components.residual_contradiction_regularization_eligible_fraction
                ),
                "gate_weighted_branch_margin": (
                    loss_components.gate_weighted_branch_margin
                ),
                "gate_weighted_branch_margin_loss": (
                    loss_components.gate_weighted_branch_margin_loss
                ),
                "gate_branch_regret": loss_components.gate_branch_regret,
                "gate_branch_regret_loss": loss_components.gate_branch_regret_loss,
                "gate_branch_regret_eligible_fraction": (
                    loss_components.gate_branch_regret_eligible_fraction
                ),
                "gate_branch_regret_weight_multiplier": (
                    loss_components.gate_branch_regret_weight_multiplier
                ),
                "gate_branch_regret_effective_weight": (
                    loss_components.gate_branch_regret_effective_weight
                ),
                "gate_bad_branch_suppression": (
                    loss_components.gate_bad_branch_suppression
                ),
                "gate_bad_branch_suppression_loss": (
                    loss_components.gate_bad_branch_suppression_loss
                ),
                "gate_bad_branch_suppression_bad_gate_mass": (
                    loss_components.gate_bad_branch_suppression_bad_gate_mass
                ),
                "gate_bad_branch_suppression_weight_multiplier": (
                    loss_components.gate_bad_branch_suppression_weight_multiplier
                ),
                "gate_bad_branch_suppression_effective_weight": (
                    loss_components.gate_bad_branch_suppression_effective_weight
                ),
                "top_branch_margin": loss_components.top_branch_margin,
                "top_branch_margin_loss": loss_components.top_branch_margin_loss,
                "top_branch_margin_effective_weight": (
                    loss_components.top_branch_margin_effective_weight
                ),
                "top_branch_margin_hardness_multiplier": (
                    loss_components.top_branch_margin_hardness_multiplier
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
                diagnostic_rows = build_diagnostic_rows(
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
                objective_rows = self._branch_objective_diagnostic_rows(
                    output,
                    batch.labels,
                    epoch=epoch,
                )
                for row, objective_row in zip(
                    diagnostic_rows,
                    objective_rows,
                    strict=True,
                ):
                    row.update(objective_row)
                diagnostics.extend(diagnostic_rows)

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
        averaged_components: dict[str, Any] = {
            name: value / float(max(1, component_counts[name]))
            for name, value in component_totals.items()
            if component_counts[name] > 0
        }
        if branch_objective_stats is not None:
            self._update_adaptive_branch_objective_state(
                epoch=epoch,
                stats=branch_objective_stats,
            )
            averaged_components.update(
                self._branch_objective_stats_summary(branch_objective_stats)
            )
        adaptive_state = self._branch_objective_state_dict()
        averaged_components.update(
            {
                "adaptive_top_branch_margin_by_label": adaptive_state[
                    "top_branch_margin_by_label"
                ],
                "adaptive_gate_branch_regret_positive_threshold_by_label": (
                    adaptive_state["gate_branch_regret_positive_threshold_by_label"]
                ),
                "adaptive_gate_bad_branch_suppression_threshold_by_label": (
                    adaptive_state["gate_bad_branch_suppression_threshold_by_label"]
                ),
                "adaptive_top_branch_violation_rate_ema_by_label": adaptive_state[
                    "top_branch_violation_rate_ema_by_label"
                ],
                "adaptive_gate_branch_regret_eligible_rate_ema_by_label": (
                    adaptive_state["gate_branch_regret_eligible_rate_ema_by_label"]
                ),
                "adaptive_gate_bad_branch_suppression_bad_gate_mass_ema_by_label": (
                    adaptive_state[
                        "gate_bad_branch_suppression_bad_gate_mass_ema_by_label"
                    ]
                ),
            }
        )
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
        train_loss_component_history: list[dict[str, Any]] = []
        val_loss_component_history: list[dict[str, Any]] = []
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

            if val_result.probabilities.ndim == 1:
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
                    reason=(
                        "threshold optimization is only supported for one-logit "
                        "binary outputs"
                    ),
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
                    "train_class_evidence_margin": train_components.get(
                        "class_evidence_margin",
                        0.0,
                    ),
                    "train_loss_class_evidence_margin": train_components.get(
                        "class_evidence_margin_loss",
                        0.0,
                    ),
                    "train_class_evidence_gap_cap_regularization": (
                        train_components.get(
                            "class_evidence_gap_cap_regularization",
                            0.0,
                        )
                    ),
                    "train_loss_class_evidence_gap_cap_regularization": (
                        train_components.get(
                            "class_evidence_gap_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "train_class_evidence_gap_cap_label_multiplier": (
                        train_components.get(
                            "class_evidence_gap_cap_label_multiplier",
                            0.0,
                        )
                    ),
                    "train_class_evidence_gap_cap_effective_cap": (
                        train_components.get(
                            "class_evidence_gap_cap_effective_cap",
                            0.0,
                        )
                    ),
                    "train_class_evidence_positive_gap_cap_regularization": (
                        train_components.get(
                            "class_evidence_positive_gap_cap_regularization",
                            0.0,
                        )
                    ),
                    "train_loss_class_evidence_positive_gap_cap_regularization": (
                        train_components.get(
                            "class_evidence_positive_gap_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "train_interaction_gap_cap_regularization": (
                        train_components.get(
                            "interaction_gap_cap_regularization",
                            0.0,
                        )
                    ),
                    "train_loss_interaction_gap_cap_regularization": (
                        train_components.get(
                            "interaction_gap_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "train_top_support_score_margin": train_components.get(
                        "top_support_score_margin",
                        0.0,
                    ),
                    "train_loss_top_support_score_margin": train_components.get(
                        "top_support_score_margin_loss",
                        0.0,
                    ),
                    "train_top_support_score_margin_label_multiplier": (
                        train_components.get(
                            "top_support_score_margin_label_multiplier",
                            0.0,
                        )
                    ),
                    "train_top_support_score_margin_support_multiplier": (
                        train_components.get(
                            "top_support_score_margin_support_multiplier",
                            0.0,
                        )
                    ),
                    "train_top_support_score_margin_hardness_multiplier": (
                        train_components.get(
                            "top_support_score_margin_hardness_multiplier",
                            0.0,
                        )
                    ),
                    "train_top_support_gap_min_constraint": train_components.get(
                        "top_support_gap_min_constraint",
                        0.0,
                    ),
                    "train_loss_top_support_gap_min_constraint": train_components.get(
                        "top_support_gap_min_constraint_loss",
                        0.0,
                    ),
                    "train_top_support_gap_min_target": train_components.get(
                        "top_support_gap_min_target",
                        0.0,
                    ),
                    "train_class_top_branch_relative_margin": train_components.get(
                        "class_top_branch_relative_margin",
                        0.0,
                    ),
                    "train_loss_class_top_branch_relative_margin": (
                        train_components.get(
                            "class_top_branch_relative_margin_loss",
                            0.0,
                        )
                    ),
                    "train_class_top_branch_relative_margin_support_multiplier": (
                        train_components.get(
                            "class_top_branch_relative_margin_support_multiplier",
                            0.0,
                        )
                    ),
                    "train_class_top_branch_relative_margin_hardness_multiplier": (
                        train_components.get(
                            "class_top_branch_relative_margin_hardness_multiplier",
                            0.0,
                        )
                    ),
                    "train_top_teacher_gap_min_constraint": train_components.get(
                        "top_teacher_gap_min_constraint",
                        0.0,
                    ),
                    "train_loss_top_teacher_gap_min_constraint": (
                        train_components.get(
                            "top_teacher_gap_min_constraint_loss",
                            0.0,
                        )
                    ),
                    "train_top_teacher_gap_min_target": train_components.get(
                        "top_teacher_gap_min_target",
                        0.0,
                    ),
                    "train_branch_direct_score_margin": train_components.get(
                        "branch_direct_score_margin",
                        0.0,
                    ),
                    "train_loss_branch_direct_score_margin": train_components.get(
                        "branch_direct_score_margin_loss",
                        0.0,
                    ),
                    "train_branch_path_dominance_constraint": train_components.get(
                        "branch_path_dominance_constraint",
                        0.0,
                    ),
                    "train_loss_branch_path_dominance_constraint": (
                        train_components.get(
                            "branch_path_dominance_constraint_loss",
                            0.0,
                        )
                    ),
                    "train_branch_path_dominance_support_multiplier": (
                        train_components.get(
                            "branch_path_dominance_support_multiplier",
                            0.0,
                        )
                    ),
                    "train_branch_path_dominance_label_multiplier": (
                        train_components.get(
                            "branch_path_dominance_label_multiplier",
                            0.0,
                        )
                    ),
                    "train_branch_path_dominance_allowed_drop": (
                        train_components.get(
                            "branch_path_dominance_allowed_drop",
                            0.0,
                        )
                    ),
                    "train_branch_support_disagreement_cap_regularization": (
                        train_components.get(
                            "branch_support_disagreement_cap_regularization",
                            0.0,
                        )
                    ),
                    "train_loss_branch_support_disagreement_cap_regularization": (
                        train_components.get(
                            "branch_support_disagreement_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "train_branch_support_disagreement_weight": train_components.get(
                        "branch_support_disagreement_weight",
                        0.0,
                    ),
                    "train_branch_support_score_margin": train_components.get(
                        "branch_support_score_margin",
                        0.0,
                    ),
                    "train_loss_branch_support_score_margin": train_components.get(
                        "branch_support_score_margin_loss",
                        0.0,
                    ),
                    "train_class_gated_branch_logit_margin": train_components.get(
                        "class_gated_branch_logit_margin",
                        0.0,
                    ),
                    "train_loss_class_gated_branch_logit_margin": (
                        train_components.get(
                            "class_gated_branch_logit_margin_loss",
                            0.0,
                        )
                    ),
                    "train_branch_to_evidence_ranking_consistency": (
                        train_components.get(
                            "branch_to_evidence_ranking_consistency",
                            0.0,
                        )
                    ),
                    "train_loss_branch_to_evidence_ranking_consistency": (
                        train_components.get(
                            "branch_to_evidence_ranking_consistency_loss",
                            0.0,
                        )
                    ),
                    "train_global_residual_anti_veto": train_components.get(
                        "global_residual_anti_veto",
                        0.0,
                    ),
                    "train_loss_global_residual_anti_veto": train_components.get(
                        "global_residual_anti_veto_loss",
                        0.0,
                    ),
                    "train_global_residual_anti_veto_eligible_fraction": (
                        train_components.get(
                            "global_residual_anti_veto_eligible_fraction",
                            0.0,
                        )
                    ),
                    "train_residual_contradiction_regularization": (
                        train_components.get(
                            "residual_contradiction_regularization",
                            0.0,
                        )
                    ),
                    "train_loss_residual_contradiction_regularization": (
                        train_components.get(
                            "residual_contradiction_regularization_loss",
                            0.0,
                        )
                    ),
                    "train_residual_contradiction_regularization_eligible_fraction": (
                        train_components.get(
                            "residual_contradiction_regularization_eligible_fraction",
                            0.0,
                        )
                    ),
                    "train_gate_weighted_branch_margin": train_components.get(
                        "gate_weighted_branch_margin",
                        0.0,
                    ),
                    "train_loss_gate_weighted_branch_margin": train_components.get(
                        "gate_weighted_branch_margin_loss",
                        0.0,
                    ),
                    "train_gate_branch_regret": train_components.get(
                        "gate_branch_regret",
                        0.0,
                    ),
                    "train_loss_gate_branch_regret": train_components.get(
                        "gate_branch_regret_loss",
                        0.0,
                    ),
                    "train_gate_branch_regret_eligible_fraction": (
                        train_components.get(
                            "gate_branch_regret_eligible_fraction",
                            0.0,
                        )
                    ),
                    "train_gate_branch_regret_weight_multiplier": (
                        train_components.get(
                            "gate_branch_regret_weight_multiplier",
                            0.0,
                        )
                    ),
                    "train_gate_branch_regret_effective_weight": (
                        train_components.get(
                            "gate_branch_regret_effective_weight",
                            0.0,
                        )
                    ),
                    "train_gate_bad_branch_suppression": train_components.get(
                        "gate_bad_branch_suppression",
                        0.0,
                    ),
                    "train_loss_gate_bad_branch_suppression": train_components.get(
                        "gate_bad_branch_suppression_loss",
                        0.0,
                    ),
                    "train_gate_bad_branch_suppression_bad_gate_mass": (
                        train_components.get(
                            "gate_bad_branch_suppression_bad_gate_mass",
                            0.0,
                        )
                    ),
                    "train_gate_bad_branch_suppression_weight_multiplier": (
                        train_components.get(
                            "gate_bad_branch_suppression_weight_multiplier",
                            0.0,
                        )
                    ),
                    "train_gate_bad_branch_suppression_effective_weight": (
                        train_components.get(
                            "gate_bad_branch_suppression_effective_weight",
                            0.0,
                        )
                    ),
                    "train_top_branch_margin": train_components.get(
                        "top_branch_margin",
                        0.0,
                    ),
                    "train_loss_top_branch_margin": train_components.get(
                        "top_branch_margin_loss",
                        0.0,
                    ),
                    "train_top_branch_margin_effective_weight": train_components.get(
                        "top_branch_margin_effective_weight",
                        0.0,
                    ),
                    "train_top_branch_margin_hardness_multiplier": (
                        train_components.get(
                            "top_branch_margin_hardness_multiplier",
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
                    "val_class_evidence_margin": val_components.get(
                        "class_evidence_margin",
                        0.0,
                    ),
                    "val_loss_class_evidence_margin": val_components.get(
                        "class_evidence_margin_loss",
                        0.0,
                    ),
                    "val_class_evidence_gap_cap_regularization": val_components.get(
                        "class_evidence_gap_cap_regularization",
                        0.0,
                    ),
                    "val_loss_class_evidence_gap_cap_regularization": (
                        val_components.get(
                            "class_evidence_gap_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "val_class_evidence_gap_cap_label_multiplier": (
                        val_components.get(
                            "class_evidence_gap_cap_label_multiplier",
                            0.0,
                        )
                    ),
                    "val_class_evidence_gap_cap_effective_cap": (
                        val_components.get(
                            "class_evidence_gap_cap_effective_cap",
                            0.0,
                        )
                    ),
                    "val_class_evidence_positive_gap_cap_regularization": (
                        val_components.get(
                            "class_evidence_positive_gap_cap_regularization",
                            0.0,
                        )
                    ),
                    "val_loss_class_evidence_positive_gap_cap_regularization": (
                        val_components.get(
                            "class_evidence_positive_gap_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "val_interaction_gap_cap_regularization": (
                        val_components.get(
                            "interaction_gap_cap_regularization",
                            0.0,
                        )
                    ),
                    "val_loss_interaction_gap_cap_regularization": (
                        val_components.get(
                            "interaction_gap_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "val_top_support_score_margin": val_components.get(
                        "top_support_score_margin",
                        0.0,
                    ),
                    "val_loss_top_support_score_margin": val_components.get(
                        "top_support_score_margin_loss",
                        0.0,
                    ),
                    "val_top_support_score_margin_label_multiplier": (
                        val_components.get(
                            "top_support_score_margin_label_multiplier",
                            0.0,
                        )
                    ),
                    "val_top_support_score_margin_support_multiplier": (
                        val_components.get(
                            "top_support_score_margin_support_multiplier",
                            0.0,
                        )
                    ),
                    "val_top_support_score_margin_hardness_multiplier": (
                        val_components.get(
                            "top_support_score_margin_hardness_multiplier",
                            0.0,
                        )
                    ),
                    "val_top_support_gap_min_constraint": val_components.get(
                        "top_support_gap_min_constraint",
                        0.0,
                    ),
                    "val_loss_top_support_gap_min_constraint": val_components.get(
                        "top_support_gap_min_constraint_loss",
                        0.0,
                    ),
                    "val_top_support_gap_min_target": val_components.get(
                        "top_support_gap_min_target",
                        0.0,
                    ),
                    "val_class_top_branch_relative_margin": val_components.get(
                        "class_top_branch_relative_margin",
                        0.0,
                    ),
                    "val_loss_class_top_branch_relative_margin": val_components.get(
                        "class_top_branch_relative_margin_loss",
                        0.0,
                    ),
                    "val_class_top_branch_relative_margin_support_multiplier": (
                        val_components.get(
                            "class_top_branch_relative_margin_support_multiplier",
                            0.0,
                        )
                    ),
                    "val_class_top_branch_relative_margin_hardness_multiplier": (
                        val_components.get(
                            "class_top_branch_relative_margin_hardness_multiplier",
                            0.0,
                        )
                    ),
                    "val_top_teacher_gap_min_constraint": val_components.get(
                        "top_teacher_gap_min_constraint",
                        0.0,
                    ),
                    "val_loss_top_teacher_gap_min_constraint": val_components.get(
                        "top_teacher_gap_min_constraint_loss",
                        0.0,
                    ),
                    "val_top_teacher_gap_min_target": val_components.get(
                        "top_teacher_gap_min_target",
                        0.0,
                    ),
                    "val_branch_direct_score_margin": val_components.get(
                        "branch_direct_score_margin",
                        0.0,
                    ),
                    "val_loss_branch_direct_score_margin": val_components.get(
                        "branch_direct_score_margin_loss",
                        0.0,
                    ),
                    "val_branch_path_dominance_constraint": val_components.get(
                        "branch_path_dominance_constraint",
                        0.0,
                    ),
                    "val_loss_branch_path_dominance_constraint": val_components.get(
                        "branch_path_dominance_constraint_loss",
                        0.0,
                    ),
                    "val_branch_path_dominance_support_multiplier": (
                        val_components.get(
                            "branch_path_dominance_support_multiplier",
                            0.0,
                        )
                    ),
                    "val_branch_path_dominance_label_multiplier": (
                        val_components.get(
                            "branch_path_dominance_label_multiplier",
                            0.0,
                        )
                    ),
                    "val_branch_path_dominance_allowed_drop": (
                        val_components.get(
                            "branch_path_dominance_allowed_drop",
                            0.0,
                        )
                    ),
                    "val_branch_support_disagreement_cap_regularization": (
                        val_components.get(
                            "branch_support_disagreement_cap_regularization",
                            0.0,
                        )
                    ),
                    "val_loss_branch_support_disagreement_cap_regularization": (
                        val_components.get(
                            "branch_support_disagreement_cap_regularization_loss",
                            0.0,
                        )
                    ),
                    "val_branch_support_disagreement_weight": val_components.get(
                        "branch_support_disagreement_weight",
                        0.0,
                    ),
                    "val_branch_support_score_margin": val_components.get(
                        "branch_support_score_margin",
                        0.0,
                    ),
                    "val_loss_branch_support_score_margin": val_components.get(
                        "branch_support_score_margin_loss",
                        0.0,
                    ),
                    "val_class_gated_branch_logit_margin": val_components.get(
                        "class_gated_branch_logit_margin",
                        0.0,
                    ),
                    "val_loss_class_gated_branch_logit_margin": val_components.get(
                        "class_gated_branch_logit_margin_loss",
                        0.0,
                    ),
                    "val_branch_to_evidence_ranking_consistency": (
                        val_components.get(
                            "branch_to_evidence_ranking_consistency",
                            0.0,
                        )
                    ),
                    "val_loss_branch_to_evidence_ranking_consistency": (
                        val_components.get(
                            "branch_to_evidence_ranking_consistency_loss",
                            0.0,
                        )
                    ),
                    "val_global_residual_anti_veto": val_components.get(
                        "global_residual_anti_veto",
                        0.0,
                    ),
                    "val_loss_global_residual_anti_veto": val_components.get(
                        "global_residual_anti_veto_loss",
                        0.0,
                    ),
                    "val_global_residual_anti_veto_eligible_fraction": (
                        val_components.get(
                            "global_residual_anti_veto_eligible_fraction",
                            0.0,
                        )
                    ),
                    "val_residual_contradiction_regularization": (
                        val_components.get(
                            "residual_contradiction_regularization",
                            0.0,
                        )
                    ),
                    "val_loss_residual_contradiction_regularization": (
                        val_components.get(
                            "residual_contradiction_regularization_loss",
                            0.0,
                        )
                    ),
                    "val_residual_contradiction_regularization_eligible_fraction": (
                        val_components.get(
                            "residual_contradiction_regularization_eligible_fraction",
                            0.0,
                        )
                    ),
                    "val_gate_weighted_branch_margin": val_components.get(
                        "gate_weighted_branch_margin",
                        0.0,
                    ),
                    "val_loss_gate_weighted_branch_margin": val_components.get(
                        "gate_weighted_branch_margin_loss",
                        0.0,
                    ),
                    "val_gate_branch_regret": val_components.get(
                        "gate_branch_regret",
                        0.0,
                    ),
                    "val_loss_gate_branch_regret": val_components.get(
                        "gate_branch_regret_loss",
                        0.0,
                    ),
                    "val_gate_branch_regret_eligible_fraction": val_components.get(
                        "gate_branch_regret_eligible_fraction",
                        0.0,
                    ),
                    "val_gate_branch_regret_weight_multiplier": val_components.get(
                        "gate_branch_regret_weight_multiplier",
                        0.0,
                    ),
                    "val_gate_branch_regret_effective_weight": val_components.get(
                        "gate_branch_regret_effective_weight",
                        0.0,
                    ),
                    "val_gate_bad_branch_suppression": val_components.get(
                        "gate_bad_branch_suppression",
                        0.0,
                    ),
                    "val_loss_gate_bad_branch_suppression": val_components.get(
                        "gate_bad_branch_suppression_loss",
                        0.0,
                    ),
                    "val_gate_bad_branch_suppression_bad_gate_mass": (
                        val_components.get(
                            "gate_bad_branch_suppression_bad_gate_mass",
                            0.0,
                        )
                    ),
                    "val_gate_bad_branch_suppression_weight_multiplier": (
                        val_components.get(
                            "gate_bad_branch_suppression_weight_multiplier",
                            0.0,
                        )
                    ),
                    "val_gate_bad_branch_suppression_effective_weight": (
                        val_components.get(
                            "gate_bad_branch_suppression_effective_weight",
                            0.0,
                        )
                    ),
                    "val_top_branch_margin": val_components.get(
                        "top_branch_margin",
                        0.0,
                    ),
                    "val_loss_top_branch_margin": val_components.get(
                        "top_branch_margin_loss",
                        0.0,
                    ),
                    "val_top_branch_margin_effective_weight": val_components.get(
                        "top_branch_margin_effective_weight",
                        0.0,
                    ),
                    "val_top_branch_margin_hardness_multiplier": val_components.get(
                        "top_branch_margin_hardness_multiplier",
                        0.0,
                    ),
                }
            )
            train_loss_component_history.append(train_components)
            val_loss_component_history.append(val_components)

            lrs = [float(group["lr"]) for group in optimizer.param_groups]
            adaptive_state = self._branch_objective_state_dict()
            diagnostics_path = None
            if val_result.diagnostics:
                diagnostics_path = write_diagnostics_jsonl(
                    val_result.diagnostics,
                    self.cfg.run_dir / "diagnostics" / f"val_epoch_{epoch:03d}.jsonl",
                )
            epoch_log_context = EpochLogContext(
                epoch=epoch,
                total_epochs=self.cfg.epochs,
                learning_rates=lrs,
                class_names=self._class_names(),
                train_loss=train_result.loss,
                val_loss=val_result.loss,
                train_metrics=train_result.metrics,
                val_metrics=val_result.metrics,
                val_metrics_optimized=val_metrics_optimized,
                val_threshold_optimization=val_threshold_optimization,
                train_components=train_components,
                val_components=val_components,
                adaptive_state=adaptive_state,
                diagnostics_path=diagnostics_path,
            )
            for log_line in format_epoch_log_block(epoch_log_context):
                self.logger.info("%s", log_line)
            append_epoch_jsonl_logs(self.cfg.run_dir, epoch_log_context)

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
                "adaptive_branch_objective_state": (
                    self._branch_objective_state_dict()
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
