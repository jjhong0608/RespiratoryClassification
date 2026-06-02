from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import EvalMetrics
from src.evaluation.thresholds import ThresholdOptimizationResult


@dataclass(frozen=True)
class EpochLogContext:
    epoch: int
    total_epochs: int
    learning_rates: Sequence[float]
    class_names: Sequence[str]
    train_loss: float
    val_loss: float
    train_metrics: EvalMetrics
    val_metrics: EvalMetrics
    val_metrics_optimized: EvalMetrics
    val_threshold_optimization: ThresholdOptimizationResult
    train_components: Mapping[str, Any]
    val_components: Mapping[str, Any]
    adaptive_state: Mapping[str, Any]
    diagnostics_path: str | Path | None = None


def format_epoch_log_block(context: EpochLogContext) -> list[str]:
    threshold_text = _format_threshold(context.val_threshold_optimization)
    lines = [
        (
            f"Epoch {context.epoch:03d}/{context.total_epochs:03d} | "
            f"lr={_format_learning_rates(context.learning_rates)} | "
            f"phase={_display_phase(context.epoch)} | threshold={threshold_text}"
        ),
        _format_metric_line(
            "Train",
            loss=context.train_loss,
            metrics=context.train_metrics,
            class_names=context.class_names,
        ),
        _format_metric_line(
            "Val  ",
            loss=context.val_loss,
            metrics=context.val_metrics,
            class_names=context.class_names,
            include_balanced_accuracy=True,
            include_confusion=True,
        ),
        _format_component_line(
            "Loss Train",
            context.train_components,
            fallback_loss=context.train_loss,
        ),
        _format_component_line(
            "Loss Val  ",
            context.val_components,
            fallback_loss=context.val_loss,
        ),
    ]
    gate_line = _format_gate_line(context.train_components, context.val_components)
    if gate_line:
        lines.append(gate_line)
    auto_line = _format_auto_line(context.train_components, context.adaptive_state)
    if auto_line:
        lines.append(auto_line)
    lines.append(
        f"Artifacts | diagnostics={_format_artifact_path(context.diagnostics_path)}"
    )
    return lines


def build_metrics_epoch_payload(context: EpochLogContext) -> dict[str, Any]:
    return _to_jsonable(
        {
            "epoch": int(context.epoch),
            "phase": _display_phase(context.epoch),
            "learning_rates": [float(lr) for lr in context.learning_rates],
            "class_names": list(context.class_names),
            "train": {
                "loss": float(context.train_loss),
                "metrics": context.train_metrics.to_dict(),
                "confusion_summary": _confusion_summary(
                    context.train_metrics,
                    context.class_names,
                ),
            },
            "val": {
                "loss": float(context.val_loss),
                "metrics": context.val_metrics.to_dict(),
                "confusion_summary": _confusion_summary(
                    context.val_metrics,
                    context.class_names,
                ),
            },
            "val_optimized": context.val_metrics_optimized.to_dict(),
            "threshold_optimization": context.val_threshold_optimization.to_dict(),
            "diagnostics_path": (
                str(context.diagnostics_path) if context.diagnostics_path else None
            ),
        }
    )


def build_loss_components_epoch_payload(context: EpochLogContext) -> dict[str, Any]:
    return _to_jsonable(
        {
            "epoch": int(context.epoch),
            "phase": _display_phase(context.epoch),
            "train": dict(context.train_components),
            "val": dict(context.val_components),
        }
    )


def build_adaptive_state_epoch_payload(context: EpochLogContext) -> dict[str, Any]:
    return _to_jsonable(
        {
            "epoch": int(context.epoch),
            "phase": _display_phase(context.epoch),
            "adaptive_state": dict(context.adaptive_state),
            "train_top_branch_violation_rate_by_label": context.train_components.get(
                "top_branch_violation_rate_by_label",
                {},
            ),
            "train_gate_branch_regret_eligible_rate_by_label": (
                context.train_components.get(
                    "gate_branch_regret_eligible_rate_by_label",
                    {},
                )
            ),
            "diagnostics_path": (
                str(context.diagnostics_path) if context.diagnostics_path else None
            ),
        }
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_to_jsonable(dict(payload)), sort_keys=True) + "\n")
    return path


def append_epoch_jsonl_logs(run_dir: Path, context: EpochLogContext) -> dict[str, Path]:
    logs_dir = run_dir / "logs"
    paths = {
        "metrics": logs_dir / "metrics_epoch.jsonl",
        "loss_components": logs_dir / "loss_components_epoch.jsonl",
        "adaptive_state": logs_dir / "adaptive_state_epoch.jsonl",
    }
    append_jsonl(paths["metrics"], build_metrics_epoch_payload(context))
    append_jsonl(paths["loss_components"], build_loss_components_epoch_payload(context))
    append_jsonl(paths["adaptive_state"], build_adaptive_state_epoch_payload(context))
    return paths


def _display_phase(epoch: int) -> int:
    if int(epoch) <= 10:
        return 1
    if int(epoch) <= 30:
        return 2
    return 3


def _format_learning_rates(learning_rates: Sequence[float]) -> str:
    values = [float(lr) for lr in learning_rates]
    if not values:
        return "[]"
    if len(values) == 1:
        return f"{values[0]:.8f}"
    return "[" + ", ".join(f"{lr:.8f}" for lr in values) + "]"


def _format_threshold(optimization: ThresholdOptimizationResult) -> str:
    if optimization.applied:
        return (
            f"opt:{optimization.selected_threshold:.4f}({optimization.selected_metric})"
        )
    if not optimization.enabled:
        return "disabled"
    return f"fallback:{optimization.selected_threshold:.4f}"


def _format_metric_line(
    prefix: str,
    *,
    loss: float,
    metrics: EvalMetrics,
    class_names: Sequence[str],
    include_balanced_accuracy: bool = False,
    include_confusion: bool = False,
) -> str:
    f1_value = metrics.macro_f1 if metrics.macro_f1 is not None else metrics.f1_score
    parts = [
        f"{prefix} | loss={loss:.4f}",
        f"acc={metrics.accuracy:.4f}",
        f"macro_f1={float(f1_value):.4f}",
    ]
    if include_balanced_accuracy:
        parts.append(f"bal_acc={metrics.balanced_accuracy:.4f}")
    recall_text = _format_per_class_values(
        values=metrics.per_class_recall,
        class_names=class_names,
        metric_name="recall",
    )
    if recall_text:
        parts.append(recall_text)
    else:
        parts.append(f"recall={metrics.recall:.4f}")
    if include_confusion:
        confusion_text = _format_confusion_directions(metrics, class_names)
        if confusion_text:
            parts.append(confusion_text)
    return " | ".join(parts)


def _format_component_line(
    prefix: str,
    components: Mapping[str, Any],
    *,
    fallback_loss: float,
) -> str:
    parts = [
        f"{prefix} | main={_component_float(components, 'main', fallback_loss):.4f}",
    ]
    _append_component(parts, "evidence_aux", components, "evidence_auxiliary_loss")
    _append_component(parts, "branch_binary", components, "branch_binary_auxiliary")
    _append_raw_loss_pair(
        parts,
        "class_margin",
        components,
        raw_key="class_evidence_margin",
        loss_key="class_evidence_margin_loss",
    )
    _append_raw_loss_pair(
        parts,
        "branch_logit_margin",
        components,
        raw_key="class_gated_branch_logit_margin",
        loss_key="class_gated_branch_logit_margin_loss",
    )
    _append_raw_loss_pair(
        parts,
        "gate_weighted",
        components,
        raw_key="gate_weighted_branch_margin",
        loss_key="gate_weighted_branch_margin_loss",
    )
    _append_raw_loss_pair(
        parts,
        "top_branch",
        components,
        raw_key="top_branch_margin",
        loss_key="top_branch_margin_loss",
    )
    parts.append(
        f"scheduled={_component_float(components, 'total_scheduled', fallback_loss):.4f}"
    )
    if "total_monitor" in components:
        parts.append(
            f"monitor={_component_float(components, 'total_monitor', fallback_loss):.4f}"
        )
    return " | ".join(parts)


def _format_gate_line(
    train_components: Mapping[str, Any],
    val_components: Mapping[str, Any],
) -> str:
    parts: list[str] = []
    if _has_component(train_components, val_components, "gate_entropy"):
        parts.append(
            "entropy="
            f"{_component_float(train_components, 'gate_entropy', 0.0):.4f}/"
            f"{_component_float(val_components, 'gate_entropy', 0.0):.4f}"
        )
    if _has_component(train_components, val_components, "class_gate_diversity"):
        parts.append(
            "diversity="
            f"{_component_float(train_components, 'class_gate_diversity', 0.0):.4f}/"
            f"{_component_float(val_components, 'class_gate_diversity', 0.0):.4f}"
        )
    if _has_component(train_components, val_components, "gate_branch_regret"):
        parts.append(
            "regret raw="
            f"{_component_float(train_components, 'gate_branch_regret', 0.0):.4f}/"
            f"{_component_float(val_components, 'gate_branch_regret', 0.0):.4f}"
        )
        parts.append(
            "loss="
            f"{_component_float(train_components, 'gate_branch_regret_loss', 0.0):.4f}/"
            f"{_component_float(val_components, 'gate_branch_regret_loss', 0.0):.4f}"
        )
        parts.append(
            "eff_w="
            f"{_component_float(train_components, 'gate_branch_regret_effective_weight', 0.0):.4f}"
        )
        parts.append(
            "multiplier="
            f"{_component_float(train_components, 'gate_branch_regret_weight_multiplier', 0.0):.4f}"
        )
        parts.append(
            "eligible="
            f"{_component_float(train_components, 'gate_branch_regret_eligible_fraction', 0.0):.4f}/"
            f"{_component_float(val_components, 'gate_branch_regret_eligible_fraction', 0.0):.4f}"
        )
    if not parts:
        return ""
    return "Gate  | " + " | ".join(parts)


def _format_auto_line(
    train_components: Mapping[str, Any],
    adaptive_state: Mapping[str, Any],
) -> str:
    parts: list[str] = []
    top_margin = adaptive_state.get("top_branch_margin_by_label")
    if isinstance(top_margin, Mapping) and top_margin:
        parts.append(f"top_margin={_format_label_values(top_margin)}")
    regret_threshold = adaptive_state.get(
        "gate_branch_regret_positive_threshold_by_label"
    )
    if isinstance(regret_threshold, Mapping) and regret_threshold:
        parts.append(f"regret_threshold={_format_label_values(regret_threshold)}")
    top_rates = train_components.get("top_branch_violation_rate_by_label")
    if isinstance(top_rates, Mapping) and top_rates:
        parts.append(f"top_violation={_format_label_values(top_rates)}")
    regret_rates = train_components.get("gate_branch_regret_eligible_rate_by_label")
    if isinstance(regret_rates, Mapping) and regret_rates:
        parts.append(f"regret_eligible={_format_label_values(regret_rates)}")
    if not parts:
        return ""
    return "Auto  | " + " | ".join(parts)


def _format_per_class_values(
    *,
    values: Sequence[float] | None,
    class_names: Sequence[str],
    metric_name: str,
) -> str:
    if values is None or not class_names:
        return ""
    usable = min(len(values), len(class_names))
    if usable == 0:
        return ""
    formatted = "/".join(
        f"{class_names[index]}={float(values[index]):.4f}" for index in range(usable)
    )
    return f"{metric_name} {formatted}"


def _format_confusion_directions(
    metrics: EvalMetrics,
    class_names: Sequence[str],
) -> str:
    summary = _confusion_summary(metrics, class_names)
    parts = [
        f"{key}={value}"
        for key, value in summary.items()
        if key in {"wheeze->normal", "normal->wheeze"}
    ]
    return " | ".join(parts)


def _confusion_summary(
    metrics: EvalMetrics,
    class_names: Sequence[str],
) -> dict[str, int]:
    label_to_index = {label: index for index, label in enumerate(class_names)}
    cm = metrics.confusion_matrix
    summary: dict[str, int] = {}
    for source, target in (("wheeze", "normal"), ("normal", "wheeze")):
        if source not in label_to_index or target not in label_to_index:
            continue
        row = label_to_index[source]
        col = label_to_index[target]
        if row >= len(cm) or col >= len(cm[row]):
            continue
        summary[f"{source}->{target}"] = int(cm[row][col])
    return summary


def _append_component(
    parts: list[str],
    label: str,
    components: Mapping[str, Any],
    key: str,
) -> None:
    if key in components:
        parts.append(f"{label}={_component_float(components, key, 0.0):.4f}")


def _append_raw_loss_pair(
    parts: list[str],
    label: str,
    components: Mapping[str, Any],
    *,
    raw_key: str,
    loss_key: str,
) -> None:
    if raw_key not in components and loss_key not in components:
        return
    raw_value = _component_float(components, raw_key, 0.0)
    loss_value = _component_float(components, loss_key, 0.0)
    parts.append(f"{label} raw={raw_value:.4f} loss={loss_value:.4f}")


def _has_component(
    train_components: Mapping[str, Any],
    val_components: Mapping[str, Any],
    key: str,
) -> bool:
    return key in train_components or key in val_components


def _component_float(
    components: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = components.get(key, default)
    if value is None:
        return float(default)
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _format_label_values(values: Mapping[str, Any]) -> str:
    if not values:
        return "{}"
    formatted: list[str] = []
    for label, value in values.items():
        if value is None:
            formatted.append(f"{label}=None")
        elif isinstance(value, (int, float, np.generic)):
            formatted.append(f"{label}={float(value):.4f}")
        else:
            formatted.append(f"{label}={value}")
    return "{" + ", ".join(formatted) + "}"


def _format_artifact_path(path: str | Path | None) -> str:
    if path is None:
        return "none"
    return str(path)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value
