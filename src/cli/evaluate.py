from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import build_clip_loader, build_dataset
from src.evaluation.diagnostics import build_diagnostic_rows, write_diagnostics_jsonl
from src.evaluation.metrics import MetricsComputer
from src.evaluation.thresholds import (
    ThresholdOptimizationResult,
    compute_metrics_at_threshold,
    load_checkpoint_threshold_optimization,
)
from src.models.model import MultiScaleRdtAstModel
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import EvalConfig, JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger


@dataclass(frozen=True)
class PredictionRow:
    audio_path: str
    true_label: int
    predicted_label: int
    predicted_probability: float
    class_probabilities: tuple[float, ...] | None = None


def _validate_eval_frontend_dims(
    *,
    checkpoint_path: str | Path,
    checkpoint_num_mel_bins: int,
    checkpoint_max_length: int,
    dataset_num_mel_bins: int,
    dataset_max_length: int,
) -> None:
    if (
        checkpoint_num_mel_bins == dataset_num_mel_bins
        and checkpoint_max_length == dataset_max_length
    ):
        return
    raise ValueError(
        "Evaluation frontend dims do not match the checkpoint encoder dims.\n"
        f"- checkpoint: num_mel_bins={checkpoint_num_mel_bins}, "
        f"max_length={checkpoint_max_length}\n"
        f"- dataset:    num_mel_bins={dataset_num_mel_bins}, "
        f"max_length={dataset_max_length}\n"
        f"- checkpoint_path: {checkpoint_path}\n\n"
        "Use an evaluation config with the same data.preprocessing.ast_fbank "
        "settings that were used for training this checkpoint."
    )


def _predict_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 1:
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(torch.long)
        return probabilities, predictions
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return probabilities, predictions


def evaluate_checkpoint(
    cfg: EvalConfig,
    checkpoint_path: str | Path,
    *,
    return_predictions: bool = False,
    return_diagnostics: bool = False,
) -> (
    tuple[dict, list[PredictionRow], list[dict]]
    | tuple[dict, list[PredictionRow]]
    | dict
):
    device = torch.device(cfg.experiment.device)
    checkpoint = load_checkpoint(str(checkpoint_path), device=device)
    model_cfg_raw = checkpoint.get("model_cfg")
    if model_cfg_raw is None:
        raise RuntimeError("Checkpoint missing model_cfg")
    model_cfg = parse_model_cfg(model_cfg_raw)
    model = MultiScaleRdtAstModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    loss_weight_summary = checkpoint.get("loss_weight_summary", {})
    main_index_to_binary_target = None
    if isinstance(loss_weight_summary, dict):
        raw_mapping = loss_weight_summary.get("main_index_to_binary_target")
        if isinstance(raw_mapping, (list, tuple)):
            main_index_to_binary_target = tuple(int(item) for item in raw_mapping)

    dataset = build_dataset(cfg.data, split="eval")
    _validate_eval_frontend_dims(
        checkpoint_path=checkpoint_path,
        checkpoint_num_mel_bins=model_cfg.encoder.feature_dims.num_mel_bins,
        checkpoint_max_length=model_cfg.encoder.feature_dims.max_length,
        dataset_num_mel_bins=dataset.num_mel_bins,
        dataset_max_length=dataset.max_length,
    )
    loader = build_clip_loader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    probabilities: list[float] | list[list[float]] = []
    predictions: list[int] = []
    targets: list[int] = []
    branch_binary_probabilities: list[list[float]] = []
    branch_binary_targets: list[int] = []
    prediction_rows: list[PredictionRow] = []
    diagnostics: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.input_values)
            batch_probs, batch_preds = _predict_from_logits(output.logits)
            probabilities.extend(batch_probs.cpu().tolist())
            predictions.extend(batch_preds.cpu().tolist())
            targets.extend(batch.labels.cpu().to(torch.long).tolist())
            binary_targets_for_batch = None
            if (
                output.branch_binary_logits is not None
                and main_index_to_binary_target is not None
            ):
                mapping = torch.tensor(
                    main_index_to_binary_target,
                    device=batch.labels.device,
                    dtype=torch.long,
                )
                binary_targets_for_batch = mapping[
                    batch.labels.to(device=batch.labels.device, dtype=torch.long)
                ]
                branch_binary_targets.extend(
                    binary_targets_for_batch.detach().cpu().to(torch.long).tolist()
                )
                branch_binary_probabilities.extend(
                    torch.sigmoid(output.branch_binary_logits.detach()).cpu().tolist()
                )

            if return_predictions or return_diagnostics:
                for index, audio_path in enumerate(batch.audio_paths):
                    predicted_label = int(batch_preds[index].item())
                    if batch_probs.ndim == 1:
                        predicted_probability = float(batch_probs[index].item())
                        class_probabilities = None
                    else:
                        predicted_probability = float(
                            batch_probs[index, predicted_label].item()
                        )
                        class_probabilities = tuple(
                            float(item) for item in batch_probs[index].tolist()
                        )
                    prediction_rows.append(
                        PredictionRow(
                            audio_path=audio_path,
                            true_label=int(batch.labels[index].item()),
                            predicted_label=predicted_label,
                            predicted_probability=predicted_probability,
                            class_probabilities=class_probabilities,
                        )
                    )
                diagnostics.extend(
                    build_diagnostic_rows(
                        batch,
                        output,
                        probabilities=batch_probs.cpu(),
                        predicted_labels=batch_preds.cpu(),
                        analysis=cfg.analysis.outputs,
                        binary_auxiliary_targets=(
                            binary_targets_for_batch.detach().cpu()
                            if binary_targets_for_batch is not None
                            else None
                        ),
                    )
                )

    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
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
    baseline_metrics = MetricsComputer.compute(
        y_true,
        y_pred,
        y_prob,
        branch_binary_probabilities=branch_binary_prob_arr,
        branch_binary_targets=branch_binary_target_arr,
    )
    if y_prob.ndim == 1 and cfg.threshold_optimization.enabled:
        threshold_optimization = load_checkpoint_threshold_optimization(
            checkpoint.get("val_threshold_optimization"),
            cfg.threshold_optimization.metric,
        )
        optimized_metrics = compute_metrics_at_threshold(
            y_true,
            y_prob,
            threshold_optimization.selected_threshold,
        )
    elif y_prob.ndim == 1:
        threshold_optimization = ThresholdOptimizationResult.disabled(
            cfg.threshold_optimization.metric
        )
        optimized_metrics = baseline_metrics
    else:
        threshold_optimization = ThresholdOptimizationResult.disabled(
            cfg.threshold_optimization.metric,
            reason=(
                "threshold optimization is only supported for one-logit binary outputs"
            ),
        )
        optimized_metrics = baseline_metrics

    metrics = baseline_metrics.to_dict()
    metrics["decision_threshold"] = 0.5 if y_prob.ndim == 1 else None
    metrics["targets"] = y_true.tolist()
    metrics["predicted_probability"] = y_prob.tolist()
    metrics["threshold_optimization"] = threshold_optimization.to_dict()
    metrics["optimized_metrics"] = optimized_metrics.to_dict()
    metrics["optimized_metrics"]["decision_threshold"] = (
        threshold_optimization.selected_threshold if y_prob.ndim == 1 else None
    )

    if return_predictions and return_diagnostics:
        return metrics, prediction_rows, diagnostics
    if return_predictions:
        return metrics, prediction_rows
    return metrics


def write_predictions_csv(rows: list[PredictionRow], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_classes = max(
        (len(row.class_probabilities or ()) for row in rows),
        default=0,
    )
    fieldnames = [
        "audio_path",
        "true_label",
        "predicted_label",
        "predicted_probability",
    ]
    fieldnames.extend(f"class_probability_{index}" for index in range(max_classes))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "audio_path": row.audio_path,
                "true_label": row.true_label,
                "predicted_label": row.predicted_label,
                "predicted_probability": row.predicted_probability,
            }
            if row.class_probabilities is not None:
                for index, probability in enumerate(row.class_probabilities):
                    payload[f"class_probability_{index}"] = probability
            writer.writerow(payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = JsonConfigLoader.load_eval(args.config)
    out_dir = Fs.ensure_dir(Path(cfg.checkpoint_path).parent)
    enable_file_logging(out_dir / "eval.log", mode="w")

    result = evaluate_checkpoint(
        cfg,
        cfg.checkpoint_path,
        return_predictions=True,
        return_diagnostics=True,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "Expected evaluate_checkpoint to return metrics, rows, diagnostics"
        )
    metrics, rows, diagnostics = result
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote %s", metrics_path)

    predictions_path = write_predictions_csv(rows, out_dir / "eval_predictions.csv")
    logger.info("Wrote %s", predictions_path)

    if diagnostics:
        diagnostics_path = write_diagnostics_jsonl(
            diagnostics,
            out_dir / "eval_diagnostics.jsonl",
        )
        logger.info("Wrote %s", diagnostics_path)


if __name__ == "__main__":
    main()
