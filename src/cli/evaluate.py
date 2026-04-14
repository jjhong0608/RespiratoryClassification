from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import build_bag_loader, build_dataset
from src.evaluation.diagnostics import build_diagnostic_rows, write_diagnostics_jsonl
from src.evaluation.metrics import MetricsComputer
from src.evaluation.thresholds import (
    ThresholdOptimizationResult,
    compute_metrics_at_threshold,
    load_checkpoint_threshold_optimization,
)
from src.models.model import RespiratoryMILModel
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import EvalConfig, JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger


@dataclass(frozen=True)
class PredictionRow:
    audio_path: str
    true_label: int
    predicted_probability: float
    predicted_label: int


def _validate_eval_frontend_dims(
    *,
    checkpoint_path: str | Path,
    checkpoint_n_mels: int,
    checkpoint_audio_ctx: int,
    dataset_n_mels: int,
    dataset_audio_ctx: int,
) -> None:
    if (
        checkpoint_n_mels == dataset_n_mels
        and checkpoint_audio_ctx == dataset_audio_ctx
    ):
        return
    raise ValueError(
        "Evaluation frontend dims do not match the checkpoint encoder dims.\n"
        f"- checkpoint: n_mels={checkpoint_n_mels}, n_audio_ctx={checkpoint_audio_ctx}\n"
        f"- dataset:    n_mels={dataset_n_mels}, n_audio_ctx={dataset_audio_ctx}\n"
        f"- checkpoint_path: {checkpoint_path}\n\n"
        "Use an evaluation config with the same data.preprocessing feature settings "
        "that were used for training this checkpoint."
    )


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
    model = RespiratoryMILModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = build_dataset(cfg.data, split="eval")
    _validate_eval_frontend_dims(
        checkpoint_path=checkpoint_path,
        checkpoint_n_mels=model_cfg.segment_encoder.dims.n_mels,
        checkpoint_audio_ctx=model_cfg.segment_encoder.dims.n_audio_ctx,
        dataset_n_mels=dataset.segment_n_mels,
        dataset_audio_ctx=dataset.segment_audio_ctx,
    )
    loader = build_bag_loader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    probabilities: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    prediction_rows: list[PredictionRow] = []
    diagnostics: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.segments, batch.instance_mask)
            bag_probs = torch.sigmoid(output.bag_logits)
            bag_preds = (bag_probs >= 0.5).to(torch.long)
            probabilities.extend(bag_probs.cpu().tolist())
            predictions.extend(bag_preds.cpu().tolist())
            targets.extend(batch.labels.cpu().to(torch.long).tolist())

            if return_predictions or return_diagnostics:
                for index, audio_path in enumerate(batch.audio_paths):
                    prediction_rows.append(
                        PredictionRow(
                            audio_path=audio_path,
                            true_label=int(batch.labels[index].item()),
                            predicted_probability=float(bag_probs[index].item()),
                            predicted_label=int(bag_preds[index].item()),
                        )
                    )
                diagnostics.extend(
                    build_diagnostic_rows(
                        batch,
                        output,
                        probabilities=bag_probs.cpu(),
                        predicted_labels=bag_preds.cpu(),
                        analysis=cfg.analysis.outputs,
                    )
                )

    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    baseline_metrics = MetricsComputer.compute(y_true, y_pred, y_prob)
    if cfg.threshold_optimization.enabled:
        threshold_optimization = load_checkpoint_threshold_optimization(
            checkpoint.get("val_threshold_optimization"),
            cfg.threshold_optimization.metric,
        )
        optimized_metrics = compute_metrics_at_threshold(
            y_true,
            y_prob,
            threshold_optimization.selected_threshold,
        )
    else:
        threshold_optimization = ThresholdOptimizationResult.disabled(
            cfg.threshold_optimization.metric
        )
        optimized_metrics = baseline_metrics

    metrics = baseline_metrics.to_dict()
    metrics["decision_threshold"] = 0.5
    metrics["targets"] = y_true.tolist()
    metrics["predicted_probability"] = y_prob.tolist()
    metrics["threshold_optimization"] = threshold_optimization.to_dict()
    metrics["optimized_metrics"] = optimized_metrics.to_dict()
    metrics["optimized_metrics"]["decision_threshold"] = (
        threshold_optimization.selected_threshold
    )

    if return_predictions and return_diagnostics:
        return metrics, prediction_rows, diagnostics
    if return_predictions:
        return metrics, prediction_rows
    return metrics


def write_predictions_csv(rows: list[PredictionRow], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio_path",
                "true_label",
                "predicted_probability",
                "predicted_label",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_path": row.audio_path,
                    "true_label": row.true_label,
                    "predicted_probability": row.predicted_probability,
                    "predicted_label": row.predicted_label,
                }
            )
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
