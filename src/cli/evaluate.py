from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.audio import AudioPreprocessConfig
from src.data.dataset import DatasetConfig, RespiratorySoundDataset
from src.evaluation.metrics import MetricsComputer
from src.evaluation.thresholds import ThresholdOptimizer
from src.models.model import WhisperEncoderClassifier
from src.utils.checkpoint import load_checkpoint, parse_model_cfg
from src.utils.config import JsonConfigLoader
from src.utils.fs import Fs
from src.utils.logging import enable_file_logging, logger


@dataclass(frozen=True)
class PredictionRow:
    filename: str
    predicted_probability: float
    predicted_label: str
    predicted_index: int


def resolve_binary_threshold(
    cfg, ckpt: dict
) -> tuple[float, Literal["manual", "checkpoint", "default"], dict | None]:
    manual_threshold = getattr(cfg.threshold, "manual", None)
    if manual_threshold is not None:
        return (
            float(manual_threshold),
            "manual",
            ckpt.get("threshold_optimization_result"),
        )

    raw = ckpt.get("threshold_optimization_result")
    if not isinstance(raw, dict):
        return 0.5, "default", None
    threshold = raw.get("selected_threshold")
    if not isinstance(threshold, (int, float)):
        return 0.5, "default", raw
    return float(threshold), "checkpoint", raw


def build_loader(
    roots: list[str],
    cfg: AudioPreprocessConfig,
    label_to_index: dict[str, int],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, list[Path]]:
    datasets = [
        RespiratorySoundDataset(
            DatasetConfig(roots=[r], label_to_index=label_to_index, preprocess=cfg)
        )
        for r in roots
    ]
    file_paths: list[Path] = []
    for dataset in datasets:
        file_paths.extend(dataset.file_paths)
    ds: Dataset[tuple[Tensor, int]] = (
        datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    )
    if len(cast(Sized, ds)) == 0:
        raise ValueError(f"No .wav files found under: {roots}")
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        file_paths,
    )


@overload
def evaluate_checkpoint(
    cfg, checkpoint_path: str | Path, *, return_predictions: Literal[False] = False
) -> dict: ...


@overload
def evaluate_checkpoint(
    cfg, checkpoint_path: str | Path, *, return_predictions: Literal[True]
) -> tuple[dict, list[PredictionRow]]: ...


def evaluate_checkpoint(
    cfg, checkpoint_path: str | Path, *, return_predictions: bool = False
) -> dict | tuple[dict, list[PredictionRow]]:
    device = torch.device(cfg.device)
    ckpt = load_checkpoint(
        str(checkpoint_path), device=device, unsafe=cfg.unsafe_pickle_load
    )
    model_state = ckpt["model_state_dict"]

    label_to_index = dict(ckpt.get("label_to_index", cfg.data.label_to_index))
    index_to_label = {int(v): str(k) for k, v in label_to_index.items()}
    preprocess_raw = ckpt.get("preprocess_cfg", None)
    if isinstance(preprocess_raw, dict):
        preprocess = AudioPreprocessConfig(**preprocess_raw)
    elif isinstance(preprocess_raw, AudioPreprocessConfig):
        preprocess = preprocess_raw
    else:
        preprocess = AudioPreprocessConfig(
            sample_rate=cfg.data.sample_rate,
            n_mels=80,
            clip_seconds=cfg.data.clip_seconds,
            source_type=cfg.data.source_type,
            bandpass_enabled=cfg.data.bandpass.enabled,
            bandpass_low_freq=cfg.data.bandpass.low_freq,
            bandpass_high_freq=cfg.data.bandpass.high_freq,
            bandpass_q=cfg.data.bandpass.q,
        )

    loader, file_paths = build_loader(
        list(cfg.data.eval_dirs),
        preprocess,
        label_to_index,
        cfg.data.batch_size,
        cfg.data.num_workers,
    )

    model_cfg_raw = ckpt.get("model_cfg", None)
    if model_cfg_raw is None:
        raise RuntimeError(
            "Checkpoint missing `model_cfg`; re-train with `src.cli.training`."
        )

    model_cfg = parse_model_cfg(model_cfg_raw)
    model = WhisperEncoderClassifier(model_cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    ys: list[int] = []
    preds: list[int] = []
    prob_rows: list[np.ndarray] = []
    pred_rows: list[PredictionRow] = []
    path_cursor = 0
    decision_threshold, threshold_source, threshold_info = resolve_binary_threshold(
        cfg, ckpt
    )
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_size = int(prob.shape[0])
            if prob.shape[1] == 2:
                positive_prob = prob[:, 1]
                pred = ThresholdOptimizer.predict(positive_prob, decision_threshold)
                pred_prob = np.where(pred == 1, positive_prob, 1.0 - positive_prob)
            else:
                if getattr(cfg.threshold, "manual", None) is not None:
                    raise ValueError(
                        "threshold.manual is only supported for binary evaluation"
                    )
                pred = prob.argmax(axis=-1)
                pred_prob = prob[np.arange(batch_size, dtype=np.int64), pred]
            batch_paths = file_paths[path_cursor : path_cursor + batch_size]
            if len(batch_paths) != batch_size:
                raise RuntimeError(
                    "File path count does not match prediction batch size during eval."
                )
            path_cursor += batch_size
            ys.extend(y.numpy().tolist())
            preds.extend(pred.tolist())
            prob_rows.extend(list(prob))
            if return_predictions:
                for i in range(batch_size):
                    pred_idx = int(pred[i])
                    pred_label = index_to_label.get(pred_idx, str(pred_idx))
                    pred_rows.append(
                        PredictionRow(
                            filename=str(batch_paths[i]),
                            predicted_probability=float(pred_prob[i]),
                            predicted_label=pred_label,
                            predicted_index=pred_idx,
                        )
                    )

    y_true = np.asarray(ys, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    y_prob_full = np.stack(prob_rows, axis=0) if prob_rows else None
    if y_prob_full is not None and y_prob_full.shape[1] == 2:
        y_prob: np.ndarray | None = y_prob_full[:, 1]
    else:
        y_prob = y_prob_full
    metrics = MetricsComputer.compute(y_true, y_pred, y_prob)
    out = metrics.to_dict()
    if y_prob_full is not None and y_prob_full.shape[1] == 2:
        out["positive_class_probability"] = y_prob_full[:, 1].tolist()
        out["decision_threshold"] = decision_threshold
        out["decision_threshold_source"] = threshold_source
        out["threshold_optimization_result"] = threshold_info

    if return_predictions:
        if path_cursor != len(file_paths):
            raise RuntimeError(
                f"Processed {path_cursor} samples but found {len(file_paths)} files."
            )
        return out, pred_rows
    return out


def write_predictions_csv(rows: list[PredictionRow], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "predicted_probability",
                "predicted_label",
                "predicted_index",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row.filename,
                    "predicted_probability": row.predicted_probability,
                    "predicted_label": row.predicted_label,
                    "predicted_index": row.predicted_index,
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

    out, pred_rows = evaluate_checkpoint(
        cfg, cfg.checkpoint_path, return_predictions=True
    )
    out_path = out_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"Wrote {out_path}")

    csv_path = write_predictions_csv(pred_rows, out_dir / "eval_predictions.csv")
    logger.info(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
