from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.loaders import BagBatch
from src.models.model import AstMilOutput
from src.utils.config import DiagnosticsConfig


def _predicted_instance_score(
    instance_probabilities: torch.Tensor,
    *,
    predicted_label: int,
) -> torch.Tensor:
    if instance_probabilities.ndim == 1:
        if predicted_label == 1:
            return instance_probabilities
        return 1.0 - instance_probabilities
    return instance_probabilities[:, predicted_label]


def _serialize_prediction_tensor(values: torch.Tensor) -> float | list[float]:
    if values.ndim == 0:
        return float(values.item())
    return values.tolist()


def _build_top_k_instances(
    *,
    start_sec: torch.Tensor,
    end_sec: torch.Tensor,
    instance_index: torch.Tensor,
    instance_probabilities: torch.Tensor,
    attention_weights: torch.Tensor | None,
    predicted_label: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if attention_weights is not None:
        ranking_scores = attention_weights
        ranking_name = "attention_weight"
    else:
        ranking_scores = _predicted_instance_score(
            instance_probabilities,
            predicted_label=predicted_label,
        )
        ranking_name = "predicted_class_probability"

    order = torch.argsort(ranking_scores, descending=True)[:top_k]
    rows: list[dict[str, Any]] = []
    for index in order.tolist():
        row: dict[str, Any] = {
            "instance_index": int(instance_index[index].item()),
            "start_sec": float(start_sec[index].item()),
            "end_sec": float(end_sec[index].item()),
            ranking_name: float(ranking_scores[index].item()),
        }
        if attention_weights is not None:
            row["predicted_class_probability"] = float(
                _predicted_instance_score(
                    instance_probabilities,
                    predicted_label=predicted_label,
                )[index].item()
            )
        rows.append(row)
    return rows


def build_diagnostic_rows(
    batch: BagBatch,
    output: AstMilOutput,
    *,
    predicted_labels: torch.Tensor,
    diagnostics: DiagnosticsConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bag_logits = output.bag_logits.detach().cpu()
    bag_probabilities = output.bag_probabilities.detach().cpu()
    bag_embedding = output.bag_embedding.detach().cpu()
    instance_logits = output.instance_logits.detach().cpu()
    instance_probabilities = output.instance_probabilities.detach().cpu()
    instance_embeddings = output.instance_embeddings.detach().cpu()
    attention_weights = (
        None
        if output.attention_weights is None
        else output.attention_weights.detach().cpu()
    )

    mask = batch.instance_mask.detach().cpu()
    start_sec = batch.instance_start_sec.detach().cpu()
    end_sec = batch.instance_end_sec.detach().cpu()
    instance_index = batch.instance_index.detach().cpu()

    for batch_index, recording_path in enumerate(batch.recording_paths):
        valid = mask[batch_index].to(dtype=torch.bool)
        predicted_label = int(predicted_labels[batch_index].item())
        row: dict[str, Any] = {
            "recording_id": batch.recording_ids[batch_index],
            "recording_path": recording_path,
            "true_label": int(batch.labels[batch_index].item()),
            "predicted_label": predicted_label,
            "num_instances": int(valid.sum().item()),
            "label_name": batch.label_names[batch_index],
        }
        if diagnostics.save_bag_logits:
            row["bag_logits"] = _serialize_prediction_tensor(bag_logits[batch_index])
        if diagnostics.save_bag_probabilities:
            row["bag_probabilities"] = _serialize_prediction_tensor(
                bag_probabilities[batch_index]
            )
        if diagnostics.save_bag_embedding:
            row["bag_embedding"] = bag_embedding[batch_index].tolist()
        if diagnostics.save_instance_metadata:
            row["instance_start_sec"] = start_sec[batch_index, valid].tolist()
            row["instance_end_sec"] = end_sec[batch_index, valid].tolist()
            row["instance_index"] = instance_index[batch_index, valid].tolist()
        if diagnostics.save_instance_logits:
            row["instance_logits"] = _serialize_prediction_tensor(
                instance_logits[batch_index, valid]
            )
        if diagnostics.save_instance_probabilities:
            row["instance_probabilities"] = _serialize_prediction_tensor(
                instance_probabilities[batch_index, valid]
            )
        if diagnostics.save_instance_embeddings:
            row["instance_embeddings"] = instance_embeddings[
                batch_index, valid
            ].tolist()
        if diagnostics.save_attention_weights and attention_weights is not None:
            row["attention_weights"] = attention_weights[batch_index, valid].tolist()
        if diagnostics.top_k_instances > 0:
            row["top_k_instances"] = _build_top_k_instances(
                start_sec=start_sec[batch_index, valid],
                end_sec=end_sec[batch_index, valid],
                instance_index=instance_index[batch_index, valid],
                instance_probabilities=instance_probabilities[batch_index, valid],
                attention_weights=(
                    None
                    if attention_weights is None
                    else attention_weights[batch_index, valid]
                ),
                predicted_label=predicted_label,
                top_k=diagnostics.top_k_instances,
            )
        rows.append(row)
    return rows


def write_diagnostics_jsonl(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path
