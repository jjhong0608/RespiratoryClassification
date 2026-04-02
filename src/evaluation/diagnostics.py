from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.data.loaders import BagBatch
from src.models.model import MILModelOutput
from src.utils.config import AnalysisOutputConfig


def build_diagnostic_rows(
    batch: BagBatch,
    output: MILModelOutput,
    *,
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
    analysis: AnalysisOutputConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    instance_scores = torch.sigmoid(output.instance_logits).detach().cpu()
    attention_weights = (
        None
        if output.attention_weights is None
        else output.attention_weights.detach().cpu()
    )
    topk_indices = (
        None if output.topk_indices is None else output.topk_indices.detach().cpu()
    )

    for bag_index, audio_path in enumerate(batch.audio_paths):
        row: dict[str, Any] = {
            "audio_path": audio_path,
            "true_label": int(batch.labels[bag_index].item()),
            "predicted_probability": float(probabilities[bag_index].item()),
            "predicted_label": int(predicted_labels[bag_index].item()),
        }
        if analysis.save_segment_scores:
            valid_count = int(batch.instance_mask[bag_index].sum().item())
            row["segment_scores"] = instance_scores[bag_index, :valid_count].tolist()
        if analysis.save_attention_weights and attention_weights is not None:
            valid_count = int(batch.instance_mask[bag_index].sum().item())
            row["attention_weights"] = attention_weights[
                bag_index, :valid_count
            ].tolist()
        if analysis.save_topk_indices and topk_indices is not None:
            indices = topk_indices[bag_index].tolist()
            row["topk_indices"] = [int(item) for item in indices if int(item) >= 0]
        if analysis.save_bag_metadata:
            row["label_name"] = batch.label_names[bag_index]
            row["segment_metadata"] = [
                asdict(item) for item in batch.segment_metadata[bag_index]
            ]
        rows.append(row)
    return rows


def write_diagnostics_jsonl(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path
