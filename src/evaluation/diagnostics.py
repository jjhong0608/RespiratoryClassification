from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.loaders import ClipBatch
from src.models.model import AstModelOutput
from src.utils.config import AnalysisOutputConfig


def build_diagnostic_rows(
    batch: ClipBatch,
    output: AstModelOutput,
    *,
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
    analysis: AnalysisOutputConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logits = output.logits.detach().cpu()
    embeddings = output.pooled_embedding.detach().cpu()
    probs = probabilities.detach().cpu()
    branch_logits = (
        output.branch_logits.detach().cpu()
        if output.branch_logits is not None
        else None
    )
    selected_evidence_tokens = (
        output.selected_evidence_tokens.detach().cpu()
        if output.selected_evidence_tokens is not None
        else None
    )
    selected_evidence_indices = (
        output.selected_evidence_indices.detach().cpu()
        if output.selected_evidence_indices is not None
        else None
    )
    selected_evidence_scores = (
        output.selected_evidence_scores.detach().cpu()
        if output.selected_evidence_scores is not None
        else None
    )
    selected_evidence_branch_ids = (
        output.selected_evidence_branch_ids.detach().cpu()
        if output.selected_evidence_branch_ids is not None
        else None
    )

    for index, audio_path in enumerate(batch.audio_paths):
        predicted_label = int(predicted_labels[index].item())
        if probs.ndim == 1:
            predicted_probability = float(probs[index].item())
            probability_payload: float | list[float] = predicted_probability
        else:
            predicted_probability = float(probs[index, predicted_label].item())
            probability_payload = probs[index].tolist()

        row: dict[str, Any] = {
            "audio_path": audio_path,
            "true_label": int(batch.labels[index].item()),
            "predicted_label": predicted_label,
            "predicted_probability": predicted_probability,
        }
        if analysis.save_logits:
            if logits.ndim == 1:
                row["logits"] = float(logits[index].item())
            else:
                row["logits"] = logits[index].tolist()
            if branch_logits is not None:
                row["branch_logits"] = branch_logits[index].tolist()
        if analysis.save_probabilities:
            row["probabilities"] = probability_payload
        if selected_evidence_indices is not None:
            row["selected_evidence_indices"] = selected_evidence_indices[index].tolist()
        if selected_evidence_scores is not None:
            row["selected_evidence_scores"] = selected_evidence_scores[index].tolist()
        if selected_evidence_branch_ids is not None:
            row["selected_evidence_branch_ids"] = selected_evidence_branch_ids[
                index
            ].tolist()
        if analysis.save_embeddings:
            row["pooled_embedding"] = embeddings[index].tolist()
            if selected_evidence_tokens is not None:
                row["selected_evidence_tokens"] = selected_evidence_tokens[
                    index
                ].tolist()
        if analysis.save_clip_metadata:
            row["label_name"] = batch.label_names[index]
        rows.append(row)
    return rows


def write_diagnostics_jsonl(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path
