from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.fsd50k_dataset import Fsd50kBatch
from src.data.loaders import ClipBatch
from src.models.model import AstModelOutput
from src.utils.config import AnalysisOutputConfig


def _optional_output_tensors(output: AstModelOutput) -> dict[str, torch.Tensor | None]:
    branch_logits = (
        output.branch_logits.detach().cpu()
        if output.branch_logits is not None
        else None
    )
    branch_binary_logits = (
        output.branch_binary_logits.detach().cpu()
        if output.branch_binary_logits is not None
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
    evidence_gate_weights = (
        output.evidence_gate_weights.detach().cpu()
        if output.evidence_gate_weights is not None
        else None
    )
    evidence_gate_entropy = (
        output.evidence_gate_entropy.detach().cpu()
        if output.evidence_gate_entropy is not None
        else None
    )
    branch_evidence_norms = (
        output.branch_evidence_norms.detach().cpu()
        if output.branch_evidence_norms is not None
        else None
    )
    selected_evidence_dropout_mask = (
        output.selected_evidence_dropout_mask.detach().cpu()
        if output.selected_evidence_dropout_mask is not None
        else None
    )
    selected_evidence_keep_ratio = (
        output.selected_evidence_keep_ratio.detach().cpu()
        if output.selected_evidence_keep_ratio is not None
        else None
    )
    return {
        "branch_logits": branch_logits,
        "branch_binary_logits": branch_binary_logits,
        "selected_evidence_tokens": selected_evidence_tokens,
        "selected_evidence_indices": selected_evidence_indices,
        "selected_evidence_scores": selected_evidence_scores,
        "selected_evidence_branch_ids": selected_evidence_branch_ids,
        "evidence_gate_weights": evidence_gate_weights,
        "evidence_gate_entropy": evidence_gate_entropy,
        "branch_evidence_norms": branch_evidence_norms,
        "selected_evidence_dropout_mask": selected_evidence_dropout_mask,
        "selected_evidence_keep_ratio": selected_evidence_keep_ratio,
    }


def _add_evidence_metadata(
    row: dict[str, Any],
    tensors: dict[str, torch.Tensor | None],
    output: AstModelOutput,
    index: int,
) -> None:
    selected_evidence_indices = tensors["selected_evidence_indices"]
    selected_evidence_scores = tensors["selected_evidence_scores"]
    selected_evidence_branch_ids = tensors["selected_evidence_branch_ids"]
    evidence_gate_weights = tensors["evidence_gate_weights"]
    evidence_gate_entropy = tensors["evidence_gate_entropy"]
    branch_evidence_norms = tensors["branch_evidence_norms"]
    selected_evidence_dropout_mask = tensors["selected_evidence_dropout_mask"]
    selected_evidence_keep_ratio = tensors["selected_evidence_keep_ratio"]

    if selected_evidence_indices is not None:
        row["selected_evidence_indices"] = selected_evidence_indices[index].tolist()
    if selected_evidence_scores is not None:
        row["selected_evidence_scores"] = selected_evidence_scores[index].tolist()
    if selected_evidence_branch_ids is not None:
        row["selected_evidence_branch_ids"] = selected_evidence_branch_ids[
            index
        ].tolist()
    if output.evidence_score_source is not None:
        row["evidence_score_source"] = output.evidence_score_source
    if output.evidence_pooling_type is not None:
        row["evidence_pooling_type"] = output.evidence_pooling_type
    if evidence_gate_weights is not None:
        row["evidence_gate_weights"] = evidence_gate_weights[index].tolist()
    if evidence_gate_entropy is not None:
        row["evidence_gate_entropy"] = float(evidence_gate_entropy[index].item())
    if branch_evidence_norms is not None:
        row["branch_evidence_norms"] = branch_evidence_norms[index].tolist()
    if selected_evidence_dropout_mask is not None:
        row["selected_evidence_dropout_mask"] = selected_evidence_dropout_mask[
            index
        ].tolist()
    if selected_evidence_keep_ratio is not None:
        row["selected_evidence_keep_ratio"] = float(
            selected_evidence_keep_ratio[index].item()
        )


def build_diagnostic_rows(
    batch: ClipBatch,
    output: AstModelOutput,
    *,
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
    analysis: AnalysisOutputConfig,
    binary_auxiliary_targets: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logits = output.logits.detach().cpu()
    embeddings = output.pooled_embedding.detach().cpu()
    probs = probabilities.detach().cpu()
    tensors = _optional_output_tensors(output)
    branch_logits = tensors["branch_logits"]
    branch_binary_logits = tensors["branch_binary_logits"]
    branch_binary_probabilities = (
        torch.sigmoid(branch_binary_logits)
        if branch_binary_logits is not None
        else None
    )
    binary_auxiliary_targets_cpu = (
        binary_auxiliary_targets.detach().cpu()
        if binary_auxiliary_targets is not None
        else None
    )
    selected_evidence_tokens = tensors["selected_evidence_tokens"]

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
            if branch_binary_logits is not None:
                row["branch_binary_logits"] = branch_binary_logits[index].tolist()
                assert branch_binary_probabilities is not None
                row["branch_binary_probabilities"] = branch_binary_probabilities[
                    index
                ].tolist()
            if binary_auxiliary_targets_cpu is not None:
                row["binary_auxiliary_target"] = int(
                    binary_auxiliary_targets_cpu[index].item()
                )
        if analysis.save_probabilities:
            row["probabilities"] = probability_payload
        _add_evidence_metadata(row, tensors, output, index)
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


def _indices_to_names(indices: list[int], class_names: tuple[str, ...]) -> list[str]:
    return [
        class_names[index] if index < len(class_names) else str(index)
        for index in indices
    ]


def build_multilabel_diagnostic_rows(
    batch: Fsd50kBatch,
    output: AstModelOutput,
    *,
    probabilities: torch.Tensor,
    class_names: tuple[str, ...],
    threshold: float,
    analysis: AnalysisOutputConfig,
) -> list[dict[str, Any]]:
    if batch.labels is None:
        raise ValueError("FSD50K supervised diagnostics require batch labels")
    rows: list[dict[str, Any]] = []
    logits = output.logits.detach().cpu()
    embeddings = output.pooled_embedding.detach().cpu()
    probs = probabilities.detach().cpu()
    labels = batch.labels.detach().cpu()
    tensors = _optional_output_tensors(output)
    selected_evidence_tokens = tensors["selected_evidence_tokens"]
    branch_logits = tensors["branch_logits"]
    top_k = min(5, int(probs.shape[1]))

    for index, clip_id in enumerate(batch.clip_ids):
        true_label_indices = torch.nonzero(
            labels[index] > 0.5, as_tuple=False
        ).flatten()
        predicted_label_indices = torch.nonzero(
            probs[index] >= float(threshold),
            as_tuple=False,
        ).flatten()
        top_values, top_indices = torch.topk(probs[index], k=top_k)
        true_indices = [int(value.item()) for value in true_label_indices]
        predicted_indices = [int(value.item()) for value in predicted_label_indices]
        top_indices_list = [int(value.item()) for value in top_indices]

        row: dict[str, Any] = {
            "clip_id": clip_id,
            "audio_path": batch.audio_paths[index],
            "true_label_indices": true_indices,
            "predicted_label_indices": predicted_indices,
            "top5_label_indices": top_indices_list,
            "top5_probabilities": top_values.tolist(),
            "threshold": float(threshold),
        }
        if analysis.save_clip_metadata:
            row["true_label_names"] = _indices_to_names(true_indices, class_names)
            row["predicted_label_names"] = _indices_to_names(
                predicted_indices,
                class_names,
            )
            row["top5_label_names"] = _indices_to_names(top_indices_list, class_names)
        if analysis.save_logits:
            row["logits"] = logits[index].tolist()
            if branch_logits is not None:
                row["branch_logits"] = branch_logits[index].tolist()
        if analysis.save_probabilities:
            row["probabilities"] = probs[index].tolist()
        _add_evidence_metadata(row, tensors, output, index)
        if analysis.save_embeddings:
            row["pooled_embedding"] = embeddings[index].tolist()
            if selected_evidence_tokens is not None:
                row["selected_evidence_tokens"] = selected_evidence_tokens[
                    index
                ].tolist()
        rows.append(row)
    return rows


def write_diagnostics_jsonl(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def write_diagnostics_json(payload: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    return path
