from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.loaders import ClipBatch
from src.models.model import AstModelOutput
from src.utils.config import AnalysisOutputConfig


def _true_vs_hardest_gap(
    values: torch.Tensor,
    label: int,
) -> tuple[float, int] | None:
    if values.ndim != 1 or values.numel() < 2 or not (0 <= label < values.numel()):
        return None
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[label] = False
    negative_value, negative_index = values.masked_fill(~mask, float("-inf")).max(dim=0)
    return float((values[label] - negative_value).item()), int(negative_index.item())


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
    class_evidence_gate_weights = (
        output.class_evidence_gate_weights.detach().cpu()
        if output.class_evidence_gate_weights is not None
        else None
    )
    class_evidence_gate_entropy = (
        output.class_evidence_gate_entropy.detach().cpu()
        if output.class_evidence_gate_entropy is not None
        else None
    )
    class_evidence_learned_gate_weights = (
        output.class_evidence_learned_gate_weights.detach().cpu()
        if output.class_evidence_learned_gate_weights is not None
        else None
    )
    class_evidence_learned_gate_entropy = (
        output.class_evidence_learned_gate_entropy.detach().cpu()
        if output.class_evidence_learned_gate_entropy is not None
        else None
    )
    class_evidence_gate_mixing_alpha = (
        output.class_evidence_gate_mixing_alpha.detach().cpu()
        if output.class_evidence_gate_mixing_alpha is not None
        else None
    )
    class_evidence_logits = (
        output.class_evidence_logits.detach().cpu()
        if output.class_evidence_logits is not None
        else None
    )
    global_residual_logits = (
        output.global_residual_logits.detach().cpu()
        if output.global_residual_logits is not None
        else None
    )
    class_gated_branch_logits = (
        output.class_gated_branch_logits.detach().cpu()
        if output.class_gated_branch_logits is not None
        else None
    )
    class_gated_branch_logit_features = (
        output.class_gated_branch_logit_features.detach().cpu()
        if output.class_gated_branch_logit_features is not None
        else None
    )
    class_gated_branch_logit_relative_features = (
        output.class_gated_branch_logit_relative_features.detach().cpu()
        if output.class_gated_branch_logit_relative_features is not None
        else None
    )
    class_top_branch_margin_features = (
        output.class_top_branch_margin_features.detach().cpu()
        if output.class_top_branch_margin_features is not None
        else None
    )
    class_top_branch_margin_relative_features = (
        output.class_top_branch_margin_relative_features.detach().cpu()
        if output.class_top_branch_margin_relative_features is not None
        else None
    )
    class_evidence_scorer_branch_raw_features = (
        output.class_evidence_scorer_branch_raw_features.detach().cpu()
        if output.class_evidence_scorer_branch_raw_features is not None
        else None
    )
    class_evidence_scorer_branch_features = (
        output.class_evidence_scorer_branch_features.detach().cpu()
        if output.class_evidence_scorer_branch_features is not None
        else None
    )
    class_evidence_embedding_scores = (
        output.class_evidence_embedding_scores.detach().cpu()
        if output.class_evidence_embedding_scores is not None
        else None
    )
    class_evidence_raw_embedding_scores = (
        output.class_evidence_raw_embedding_scores.detach().cpu()
        if output.class_evidence_raw_embedding_scores is not None
        else None
    )
    class_evidence_bounded_embedding_scores = (
        output.class_evidence_bounded_embedding_scores.detach().cpu()
        if output.class_evidence_bounded_embedding_scores is not None
        else None
    )
    class_evidence_top_support_scores = (
        output.class_evidence_top_support_scores.detach().cpu()
        if output.class_evidence_top_support_scores is not None
        else None
    )
    class_evidence_gated_support_scores = (
        output.class_evidence_gated_support_scores.detach().cpu()
        if output.class_evidence_gated_support_scores is not None
        else None
    )
    class_evidence_top_raw_existential_scores = (
        output.class_evidence_top_raw_existential_scores.detach().cpu()
        if output.class_evidence_top_raw_existential_scores is not None
        else None
    )
    class_evidence_top_relative_correction_scores = (
        output.class_evidence_top_relative_correction_scores.detach().cpu()
        if output.class_evidence_top_relative_correction_scores is not None
        else None
    )
    class_evidence_top_support_raw_positive_component = (
        output.class_evidence_top_support_raw_positive_component.detach().cpu()
        if output.class_evidence_top_support_raw_positive_component is not None
        else None
    )
    class_evidence_top_support_relative_positive_component = (
        output.class_evidence_top_support_relative_positive_component.detach().cpu()
        if output.class_evidence_top_support_relative_positive_component is not None
        else None
    )
    class_evidence_top_support_relative_negative_component = (
        output.class_evidence_top_support_relative_negative_component.detach().cpu()
        if output.class_evidence_top_support_relative_negative_component is not None
        else None
    )
    class_evidence_top_support_direct_raw_scale = (
        output.class_evidence_top_support_direct_raw_scale.detach().cpu()
        if output.class_evidence_top_support_direct_raw_scale is not None
        else None
    )
    class_evidence_top_support_direct_relative_positive_scale = (
        output.class_evidence_top_support_direct_relative_positive_scale.detach().cpu()
        if output.class_evidence_top_support_direct_relative_positive_scale is not None
        else None
    )
    class_evidence_top_support_direct_relative_negative_scale = (
        output.class_evidence_top_support_direct_relative_negative_scale.detach().cpu()
        if output.class_evidence_top_support_direct_relative_negative_scale is not None
        else None
    )
    class_evidence_top_relative_positive = (
        output.class_evidence_top_relative_positive.detach().cpu()
        if output.class_evidence_top_relative_positive is not None
        else None
    )
    class_evidence_top_relative_negative = (
        output.class_evidence_top_relative_negative.detach().cpu()
        if output.class_evidence_top_relative_negative is not None
        else None
    )
    class_evidence_gate_reliability = (
        output.class_evidence_gate_reliability.detach().cpu()
        if output.class_evidence_gate_reliability is not None
        else None
    )
    class_evidence_gate_reliability_regret = (
        output.class_evidence_gate_reliability_regret.detach().cpu()
        if output.class_evidence_gate_reliability_regret is not None
        else None
    )
    class_evidence_top_margin = (
        output.class_evidence_top_margin.detach().cpu()
        if output.class_evidence_top_margin is not None
        else None
    )
    class_evidence_gated_margin = (
        output.class_evidence_gated_margin.detach().cpu()
        if output.class_evidence_gated_margin is not None
        else None
    )
    class_evidence_branch_existential_scores = (
        output.class_evidence_branch_existential_scores.detach().cpu()
        if output.class_evidence_branch_existential_scores is not None
        else None
    )
    class_evidence_branch_competitive_scores = (
        output.class_evidence_branch_competitive_scores.detach().cpu()
        if output.class_evidence_branch_competitive_scores is not None
        else None
    )
    class_evidence_branch_direct_scores = (
        output.class_evidence_branch_direct_scores.detach().cpu()
        if output.class_evidence_branch_direct_scores is not None
        else None
    )
    class_evidence_branch_residual_scores = (
        output.class_evidence_branch_residual_scores.detach().cpu()
        if output.class_evidence_branch_residual_scores is not None
        else None
    )
    class_evidence_branch_support_scores = (
        output.class_evidence_branch_support_scores.detach().cpu()
        if output.class_evidence_branch_support_scores is not None
        else None
    )
    class_evidence_interaction_scores = (
        output.class_evidence_interaction_scores.detach().cpu()
        if output.class_evidence_interaction_scores is not None
        else None
    )
    class_evidence_raw_interaction_scores = (
        output.class_evidence_raw_interaction_scores.detach().cpu()
        if output.class_evidence_raw_interaction_scores is not None
        else None
    )
    class_evidence_bounded_interaction_scores = (
        output.class_evidence_bounded_interaction_scores.detach().cpu()
        if output.class_evidence_bounded_interaction_scores is not None
        else None
    )
    class_evidence_branch_scale = (
        output.class_evidence_branch_scale.detach().cpu()
        if output.class_evidence_branch_scale is not None
        else None
    )
    class_evidence_branch_direct_top_scale = (
        output.class_evidence_branch_direct_top_scale.detach().cpu()
        if output.class_evidence_branch_direct_top_scale is not None
        else None
    )
    class_evidence_branch_direct_gated_scale = (
        output.class_evidence_branch_direct_gated_scale.detach().cpu()
        if output.class_evidence_branch_direct_gated_scale is not None
        else None
    )
    class_evidence_branch_direct_raw_scale = (
        output.class_evidence_branch_direct_raw_scale.detach().cpu()
        if output.class_evidence_branch_direct_raw_scale is not None
        else None
    )
    class_evidence_branch_direct_relative_scale = (
        output.class_evidence_branch_direct_relative_scale.detach().cpu()
        if output.class_evidence_branch_direct_relative_scale is not None
        else None
    )
    class_evidence_branch_direct_residual_scale = (
        output.class_evidence_branch_direct_residual_scale.detach().cpu()
        if output.class_evidence_branch_direct_residual_scale is not None
        else None
    )
    class_evidence_branch_direct_top_weights = (
        output.class_evidence_branch_direct_top_weights.detach().cpu()
        if output.class_evidence_branch_direct_top_weights is not None
        else None
    )
    class_evidence_branch_direct_gated_weights = (
        output.class_evidence_branch_direct_gated_weights.detach().cpu()
        if output.class_evidence_branch_direct_gated_weights is not None
        else None
    )
    class_evidence_branch_direct_existential_weights = (
        output.class_evidence_branch_direct_existential_weights.detach().cpu()
        if output.class_evidence_branch_direct_existential_weights is not None
        else None
    )
    class_evidence_branch_direct_competitive_weights = (
        output.class_evidence_branch_direct_competitive_weights.detach().cpu()
        if output.class_evidence_branch_direct_competitive_weights is not None
        else None
    )
    class_evidence_interaction_scale = (
        output.class_evidence_interaction_scale.detach().cpu()
        if output.class_evidence_interaction_scale is not None
        else None
    )
    class_evidence_interaction_scale_multiplier = (
        output.class_evidence_interaction_scale_multiplier.detach().cpu()
        if output.class_evidence_interaction_scale_multiplier is not None
        else None
    )
    class_evidence_interaction_effective_scale = (
        output.class_evidence_interaction_effective_scale.detach().cpu()
        if output.class_evidence_interaction_effective_scale is not None
        else None
    )
    class_evidence_embedding_score_bound = (
        output.class_evidence_embedding_score_bound.detach().cpu()
        if output.class_evidence_embedding_score_bound is not None
        else None
    )
    class_evidence_embedding_score_temperature = (
        output.class_evidence_embedding_score_temperature.detach().cpu()
        if output.class_evidence_embedding_score_temperature is not None
        else None
    )
    class_evidence_interaction_score_bound = (
        output.class_evidence_interaction_score_bound.detach().cpu()
        if output.class_evidence_interaction_score_bound is not None
        else None
    )
    class_evidence_interaction_score_temperature = (
        output.class_evidence_interaction_score_temperature.detach().cpu()
        if output.class_evidence_interaction_score_temperature is not None
        else None
    )
    class_evidence_scorer_branch_feature_transform_temperature = (
        output.class_evidence_scorer_branch_feature_transform_temperature.detach().cpu()
        if output.class_evidence_scorer_branch_feature_transform_temperature is not None
        else None
    )
    global_residual_scale = (
        output.global_residual_scale.detach().cpu()
        if output.global_residual_scale is not None
        else None
    )
    global_residual_schedule_multiplier = (
        output.global_residual_schedule_multiplier.detach().cpu()
        if output.global_residual_schedule_multiplier is not None
        else None
    )
    global_residual_effective_scale = (
        output.global_residual_effective_scale.detach().cpu()
        if output.global_residual_effective_scale is not None
        else None
    )
    bounded_global_residual_logits = (
        output.bounded_global_residual_logits.detach().cpu()
        if output.bounded_global_residual_logits is not None
        else None
    )
    centered_bounded_global_residual_logits = (
        output.centered_bounded_global_residual_logits.detach().cpu()
        if output.centered_bounded_global_residual_logits is not None
        else None
    )
    global_residual_gate = (
        output.global_residual_gate.detach().cpu()
        if output.global_residual_gate is not None
        else None
    )
    global_residual_learned_gate = (
        output.global_residual_learned_gate.detach().cpu()
        if output.global_residual_learned_gate is not None
        else None
    )
    global_residual_evidence_confidence = (
        output.global_residual_evidence_confidence.detach().cpu()
        if output.global_residual_evidence_confidence is not None
        else None
    )
    global_residual_confidence_factor = (
        output.global_residual_confidence_factor.detach().cpu()
        if output.global_residual_confidence_factor is not None
        else None
    )
    global_residual_contribution = (
        output.global_residual_contribution.detach().cpu()
        if output.global_residual_contribution is not None
        else None
    )
    global_residual_bound = (
        output.global_residual_bound.detach().cpu()
        if output.global_residual_bound is not None
        else None
    )
    global_residual_temperature = (
        output.global_residual_temperature.detach().cpu()
        if output.global_residual_temperature is not None
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
            if class_evidence_logits is not None:
                row["class_evidence_logits"] = class_evidence_logits[index].tolist()
            if global_residual_logits is not None:
                row["global_residual_logits"] = global_residual_logits[index].tolist()
            if bounded_global_residual_logits is not None:
                row["bounded_global_residual_logits"] = bounded_global_residual_logits[
                    index
                ].tolist()
            if centered_bounded_global_residual_logits is not None:
                row["centered_bounded_global_residual_logits"] = (
                    centered_bounded_global_residual_logits[index].tolist()
                )
            if global_residual_learned_gate is not None:
                row["global_residual_learned_gate"] = global_residual_learned_gate[
                    index
                ].tolist()
            if global_residual_evidence_confidence is not None:
                row["global_residual_evidence_confidence"] = (
                    global_residual_evidence_confidence[index].tolist()
                )
            if global_residual_confidence_factor is not None:
                row["global_residual_confidence_factor"] = (
                    global_residual_confidence_factor[index].tolist()
                )
            if global_residual_gate is not None:
                row["global_residual_gate"] = global_residual_gate[index].tolist()
            if global_residual_contribution is not None:
                row["global_residual_contribution"] = global_residual_contribution[
                    index
                ].tolist()
            if class_gated_branch_logits is not None:
                row["class_gated_branch_logits"] = class_gated_branch_logits[
                    index
                ].tolist()
            if class_gated_branch_logit_features is not None:
                row["class_gated_branch_logit_features"] = (
                    class_gated_branch_logit_features[index].tolist()
                )
            if class_gated_branch_logit_relative_features is not None:
                row["class_gated_branch_logit_relative_features"] = (
                    class_gated_branch_logit_relative_features[index].tolist()
                )
            if class_top_branch_margin_features is not None:
                row["class_top_branch_margin_features"] = (
                    class_top_branch_margin_features[index].tolist()
                )
            if class_top_branch_margin_relative_features is not None:
                row["class_top_branch_margin_relative_features"] = (
                    class_top_branch_margin_relative_features[index].tolist()
                )
            if class_evidence_scorer_branch_raw_features is not None:
                row["class_evidence_scorer_branch_raw_features"] = (
                    class_evidence_scorer_branch_raw_features[index].tolist()
                )
            if class_evidence_scorer_branch_features is not None:
                row["class_evidence_scorer_branch_features"] = (
                    class_evidence_scorer_branch_features[index].tolist()
                )
            true_label = int(batch.labels[index].item())
            if class_evidence_embedding_scores is not None:
                row["class_evidence_embedding_scores"] = (
                    class_evidence_embedding_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_embedding_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_embedding_score_gap"] = gap
                    row["class_evidence_embedding_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_raw_embedding_scores is not None:
                row["class_evidence_raw_embedding_scores"] = (
                    class_evidence_raw_embedding_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_raw_embedding_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_raw_embedding_score_gap"] = gap
                    row["class_evidence_raw_embedding_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_bounded_embedding_scores is not None:
                row["class_evidence_bounded_embedding_scores"] = (
                    class_evidence_bounded_embedding_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_bounded_embedding_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_bounded_embedding_score_gap"] = gap
                    row["class_evidence_bounded_embedding_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_top_support_scores is not None:
                row["class_evidence_top_support_scores"] = (
                    class_evidence_top_support_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_top_support_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_top_support_score_gap"] = gap
                    row["class_evidence_top_support_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_gated_support_scores is not None:
                row["class_evidence_gated_support_scores"] = (
                    class_evidence_gated_support_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_gated_support_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_gated_support_score_gap"] = gap
                    row["class_evidence_gated_support_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_top_raw_existential_scores is not None:
                row["class_evidence_top_raw_existential_scores"] = (
                    class_evidence_top_raw_existential_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_top_raw_existential_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_top_raw_existential_score_gap"] = gap
                    row["class_evidence_top_raw_existential_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_top_relative_correction_scores is not None:
                row["class_evidence_top_relative_correction_scores"] = (
                    class_evidence_top_relative_correction_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_top_relative_correction_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_top_relative_correction_score_gap"] = gap
                    row[
                        "class_evidence_top_relative_correction_score_negative_class"
                    ] = negative_class
            if class_evidence_top_support_raw_positive_component is not None:
                row["class_evidence_top_support_raw_positive_component"] = (
                    class_evidence_top_support_raw_positive_component[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_top_support_raw_positive_component[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_top_support_raw_positive_component_gap"] = gap
                    row[
                        "class_evidence_top_support_raw_positive_component_negative_class"
                    ] = negative_class
            if class_evidence_top_support_relative_positive_component is not None:
                row["class_evidence_top_support_relative_positive_component"] = (
                    class_evidence_top_support_relative_positive_component[
                        index
                    ].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_top_support_relative_positive_component[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row[
                        "class_evidence_top_support_relative_positive_component_gap"
                    ] = gap
                    row[
                        "class_evidence_top_support_relative_positive_component_negative_class"
                    ] = negative_class
            if class_evidence_top_support_relative_negative_component is not None:
                row["class_evidence_top_support_relative_negative_component"] = (
                    class_evidence_top_support_relative_negative_component[
                        index
                    ].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_top_support_relative_negative_component[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row[
                        "class_evidence_top_support_relative_negative_component_gap"
                    ] = gap
                    row[
                        "class_evidence_top_support_relative_negative_component_negative_class"
                    ] = negative_class
            if class_evidence_top_relative_positive is not None:
                row["class_evidence_top_relative_positive"] = (
                    class_evidence_top_relative_positive[index].tolist()
                )
            if class_evidence_top_relative_negative is not None:
                row["class_evidence_top_relative_negative"] = (
                    class_evidence_top_relative_negative[index].tolist()
                )
            if class_evidence_gate_reliability is not None:
                row["class_evidence_gate_reliability"] = (
                    class_evidence_gate_reliability[index].tolist()
                )
            if class_evidence_gate_reliability_regret is not None:
                row["class_evidence_gate_reliability_regret"] = (
                    class_evidence_gate_reliability_regret[index].tolist()
                )
            if class_evidence_top_margin is not None:
                row["class_evidence_top_margin"] = class_evidence_top_margin[
                    index
                ].tolist()
            if class_evidence_gated_margin is not None:
                row["class_evidence_gated_margin"] = class_evidence_gated_margin[
                    index
                ].tolist()
            if class_evidence_branch_existential_scores is not None:
                row["class_evidence_branch_existential_scores"] = (
                    class_evidence_branch_existential_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_branch_existential_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_branch_existential_score_gap"] = gap
                    row["class_evidence_branch_existential_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_branch_competitive_scores is not None:
                row["class_evidence_branch_competitive_scores"] = (
                    class_evidence_branch_competitive_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_branch_competitive_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_branch_competitive_score_gap"] = gap
                    row["class_evidence_branch_competitive_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_branch_direct_scores is not None:
                row["class_evidence_branch_direct_scores"] = (
                    class_evidence_branch_direct_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_branch_direct_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_branch_direct_score_gap"] = gap
                    row["class_evidence_branch_direct_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_branch_residual_scores is not None:
                row["class_evidence_branch_residual_scores"] = (
                    class_evidence_branch_residual_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_branch_residual_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_branch_residual_score_gap"] = gap
                    row["class_evidence_branch_residual_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_branch_support_scores is not None:
                row["class_evidence_branch_support_scores"] = (
                    class_evidence_branch_support_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_branch_support_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_branch_support_score_gap"] = gap
                    row["class_evidence_branch_support_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_interaction_scores is not None:
                row["class_evidence_interaction_scores"] = (
                    class_evidence_interaction_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_interaction_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_interaction_score_gap"] = gap
                    row["class_evidence_interaction_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_raw_interaction_scores is not None:
                row["class_evidence_raw_interaction_scores"] = (
                    class_evidence_raw_interaction_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_raw_interaction_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_raw_interaction_score_gap"] = gap
                    row["class_evidence_raw_interaction_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_bounded_interaction_scores is not None:
                row["class_evidence_bounded_interaction_scores"] = (
                    class_evidence_bounded_interaction_scores[index].tolist()
                )
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_bounded_interaction_scores[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_bounded_interaction_score_gap"] = gap
                    row["class_evidence_bounded_interaction_score_negative_class"] = (
                        negative_class
                    )
            if class_evidence_logits is not None:
                gap_payload = _true_vs_hardest_gap(
                    class_evidence_logits[index],
                    true_label,
                )
                if gap_payload is not None:
                    gap, negative_class = gap_payload
                    row["class_evidence_total_gap"] = gap
                    row["class_evidence_total_negative_class"] = negative_class
            if class_evidence_branch_scale is not None:
                row["class_evidence_branch_scale"] = float(
                    class_evidence_branch_scale.item()
                )
            if class_evidence_branch_direct_top_scale is not None:
                row["class_evidence_branch_direct_top_scale"] = float(
                    class_evidence_branch_direct_top_scale.item()
                )
            if class_evidence_branch_direct_gated_scale is not None:
                row["class_evidence_branch_direct_gated_scale"] = float(
                    class_evidence_branch_direct_gated_scale.item()
                )
            if class_evidence_branch_direct_raw_scale is not None:
                row["class_evidence_branch_direct_raw_scale"] = float(
                    class_evidence_branch_direct_raw_scale.item()
                )
            if class_evidence_branch_direct_relative_scale is not None:
                row["class_evidence_branch_direct_relative_scale"] = float(
                    class_evidence_branch_direct_relative_scale.item()
                )
            if class_evidence_branch_direct_residual_scale is not None:
                row["class_evidence_branch_direct_residual_scale"] = float(
                    class_evidence_branch_direct_residual_scale.item()
                )
            if class_evidence_top_support_direct_raw_scale is not None:
                row["class_evidence_top_support_direct_raw_scale"] = float(
                    class_evidence_top_support_direct_raw_scale.item()
                )
            if class_evidence_top_support_direct_relative_positive_scale is not None:
                row["class_evidence_top_support_direct_relative_positive_scale"] = (
                    float(
                        class_evidence_top_support_direct_relative_positive_scale.item()
                    )
                )
            if class_evidence_top_support_direct_relative_negative_scale is not None:
                row["class_evidence_top_support_direct_relative_negative_scale"] = (
                    float(
                        class_evidence_top_support_direct_relative_negative_scale.item()
                    )
                )
            if class_evidence_branch_direct_top_weights is not None:
                row["class_evidence_branch_direct_top_weights"] = (
                    class_evidence_branch_direct_top_weights.tolist()
                )
            if class_evidence_branch_direct_gated_weights is not None:
                row["class_evidence_branch_direct_gated_weights"] = (
                    class_evidence_branch_direct_gated_weights.tolist()
                )
            if class_evidence_branch_direct_existential_weights is not None:
                row["class_evidence_branch_direct_existential_weights"] = (
                    class_evidence_branch_direct_existential_weights.tolist()
                )
            if class_evidence_branch_direct_competitive_weights is not None:
                row["class_evidence_branch_direct_competitive_weights"] = (
                    class_evidence_branch_direct_competitive_weights.tolist()
                )
            if class_evidence_interaction_scale is not None:
                row["class_evidence_interaction_scale"] = float(
                    class_evidence_interaction_scale.item()
                )
            if class_evidence_interaction_scale_multiplier is not None:
                row["class_evidence_interaction_scale_multiplier"] = float(
                    class_evidence_interaction_scale_multiplier.item()
                )
            if class_evidence_interaction_effective_scale is not None:
                row["class_evidence_interaction_effective_scale"] = float(
                    class_evidence_interaction_effective_scale.item()
                )
            if class_evidence_embedding_score_bound is not None:
                row["class_evidence_embedding_score_bound"] = float(
                    class_evidence_embedding_score_bound.item()
                )
            if class_evidence_embedding_score_temperature is not None:
                row["class_evidence_embedding_score_temperature"] = float(
                    class_evidence_embedding_score_temperature.item()
                )
            if class_evidence_interaction_score_bound is not None:
                row["class_evidence_interaction_score_bound"] = float(
                    class_evidence_interaction_score_bound.item()
                )
            if class_evidence_interaction_score_temperature is not None:
                row["class_evidence_interaction_score_temperature"] = float(
                    class_evidence_interaction_score_temperature.item()
                )
            if output.class_evidence_scorer_type is not None:
                row["class_evidence_scorer_type"] = output.class_evidence_scorer_type
            if output.class_evidence_scorer_branch_feature_transform_mode is not None:
                row["class_evidence_scorer_branch_feature_transform_mode"] = (
                    output.class_evidence_scorer_branch_feature_transform_mode
                )
            if class_evidence_scorer_branch_feature_transform_temperature is not None:
                row["class_evidence_scorer_branch_feature_transform_temperature"] = (
                    float(
                        class_evidence_scorer_branch_feature_transform_temperature.item()
                    )
                )
            if output.class_gated_branch_logit_feature_mode is not None:
                row["class_gated_branch_logit_feature_mode"] = (
                    output.class_gated_branch_logit_feature_mode
                )
            if output.global_residual_zero_mean_enabled is not None:
                row["global_residual_zero_mean_enabled"] = (
                    output.global_residual_zero_mean_enabled
                )
            if output.global_residual_rebound_enabled is not None:
                row["global_residual_rebound_enabled"] = (
                    output.global_residual_rebound_enabled
                )
            if binary_auxiliary_targets_cpu is not None:
                row["binary_auxiliary_target"] = int(
                    binary_auxiliary_targets_cpu[index].item()
                )
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
        if output.evidence_score_source is not None:
            row["evidence_score_source"] = output.evidence_score_source
        if output.evidence_pooling_type is not None:
            row["evidence_pooling_type"] = output.evidence_pooling_type
        if evidence_gate_weights is not None:
            row["evidence_gate_weights"] = evidence_gate_weights[index].tolist()
        if evidence_gate_entropy is not None:
            row["evidence_gate_entropy"] = float(evidence_gate_entropy[index].item())
        if class_evidence_gate_weights is not None:
            row["class_evidence_gate_weights"] = class_evidence_gate_weights[
                index
            ].tolist()
            true_label = int(batch.labels[index].item())
            if 0 <= true_label < int(class_evidence_gate_weights.shape[1]):
                row["true_class_gate_weights"] = class_evidence_gate_weights[
                    index,
                    true_label,
                ].tolist()
            if 0 <= predicted_label < int(class_evidence_gate_weights.shape[1]):
                row["predicted_class_gate_weights"] = class_evidence_gate_weights[
                    index,
                    predicted_label,
                ].tolist()
        if class_evidence_gate_entropy is not None:
            row["class_evidence_gate_entropy"] = class_evidence_gate_entropy[
                index
            ].tolist()
        if class_evidence_learned_gate_weights is not None:
            row["class_evidence_learned_gate_weights"] = (
                class_evidence_learned_gate_weights[index].tolist()
            )
            true_label = int(batch.labels[index].item())
            if 0 <= true_label < int(class_evidence_learned_gate_weights.shape[1]):
                row["true_class_learned_gate_weights"] = (
                    class_evidence_learned_gate_weights[index, true_label].tolist()
                )
            if 0 <= predicted_label < int(class_evidence_learned_gate_weights.shape[1]):
                row["predicted_class_learned_gate_weights"] = (
                    class_evidence_learned_gate_weights[
                        index,
                        predicted_label,
                    ].tolist()
                )
        if class_evidence_learned_gate_entropy is not None:
            row["class_evidence_learned_gate_entropy"] = (
                class_evidence_learned_gate_entropy[index].tolist()
            )
        if class_evidence_gate_mixing_alpha is not None:
            row["class_evidence_gate_mixing_alpha"] = float(
                class_evidence_gate_mixing_alpha.item()
            )
        if global_residual_scale is not None:
            row["global_residual_scale"] = float(global_residual_scale.item())
        if global_residual_schedule_multiplier is not None:
            row["global_residual_schedule_multiplier"] = float(
                global_residual_schedule_multiplier.item()
            )
        if global_residual_effective_scale is not None:
            row["global_residual_effective_scale"] = float(
                global_residual_effective_scale.item()
            )
        if global_residual_bound is not None:
            row["global_residual_bound"] = float(global_residual_bound.item())
        if global_residual_temperature is not None:
            row["global_residual_temperature"] = float(
                global_residual_temperature.item()
            )
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
