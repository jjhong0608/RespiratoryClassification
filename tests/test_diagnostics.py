from __future__ import annotations

import pytest
import torch
from src.data.loaders import ClipBatch
from src.evaluation.diagnostics import build_diagnostic_rows
from src.models.model import AstModelOutput
from src.utils.config import AnalysisOutputConfig


def _batch() -> ClipBatch:
    return ClipBatch(
        input_values=torch.zeros(1, 4, 4),
        labels=torch.tensor([1]),
        audio_paths=("clip.wav",),
        label_names=("abnormal",),
    )


def _output() -> AstModelOutput:
    return AstModelOutput(
        logits=torch.tensor([0.25]),
        pooled_embedding=torch.tensor([[0.1, 0.2]]),
        branch_logits=torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
        branch_binary_logits=torch.tensor([[0.1, -0.2, 0.3, -0.4]]),
        selected_evidence_tokens=torch.ones(1, 8, 2),
        selected_evidence_indices=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        selected_evidence_scores=torch.tensor(
            [[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
        ),
        selected_evidence_branch_ids=torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]),
        evidence_score_source="attention_logit",
    )


def _branch_gated_output() -> AstModelOutput:
    return AstModelOutput(
        logits=torch.tensor([0.25]),
        pooled_embedding=torch.tensor([[0.1, 0.2]]),
        selected_evidence_indices=torch.tensor([[1, 2]]),
        selected_evidence_scores=torch.tensor([[0.8, 0.7]]),
        selected_evidence_branch_ids=torch.tensor([[0, 1]]),
        evidence_score_source="attention_weight",
        evidence_pooling_type="branch_gated",
        evidence_gate_weights=torch.tensor([[0.4, 0.6]]),
        evidence_gate_entropy=torch.tensor([0.673]),
        branch_evidence_norms=torch.tensor([[1.0, 2.0]]),
    )


def _class_aware_output() -> AstModelOutput:
    return AstModelOutput(
        logits=torch.tensor([[0.1, 0.7, -0.2]]),
        pooled_embedding=torch.tensor([[0.1, 0.2]]),
        evidence_pooling_type="class_aware_branch_gated",
        evidence_gate_weights=torch.tensor([[0.3, 0.7]]),
        evidence_gate_entropy=torch.tensor([0.61]),
        class_evidence_logits=torch.tensor([[0.2, 0.8, -0.1]]),
        global_residual_logits=torch.tensor([[0.01, 0.02, 0.03]]),
        bounded_global_residual_logits=torch.tensor([[0.01, 0.02, 0.03]]),
        global_residual_bound=torch.tensor(1.0),
        global_residual_temperature=torch.tensor(1.0),
        class_gated_branch_logits=torch.tensor([[0.12, 0.34, 0.56]]),
        class_gated_branch_logit_features=torch.tensor([[-0.44, -0.22, 0.22]]),
        class_gated_branch_logit_feature_mode="hardest_negative_margin",
        class_top_branch_margin_features=torch.tensor([[0.2, 0.4, 0.6]]),
        class_evidence_scorer_branch_features=torch.tensor(
            [[[-0.44, 0.2], [-0.22, 0.4], [0.22, 0.6]]]
        ),
        class_evidence_scorer_type="two_tower_mlp",
        class_evidence_scorer_branch_feature_transform_mode="tanh",
        class_evidence_scorer_branch_feature_transform_temperature=torch.tensor(1.0),
        class_evidence_gate_weights=torch.tensor(
            [[[0.8, 0.2], [0.25, 0.75], [0.5, 0.5]]]
        ),
        class_evidence_gate_entropy=torch.tensor([[0.50, 0.56, 0.69]]),
        class_evidence_learned_gate_weights=torch.tensor(
            [[[0.7, 0.3], [0.35, 0.65], [0.45, 0.55]]]
        ),
        class_evidence_learned_gate_entropy=torch.tensor([[0.61, 0.65, 0.69]]),
        class_evidence_gate_mixing_alpha=torch.tensor(0.5),
        global_residual_scale=torch.tensor(0.1),
        global_residual_schedule_multiplier=torch.tensor(0.5),
        global_residual_effective_scale=torch.tensor(0.05),
    )


def _dropout_output() -> AstModelOutput:
    return AstModelOutput(
        logits=torch.tensor([0.25]),
        pooled_embedding=torch.tensor([[0.1, 0.2]]),
        selected_evidence_indices=torch.tensor([[1, 2, 3, 4]]),
        selected_evidence_scores=torch.tensor([[0.8, 0.7, 0.6, 0.5]]),
        selected_evidence_branch_ids=torch.tensor([[0, 0, 1, 1]]),
        selected_evidence_dropout_mask=torch.tensor([[True, False, True, True]]),
        selected_evidence_keep_ratio=torch.tensor([0.75]),
    )


def test_diagnostics_always_include_selected_evidence_metadata() -> None:
    rows = build_diagnostic_rows(
        _batch(),
        _output(),
        probabilities=torch.tensor([0.75]),
        predicted_labels=torch.tensor([1]),
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=False,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert row["selected_evidence_indices"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert row["selected_evidence_branch_ids"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(row["selected_evidence_scores"]) == 8
    assert row["evidence_score_source"] == "attention_logit"
    assert "selected_evidence_tokens" not in row
    assert row["branch_binary_logits"] == [
        0.10000000149011612,
        -0.20000000298023224,
        0.30000001192092896,
        -0.4000000059604645,
    ]
    assert len(row["branch_binary_probabilities"]) == 4


def test_diagnostics_include_binary_auxiliary_target_when_available() -> None:
    rows = build_diagnostic_rows(
        _batch(),
        _output(),
        probabilities=torch.tensor([0.75]),
        predicted_labels=torch.tensor([1]),
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=False,
            save_clip_metadata=True,
        ),
        binary_auxiliary_targets=torch.tensor([1]),
    )

    assert rows[0]["binary_auxiliary_target"] == 1


def test_diagnostics_include_selected_evidence_tokens_only_with_embeddings() -> None:
    rows = build_diagnostic_rows(
        _batch(),
        _output(),
        probabilities=torch.tensor([0.75]),
        predicted_labels=torch.tensor([1]),
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=True,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert "pooled_embedding" in row
    assert "selected_evidence_tokens" in row
    assert len(row["selected_evidence_tokens"]) == 8


def test_diagnostics_include_branch_gated_pooling_metadata() -> None:
    rows = build_diagnostic_rows(
        _batch(),
        _branch_gated_output(),
        probabilities=torch.tensor([0.75]),
        predicted_labels=torch.tensor([1]),
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=False,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert row["evidence_pooling_type"] == "branch_gated"
    assert row["evidence_gate_weights"] == [0.4000000059604645, 0.6000000238418579]
    assert row["evidence_gate_entropy"] == pytest.approx(0.673)
    assert row["branch_evidence_norms"] == [1.0, 2.0]


def test_diagnostics_include_class_aware_gate_metadata() -> None:
    rows = build_diagnostic_rows(
        _batch(),
        _class_aware_output(),
        probabilities=torch.tensor([[0.2, 0.7, 0.1]]),
        predicted_labels=torch.tensor([1]),
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=False,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert row["evidence_pooling_type"] == "class_aware_branch_gated"
    assert row["class_evidence_logits"] == [
        0.20000000298023224,
        0.800000011920929,
        -0.10000000149011612,
    ]
    assert row["global_residual_logits"] == [
        0.009999999776482582,
        0.019999999552965164,
        0.029999999329447746,
    ]
    assert row["bounded_global_residual_logits"] == [
        0.009999999776482582,
        0.019999999552965164,
        0.029999999329447746,
    ]
    assert row["global_residual_bound"] == pytest.approx(1.0)
    assert row["global_residual_temperature"] == pytest.approx(1.0)
    assert row["class_gated_branch_logits"] == [
        0.11999999731779099,
        0.3400000035762787,
        0.5600000023841858,
    ]
    assert row["class_gated_branch_logit_features"] == [
        -0.4399999976158142,
        -0.2199999988079071,
        0.2199999988079071,
    ]
    assert row["class_gated_branch_logit_feature_mode"] == ("hardest_negative_margin")
    assert row["class_top_branch_margin_features"] == [
        0.20000000298023224,
        0.4000000059604645,
        0.6000000238418579,
    ]
    assert row["class_evidence_scorer_branch_features"][1] == [
        -0.2199999988079071,
        0.4000000059604645,
    ]
    assert row["class_evidence_scorer_type"] == "two_tower_mlp"
    assert row["class_evidence_scorer_branch_feature_transform_mode"] == "tanh"
    assert row["class_evidence_scorer_branch_feature_transform_temperature"] == 1.0
    assert row["class_evidence_gate_weights"][1] == [
        0.25,
        0.75,
    ]
    assert row["true_class_gate_weights"] == [0.25, 0.75]
    assert row["predicted_class_gate_weights"] == [0.25, 0.75]
    assert row["class_evidence_gate_entropy"] == pytest.approx([0.50, 0.56, 0.69])
    assert row["class_evidence_learned_gate_weights"][1] == [
        0.3499999940395355,
        0.6499999761581421,
    ]
    assert row["true_class_learned_gate_weights"] == [
        0.3499999940395355,
        0.6499999761581421,
    ]
    assert row["predicted_class_learned_gate_weights"] == [
        0.3499999940395355,
        0.6499999761581421,
    ]
    assert row["class_evidence_learned_gate_entropy"] == pytest.approx(
        [0.61, 0.65, 0.69]
    )
    assert row["class_evidence_gate_mixing_alpha"] == pytest.approx(0.5)
    assert row["global_residual_scale"] == pytest.approx(0.1)
    assert row["global_residual_schedule_multiplier"] == pytest.approx(0.5)
    assert row["global_residual_effective_scale"] == pytest.approx(0.05)


def test_diagnostics_include_selected_evidence_dropout_metadata() -> None:
    rows = build_diagnostic_rows(
        _batch(),
        _dropout_output(),
        probabilities=torch.tensor([0.75]),
        predicted_labels=torch.tensor([1]),
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=False,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert row["selected_evidence_dropout_mask"] == [True, False, True, True]
    assert row["selected_evidence_keep_ratio"] == pytest.approx(0.75)
