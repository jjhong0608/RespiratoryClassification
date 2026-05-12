from __future__ import annotations

import pytest
import torch
from src.data.fsd50k_dataset import Fsd50kBatch
from src.data.loaders import ClipBatch
from src.evaluation.diagnostics import (
    build_diagnostic_rows,
    build_multilabel_diagnostic_rows,
)
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


def test_multilabel_diagnostics_use_compact_default_fields() -> None:
    batch = Fsd50kBatch(
        input_values=torch.zeros(1, 4, 4),
        labels=torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        clip_ids=("clip_001",),
        audio_paths=("clip_001.wav",),
    )
    output = AstModelOutput(
        logits=torch.tensor([[4.0, -1.0, 0.2, 3.0, 2.0, -2.0]]),
        pooled_embedding=torch.tensor([[0.1, 0.2]]),
        selected_evidence_indices=torch.tensor([[1, 2]]),
        selected_evidence_scores=torch.tensor([[0.8, 0.7]]),
        selected_evidence_branch_ids=torch.tensor([[0, 1]]),
        evidence_score_source="attention_weight",
    )
    rows = build_multilabel_diagnostic_rows(
        batch,
        output,
        probabilities=torch.sigmoid(output.logits),
        class_names=("a", "b", "c", "d", "e", "f"),
        threshold=0.5,
        analysis=AnalysisOutputConfig(
            save_logits=False,
            save_probabilities=False,
            save_embeddings=False,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert row["clip_id"] == "clip_001"
    assert row["audio_path"] == "clip_001.wav"
    assert row["true_label_indices"] == [0, 2]
    assert row["true_label_names"] == ["a", "c"]
    assert row["predicted_label_indices"] == [0, 2, 3, 4]
    assert row["predicted_label_names"] == ["a", "c", "d", "e"]
    assert row["top5_label_names"][0] == "a"
    assert row["threshold"] == pytest.approx(0.5)
    assert row["selected_evidence_indices"] == [1, 2]
    assert row["evidence_score_source"] == "attention_weight"
    assert "logits" not in row
    assert "probabilities" not in row
    assert "pooled_embedding" not in row


def test_multilabel_diagnostics_respect_output_toggles() -> None:
    batch = Fsd50kBatch(
        input_values=torch.zeros(1, 4, 4),
        labels=torch.tensor([[1.0, 0.0, 0.0]]),
        clip_ids=("clip_002",),
        audio_paths=("clip_002.wav",),
    )
    output = AstModelOutput(
        logits=torch.tensor([[1.0, -1.0, 0.0]]),
        pooled_embedding=torch.tensor([[0.1, 0.2]]),
        branch_logits=torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
        selected_evidence_tokens=torch.ones(1, 2, 2),
        selected_evidence_indices=torch.tensor([[1, 2]]),
        selected_evidence_scores=torch.tensor([[0.8, 0.7]]),
        selected_evidence_branch_ids=torch.tensor([[0, 1]]),
        evidence_pooling_type="branch_gated",
        evidence_gate_weights=torch.tensor([[0.4, 0.6]]),
        evidence_gate_entropy=torch.tensor([0.673]),
        branch_evidence_norms=torch.tensor([[1.0, 2.0]]),
    )
    rows = build_multilabel_diagnostic_rows(
        batch,
        output,
        probabilities=torch.sigmoid(output.logits),
        class_names=("a", "b", "c"),
        threshold=0.5,
        analysis=AnalysisOutputConfig(
            save_logits=True,
            save_probabilities=True,
            save_embeddings=True,
            save_clip_metadata=True,
        ),
    )

    row = rows[0]
    assert "logits" in row
    assert "branch_logits" in row
    assert "probabilities" in row
    assert "pooled_embedding" in row
    assert "selected_evidence_tokens" in row
    assert row["evidence_pooling_type"] == "branch_gated"
    assert row["evidence_gate_entropy"] == pytest.approx(0.673)
