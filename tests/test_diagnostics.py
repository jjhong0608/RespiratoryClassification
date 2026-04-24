from __future__ import annotations

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
        selected_evidence_tokens=torch.ones(1, 8, 2),
        selected_evidence_indices=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        selected_evidence_scores=torch.tensor(
            [[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
        ),
        selected_evidence_branch_ids=torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]),
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
    assert "selected_evidence_tokens" not in row


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
