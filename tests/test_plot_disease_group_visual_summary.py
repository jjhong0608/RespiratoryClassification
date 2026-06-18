from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pytest
from src.cli.plot_disease_group_visual_summary import (
    CvContextRecord,
    DiseaseGroupReportLoader,
    FigureSpec,
    FoldMetricRecord,
    LoadedReports,
    PredictionRecord,
    VisualSummaryWriter,
    build_sankey_components,
    error_type_by_label,
    final_outcome_table,
    metric_mean_std,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fold_metric_row(fold: str, direct: float, cascade: float) -> dict[str, Any]:
    row: dict[str, Any] = {"fold": fold}
    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        row[f"direct_{metric}"] = direct
        row[f"cascade_{metric}"] = cascade
    return row


def _prediction_row(
    *,
    fold: str,
    true_label: str,
    direct_pred: str,
    stage1_pred: str,
    stage2_pred: str,
    final_pred: str,
    error_type: str,
) -> dict[str, Any]:
    stage2_executed = stage1_pred == "Abnormal"
    return {
        "fold": fold,
        "audio_path": f"/tmp/{fold}_{true_label}.wav",
        "true_label": true_label,
        "direct_pred_label": direct_pred,
        "direct_correct": str(direct_pred == true_label),
        "direct_prob_normal": 0.7 if direct_pred == "Normal" else 0.1,
        "direct_prob_airway": 0.7 if direct_pred == "Airway" else 0.1,
        "direct_prob_lung_parenchymal": 0.7
        if direct_pred == "Lung_Parenchymal"
        else 0.1,
        "stage1_pred_label": stage1_pred,
        "stage1_prob_normal": 0.8 if stage1_pred == "Normal" else 0.2,
        "stage1_prob_abnormal": 0.8 if stage1_pred == "Abnormal" else 0.2,
        "stage1_threshold": 0.5,
        "stage1_threshold_source": "checkpoint_validation",
        "stage2_executed": str(stage2_executed),
        "stage2_pred_label": stage2_pred if stage2_executed else "",
        "stage2_prob_airway": 0.7
        if stage2_executed and stage2_pred == "Airway"
        else (0.3 if stage2_executed else ""),
        "stage2_prob_lung_parenchymal": 0.7
        if stage2_executed and stage2_pred == "Lung_Parenchymal"
        else (0.3 if stage2_executed else ""),
        "stage2_threshold": 0.5 if stage2_executed else "",
        "stage2_threshold_source": "checkpoint_validation" if stage2_executed else "",
        "cascade_prob_normal": 0.8 if final_pred == "Normal" else 0.1,
        "cascade_prob_airway": 0.8 if final_pred == "Airway" else 0.1,
        "cascade_prob_lung_parenchymal": 0.8
        if final_pred == "Lung_Parenchymal"
        else 0.1,
        "cascade_final_pred_label": final_pred,
        "cascade_correct": str(final_pred == true_label),
        "cascade_error_type": error_type,
        "direct_checkpoint": "direct.pt",
        "stage1_checkpoint": "stage1.pt",
        "stage2_checkpoint": "stage2.pt",
    }


def test_loader_validates_missing_prediction_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_predictions.csv"
    _write_csv(path, [{"fold": "fold_0"}])

    loader = DiseaseGroupReportLoader(tmp_path)

    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_predictions(path)


def test_loader_parses_metrics_and_predictions(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    _write_csv(
        metrics_path,
        [
            _fold_metric_row("fold_0", 0.8, 0.7),
            _fold_metric_row("fold_1", 0.6, 0.9),
        ],
    )
    _write_csv(
        predictions_path,
        [
            _prediction_row(
                fold="fold_0",
                true_label="Normal",
                direct_pred="Normal",
                stage1_pred="Normal",
                stage2_pred="",
                final_pred="Normal",
                error_type="correct_normal",
            ),
            _prediction_row(
                fold="fold_0",
                true_label="Airway",
                direct_pred="Lung_Parenchymal",
                stage1_pred="Abnormal",
                stage2_pred="Airway",
                final_pred="Airway",
                error_type="airway_correct",
            ),
        ],
    )

    loader = DiseaseGroupReportLoader(tmp_path)
    metrics = loader.load_fold_metrics(metrics_path)
    predictions = loader.load_predictions(predictions_path)

    assert metrics[0].values["direct_macro_f1"] == 0.8
    assert metrics[1].values["cascade_brier_score"] == 0.9
    assert predictions[0].stage2_prob_airway is None
    assert predictions[1].stage2_prob_airway == 0.7
    assert predictions[1].cascade_correct is True


def test_metric_and_outcome_aggregation() -> None:
    metrics = [
        FoldMetricRecord(
            fold="fold_0",
            values={"direct_accuracy": 0.8, "cascade_accuracy": 0.6}
            | _metric_fill(0.8, 0.6),
        ),
        FoldMetricRecord(
            fold="fold_1",
            values={"direct_accuracy": 0.6, "cascade_accuracy": 1.0}
            | _metric_fill(0.6, 1.0),
        ),
    ]
    predictions = _records()

    summary = metric_mean_std(metrics)
    outcome = final_outcome_table(predictions, "cascade")
    errors = error_type_by_label(
        predictions,
        ["correct_normal", "airway_correct", "lung_parenchymal_to_airway"],
    )

    assert summary["direct"]["accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["cascade"]["accuracy"]["mean"] == pytest.approx(0.8)
    assert outcome[("Lung_Parenchymal", "Airway")] == 1
    assert errors[("Lung_Parenchymal", "lung_parenchymal_to_airway")] == 1


def test_sankey_components_include_stage_paths() -> None:
    labels, sources, targets, values, hovers = build_sankey_components(_records())

    assert "True: Normal" in labels
    assert "Stage1: Abnormal" in labels
    assert "Final: Airway" in labels
    assert sum(values) >= 3
    assert len(sources) == len(targets) == len(values) == len(hovers)


def test_visual_summary_writer_exports_index_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _write_html(self: go.Figure, path: str | Path) -> None:
        Path(path).write_text("<html></html>", encoding="utf-8")

    def _write_image(self: go.Figure, path: str | Path) -> None:
        Path(path).write_bytes(b"image")

    monkeypatch.setattr(go.Figure, "write_html", _write_html)
    monkeypatch.setattr(go.Figure, "write_image", _write_image)

    data = LoadedReports(
        reports_root=tmp_path,
        fold_metrics_path=tmp_path / "fold_metrics.csv",
        predictions_path=tmp_path / "predictions.csv",
        cascade_summary_path=tmp_path / "summary.json",
        cv_summary_paths=[],
        markdown_paths=[],
        fold_metrics=[
            FoldMetricRecord(fold="fold_0", values=_metric_fill(0.8, 0.7)),
        ],
        predictions=_records(),
        cascade_summary={},
        cv_context=[
            CvContextRecord(
                method="mock",
                metric_means={},
                metric_stds={},
                selected_f1_by_fold={},
            )
        ],
        markdown_captions={},
    )
    figure = FigureSpec(
        stem="figure",
        title="Figure",
        caption="Caption",
        source_files=["predictions.csv"],
        source_columns=["true_label"],
        figure=go.Figure(go.Bar(x=["a"], y=[1])),
    )

    VisualSummaryWriter(tmp_path / "out", {"html", "png", "pdf"}).write([figure], data)

    assert (tmp_path / "out" / "figure.html").exists()
    assert (tmp_path / "out" / "figure.png").exists()
    assert (tmp_path / "out" / "figure.pdf").exists()
    assert (tmp_path / "out" / "visual_summary_index.md").exists()
    assert (tmp_path / "out" / "visual_summary_manifest.json").exists()


def _metric_fill(direct: float, cascade: float) -> dict[str, float]:
    payload: dict[str, float] = {}
    for metric in [
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        payload[f"direct_{metric}"] = direct
        payload[f"cascade_{metric}"] = cascade
    payload.setdefault("direct_accuracy", direct)
    payload.setdefault("cascade_accuracy", cascade)
    return payload


def _records() -> list[PredictionRecord]:
    return [
        PredictionRecord(
            fold="fold_0",
            audio_path="/tmp/n.wav",
            true_label="Normal",
            direct_pred_label="Normal",
            direct_correct=True,
            direct_probabilities={
                "Normal": 0.9,
                "Airway": 0.05,
                "Lung_Parenchymal": 0.05,
            },
            stage1_pred_label="Normal",
            stage1_prob_normal=0.9,
            stage1_prob_abnormal=0.1,
            stage1_threshold=0.5,
            stage1_threshold_source="checkpoint_validation",
            stage2_executed=False,
            stage2_pred_label="",
            stage2_prob_airway=None,
            stage2_prob_lung_parenchymal=None,
            stage2_threshold=None,
            stage2_threshold_source="",
            cascade_probabilities={
                "Normal": 0.9,
                "Airway": 0.05,
                "Lung_Parenchymal": 0.05,
            },
            cascade_final_pred_label="Normal",
            cascade_correct=True,
            cascade_error_type="correct_normal",
            direct_checkpoint="d.pt",
            stage1_checkpoint="s1.pt",
            stage2_checkpoint="s2.pt",
        ),
        PredictionRecord(
            fold="fold_0",
            audio_path="/tmp/a.wav",
            true_label="Airway",
            direct_pred_label="Airway",
            direct_correct=True,
            direct_probabilities={
                "Normal": 0.1,
                "Airway": 0.8,
                "Lung_Parenchymal": 0.1,
            },
            stage1_pred_label="Abnormal",
            stage1_prob_normal=0.1,
            stage1_prob_abnormal=0.9,
            stage1_threshold=0.5,
            stage1_threshold_source="checkpoint_validation",
            stage2_executed=True,
            stage2_pred_label="Airway",
            stage2_prob_airway=0.8,
            stage2_prob_lung_parenchymal=0.2,
            stage2_threshold=0.5,
            stage2_threshold_source="checkpoint_validation",
            cascade_probabilities={
                "Normal": 0.1,
                "Airway": 0.8,
                "Lung_Parenchymal": 0.1,
            },
            cascade_final_pred_label="Airway",
            cascade_correct=True,
            cascade_error_type="airway_correct",
            direct_checkpoint="d.pt",
            stage1_checkpoint="s1.pt",
            stage2_checkpoint="s2.pt",
        ),
        PredictionRecord(
            fold="fold_0",
            audio_path="/tmp/l.wav",
            true_label="Lung_Parenchymal",
            direct_pred_label="Lung_Parenchymal",
            direct_correct=True,
            direct_probabilities={
                "Normal": 0.1,
                "Airway": 0.1,
                "Lung_Parenchymal": 0.8,
            },
            stage1_pred_label="Abnormal",
            stage1_prob_normal=0.1,
            stage1_prob_abnormal=0.9,
            stage1_threshold=0.5,
            stage1_threshold_source="checkpoint_validation",
            stage2_executed=True,
            stage2_pred_label="Airway",
            stage2_prob_airway=0.7,
            stage2_prob_lung_parenchymal=0.3,
            stage2_threshold=0.5,
            stage2_threshold_source="checkpoint_validation",
            cascade_probabilities={
                "Normal": 0.1,
                "Airway": 0.7,
                "Lung_Parenchymal": 0.2,
            },
            cascade_final_pred_label="Airway",
            cascade_correct=False,
            cascade_error_type="lung_parenchymal_to_airway",
            direct_checkpoint="d.pt",
            stage1_checkpoint="s1.pt",
            stage2_checkpoint="s2.pt",
        ),
    ]
