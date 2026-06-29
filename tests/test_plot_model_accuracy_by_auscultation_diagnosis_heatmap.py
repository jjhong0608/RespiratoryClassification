from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from src.cli import plot_model_accuracy_by_auscultation_diagnosis_heatmap as heatmap


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _catalog_rows(tmp_path: Path) -> list[dict[str, object]]:
    del tmp_path
    return [
        {
            "filename_stem": "normal_lung_cancer",
            "split": "test",
            "is_test": "True",
            "auscultation_label": "non specific",
            "diagnosis_label": "lung cancer",
        },
        {
            "filename_stem": "airway_lung_cancer",
            "split": "test",
            "is_test": "True",
            "auscultation_label": "Crackle",
            "diagnosis_label": "lung cancer",
        },
        {
            "filename_stem": "airway_ild",
            "split": "test",
            "is_test": "True",
            "auscultation_label": "wheezing",
            "diagnosis_label": "ILD except IPF (CPFE included)",
        },
        {
            "filename_stem": "train_ignored",
            "split": "fold_0",
            "is_test": "False",
            "auscultation_label": "rhonchi",
            "diagnosis_label": "COPD",
        },
        {
            "filename_stem": "unknown_axis",
            "split": "test",
            "is_test": "True",
            "auscultation_label": "Obscure",
            "diagnosis_label": "unknown diagnosis",
        },
    ]


def _prediction_rows(tmp_path: Path) -> list[dict[str, object]]:
    wav_dir = tmp_path / "data" / "5_Folds" / "test" / "Airway"
    return [
        {
            "fold": "fold_0",
            "audio_path": wav_dir / "normal_lung_cancer.wav",
            "true_label": "Normal",
            "direct_pred_label": "Normal",
            "direct_correct": "True",
            "cascade_final_pred_label": "Normal",
            "cascade_correct": "True",
        },
        {
            "fold": "fold_1",
            "audio_path": wav_dir / "normal_lung_cancer.wav",
            "true_label": "Normal",
            "direct_pred_label": "Airway",
            "direct_correct": "False",
            "cascade_final_pred_label": "Normal",
            "cascade_correct": "True",
        },
        {
            "fold": "fold_0",
            "audio_path": wav_dir / "airway_lung_cancer.wav",
            "true_label": "Airway",
            "direct_pred_label": "Airway",
            "direct_correct": "True",
            "cascade_final_pred_label": "Normal",
            "cascade_correct": "False",
        },
        {
            "fold": "fold_0",
            "audio_path": wav_dir / "airway_ild.wav",
            "true_label": "Airway",
            "direct_pred_label": "Lung_Parenchymal",
            "direct_correct": "False",
            "cascade_final_pred_label": "Airway",
            "cascade_correct": "True",
        },
        {
            "fold": "fold_0",
            "audio_path": wav_dir / "unknown_axis.wav",
            "true_label": "Airway",
            "direct_pred_label": "Airway",
            "direct_correct": "True",
            "cascade_final_pred_label": "Airway",
            "cascade_correct": "True",
        },
    ]


def _build_fixture(tmp_path: Path) -> heatmap.AccuracyHeatmapConfig:
    catalog = tmp_path / "catalog.csv"
    catalog_fieldnames = [
        "filename_stem",
        "split",
        "is_test",
        "auscultation_label",
        "diagnosis_label",
    ]
    _write_csv(catalog, catalog_fieldnames, _catalog_rows(tmp_path))

    prediction_fieldnames = [
        "fold",
        "audio_path",
        "true_label",
        "direct_pred_label",
        "direct_correct",
        "cascade_final_pred_label",
        "cascade_correct",
    ]
    reports: list[heatmap.ModelReportSpec] = []
    for model_family, stem in (
        ("AST", "ast"),
        ("Whisper", "whisper"),
        ("ResNet50", "resnet50"),
    ):
        path = tmp_path / f"{stem}_predictions.csv"
        _write_csv(path, prediction_fieldnames, _prediction_rows(tmp_path))
        reports.append(heatmap.ModelReportSpec(model_family, stem, path))

    return heatmap.AccuracyHeatmapConfig(
        catalog=catalog,
        model_reports=tuple(reports),
        out_dir=tmp_path / "out",
        formats={"json"},
        install_chrome=False,
    )


def _cell(
    cells: list[heatmap.AccuracyCell],
    *,
    model: str,
    strategy: heatmap.Strategy,
    auscultation: str,
    diagnosis: str,
) -> heatmap.AccuracyCell:
    return next(
        cell
        for cell in cells
        if cell.model_family == model
        and cell.strategy == strategy
        and cell.auscultation_display == auscultation
        and cell.diagnosis_display == diagnosis
    )


def test_diagnosis_group_mapping_and_spans_follow_axis_order() -> None:
    assert heatmap.diagnosis_group_by_diagnosis() == {
        "healthy": "Normal",
        "lung cancer": "Lung Parenchymal",
        "lung nodule": "Lung Parenchymal",
        "pneumonia": "Lung Parenchymal",
        "IPF": "Lung Parenchymal",
        "ILD except IPF": "Lung Parenchymal",
        "COPD": "Airway",
        "asthma": "Airway",
    }

    spans = heatmap.build_diagnosis_group_spans()

    assert [span.group for span in spans] == [
        "Normal",
        "Lung Parenchymal",
        "Airway",
    ]
    assert [(span.start_index, span.end_index_exclusive) for span in spans] == [
        (0, 1),
        (1, 6),
        (6, 8),
    ]
    assert [span.x0_domain for span in spans] == pytest.approx([0.0, 0.125, 0.75])
    assert [span.x1_domain for span in spans] == pytest.approx([0.125, 0.75, 1.0])
    assert [span.center_domain for span in spans] == pytest.approx(
        [0.0625, 0.4375, 0.875]
    )


def test_builder_joins_test_catalog_and_calculates_fold_level_accuracy(
    tmp_path: Path,
) -> None:
    cfg = _build_fixture(tmp_path)

    result = heatmap.AccuracyHeatmapBuilder(cfg).build()

    assert result.input_prediction_row_counts == {
        "AST": 5,
        "Whisper": 5,
        "ResNet50": 5,
    }
    assert result.excluded_prediction_row_counts == {
        "unsupported_auscultation_and_diagnosis": 3
    }
    assert len(result.joined_rows) == 24
    assert len(result.cells) == 192

    direct_cell = _cell(
        result.cells,
        model="AST",
        strategy="direct",
        auscultation="non-specific",
        diagnosis="lung cancer",
    )
    cascade_cell = _cell(
        result.cells,
        model="AST",
        strategy="cascade",
        auscultation="non-specific",
        diagnosis="lung cancer",
    )
    ild_cell = _cell(
        result.cells,
        model="AST",
        strategy="cascade",
        auscultation="wheeze",
        diagnosis="ILD except IPF",
    )
    zero_cell = _cell(
        result.cells,
        model="AST",
        strategy="direct",
        auscultation="rhonchi",
        diagnosis="COPD",
    )

    assert direct_cell.total == 2
    assert direct_cell.correct == 1
    assert direct_cell.accuracy == pytest.approx(0.5)
    assert direct_cell.text == "50.0%<br>1/2"
    assert cascade_cell.total == 2
    assert cascade_cell.correct == 2
    assert cascade_cell.accuracy == pytest.approx(1.0)
    assert ild_cell.total == 1
    assert ild_cell.correct == 1
    assert zero_cell.total == 0
    assert zero_cell.accuracy is None
    assert zero_cell.text == "N/A<br>0/0"


def test_writer_creates_summary_csv_json_and_six_plotly_json_files(
    tmp_path: Path,
) -> None:
    cfg = _build_fixture(tmp_path)
    result = heatmap.AccuracyHeatmapBuilder(cfg).build()

    summary = heatmap.AccuracyHeatmapWriter(cfg).write(result)

    assert summary["summary_cell_count"] == 192
    assert summary["total_input_prediction_rows"] == 15
    assert summary["total_joined_strategy_rows"] == 24
    assert len(summary["figures"]) == 6
    assert (cfg.out_dir / heatmap.ROW_AUDIT_CSV).exists()
    assert (cfg.out_dir / heatmap.SUMMARY_CSV).exists()
    assert (cfg.out_dir / heatmap.SUMMARY_JSON).exists()
    for stem in summary["figures"]:
        assert (cfg.out_dir / f"{stem}.json").exists()

    with (cfg.out_dir / heatmap.SUMMARY_CSV).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 192
    zero = next(
        row
        for row in rows
        if row["model_family"] == "AST"
        and row["strategy"] == "direct"
        and row["auscultation_display"] == "rhonchi"
        and row["diagnosis_display"] == "COPD"
    )
    assert zero["accuracy"] == ""
    assert zero["accuracy_percent"] == ""

    payload = json.loads((cfg.out_dir / heatmap.SUMMARY_JSON).read_text())
    assert payload["auscultation_order"] == list(heatmap.AUSCULTATION_ORDER)
    assert payload["diagnosis_order"] == list(heatmap.DIAGNOSIS_ORDER)
    assert payload["diagnosis_group_by_diagnosis"] == (
        heatmap.diagnosis_group_by_diagnosis()
    )
    assert payload["diagnosis_group_order"] == list(heatmap.DIAGNOSIS_GROUP_ORDER)
    assert payload["diagnosis_group_spans"] == [
        span.to_row() for span in heatmap.build_diagnosis_group_spans()
    ]

    figure_path = cfg.out_dir / "ast_direct_accuracy_by_auscultation_x_diagnosis_heatmap.json"
    figure_payload = json.loads(figure_path.read_text())
    assert figure_payload["layout"]["height"] == 700
    assert figure_payload["layout"]["yaxis"]["domain"] == list(heatmap.Y_AXIS_DOMAIN)
    annotation_texts = [
        annotation["text"] for annotation in figure_payload["layout"]["annotations"]
    ]
    assert annotation_texts == ["Normal", "Lung Parenchymal", "Airway"]
    assert {
        annotation["font"]["size"]
        for annotation in figure_payload["layout"]["annotations"]
    } == {14}
    assert {
        (annotation["xanchor"], annotation["yanchor"], annotation["yshift"])
        for annotation in figure_payload["layout"]["annotations"]
    } == {("center", "middle", 2)}
    rectangles = [
        shape
        for shape in figure_payload["layout"]["shapes"]
        if shape["type"] == "rect"
    ]
    assert len(rectangles) == 3
    assert rectangles[0]["y0"] - heatmap.Y_AXIS_DOMAIN[1] == pytest.approx(0.015)
    assert rectangles[0]["y1"] - rectangles[0]["y0"] == pytest.approx(0.05)
    boundary_lines = [
        shape
        for shape in figure_payload["layout"]["shapes"]
        if shape["type"] == "line"
    ]
    assert len(boundary_lines) == 2
    assert [line["x0"] for line in boundary_lines] == pytest.approx(
        [
            heatmap.X_AXIS_DOMAIN[1] * 0.125,
            heatmap.X_AXIS_DOMAIN[1] * 0.75,
        ]
    )


def test_missing_catalog_join_raises_clear_error(tmp_path: Path) -> None:
    cfg = _build_fixture(tmp_path)
    bad_path = tmp_path / "bad_ast.csv"
    with cfg.model_reports[0].predictions_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["audio_path"] = str(tmp_path / "missing.wav")
    _write_csv(bad_path, list(rows[0]), rows)
    bad_cfg = heatmap.AccuracyHeatmapConfig(
        catalog=cfg.catalog,
        model_reports=(
            heatmap.ModelReportSpec("AST", "ast", bad_path),
            *cfg.model_reports[1:],
        ),
        out_dir=tmp_path / "bad_out",
        formats={"json"},
        install_chrome=False,
    )

    with pytest.raises(ValueError, match="does not join to test catalog"):
        heatmap.AccuracyHeatmapBuilder(bad_cfg).build()


def test_parse_args_defaults_and_formats() -> None:
    cfg = heatmap.parse_args(
        [
            "--catalog",
            "catalog.csv",
            "--ast-predictions",
            "ast.csv",
            "--whisper-predictions",
            "whisper.csv",
            "--resnet50-predictions",
            "resnet.csv",
            "--out-dir",
            "out",
            "--formats",
            "html,json",
        ]
    )

    assert cfg.catalog == Path("catalog.csv")
    assert cfg.formats == {"html", "json"}
    assert [report.model_family for report in cfg.model_reports] == [
        "AST",
        "Whisper",
        "ResNet50",
    ]
