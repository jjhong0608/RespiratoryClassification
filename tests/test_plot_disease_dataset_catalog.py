from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pytest
from src.cli.build_disease_dataset_catalog import CATALOG_COLUMNS, WARNING_COLUMNS
from src.cli.plot_disease_dataset_catalog import (
    CatalogAggregationBuilder,
    CatalogVisualSummaryWriter,
    DiseaseCatalogLoader,
    collect_figures,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_fieldnames = list(dict.fromkeys(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=unique_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _catalog_row(
    *,
    stem: str,
    primary_label: str,
    binary_label: str,
    split: str,
    auscultation_label: str,
    diagnosis_label: str,
    is_test: bool,
    appears_in_abnormal_overlay: bool,
) -> dict[str, Any]:
    return {
        "filename_stem": stem,
        "filename": f"{stem}.wav",
        "audio_path": f"/tmp/root/{primary_label}/{stem}.wav",
        "primary_label": primary_label,
        "binary_label": binary_label,
        "is_abnormal": str(binary_label == "Abnormal"),
        "excel_source_file": f"{primary_label}.xlsx",
        "excel_row_number": "2",
        "auscultation_raw": auscultation_label,
        "auscultation_code": "1",
        "auscultation_label": auscultation_label,
        "diagnosis_raw": diagnosis_label,
        "diagnosis_code": "1",
        "diagnosis_label": diagnosis_label,
        "split": split,
        "split_role": "test" if is_test else "validation",
        "cv_validation_fold": "" if is_test else split.removeprefix("fold_"),
        "is_test": str(is_test),
        "split_primary_label": primary_label,
        "split_audio_path": f"/tmp/split/{split}/{primary_label}/{stem}.wav",
        "appears_in_abnormal_overlay": str(appears_in_abnormal_overlay),
        "abnormal_overlay_path": (
            f"/tmp/split/{split}/Abnormal/{stem}.wav"
            if appears_in_abnormal_overlay
            else ""
        ),
        "has_excel_metadata": "True",
        "has_split_membership": "True",
        "label_matches_excel_source": "True",
        "label_matches_split_folder": "True",
        "notes": "",
    }


def _warning_row() -> dict[str, Any]:
    return {
        "warning_type": "excel_row_missing_wav",
        "filename_stem": "airway_extra",
        "filename": "",
        "primary_label": "",
        "source_label": "Airway",
        "path": "",
        "excel_source_file": "Airway.xlsx",
        "excel_row_number": "4",
        "column": "filename",
        "raw_value": "airway_extra",
        "message": "Excel row has no matching wav file",
    }


def _build_fixture(tmp_path: Path) -> Path:
    catalog_dir = tmp_path / "disease_dataset_catalog"
    rows = [
        _catalog_row(
            stem="normal_a",
            primary_label="Normal",
            binary_label="Normal",
            split="fold_0",
            auscultation_label="non specific",
            diagnosis_label="healthy",
            is_test=False,
            appears_in_abnormal_overlay=False,
        ),
        _catalog_row(
            stem="airway_a",
            primary_label="Airway",
            binary_label="Abnormal",
            split="fold_0",
            auscultation_label="wheezing",
            diagnosis_label="COPD",
            is_test=False,
            appears_in_abnormal_overlay=True,
        ),
        _catalog_row(
            stem="lung_a",
            primary_label="Lung_Parenchymal",
            binary_label="Abnormal",
            split="test",
            auscultation_label="Crackle",
            diagnosis_label="IPF",
            is_test=True,
            appears_in_abnormal_overlay=True,
        ),
    ]
    warnings = [_warning_row()]
    summary = {
        "row_count": 3,
        "primary_label_counts": {
            "Normal": 1,
            "Airway": 1,
            "Lung_Parenchymal": 1,
        },
        "binary_label_counts": {"Normal": 1, "Abnormal": 2},
        "split_counts": {"fold_0": 2, "test": 1},
        "auscultation_label_counts": {
            "Crackle": 1,
            "non specific": 1,
            "wheezing": 1,
        },
        "diagnosis_label_counts": {"COPD": 1, "IPF": 1, "healthy": 1},
        "warning_counts": {"excel_row_missing_wav": 1},
        "warning_count": 1,
    }
    _write_csv(catalog_dir / "disease_dataset_catalog.csv", CATALOG_COLUMNS, rows)
    _write_csv(
        catalog_dir / "disease_dataset_catalog_warnings.csv",
        WARNING_COLUMNS,
        warnings,
    )
    (catalog_dir / "disease_dataset_catalog_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return catalog_dir


def test_loader_validates_required_schema(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "bad"
    _write_csv(
        catalog_dir / "disease_dataset_catalog.csv",
        ["filename_stem"],
        [{"filename_stem": "sample"}],
    )
    _write_csv(
        catalog_dir / "disease_dataset_catalog_warnings.csv", WARNING_COLUMNS, []
    )
    (catalog_dir / "disease_dataset_catalog_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Required column"):
        DiseaseCatalogLoader(catalog_dir).load()


def test_loader_rejects_invalid_boolean_values(tmp_path: Path) -> None:
    catalog_dir = _build_fixture(tmp_path)
    catalog_path = catalog_dir / "disease_dataset_catalog.csv"
    rows = list(csv.DictReader(catalog_path.open(encoding="utf-8")))
    rows[0]["is_abnormal"] = "maybe"
    _write_csv(catalog_path, list(rows[0]), rows)

    with pytest.raises(ValueError, match="Invalid boolean value"):
        DiseaseCatalogLoader(catalog_dir).load()


def test_aggregation_counts_crosstab_and_fold_delta(tmp_path: Path) -> None:
    data = DiseaseCatalogLoader(_build_fixture(tmp_path)).load()
    agg = CatalogAggregationBuilder(data.records, data.warnings)

    assert agg.count_by("primary_label")["Normal"] == 1
    assert agg.warning_count_by("source_label")["Airway"] == 1

    rows, cols, counts, ratios = agg.crosstab("binary_label", "auscultation_label")
    abnormal_row = rows.index("Abnormal")
    crackle_col = cols.index("Crackle")
    assert counts[abnormal_row, crackle_col] == 1
    assert ratios[abnormal_row, crackle_col] == pytest.approx(0.5)

    deltas = {row["category"]: row for row in agg.fold_vs_test_delta("primary_label")}
    assert deltas["Lung_Parenchymal"]["test_percent"] == pytest.approx(100.0)
    assert deltas["Lung_Parenchymal"]["delta_percentage_points"] > 0


def test_collect_figures_and_sankey_nodes(tmp_path: Path) -> None:
    data = DiseaseCatalogLoader(_build_fixture(tmp_path)).load()
    figures = collect_figures(data)
    figure_by_stem = {figure.stem: figure for figure in figures}

    assert len(figures) == 28
    assert "21_label_to_auscultation_to_diagnosis_sankey" in figure_by_stem
    assert "28_catalog_file_index" in figure_by_stem
    sankey = figure_by_stem["22_primary_to_binary_sankey"].figure
    node_labels = list(sankey.data[0].node.label)
    assert "Primary: Airway" in node_labels
    assert "Binary: Abnormal" in node_labels


def test_writer_exports_outputs_index_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _write_html(self: go.Figure, path: str | Path) -> None:
        Path(path).write_text("<html></html>", encoding="utf-8")

    def _write_image(self: go.Figure, path: str | Path) -> None:
        Path(path).write_bytes(b"image")

    monkeypatch.setattr(go.Figure, "write_html", _write_html)
    monkeypatch.setattr(go.Figure, "write_image", _write_image)

    data = DiseaseCatalogLoader(_build_fixture(tmp_path)).load()
    figures = collect_figures(data)[:2]

    CatalogVisualSummaryWriter(tmp_path / "out", {"html", "png", "pdf"}).write(
        figures,
        data,
    )

    assert (tmp_path / "out" / "01_dataset_overview_dashboard.html").exists()
    assert (tmp_path / "out" / "01_dataset_overview_dashboard.png").exists()
    assert (tmp_path / "out" / "01_dataset_overview_dashboard.pdf").exists()
    index = tmp_path / "out" / "disease_dataset_visual_summary_index.md"
    manifest = tmp_path / "out" / "disease_dataset_visual_summary_manifest.json"
    assert index.exists()
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["figure_count"] == 2
    assert payload["recomputed_summary"]["row_count"] == 3
