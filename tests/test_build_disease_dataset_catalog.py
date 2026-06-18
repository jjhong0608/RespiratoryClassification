from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import soundfile as sf
from openpyxl import Workbook
from src.cli.build_disease_dataset_catalog import (
    AUSCULTATION_COLUMN,
    DIAGNOSIS_COLUMN,
    AnnotationCodeParser,
    CatalogWriter,
    DiseaseDatasetCatalogBuilder,
    normalize_filename_stem,
)


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, [0.0, 0.1, -0.1, 0.0], 16000)


def _write_excel(path: Path, rows: list[tuple[str, object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet.append(["filename", AUSCULTATION_COLUMN, DIAGNOSIS_COLUMN])
    for row in rows:
        worksheet.append(list(row))
    workbook.save(path)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "DISEASE_CNUH_DATA"
    excel_root = tmp_path / "Disease_Data"
    out_dir = data_root / "disease_dataset_catalog"

    primary_files = {
        "Normal": ["normal_a"],
        "Airway": ["airway_a"],
        "Lung_Parenchymal": ["lung_a"],
    }
    for label, stems in primary_files.items():
        for stem in stems:
            _write_wav(data_root / label / f"{stem}.wav")
    for stem in ["airway_a", "lung_a"]:
        _write_wav(data_root / "Abnormal" / f"{stem}.wav")

    for split, split_files in {
        "fold_0": {
            "Normal": ["normal_a"],
            "Airway": ["airway_a"],
            "Lung_Parenchymal": [],
        },
        "test": {"Normal": [], "Airway": [], "Lung_Parenchymal": ["lung_a"]},
        "fold_1": {"Normal": [], "Airway": [], "Lung_Parenchymal": []},
        "fold_2": {"Normal": [], "Airway": [], "Lung_Parenchymal": []},
        "fold_3": {"Normal": [], "Airway": [], "Lung_Parenchymal": []},
        "fold_4": {"Normal": [], "Airway": [], "Lung_Parenchymal": []},
    }.items():
        for label in ("Normal", "Airway", "Lung_Parenchymal", "Abnormal"):
            (data_root / "5_Folds" / split / label).mkdir(parents=True, exist_ok=True)
        for label, stems in split_files.items():
            for stem in stems:
                _write_wav(data_root / "5_Folds" / split / label / f"{stem}.wav")
                if label in {"Airway", "Lung_Parenchymal"}:
                    _write_wav(
                        data_root / "5_Folds" / split / "Abnormal" / f"{stem}.wav"
                    )

    _write_excel(excel_root / "Normal.xlsx", [("normal_a", 5, "R/O 5")])
    _write_excel(
        excel_root / "Airway.xlsx",
        [
            ("airway_a", "2 (end exp.)", 3),
            ("airway_excel_only", "2 (R/O early)", 4),
        ],
    )
    _write_excel(excel_root / "Lung_Parenchymal.xlsx", [("lung_a", 7, "2 (R/O early)")])
    return data_root, excel_root, out_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_annotation_parser_extracts_first_integer_code() -> None:
    parser = AnnotationCodeParser(
        code_map={2: "wheezing", 5: "lung cancer"}, column_name="x"
    )

    assert parser.parse("2 (end exp.)").code == 2
    assert parser.parse("2 (R/O early)").label == "wheezing"
    assert parser.parse("R/O 5").code == 5
    assert parser.parse("R/O 5").label == "lung cancer"
    assert parser.parse("unknown").code is None
    assert normalize_filename_stem("sample.wav") == "sample"


def test_catalog_builder_outputs_expected_rows_and_warnings(tmp_path: Path) -> None:
    data_root, excel_root, out_dir = _build_fixture(tmp_path)

    result = DiseaseDatasetCatalogBuilder(
        data_root=data_root,
        excel_root=excel_root,
        out_dir=out_dir,
    ).build()

    rows_by_stem = {row["filename_stem"]: row for row in result.rows}
    assert set(rows_by_stem) == {"normal_a", "airway_a", "lung_a"}
    assert rows_by_stem["normal_a"]["binary_label"] == "Normal"
    assert rows_by_stem["airway_a"]["binary_label"] == "Abnormal"
    assert rows_by_stem["lung_a"]["binary_label"] == "Abnormal"
    assert rows_by_stem["airway_a"]["auscultation_code"] == 2
    assert rows_by_stem["airway_a"]["auscultation_label"] == "wheezing"
    assert rows_by_stem["normal_a"]["diagnosis_code"] == 5
    assert rows_by_stem["normal_a"]["diagnosis_label"] == "lung cancer"
    assert rows_by_stem["airway_a"]["split"] == "fold_0"
    assert rows_by_stem["airway_a"]["split_role"] == "validation"
    assert rows_by_stem["airway_a"]["cv_validation_fold"] == 0
    assert rows_by_stem["airway_a"]["is_test"] is False
    assert rows_by_stem["airway_a"]["appears_in_abnormal_overlay"] is True
    assert rows_by_stem["lung_a"]["split"] == "test"
    assert rows_by_stem["lung_a"]["split_role"] == "test"
    assert rows_by_stem["lung_a"]["cv_validation_fold"] == ""
    assert rows_by_stem["lung_a"]["is_test"] is True
    assert result.summary["row_count"] == 3
    assert result.summary["warning_counts"] == {"excel_row_missing_wav": 1}


def test_catalog_writer_persists_csv_json_outputs(tmp_path: Path) -> None:
    data_root, excel_root, out_dir = _build_fixture(tmp_path)
    result = DiseaseDatasetCatalogBuilder(
        data_root=data_root,
        excel_root=excel_root,
        out_dir=out_dir,
    ).build()

    paths = CatalogWriter(out_dir).write(result)

    catalog_rows = _read_csv(paths["catalog"])
    warning_rows = _read_csv(paths["warnings"])
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert len(catalog_rows) == 3
    assert warning_rows[0]["warning_type"] == "excel_row_missing_wav"
    assert summary["primary_label_counts"] == {
        "Airway": 1,
        "Lung_Parenchymal": 1,
        "Normal": 1,
    }


def test_missing_excel_metadata_stays_in_catalog_with_warning(tmp_path: Path) -> None:
    data_root, excel_root, out_dir = _build_fixture(tmp_path)
    _write_wav(data_root / "Normal" / "normal_missing_excel.wav")
    _write_wav(data_root / "5_Folds" / "fold_1" / "Normal" / "normal_missing_excel.wav")

    result = DiseaseDatasetCatalogBuilder(
        data_root=data_root,
        excel_root=excel_root,
        out_dir=out_dir,
    ).build()

    missing = next(
        row for row in result.rows if row["filename_stem"] == "normal_missing_excel"
    )
    assert missing["has_excel_metadata"] is False
    assert missing["notes"] == "missing_excel_metadata"
    assert any(
        warning.warning_type == "wav_missing_excel_metadata"
        and warning.filename_stem == "normal_missing_excel"
        for warning in result.warnings
    )


def test_abnormal_overlay_mismatch_raises_error(tmp_path: Path) -> None:
    data_root, excel_root, out_dir = _build_fixture(tmp_path)
    (data_root / "5_Folds" / "fold_0" / "Abnormal" / "airway_a.wav").unlink()

    with pytest.raises(ValueError, match="Abnormal overlay differs"):
        DiseaseDatasetCatalogBuilder(
            data_root=data_root,
            excel_root=excel_root,
            out_dir=out_dir,
        ).build()
