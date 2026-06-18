from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
from src.cli.build_direct_vs_cascade_structure_svg import (
    DirectCascadeStructureBuilder,
    StructureComparisonWriter,
    StructureDiagramConfig,
    parse_bool,
)


def test_svg_contains_required_structure_labels() -> None:
    svg = DirectCascadeStructureBuilder(StructureDiagramConfig()).build_svg()

    for expected in [
        "Direct 3-class",
        "Cascade",
        "Stage 1",
        "Stage 2",
        "Normal vs Abnormal",
        "Airway vs Lung_Parenchymal",
        "Same final label space",
        "Single wav file",
    ]:
        assert expected in svg


def test_error_path_option_controls_error_annotations() -> None:
    with_errors = DirectCascadeStructureBuilder(
        StructureDiagramConfig(show_error_paths=True)
    ).build_svg()
    without_errors = DirectCascadeStructureBuilder(
        StructureDiagramConfig(show_error_paths=False)
    ).build_svg()

    assert "True Normal routed to Abnormal" in with_errors
    assert "Disease routed to Normal" in with_errors
    assert "stroke-dasharray" in with_errors
    assert "True Normal routed to Abnormal" not in without_errors
    assert "Disease routed to Normal" not in without_errors


def test_svg_does_not_include_performance_comparison_terms() -> None:
    svg = DirectCascadeStructureBuilder(StructureDiagramConfig()).build_svg()
    root = ElementTree.fromstring(svg)
    visible_text = " ".join(
        text
        for element in root.iter()
        if element.tag.rsplit("}", maxsplit=1)[-1] in {"title", "desc", "text"}
        for text in element.itertext()
    ).lower()

    for forbidden in ["accuracy", "f1", "auc", "confusion", "performance", "metric"]:
        assert forbidden not in visible_text


def test_writer_creates_svg_html_metadata_and_skips_raster_when_tools_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.cli.build_direct_vs_cascade_structure_svg.shutil.which",
        lambda _name: None,
    )

    config = StructureDiagramConfig(out_dir=tmp_path, export_raster=True)
    metadata = StructureComparisonWriter(config).write()

    svg_path = tmp_path / "direct_vs_cascade_structure_comparison.svg"
    html_path = tmp_path / "direct_vs_cascade_structure_comparison.html"
    metadata_path = tmp_path / "direct_vs_cascade_structure_comparison_metadata.json"

    assert svg_path.exists()
    assert html_path.exists()
    assert metadata_path.exists()
    assert not (tmp_path / "direct_vs_cascade_structure_comparison.png").exists()
    assert metadata["raster_export"]["status"] == "skipped"
    assert "No SVG raster export tool found" in metadata["raster_export"]["reason"]

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["options"]["show_error_paths"] is True
    assert payload["outputs"] == [str(svg_path), str(html_path)]


def test_writer_can_disable_raster_export(tmp_path: Path) -> None:
    config = StructureDiagramConfig(out_dir=tmp_path, export_raster=False)
    metadata = StructureComparisonWriter(config).write()

    assert metadata["raster_export"]["requested"] is False
    assert metadata["raster_export"]["status"] == "skipped"
    assert metadata["raster_export"]["reason"] == "raster export disabled"


def test_parse_bool_accepts_and_rejects_expected_values() -> None:
    assert parse_bool("true") is True
    assert parse_bool("0") is False

    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("maybe")
