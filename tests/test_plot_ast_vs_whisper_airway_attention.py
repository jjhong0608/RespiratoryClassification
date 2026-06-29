from __future__ import annotations

import csv
from pathlib import Path

import pytest
from src.cli import plot_ast_vs_whisper_airway_attention as comparison


def _write_prediction_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "audio_path",
        "true_label",
        "direct_pred_label",
        "direct_prob_normal",
        "direct_prob_airway",
        "direct_prob_lung_parenchymal",
        "direct_checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_case_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "airway_a": tmp_path / "data" / "Airway" / "airway_a.wav",
        "airway_b": tmp_path / "data" / "Airway" / "airway_b.wav",
        "airway_success": tmp_path / "data" / "Airway" / "airway_success.wav",
        "normal": tmp_path / "data" / "Normal" / "normal.wav",
        "ast_ckpt_0": tmp_path / "ckpts" / "ast_fold0.pt",
        "ast_ckpt_1": tmp_path / "ckpts" / "ast_fold1.pt",
        "whisper_ckpt_0": tmp_path / "ckpts" / "whisper_fold0.pt",
        "whisper_ckpt_1": tmp_path / "ckpts" / "whisper_fold1.pt",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    return paths


def _base_rows(paths: dict[str, Path]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ast_rows = [
        {
            "fold": "fold_0",
            "audio_path": str(paths["airway_a"]),
            "true_label": "Airway",
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.01",
            "direct_prob_airway": "0.97",
            "direct_prob_lung_parenchymal": "0.02",
            "direct_checkpoint": str(paths["ast_ckpt_0"]),
        },
        {
            "fold": "fold_1",
            "audio_path": str(paths["airway_a"]),
            "true_label": "Airway",
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.02",
            "direct_prob_airway": "0.95",
            "direct_prob_lung_parenchymal": "0.03",
            "direct_checkpoint": str(paths["ast_ckpt_1"]),
        },
        {
            "fold": "fold_0",
            "audio_path": str(paths["airway_b"]),
            "true_label": "Airway",
            "direct_pred_label": "Lung_Parenchymal",
            "direct_prob_normal": "0.03",
            "direct_prob_airway": "0.25",
            "direct_prob_lung_parenchymal": "0.72",
            "direct_checkpoint": str(paths["ast_ckpt_0"]),
        },
        {
            "fold": "fold_0",
            "audio_path": str(paths["airway_success"]),
            "true_label": "Airway",
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.04",
            "direct_prob_airway": "0.91",
            "direct_prob_lung_parenchymal": "0.05",
            "direct_checkpoint": str(paths["ast_ckpt_0"]),
        },
        {
            "fold": "fold_0",
            "audio_path": str(paths["normal"]),
            "true_label": "Normal",
            "direct_pred_label": "Normal",
            "direct_prob_normal": "0.99",
            "direct_prob_airway": "0.01",
            "direct_prob_lung_parenchymal": "0.00",
            "direct_checkpoint": str(paths["ast_ckpt_0"]),
        },
    ]
    whisper_rows = [
        {
            "fold": "fold_0",
            "audio_path": str(paths["airway_a"]),
            "true_label": "Airway",
            "direct_pred_label": "Normal",
            "direct_prob_normal": "0.61",
            "direct_prob_airway": "0.31",
            "direct_prob_lung_parenchymal": "0.08",
            "direct_checkpoint": str(paths["whisper_ckpt_0"]),
        },
        {
            "fold": "fold_1",
            "audio_path": str(paths["airway_a"]),
            "true_label": "Airway",
            "direct_pred_label": "Normal",
            "direct_prob_normal": "0.58",
            "direct_prob_airway": "0.37",
            "direct_prob_lung_parenchymal": "0.05",
            "direct_checkpoint": str(paths["whisper_ckpt_1"]),
        },
        {
            "fold": "fold_0",
            "audio_path": str(paths["airway_b"]),
            "true_label": "Airway",
            "direct_pred_label": "Normal",
            "direct_prob_normal": "0.63",
            "direct_prob_airway": "0.27",
            "direct_prob_lung_parenchymal": "0.10",
            "direct_checkpoint": str(paths["whisper_ckpt_0"]),
        },
        {
            "fold": "fold_0",
            "audio_path": str(paths["airway_success"]),
            "true_label": "Airway",
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.20",
            "direct_prob_airway": "0.73",
            "direct_prob_lung_parenchymal": "0.07",
            "direct_checkpoint": str(paths["whisper_ckpt_0"]),
        },
        {
            "fold": "fold_0",
            "audio_path": str(paths["normal"]),
            "true_label": "Normal",
            "direct_pred_label": "Normal",
            "direct_prob_normal": "0.90",
            "direct_prob_airway": "0.08",
            "direct_prob_lung_parenchymal": "0.02",
            "direct_checkpoint": str(paths["whisper_ckpt_0"]),
        },
    ]
    return ast_rows, whisper_rows


def _write_reports(tmp_path: Path) -> tuple[Path, Path]:
    paths = _make_case_files(tmp_path)
    ast_rows, whisper_rows = _base_rows(paths)
    ast_dir = tmp_path / "reports" / "AST_Partial_L1"
    whisper_dir = tmp_path / "reports" / "Whisper_Partial_L1"
    _write_prediction_csv(ast_dir / comparison.PREDICTIONS_FILENAME, ast_rows)
    _write_prediction_csv(whisper_dir / comparison.PREDICTIONS_FILENAME, whisper_rows)
    return ast_dir, whisper_dir


def _config(
    ast_dir: Path,
    whisper_dir: Path,
    out_dir: Path,
    **kwargs: object,
) -> comparison.ComparisonCliConfig:
    return comparison.ComparisonCliConfig(
        ast_report_dir=ast_dir,
        whisper_report_dir=whisper_dir,
        out_dir=out_dir,
        selection=kwargs.get("selection", "ast_correct_whisper_normal"),  # type: ignore[arg-type]
        formats={"html", "json"},
        top_k=2,
        device="cpu",
        limit=kwargs.get("limit"),  # type: ignore[arg-type]
        folds=kwargs.get("folds"),  # type: ignore[arg-type]
        audio_stems=kwargs.get("audio_stems"),  # type: ignore[arg-type]
        dry_run=bool(kwargs.get("dry_run", False)),
        overwrite=bool(kwargs.get("overwrite", False)),
        install_chrome=False,
    )


def test_selector_keeps_fold_level_ast_correct_whisper_normal_cases(
    tmp_path: Path,
) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(ast_dir, whisper_dir, tmp_path / "out")

    cases, total_before_filters = comparison.AirwayCaseSelector(cfg).select()

    assert total_before_filters == 2
    assert len(cases) == 2
    assert [case.fold for case in cases] == ["fold_0", "fold_1"]
    assert {case.audio_stem for case in cases} == {"airway_a"}
    assert all(case.ast_direct_pred_label == "Airway" for case in cases)
    assert all(case.whisper_direct_pred_label == "Normal" for case in cases)
    assert all(case.selection == "ast_correct_whisper_normal" for case in cases)


def test_selector_supports_both_correct_airway_selection(tmp_path: Path) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(
        ast_dir,
        whisper_dir,
        tmp_path / "out",
        selection="both_correct_airway",
    )

    cases, total_before_filters = comparison.AirwayCaseSelector(cfg).select()

    assert total_before_filters == 1
    assert len(cases) == 1
    assert cases[0].audio_stem == "airway_success"
    assert cases[0].selection == "both_correct_airway"
    assert cases[0].ast_direct_pred_label == "Airway"
    assert cases[0].whisper_direct_pred_label == "Airway"


def test_selector_filters_limit_fold_and_audio_stem(tmp_path: Path) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(
        ast_dir,
        whisper_dir,
        tmp_path / "out",
        folds={"fold_1"},
        audio_stems={"airway_a"},
        limit=1,
    )

    cases, total_before_filters = comparison.AirwayCaseSelector(cfg).select()

    assert total_before_filters == 2
    assert len(cases) == 1
    assert cases[0].fold == "fold_1"


def test_dry_run_writes_manifest_without_visualizers(tmp_path: Path) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    out_dir = tmp_path / "out"
    cfg = _config(ast_dir, whisper_dir, out_dir, dry_run=True)

    summary = comparison.AttentionComparisonRunner(cfg).run()

    assert summary.dry_run is True
    assert summary.total_selected_after_filters == 2
    assert summary.selection == "ast_correct_whisper_normal"
    assert summary.executed == 0
    assert (out_dir / comparison.MANIFEST_CSV).exists()
    assert (out_dir / comparison.MANIFEST_JSON).exists()
    manifest = (out_dir / comparison.MANIFEST_CSV).read_text(encoding="utf-8")
    assert "selection" in manifest
    assert "ast_correct_whisper_normal" in manifest
    assert "dry_run_selected" in manifest


def test_runner_calls_ast_and_whisper_visualizers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    out_dir = tmp_path / "out"
    calls: list[tuple[str, Path]] = []

    class FakeAstVisualizer:
        def __init__(self, config: object):
            self.config = config

        def run(self) -> dict[str, str]:
            calls.append(("ast", self.config.out_dir))  # type: ignore[attr-defined]
            self.config.out_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.config.out_dir / "sample_attention_metadata.json").write_text(  # type: ignore[attr-defined]
                "{}",
                encoding="utf-8",
            )
            return {"family": "ast"}

    class FakeWhisperVisualizer:
        def __init__(self, config: object):
            self.config = config

        def run(self) -> dict[str, str]:
            calls.append(("whisper", self.config.out_dir))  # type: ignore[attr-defined]
            self.config.out_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.config.out_dir / "sample_attention_metadata.json").write_text(  # type: ignore[attr-defined]
                "{}",
                encoding="utf-8",
            )
            return {"family": "whisper"}

    monkeypatch.setattr(comparison, "AstAttentionVisualizer", FakeAstVisualizer)
    monkeypatch.setattr(
        comparison,
        "WhisperAttentionVisualizer",
        FakeWhisperVisualizer,
    )
    cfg = _config(ast_dir, whisper_dir, out_dir, limit=1)

    summary = comparison.AttentionComparisonRunner(cfg).run()

    assert summary.executed == 1
    assert summary.skipped == 0
    assert [name for name, _ in calls] == ["ast", "whisper"]
    assert (out_dir / comparison.MANIFEST_CSV).read_text(encoding="utf-8").count(
        "generated"
    ) == 1

    second_summary = comparison.AttentionComparisonRunner(cfg).run()
    assert second_summary.executed == 0
    assert second_summary.skipped == 1


def test_missing_required_column_raises_clear_error(tmp_path: Path) -> None:
    ast_dir = tmp_path / "reports" / "AST_Partial_L1"
    whisper_dir = tmp_path / "reports" / "Whisper_Partial_L1"
    ast_dir.mkdir(parents=True)
    whisper_dir.mkdir(parents=True)
    (ast_dir / comparison.PREDICTIONS_FILENAME).write_text(
        "fold,audio_path,true_label\n",
        encoding="utf-8",
    )
    (whisper_dir / comparison.PREDICTIONS_FILENAME).write_text(
        "fold,audio_path,true_label\n",
        encoding="utf-8",
    )
    cfg = _config(ast_dir, whisper_dir, tmp_path / "out")

    with pytest.raises(ValueError, match="missing required columns"):
        comparison.AirwayCaseSelector(cfg).select()


def test_parse_args_defaults_and_formats() -> None:
    cfg = comparison.parse_args(
        [
            "--ast-report-dir",
            "ast",
            "--whisper-report-dir",
            "whisper",
            "--out-dir",
            "out",
            "--formats",
            "html,json",
            "--selection",
            "both_correct_airway",
            "--fold",
            "fold_0",
            "--audio-stem",
            "sample",
            "--limit",
            "1",
            "--dry-run",
        ]
    )

    assert cfg.formats == {"html", "json"}
    assert cfg.selection == "both_correct_airway"
    assert cfg.folds == {"fold_0"}
    assert cfg.audio_stems == {"sample"}
    assert cfg.limit == 1
    assert cfg.dry_run is True
