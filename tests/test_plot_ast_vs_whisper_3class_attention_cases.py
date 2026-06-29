from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.cli import plot_ast_vs_whisper_3class_attention_cases as comparison


FIELDNAMES = [
    "fold",
    "audio_path",
    "true_label",
    "direct_pred_label",
    "direct_prob_normal",
    "direct_prob_airway",
    "direct_prob_lung_parenchymal",
    "direct_checkpoint",
]


def _write_prediction_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    return path


def _probabilities(pred_label: str, confidence: float) -> tuple[str, str, str]:
    low = (1.0 - confidence) / 2.0
    values = {
        "Normal": low,
        "Airway": low,
        "Lung_Parenchymal": low,
    }
    values[pred_label] = confidence
    return (
        f"{values['Normal']:.4f}",
        f"{values['Airway']:.4f}",
        f"{values['Lung_Parenchymal']:.4f}",
    )


def _row(
    *,
    tmp_path: Path,
    family: str,
    fold: str,
    stem: str,
    true_label: str,
    pred_label: str,
    confidence: float,
) -> dict[str, str]:
    wav = _touch(tmp_path / "data" / f"{stem}.wav")
    ckpt = _touch(tmp_path / "ckpts" / family / fold / f"{family}_{fold}.pt")
    prob_normal, prob_airway, prob_lung = _probabilities(pred_label, confidence)
    return {
        "fold": fold,
        "audio_path": str(wav),
        "true_label": true_label,
        "direct_pred_label": pred_label,
        "direct_prob_normal": prob_normal,
        "direct_prob_airway": prob_airway,
        "direct_prob_lung_parenchymal": prob_lung,
        "direct_checkpoint": str(ckpt),
    }


def _write_reports(tmp_path: Path) -> tuple[Path, Path]:
    ast_rows = [
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_0",
            stem="normal_correct",
            true_label="Normal",
            pred_label="Normal",
            confidence=0.80,
        ),
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_1",
            stem="normal_correct",
            true_label="Normal",
            pred_label="Normal",
            confidence=0.99,
        ),
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_0",
            stem="airway_correct",
            true_label="Airway",
            pred_label="Airway",
            confidence=0.98,
        ),
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_0",
            stem="lung_correct",
            true_label="Lung_Parenchymal",
            pred_label="Lung_Parenchymal",
            confidence=0.97,
        ),
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_0",
            stem="normal_wrong",
            true_label="Normal",
            pred_label="Airway",
            confidence=0.93,
        ),
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_0",
            stem="airway_wrong",
            true_label="Airway",
            pred_label="Normal",
            confidence=0.92,
        ),
        _row(
            tmp_path=tmp_path,
            family="ast",
            fold="fold_0",
            stem="lung_wrong",
            true_label="Lung_Parenchymal",
            pred_label="Normal",
            confidence=0.91,
        ),
    ]
    whisper_rows = [
        _row(
            tmp_path=tmp_path,
            family="whisper",
            fold=row["fold"],
            stem=Path(row["audio_path"]).stem,
            true_label=row["true_label"],
            pred_label=(
                "Normal"
                if row["true_label"] == row["direct_pred_label"]
                else row["direct_pred_label"]
            ),
            confidence=0.70,
        )
        for row in ast_rows
    ]
    whisper_rows[1]["direct_prob_normal"] = "0.9600"
    whisper_rows[1]["direct_prob_airway"] = "0.0200"
    whisper_rows[1]["direct_prob_lung_parenchymal"] = "0.0200"
    whisper_rows[2].update(
        {
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.0400",
            "direct_prob_airway": "0.9200",
            "direct_prob_lung_parenchymal": "0.0400",
        }
    )
    whisper_rows[3].update(
        {
            "direct_pred_label": "Lung_Parenchymal",
            "direct_prob_normal": "0.0300",
            "direct_prob_airway": "0.0300",
            "direct_prob_lung_parenchymal": "0.9400",
        }
    )
    whisper_rows[4].update(
        {
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.0450",
            "direct_prob_airway": "0.9100",
            "direct_prob_lung_parenchymal": "0.0450",
        }
    )
    whisper_rows[5].update(
        {
            "direct_pred_label": "Normal",
            "direct_prob_normal": "0.9000",
            "direct_prob_airway": "0.0500",
            "direct_prob_lung_parenchymal": "0.0500",
        }
    )
    whisper_rows[6].update(
        {
            "direct_pred_label": "Airway",
            "direct_prob_normal": "0.0600",
            "direct_prob_airway": "0.8800",
            "direct_prob_lung_parenchymal": "0.0600",
        }
    )

    ast_dir = tmp_path / "reports" / "AST_Partial_L1"
    whisper_dir = tmp_path / "reports" / "Whisper_Partial_L1"
    _write_prediction_csv(ast_dir / comparison.PREDICTIONS_FILENAME, ast_rows)
    _write_prediction_csv(whisper_dir / comparison.PREDICTIONS_FILENAME, whisper_rows)
    return ast_dir, whisper_dir


def _config(
    ast_dir: Path,
    whisper_dir: Path,
    out_dir: Path,
    *,
    selection_sets: tuple[comparison.SelectionSet, ...] = (
        "ast_correct_high_confidence",
    ),
    per_label: int = 1,
    dry_run: bool = False,
    overwrite: bool = False,
) -> comparison.ComparisonCliConfig:
    return comparison.ComparisonCliConfig(
        ast_report_dir=ast_dir,
        whisper_report_dir=whisper_dir,
        out_dir=out_dir,
        selection_sets=selection_sets,
        per_label=per_label,
        formats={"html", "json"},
        top_k=2,
        device="cpu",
        folds=None,
        audio_stems=None,
        dry_run=dry_run,
        overwrite=overwrite,
        install_chrome=False,
    )


def test_confidence_mapping_uses_predicted_label_probability(tmp_path: Path) -> None:
    normal = comparison.PredictionRecord(
        fold="fold_0",
        audio_path=tmp_path / "normal.wav",
        true_label="Normal",
        direct_pred_label="Normal",
        direct_prob_normal=0.91,
        direct_prob_airway=0.05,
        direct_prob_lung_parenchymal=0.04,
        direct_checkpoint=tmp_path / "normal.pt",
    )
    airway = comparison.PredictionRecord(
        fold="fold_0",
        audio_path=tmp_path / "airway.wav",
        true_label="Normal",
        direct_pred_label="Airway",
        direct_prob_normal=0.05,
        direct_prob_airway=0.88,
        direct_prob_lung_parenchymal=0.07,
        direct_checkpoint=tmp_path / "airway.pt",
    )
    lung = comparison.PredictionRecord(
        fold="fold_0",
        audio_path=tmp_path / "lung.wav",
        true_label="Normal",
        direct_pred_label="Lung_Parenchymal",
        direct_prob_normal=0.05,
        direct_prob_airway=0.07,
        direct_prob_lung_parenchymal=0.86,
        direct_checkpoint=tmp_path / "lung.pt",
    )

    assert normal.prediction_confidence == pytest.approx(0.91)
    assert airway.prediction_confidence == pytest.approx(0.88)
    assert lung.prediction_confidence == pytest.approx(0.86)


def test_selector_deduplicates_unique_audio_and_picks_highest_confidence_fold(
    tmp_path: Path,
) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(ast_dir, whisper_dir, tmp_path / "out")

    cases, total_before_filters = comparison.ThreeClassCaseSelector(cfg).select()

    assert total_before_filters == 3
    assert len(cases) == 3
    normal_case = next(case for case in cases if case.true_label == "Normal")
    assert normal_case.fold == "fold_1"
    assert normal_case.audio_stem == "normal_correct"
    assert normal_case.reference_confidence == pytest.approx(0.99)
    assert {case.true_label for case in cases} == {
        "Normal",
        "Airway",
        "Lung_Parenchymal",
    }


def test_wrong_selection_uses_true_label_buckets_and_wrong_pred_confidence(
    tmp_path: Path,
) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(
        ast_dir,
        whisper_dir,
        tmp_path / "out",
        selection_sets=("ast_wrong_high_confidence",),
    )

    cases, _ = comparison.ThreeClassCaseSelector(cfg).select()

    airway_case = next(case for case in cases if case.true_label == "Airway")
    assert airway_case.reference_pred_label == "Normal"
    assert airway_case.reference_confidence == pytest.approx(0.92)
    assert airway_case.correctness == "wrong"


def test_selector_supports_whisper_selection_sets(tmp_path: Path) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(
        ast_dir,
        whisper_dir,
        tmp_path / "out",
        selection_sets=(
            "whisper_correct_high_confidence",
            "whisper_wrong_high_confidence",
        ),
    )

    cases, _ = comparison.ThreeClassCaseSelector(cfg).select()

    assert len(cases) == 6
    assert {case.reference_model for case in cases} == {"Whisper"}
    assert {case.correctness for case in cases} == {"correct", "wrong"}


def test_insufficient_unique_audio_raises_clear_error(tmp_path: Path) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(
        ast_dir,
        whisper_dir,
        tmp_path / "out",
        selection_sets=("ast_wrong_high_confidence",),
        per_label=2,
    )

    with pytest.raises(ValueError, match="Insufficient unique audio"):
        comparison.ThreeClassCaseSelector(cfg).select()


def test_dry_run_writes_manifest_without_visualizers(tmp_path: Path) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    cfg = _config(ast_dir, whisper_dir, tmp_path / "out", dry_run=True)

    summary = comparison.AttentionComparisonRunner(cfg).run()

    assert summary.dry_run is True
    assert summary.total_selected_after_filters == 3
    assert summary.executed == 0
    assert (cfg.out_dir / comparison.MANIFEST_CSV).exists()
    assert {case["status"] for case in summary.cases} == {"dry_run_selected"}


def test_runner_calls_visualizers_and_skips_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ast_dir, whisper_dir = _write_reports(tmp_path)
    calls: list[tuple[str, Path]] = []

    class FakeAstVisualizer:
        def __init__(self, config: object):
            self.config = config

        def run(self) -> dict[str, str]:
            out_dir = self.config.out_dir  # type: ignore[attr-defined]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fake_attention_metadata.json").write_text(
                "{}",
                encoding="utf-8",
            )
            calls.append(("ast", out_dir))
            return {}

    class FakeWhisperVisualizer:
        def __init__(self, config: object):
            self.config = config

        def run(self) -> dict[str, str]:
            out_dir = self.config.out_dir  # type: ignore[attr-defined]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fake_attention_metadata.json").write_text(
                "{}",
                encoding="utf-8",
            )
            calls.append(("whisper", out_dir))
            return {}

    monkeypatch.setattr(comparison, "AstAttentionVisualizer", FakeAstVisualizer)
    monkeypatch.setattr(
        comparison,
        "WhisperAttentionVisualizer",
        FakeWhisperVisualizer,
    )

    cfg = _config(ast_dir, whisper_dir, tmp_path / "out")
    first_summary = comparison.AttentionComparisonRunner(cfg).run()
    second_summary = comparison.AttentionComparisonRunner(cfg).run()

    assert first_summary.executed == 3
    assert first_summary.skipped == 0
    assert second_summary.executed == 0
    assert second_summary.skipped == 3
    assert [name for name, _ in calls] == [
        "ast",
        "whisper",
        "ast",
        "whisper",
        "ast",
        "whisper",
    ]


def test_missing_required_column_raises_clear_error(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "AST_Partial_L1"
    csv_path = report_dir / comparison.PREDICTIONS_FILENAME
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("fold,audio_path\nfold_0,x.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        comparison.PredictionCsvReader(report_dir).read()


def test_parse_args_defaults_and_formats(tmp_path: Path) -> None:
    config = comparison.parse_args(
        [
            "--ast-report-dir",
            str(tmp_path / "ast"),
            "--whisper-report-dir",
            str(tmp_path / "whisper"),
            "--out-dir",
            str(tmp_path / "out"),
            "--formats",
            "html,json",
        ]
    )

    assert config.selection_sets == comparison.DEFAULT_SELECTION_SETS
    assert config.per_label == 5
    assert config.formats == {"html", "json"}

    custom = comparison.parse_args(
        [
            "--ast-report-dir",
            str(tmp_path / "ast"),
            "--whisper-report-dir",
            str(tmp_path / "whisper"),
            "--out-dir",
            str(tmp_path / "out"),
            "--selection-set",
            "whisper_wrong_high_confidence",
            "--formats",
            "html",
        ]
    )
    assert custom.selection_sets == ("whisper_wrong_high_confidence",)
    assert custom.formats == {"html"}
