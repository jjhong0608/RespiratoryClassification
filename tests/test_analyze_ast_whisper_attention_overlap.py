from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from src.cli import analyze_ast_whisper_attention_overlap as overlap


def _write_interval_csv(path: Path, intervals: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "time_start_sec", "time_end_sec"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (start, end) in enumerate(intervals, start=1):
            writer.writerow(
                {
                    "rank": str(index),
                    "time_start_sec": str(start),
                    "time_end_sec": str(end),
                }
            )


def _write_metadata(
    path: Path,
    *,
    wav: Path,
    predicted_label: str,
    max_length: int | None = 1000,
) -> None:
    payload: dict[str, object] = {
        "wav": str(wav),
        "prediction": {"predicted_label": predicted_label},
    }
    if max_length is not None:
        payload["patch_geometry"] = {"max_length": max_length}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_case(
    root: Path,
    case_name: str,
    *,
    ast_intervals: list[tuple[float, float]],
    whisper_intervals: list[tuple[float, float]],
    ast_label: str = "Airway",
    whisper_label: str = "Normal",
    max_length: int | None = 1000,
) -> Path:
    case_dir = root / case_name
    ast_dir = case_dir / "ast"
    whisper_dir = case_dir / "whisper"
    wav = Path("/tmp") / f"{case_name}.wav"
    _write_interval_csv(ast_dir / "sample_top_patches.csv", ast_intervals)
    _write_interval_csv(
        whisper_dir / "sample_top_time_spans.csv",
        whisper_intervals,
    )
    _write_metadata(
        ast_dir / "sample_attention_metadata.json",
        wav=wav,
        predicted_label=ast_label,
        max_length=max_length,
    )
    _write_metadata(
        whisper_dir / "sample_attention_metadata.json",
        wav=wav,
        predicted_label=whisper_label,
        max_length=None,
    )
    return case_dir


def test_interval_union_merges_overlapping_and_duplicate_intervals() -> None:
    intervals = [
        overlap.TimeInterval(0.0, 1.0),
        overlap.TimeInterval(0.0, 1.0),
        overlap.TimeInterval(0.5, 2.0),
        overlap.TimeInterval(3.0, 4.0),
    ]

    merged = overlap.IntervalSet.union(intervals)

    assert merged == [
        overlap.TimeInterval(0.0, 2.0),
        overlap.TimeInterval(3.0, 4.0),
    ]
    assert overlap.IntervalSet.duration(merged) == pytest.approx(3.0)


def test_overlap_metrics_keep_whisper_time_past_ast_max_as_non_overlap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "failure"
    case_dir = _write_case(
        root,
        "fold_0__sample",
        ast_intervals=[(0.0, 2.0), (0.0, 2.0), (5.0, 6.0)],
        whisper_intervals=[(1.0, 3.0), (11.0, 12.0)],
        max_length=1000,
    )
    cfg = overlap.OverlapCliConfig(
        failure_root=root,
        both_correct_root=tmp_path / "both",
        out_dir=tmp_path / "out",
        top_k=10,
        formats={"json"},
        install_chrome=False,
    )
    case = overlap.CaseFileCollector._case_files(
        "ast_correct_whisper_normal",
        case_dir,
    )

    result = overlap.AttentionOverlapAnalyzer(cfg)._analyze_case(case)

    assert result.ast_max_time_sec == pytest.approx(10.0)
    assert result.ast_top10_time_sec == pytest.approx(3.0)
    assert result.whisper_top10_time_sec == pytest.approx(3.0)
    assert result.whisper_inside_ast_time_sec == pytest.approx(2.0)
    assert result.whisper_outside_ast_time_sec == pytest.approx(1.0)
    assert result.overlap_time_sec == pytest.approx(1.0)
    assert result.union_time_sec == pytest.approx(5.0)
    assert result.ast_overlap_ratio == pytest.approx(1.0 / 3.0)
    assert result.whisper_overlap_ratio == pytest.approx(1.0 / 3.0)
    assert result.time_iou == pytest.approx(0.2)
    assert result.time_dice == pytest.approx(1.0 / 3.0)


def test_invalid_or_zero_duration_interval_raises(tmp_path: Path) -> None:
    root = tmp_path / "failure"
    case_dir = _write_case(
        root,
        "fold_0__sample",
        ast_intervals=[(1.0, 1.0)],
        whisper_intervals=[(0.0, 1.0)],
    )
    cfg = overlap.OverlapCliConfig(
        failure_root=root,
        both_correct_root=tmp_path / "both",
        out_dir=tmp_path / "out",
        top_k=10,
        formats={"json"},
        install_chrome=False,
    )
    case = overlap.CaseFileCollector._case_files(
        "ast_correct_whisper_normal",
        case_dir,
    )

    with pytest.raises(ValueError, match="positive duration"):
        overlap.AttentionOverlapAnalyzer(cfg)._analyze_case(case)


def test_cli_writes_case_summary_and_plotly_json(tmp_path: Path) -> None:
    failure_root = tmp_path / "ast_vs_whisper_airway_attention"
    both_root = tmp_path / "ast_vs_whisper_airway_attention_both_correct"
    _write_case(
        failure_root,
        "fold_0__failure",
        ast_intervals=[(0.0, 2.0), (5.0, 6.0)],
        whisper_intervals=[(1.0, 3.0), (11.0, 12.0)],
        whisper_label="Normal",
    )
    _write_case(
        both_root,
        "fold_1__success",
        ast_intervals=[(2.0, 3.0), (7.0, 8.0)],
        whisper_intervals=[(2.5, 3.0), (7.2, 7.4)],
        whisper_label="Airway",
    )
    out_dir = tmp_path / "overlap"
    cfg = overlap.OverlapCliConfig(
        failure_root=failure_root,
        both_correct_root=both_root,
        out_dir=out_dir,
        top_k=10,
        formats={"json"},
        install_chrome=False,
    )

    results, summaries = overlap.AttentionOverlapAnalyzer(cfg).analyze()
    report = overlap.OverlapReportWriter(cfg).write(results, summaries)

    assert report.total_cases == 2
    assert (out_dir / overlap.CASE_CSV).exists()
    assert (out_dir / overlap.SUMMARY_CSV).exists()
    assert (out_dir / overlap.SUMMARY_JSON).exists()
    assert (out_dir / "overlap_metric_distributions.json").exists()
    assert (out_dir / "overlap_scatter.json").exists()
    assert (out_dir / "overlap_time_breakdown.json").exists()

    with (out_dir / overlap.CASE_CSV).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["selection"] for row in rows} == {
        "ast_correct_whisper_normal",
        "both_correct_airway",
    }

    summary_payload = json.loads((out_dir / overlap.SUMMARY_JSON).read_text())
    assert summary_payload["total_cases"] == 2
    assert {row["metric"] for row in summary_payload["summary_records"]} >= {
        "time_iou",
        "whisper_outside_ast_ratio",
    }


def test_parse_args_defaults_and_formats() -> None:
    cfg = overlap.parse_args(
        [
            "--failure-root",
            "failure",
            "--both-correct-root",
            "both",
            "--out-dir",
            "out",
            "--top-k",
            "7",
            "--formats",
            "html,json",
        ]
    )

    assert cfg.failure_root == Path("failure")
    assert cfg.both_correct_root == Path("both")
    assert cfg.out_dir == Path("out")
    assert cfg.top_k == 7
    assert cfg.formats == {"html", "json"}
