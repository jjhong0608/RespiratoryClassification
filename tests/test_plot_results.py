from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.cli.plot_results import ResultsPlotter


def test_plot_results_aggregate_models_metrics(tmp_path: Path) -> None:
    raw = {
        "m1": {
            "fold0": {"accuracy": 0.5, "f1_score": 0.6},
            "fold1": {"accuracy": 0.7, "f1_score": 0.8},
        },
        "m2": {
            "fold0": {"accuracy": 0.2, "f1_score": 0.4},
            "fold1": {"accuracy": 0.4, "f1_score": 0.6},
        },
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(raw), encoding="utf-8")

    plotter = ResultsPlotter(input_path=p, out_base=tmp_path / "out")
    loaded = plotter.load_results()
    models, metrics, aggs = plotter.aggregate(loaded)

    assert models == ["m1", "m2"]
    assert "accuracy" in metrics
    assert "f1_score" in metrics

    agg_map = {(a.model, a.metric): a for a in aggs}
    assert agg_map[("m1", "accuracy")].mean == pytest.approx(0.6)
    assert agg_map[("m2", "accuracy")].mean == pytest.approx(0.3)


def test_plot_results_excludes_non_scalar_metrics(tmp_path: Path) -> None:
    raw = {
        "m1": {
            "fold0": {
                "accuracy": 0.5,
                "confusion_matrix": [[1, 0], [0, 1]],
                "positive_class_probability": [0.1, 0.9],
            }
        }
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(raw), encoding="utf-8")

    plotter = ResultsPlotter(input_path=p, out_base=tmp_path / "out")
    loaded = plotter.load_results()
    _, metrics, _ = plotter.aggregate(loaded)
    assert metrics == ["accuracy"]
