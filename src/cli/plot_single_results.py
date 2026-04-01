from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from src.plots.export import PlotlyExportMixin
from src.utils.logging import enable_file_logging, logger

REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "brier_score",
]


@dataclass(frozen=True)
class MetricAggregate:
    metric: str
    mean: float
    std: float
    n: int


class SingleResultsPlotter(PlotlyExportMixin):
    def __init__(
        self,
        input_path: str | Path,
        out_base: str | Path,
        *,
        error: Literal["std", "sem", "ci95"] = "std",
        title: str = "Single Result Metrics (mean ± error)",
    ):
        self.input_path = Path(input_path)
        self.out_base = Path(out_base)
        self.error = error
        self.title = title

    def load_results(self) -> dict[str, Any]:
        raw = json.loads(self.input_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("results file must be a JSON object")
        return raw

    def _resolve_folds(self, raw: dict[str, Any]) -> dict[str, Any]:
        if all(str(k).startswith("fold") for k in raw):
            return raw

        if len(raw) == 1:
            only_value = next(iter(raw.values()))
            if isinstance(only_value, dict) and all(
                str(k).startswith("fold") for k in only_value
            ):
                self.logger.info(
                    "Detected single top-level wrapper key; using inner fold results."
                )
                return only_value

        raise ValueError(
            "Expected fold-wise JSON structure like {'fold0': {...}, 'fold1': {...}}."
        )

    def aggregate(self, raw: dict[str, Any]) -> tuple[list[str], list[MetricAggregate]]:
        folds = self._resolve_folds(raw)

        metric_values: dict[str, list[float]] = {m: [] for m in REQUIRED_METRICS}
        for _, metrics in folds.items():
            if not isinstance(metrics, dict):
                continue
            for metric in REQUIRED_METRICS:
                value = metrics.get(metric)
                if isinstance(value, bool):
                    continue
                if value is None:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                value_f = float(value)
                if math.isnan(value_f) or math.isinf(value_f):
                    continue
                metric_values[metric].append(value_f)

        missing = [m for m in REQUIRED_METRICS if not metric_values[m]]
        if missing:
            raise ValueError(f"Missing required metrics with numeric values: {missing}")

        aggs: list[MetricAggregate] = []
        for metric in REQUIRED_METRICS:
            arr = np.asarray(metric_values[metric], dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
            aggs.append(
                MetricAggregate(metric=metric, mean=mean, std=std, n=int(arr.size))
            )

        return REQUIRED_METRICS, aggs

    def _error_array(self, agg: MetricAggregate) -> float:
        if self.error == "std":
            return agg.std
        if self.error == "sem":
            return agg.std / math.sqrt(max(1, agg.n))
        if self.error == "ci95":
            return 1.96 * (agg.std / math.sqrt(max(1, agg.n)))
        raise ValueError(f"Unsupported error mode: {self.error}")

    @staticmethod
    def _padded_range(
        values_low: list[float], values_high: list[float]
    ) -> tuple[float, float]:
        low = min(values_low)
        high = max(values_high)
        span = high - low
        pad = max(span * 0.08, 1e-6)
        return (low - pad, high + pad)

    def build_figure(
        self, metrics: list[str], aggs: list[MetricAggregate]
    ) -> go.Figure:
        agg_map = {a.metric: a for a in aggs}
        higher_is_better_metrics = [m for m in metrics if m != "brier_score"]
        higher_means: list[float] = []
        higher_errors: list[float] = []
        higher_hover: list[str] = []

        for metric in higher_is_better_metrics:
            agg = agg_map[metric]
            err = self._error_array(agg)
            higher_means.append(agg.mean)
            higher_errors.append(err)
            higher_hover.append(
                f"metric={metric}<br>mean={agg.mean:.6f}<br>"
                f"std={agg.std:.6f}<br>n={agg.n}<br>{self.error}={err:.6f}<br>"
                "direction=higher is better"
            )

        brier_agg = agg_map["brier_score"]
        brier_error = self._error_array(brier_agg)
        brier_hover = (
            f"metric=brier_score<br>mean={brier_agg.mean:.6f}<br>"
            f"std={brier_agg.std:.6f}<br>n={brier_agg.n}<br>"
            f"{self.error}={brier_error:.6f}<br>direction=lower is better"
        )
        left_lows = [m - e for m, e in zip(higher_means, higher_errors, strict=False)]
        left_highs = [m + e for m, e in zip(higher_means, higher_errors, strict=False)]
        left_range = self._padded_range(left_lows, left_highs)
        brier_lows = [brier_agg.mean - brier_error]
        brier_highs = [brier_agg.mean + brier_error]
        brier_range_numeric = self._padded_range(brier_lows, brier_highs)
        brier_range_reversed = [brier_range_numeric[1], brier_range_numeric[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=higher_is_better_metrics,
                y=higher_means,
                error_y={"type": "data", "array": higher_errors, "visible": True},
                hovertext=higher_hover,
                hoverinfo="text",
                name="Higher is better",
                marker={"color": "#1f77b4"},
            )
        )
        fig.add_trace(
            go.Bar(
                x=["brier_score"],
                y=[brier_agg.mean],
                error_y={"type": "data", "array": [brier_error], "visible": True},
                hovertext=[brier_hover],
                hoverinfo="text",
                name="Brier score (lower is better)",
                yaxis="y2",
                marker={"color": "#d62728"},
            )
        )

        fig.update_layout(
            title=self.title,
            xaxis_title="Metric",
            yaxis={
                "title": f"Score (higher is better, mean ± {self.error})",
                "range": list(left_range),
            },
            yaxis2={
                "title": f"Brier score (lower is better, mean ± {self.error})",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "range": brier_range_reversed,
            },
            height=550,
            width=1200,
            barmode="group",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        fig.update_xaxes(
            type="category", tickangle=-30, categoryorder="array", categoryarray=metrics
        )
        return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/single_results.json")
    parser.add_argument("--out", default="plots/single_results_metrics")
    parser.add_argument(
        "--formats",
        default="html,png,pdf",
        help="Comma-separated list: html,png,pdf",
    )
    parser.add_argument(
        "--error",
        default="std",
        choices=["std", "sem", "ci95"],
        help="Error bar mode across folds.",
    )
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom figure title.",
    )
    args = parser.parse_args()

    out_base = Path(args.out)
    enable_file_logging(out_base.parent / "plot_single_results.log", mode="w")

    title = (
        str(args.title)
        if args.title is not None
        else f"Single Result Metrics (mean ± {args.error})"
    )
    plotter = SingleResultsPlotter(
        input_path=args.input,
        out_base=out_base,
        error=args.error,
        title=title,
    )
    raw = plotter.load_results()
    metrics, aggs = plotter.aggregate(raw)
    logger.info(f"Found {len(metrics)} metrics.")

    fig = plotter.build_figure(metrics, aggs)
    formats = {s.strip() for s in str(args.formats).split(",") if s.strip()}
    plotter.write_outputs(
        fig,
        plotter.out_base,
        formats=formats,
        install_chrome=args.install_chrome,
    )


if __name__ == "__main__":
    main()
