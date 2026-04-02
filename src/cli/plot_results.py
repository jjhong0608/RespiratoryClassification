from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.plots.export import PlotlyExportMixin
from src.utils.logging import enable_file_logging, logger


@dataclass(frozen=True)
class MetricAggregate:
    model: str
    metric: str
    mean: float
    std: float
    n: int


class ResultsPlotter(PlotlyExportMixin):
    def __init__(
        self,
        input_path: str | Path,
        out_base: str | Path,
        *,
        error: Literal["std", "sem", "ci95"] = "std",
        cols: int = 3,
        title: str = "Evaluation Metrics (mean ± error)",
    ):
        self.input_path = Path(input_path)
        self.out_base = Path(out_base)
        self.error = error
        self.cols = max(1, cols)
        self.title = title

    def load_results(self) -> dict[str, Any]:
        raw = json.loads(self.input_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("results.json must be a JSON object")
        return raw

    def aggregate(
        self, raw: dict[str, Any]
    ) -> tuple[list[str], list[str], list[MetricAggregate]]:
        # Expected structure:
        # {
        #   "model_name": {
        #      "fold0": {"accuracy": 0.9, ...},
        #      ...
        #   },
        #   ...
        # }
        excludes = {
            "confusion_matrix",
            "positive_class_probability",
            "y_true",
            "y_pred",
            "y_prob",
            "probs",
        }

        rows: list[tuple[str, str, float]] = []
        metric_order: list[str] = []

        for model_name, folds in raw.items():
            if not isinstance(model_name, str):
                continue
            if not isinstance(folds, dict):
                continue
            for _, metrics in folds.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    if not isinstance(metric_name, str) or metric_name in excludes:
                        continue
                    if isinstance(value, bool):
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                    if math.isnan(float(value)) or math.isinf(float(value)):
                        continue
                    rows.append((model_name, metric_name, float(value)))
                    if metric_name not in metric_order:
                        metric_order.append(metric_name)

        if not rows:
            raise ValueError("No scalar numeric metrics found in results.json")

        models = sorted({m for m, _, _ in rows})
        metrics = metric_order

        aggs: list[MetricAggregate] = []
        for model in models:
            for metric in metrics:
                values = [v for m, met, v in rows if m == model and met == metric]
                if not values:
                    continue
                arr = np.asarray(values, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
                n = int(arr.size)
                aggs.append(
                    MetricAggregate(model=model, metric=metric, mean=mean, std=std, n=n)
                )

        return models, metrics, aggs

    def _error_array(self, agg: MetricAggregate) -> float:
        if self.error == "std":
            return agg.std
        if self.error == "sem":
            return agg.std / math.sqrt(max(1, agg.n))
        if self.error == "ci95":
            return 1.96 * (agg.std / math.sqrt(max(1, agg.n)))
        raise ValueError(f"Unsupported error mode: {self.error}")

    def build_figure(
        self, models: list[str], metrics: list[str], aggs: list[MetricAggregate]
    ) -> go.Figure:
        metric_to_model: dict[tuple[str, str], MetricAggregate] = {
            (a.model, a.metric): a for a in aggs
        }
        n_metrics = len(metrics)
        rows = math.ceil(n_metrics / self.cols)
        fig = make_subplots(
            rows=rows,
            cols=self.cols,
            subplot_titles=metrics,
            horizontal_spacing=0.06,
            vertical_spacing=0.12,
        )

        for i, metric in enumerate(metrics):
            r = i // self.cols + 1
            c = i % self.cols + 1
            means: list[float] = []
            errors: list[float] = []
            hover: list[str] = []
            for model in models:
                agg = metric_to_model.get((model, metric))
                if agg is None:
                    means.append(float("nan"))
                    errors.append(0.0)
                    hover.append(f"model={model}<br>metric={metric}<br>missing")
                    continue
                means.append(agg.mean)
                errors.append(self._error_array(agg))
                hover.append(
                    f"model={model}<br>metric={metric}<br>mean={agg.mean:.6f}<br>"
                    f"std={agg.std:.6f}<br>n={agg.n}"
                )

            fig.add_trace(
                go.Scatter(
                    x=models,
                    y=means,
                    mode="markers",
                    error_y={"type": "data", "array": errors, "visible": True},
                    hovertext=hover,
                    hoverinfo="text",
                    showlegend=False,
                    marker={"size": 8},
                ),
                row=r,
                col=c,
            )

        fig.update_layout(
            title=self.title,
            height=max(350, 280 * rows),
            width=max(900, 340 * self.cols),
        )
        fig.update_xaxes(type="category", tickangle=-30)
        return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.json")
    parser.add_argument("--out", default="plots/results_metrics")
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
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument(
        "--install_chrome",
        action="store_true",
        help="If PNG/PDF export fails due to missing Chrome, attempt installing Chrome via kaleido.",
    )
    args = parser.parse_args()

    out_base = Path(args.out)
    enable_file_logging(out_base.parent / "plot_results.log", mode="w")

    plotter = ResultsPlotter(
        input_path=args.input,
        out_base=out_base,
        error=args.error,
        cols=args.cols,
        title=f"Evaluation Metrics (mean ± {args.error})",
    )
    raw = plotter.load_results()
    models, metrics, aggs = plotter.aggregate(raw)
    logger.info(f"Found {len(models)} models and {len(metrics)} metrics.")

    fig = plotter.build_figure(models, metrics, aggs)
    formats = {s.strip() for s in str(args.formats).split(",") if s.strip()}
    plotter.write_outputs(
        fig,
        plotter.out_base,
        formats=formats,
        install_chrome=args.install_chrome,
    )


if __name__ == "__main__":
    main()
