from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from src.utils.logging import LoggingMixin


class PlotlyExportMixin(LoggingMixin):
    def write_outputs(
        self,
        fig: go.Figure,
        out_base: str | Path,
        *,
        formats: set[str],
        install_chrome: bool = False,
    ) -> list[Path]:
        del install_chrome

        out_base_path = Path(out_base)
        out_base_path.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        if "html" in formats:
            html_path = out_base_path.with_suffix(".html")
            fig.write_html(html_path)
            written.append(html_path)

        for image_format in ("png", "pdf"):
            if image_format not in formats:
                continue
            image_path = out_base_path.with_suffix(f".{image_format}")
            fig.write_image(image_path)
            written.append(image_path)

        return written
