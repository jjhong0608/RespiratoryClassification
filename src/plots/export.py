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

        base = Path(out_base)
        base.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        if "html" in formats:
            html_path = base.with_suffix(".html")
            fig.write_html(html_path)
            written.append(html_path)

        for suffix in ("png", "pdf"):
            if suffix not in formats:
                continue
            image_path = base.with_suffix(f".{suffix}")
            fig.write_image(image_path)
            written.append(image_path)

        return written
