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

        output_base = Path(out_base)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for fmt in sorted(formats):
            out_path = output_base.with_suffix(f".{fmt}")
            if fmt == "html":
                fig.write_html(out_path)
            elif fmt in {"png", "pdf"}:
                fig.write_image(out_path)
            else:
                raise ValueError(f"Unsupported format: {fmt}")
            self.logger.info("Wrote %s", out_path)
            written.append(out_path)
        return written
