from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from src.utils.logging import LoggingMixin, enable_file_logging, logger

DEFAULT_OUT_DIR = Path("Disease_Group_Results/reports/structure_comparison")
FIGURE_STEM = "direct_vs_cascade_structure_comparison"
CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 1100
LABEL_COLORS = {
    "Normal": "#2f80a7",
    "Airway": "#d97925",
    "Lung_Parenchymal": "#c44e52",
    "Abnormal": "#6b7280",
    "Direct": "#2a6fbb",
    "Cascade": "#4b5563",
    "Error": "#b94a48",
}


@dataclass(frozen=True)
class StructureDiagramConfig:
    out_dir: Path = DEFAULT_OUT_DIR
    show_error_paths: bool = True
    export_raster: bool = True


@dataclass(frozen=True)
class RasterExportResult:
    requested: bool
    status: str
    tool: str | None
    outputs: list[str]
    reason: str


class DirectCascadeStructureBuilder:
    def __init__(self, config: StructureDiagramConfig):
        self.config = config

    def build_svg(self) -> str:
        parts = [
            self._header(),
            self._defs(),
            self._background(),
            self._common_input(),
            self._direct_branch(),
            self._cascade_branch(),
            self._shared_final_space(),
        ]
        if self.config.show_error_paths:
            parts.append(self._cascade_error_paths())
        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _header() -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" '
            f'height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
            "\n"
            '<title id="title">Direct 3-class vs Cascade structure comparison</title>'
            "\n"
            '<desc id="desc">Schematic comparison of direct three-class '
            "classification and cascade routing for respiratory disease labels.</desc>"
        )

    @staticmethod
    def _defs() -> str:
        return """
<defs>
  <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L12,5 L0,10 z" fill="#334155"/>
  </marker>
  <marker id="arrow-blue" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L12,5 L0,10 z" fill="#2a6fbb"/>
  </marker>
  <marker id="arrow-red" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L12,5 L0,10 z" fill="#b94a48"/>
  </marker>
  <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="10" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.12"/>
  </filter>
  <style>
    .title { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 34px; font-weight: 800; fill: #172554; }
    .panel-title { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 30px; font-weight: 800; fill: #111827; }
    .box-title { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 22px; font-weight: 800; fill: #0f172a; }
    .box-line { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 500; fill: #334155; }
    .small { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 15px; font-weight: 600; fill: #475569; }
    .pill { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 800; fill: #ffffff; }
    .arrow-label { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 800; fill: #334155; }
    .error-text { font-family: Avenir, Helvetica, Arial, sans-serif; font-size: 15px; font-weight: 800; fill: #8a2f2d; }
  </style>
</defs>
""".strip()

    @staticmethod
    def _background() -> str:
        return """
<rect x="0" y="0" width="1800" height="1100" fill="#f8fafc"/>
<text x="900" y="48" text-anchor="middle" class="title">Direct 3-class vs Cascade: Classification Structure</text>
<text x="900" y="82" text-anchor="middle" class="small">Same respiratory wav input, different decision routing, same final three-label output space</text>
<rect x="70" y="175" width="780" height="745" rx="28" fill="#eef6ff" stroke="#9cc3e6" stroke-width="3"/>
<rect x="950" y="175" width="780" height="745" rx="28" fill="#f7f3ed" stroke="#d7c7aa" stroke-width="3"/>
<text x="460" y="225" text-anchor="middle" class="panel-title">Direct 3-class</text>
<text x="1340" y="225" text-anchor="middle" class="panel-title">Cascade</text>
""".strip()

    def _common_input(self) -> str:
        return "\n".join(
            [
                self._box(
                    760,
                    105,
                    280,
                    76,
                    "Same input",
                    ["Single wav file"],
                    fill="#ffffff",
                    stroke="#64748b",
                ),
                self._arrow(900, 181, 460, 285, stroke="#2a6fbb", marker="arrow-blue"),
                self._arrow(900, 181, 1340, 285, stroke="#4b5563"),
            ]
        )

    def _direct_branch(self) -> str:
        output_y = 740
        return "\n".join(
            [
                self._box(
                    210,
                    285,
                    500,
                    118,
                    "AST encoder + 3-class classifier",
                    ["One model, one decision stage"],
                    fill="#ffffff",
                    stroke="#2a6fbb",
                ),
                self._arrow(460, 403, 460, 496, stroke="#2a6fbb", marker="arrow-blue"),
                self._box(
                    210,
                    496,
                    500,
                    128,
                    "Softmax over 3 labels",
                    ["Normal", "Airway", "Lung_Parenchymal"],
                    fill="#ffffff",
                    stroke="#2a6fbb",
                ),
                self._arrow(460, 624, 460, 700, stroke="#2a6fbb", marker="arrow-blue"),
                self._pill(
                    140, output_y, 190, 68, "Final Normal", LABEL_COLORS["Normal"]
                ),
                self._pill(
                    365, output_y, 190, 68, "Final Airway", LABEL_COLORS["Airway"]
                ),
                self._pill(
                    590,
                    output_y,
                    210,
                    68,
                    "Final Lung_Parenchymal",
                    LABEL_COLORS["Lung_Parenchymal"],
                ),
                self._polyline(
                    [(460, 700), (235, 700), (235, output_y)],
                    stroke="#2a6fbb",
                    marker="arrow-blue",
                ),
                self._arrow(
                    460, 700, 460, output_y, stroke="#2a6fbb", marker="arrow-blue"
                ),
                self._polyline(
                    [(460, 700), (695, 700), (695, output_y)],
                    stroke="#2a6fbb",
                    marker="arrow-blue",
                ),
                self._note(
                    185,
                    835,
                    550,
                    "Direct path",
                    "Every sample passes through the same model and directly receives one final label.",
                    "#e0f2fe",
                    "#2a6fbb",
                ),
            ]
        )

    def _cascade_branch(self) -> str:
        return "\n".join(
            [
                self._box(
                    1090,
                    285,
                    500,
                    118,
                    "Stage 1 AST",
                    ["Normal vs Abnormal", "Gate decides whether Stage 2 runs"],
                    fill="#ffffff",
                    stroke="#6b7280",
                ),
                self._polyline(
                    [(1340, 403), (1115, 470), (1115, 535)],
                    stroke="#2f80a7",
                    marker="arrow",
                ),
                self._text(1080, 480, "Stage 1 = Normal", css_class="arrow-label"),
                self._pill(995, 535, 240, 72, "Final Normal", LABEL_COLORS["Normal"]),
                self._polyline(
                    [(1340, 403), (1510, 470), (1510, 525)],
                    stroke="#6b7280",
                    marker="arrow",
                ),
                self._text(1510, 480, "Stage 1 = Abnormal", css_class="arrow-label"),
                self._box(
                    1345,
                    525,
                    330,
                    126,
                    "Stage 2 AST",
                    ["Airway vs Lung_Parenchymal", "Executed only for Abnormal"],
                    fill="#ffffff",
                    stroke="#d97925",
                ),
                self._polyline(
                    [(1510, 651), (1370, 700), (1370, 740)],
                    stroke="#d97925",
                    marker="arrow",
                ),
                self._polyline(
                    [(1510, 651), (1615, 700), (1615, 740)],
                    stroke="#c44e52",
                    marker="arrow",
                ),
                self._pill(1260, 740, 220, 68, "Final Airway", LABEL_COLORS["Airway"]),
                self._pill(
                    1500,
                    740,
                    230,
                    68,
                    "Final Lung_Parenchymal",
                    LABEL_COLORS["Lung_Parenchymal"],
                ),
                self._note(
                    1040,
                    835,
                    610,
                    "Cascade path",
                    "Stage 1 routes the sample. Stage 2 is conditional, not universal.",
                    "#fef3c7",
                    "#b7791f",
                ),
            ]
        )

    def _cascade_error_paths(self) -> str:
        return "\n".join(
            [
                self._polyline(
                    [(1125, 585), (1265, 675), (1370, 740)],
                    stroke=LABEL_COLORS["Error"],
                    marker="arrow-red",
                    dash=True,
                ),
                self._text(
                    1195,
                    670,
                    "Disease routed to Normal -> Stage 2 skipped",
                    css_class="error-text",
                ),
                self._polyline(
                    [(1510, 585), (1620, 690), (1620, 740)],
                    stroke=LABEL_COLORS["Error"],
                    marker="arrow-red",
                    dash=True,
                ),
                self._text(
                    1395,
                    690,
                    "True Normal routed to Abnormal -> forced disease label",
                    css_class="error-text",
                ),
                self._note(
                    1015,
                    910,
                    645,
                    "Routing consequence",
                    "Dashed red annotations describe structure-specific routing consequences only.",
                    "#fff1f2",
                    LABEL_COLORS["Error"],
                ),
            ]
        )

    def _shared_final_space(self) -> str:
        return "\n".join(
            [
                self._box(
                    420,
                    970,
                    960,
                    78,
                    "Same final label space",
                    ["Normal / Airway / Lung_Parenchymal"],
                    fill="#ffffff",
                    stroke="#64748b",
                ),
                self._arrow(460, 808, 670, 970, stroke="#64748b"),
                self._arrow(1510, 808, 1130, 970, stroke="#64748b"),
            ]
        )

    def _box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        lines: Sequence[str],
        *,
        fill: str,
        stroke: str,
    ) -> str:
        text_lines = [
            f'<text x="{x + width / 2:.1f}" y="{y + 36}" text-anchor="middle" class="box-title">{escape(title)}</text>'
        ]
        for index, line in enumerate(lines):
            text_lines.append(
                f'<text x="{x + width / 2:.1f}" y="{y + 68 + 27 * index}" text-anchor="middle" class="box-line">{escape(line)}</text>'
            )
        return "\n".join(
            [
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="3" filter="url(#soft-shadow)"/>',
                *text_lines,
            ]
        )

    @staticmethod
    def _pill(x: int, y: int, width: int, height: int, label: str, fill: str) -> str:
        return "\n".join(
            [
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{fill}" stroke="#ffffff" stroke-width="3" filter="url(#soft-shadow)"/>',
                f'<text x="{x + width / 2:.1f}" y="{y + height / 2 + 6:.1f}" text-anchor="middle" class="pill">{escape(label)}</text>',
            ]
        )

    @staticmethod
    def _note(
        x: int,
        y: int,
        width: int,
        title: str,
        body: str,
        fill: str,
        stroke: str,
    ) -> str:
        return "\n".join(
            [
                f'<rect x="{x}" y="{y}" width="{width}" height="66" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                f'<text x="{x + 18}" y="{y + 27}" class="box-title" font-size="18">{escape(title)}</text>',
                f'<text x="{x + 18}" y="{y + 50}" class="small">{escape(body)}</text>',
            ]
        )

    @staticmethod
    def _arrow(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        stroke: str,
        marker: str = "arrow",
        dash: bool = False,
    ) -> str:
        dash_attr = ' stroke-dasharray="10 8"' if dash else ""
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="4" stroke-linecap="round"{dash_attr} marker-end="url(#{marker})"/>'
        )

    @staticmethod
    def _polyline(
        points: Sequence[tuple[int, int]],
        *,
        stroke: str,
        marker: str = "arrow",
        dash: bool = False,
    ) -> str:
        encoded_points = " ".join(f"{x},{y}" for x, y in points)
        dash_attr = ' stroke-dasharray="10 8"' if dash else ""
        return (
            f'<polyline points="{encoded_points}" fill="none" stroke="{stroke}" '
            f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"{dash_attr} '
            f'marker-end="url(#{marker})"/>'
        )

    @staticmethod
    def _text(x: int, y: int, text: str, *, css_class: str) -> str:
        return f'<text x="{x}" y="{y}" text-anchor="middle" class="{css_class}">{escape(text)}</text>'


class RasterExporter(LoggingMixin):
    def export(
        self, svg_path: Path, out_base: Path, requested: bool
    ) -> RasterExportResult:
        if not requested:
            return RasterExportResult(
                requested=False,
                status="skipped",
                tool=None,
                outputs=[],
                reason="raster export disabled",
            )

        rsvg = shutil.which("rsvg-convert")
        if rsvg is not None:
            return self._export_with_rsvg(rsvg, svg_path, out_base)

        inkscape = shutil.which("inkscape")
        if inkscape is not None:
            return self._export_with_inkscape(inkscape, svg_path, out_base)

        return RasterExportResult(
            requested=True,
            status="skipped",
            tool=None,
            outputs=[],
            reason="No SVG raster export tool found: rsvg-convert or inkscape",
        )

    def _export_with_rsvg(
        self, executable: str, svg_path: Path, out_base: Path
    ) -> RasterExportResult:
        outputs: list[str] = []
        for fmt in ("png", "pdf"):
            out_path = out_base.with_suffix(f".{fmt}")
            command = [executable, "-f", fmt, "-o", str(out_path), str(svg_path)]
            self._run(command)
            outputs.append(str(out_path))
        return RasterExportResult(
            requested=True,
            status="written",
            tool="rsvg-convert",
            outputs=outputs,
            reason="",
        )

    def _export_with_inkscape(
        self, executable: str, svg_path: Path, out_base: Path
    ) -> RasterExportResult:
        outputs: list[str] = []
        for fmt in ("png", "pdf"):
            out_path = out_base.with_suffix(f".{fmt}")
            command = [
                executable,
                str(svg_path),
                f"--export-type={fmt}",
                f"--export-filename={out_path}",
            ]
            self._run(command)
            outputs.append(str(out_path))
        return RasterExportResult(
            requested=True,
            status="written",
            tool="inkscape",
            outputs=outputs,
            reason="",
        )

    def _run(self, command: Sequence[str]) -> None:
        self.logger.info("Running raster export command: %s", " ".join(command))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Raster export failed with command "
                f"{command!r}: stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )


class StructureComparisonWriter(LoggingMixin):
    def __init__(self, config: StructureDiagramConfig):
        self.config = config

    def write(self) -> dict[str, Any]:
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        svg = DirectCascadeStructureBuilder(self.config).build_svg()
        out_base = self.config.out_dir / FIGURE_STEM
        svg_path = out_base.with_suffix(".svg")
        html_path = out_base.with_suffix(".html")
        metadata_path = out_base.with_name(f"{FIGURE_STEM}_metadata.json")

        svg_path.write_text(svg, encoding="utf-8")
        html_path.write_text(self._build_html(svg), encoding="utf-8")
        self.logger.info("Wrote %s", svg_path)
        self.logger.info("Wrote %s", html_path)

        raster = RasterExporter().export(
            svg_path,
            out_base,
            requested=self.config.export_raster,
        )
        output_files = [str(svg_path), str(html_path), *raster.outputs]
        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "figure_stem": FIGURE_STEM,
            "out_dir": str(self.config.out_dir),
            "options": {
                "show_error_paths": self.config.show_error_paths,
                "export_raster": self.config.export_raster,
                "canvas_width": CANVAS_WIDTH,
                "canvas_height": CANVAS_HEIGHT,
            },
            "label_colors": LABEL_COLORS,
            "outputs": output_files,
            "raster_export": {
                "requested": raster.requested,
                "status": raster.status,
                "tool": raster.tool,
                "outputs": raster.outputs,
                "reason": raster.reason,
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info("Wrote %s", metadata_path)
        return metadata

    @staticmethod
    def _build_html(svg: str) -> str:
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Direct 3-class vs Cascade Structure</title>
  <style>
    body {{
      margin: 0;
      background: #0f172a;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }}
    .figure-frame {{
      width: min(96vw, 1800px);
      background: white;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.25);
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <main class="figure-frame">
{svg}
  </main>
</body>
</html>
"""


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false value, got {raw!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--export-raster", nargs="?", const="true", default="true")
    parser.add_argument("--show-error-paths", nargs="?", const="true", default="true")
    args = parser.parse_args()

    config = StructureDiagramConfig(
        out_dir=Path(args.out_dir),
        show_error_paths=parse_bool(args.show_error_paths),
        export_raster=parse_bool(args.export_raster),
    )
    enable_file_logging(
        config.out_dir / "build_direct_vs_cascade_structure_svg.log",
        mode="w",
    )
    metadata = StructureComparisonWriter(config).write()
    logger.info("Completed structure comparison SVG build: %s", metadata)


if __name__ == "__main__":
    main()
