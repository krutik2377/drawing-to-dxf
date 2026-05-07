"""Command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import drawing_to_dxf
from drawing_to_dxf.panel_dxf_pipeline import PanelDxfRunConfig, run_panel_dxfs
from drawing_to_dxf.pipeline import RunConfig, run
from drawing_to_dxf.sheet_pipeline import SheetRunConfig, run_sheet


def _sheet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="PNG/JPG/TIFF/BMP or PDF shop sheet")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("out_sheet"))
    p.add_argument("--pdf-page", type=int, default=0)
    p.add_argument("--pdf-dpi", type=float, default=150.0)
    p.add_argument("--max-side", type=int, default=4096)
    p.add_argument("--panel-min-area", type=int, default=15_000, help="Min panel area (px²) after preprocess")
    p.add_argument("--panel-min-gap", type=int, default=48, help="Min white gutter width for gap splitter")
    p.add_argument("--ocr-gpu", action="store_true")
    p.add_argument(
        "--ai",
        choices=("none", "openai", "ollama"),
        default="none",
        help="Structured extraction: paid API (openai) or free local (ollama + vision model)",
    )
    p.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    p.add_argument(
        "--openai-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Override with Azure/OpenAI-compatible gateway (env OPENAI_BASE_URL also works)",
    )
    p.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", type=str, default="llava")
    p.add_argument(
        "--layout-cols",
        type=int,
        default=None,
        help="Columns in composite PNG (default: ~sqrt(N))",
    )


def sheet_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split multi-part shop sheet → panels + optional AI JSON + composite PNG.")
    _sheet_args(p)
    args = p.parse_args(argv)
    inp = args.input
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2
    is_pdf = inp.suffix.lower() == ".pdf"
    cfg = SheetRunConfig(
        input_path=inp,
        output_dir=args.output_dir,
        is_pdf=is_pdf,
        pdf_page=args.pdf_page,
        pdf_dpi=args.pdf_dpi,
        max_side=args.max_side,
        panel_min_area=args.panel_min_area,
        panel_min_gap=args.panel_min_gap,
        ocr_gpu=args.ocr_gpu,
        ai_provider=args.ai,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        layout_cols=args.layout_cols,
    )
    try:
        res = run_sheet(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    print(f"Manifest: {res.manifest_path}")
    print(f"Composite: {res.composite_png}")
    print(f"Panels: {res.panel_count}")
    for w in res.warnings:
        print(f"Warning: {w}", file=sys.stderr)
    return 0


def panels_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Split sheet into gutter-separated panels; write one DXF per panel "
        "(vectorize inside each crop; optional OCR filenames). Safer than full-page part linking.",
    )
    p.add_argument("input", type=Path, help="PNG/JPG/TIFF/BMP or PDF shop sheet")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("out_panels"))
    p.add_argument("--pdf-page", type=int, default=0)
    p.add_argument("--pdf-dpi", type=float, default=150.0)
    p.add_argument("--max-side", type=int, default=4096)
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--min-line-length", type=int, default=20)
    p.add_argument("--panel-min-area", type=int, default=15_000)
    p.add_argument("--panel-min-gap", type=int, default=48)
    p.add_argument(
        "--panel-min-short-side",
        type=int,
        default=80,
        help="Discard grid cells where min(width,height) is below this (drops gutter strips)",
    )
    p.add_argument(
        "--panel-max-aspect",
        type=float,
        default=10.0,
        help="Discard cells with max_side/min_side above this (narrow ribbons)",
    )
    p.add_argument(
        "--segment-merge-distance",
        type=float,
        default=5.0,
        help="pixels: merge nearby Hough endpoints for cleaner DXF lines (0 disables)",
    )
    p.add_argument(
        "--vector-collinear-angle-deg",
        type=float,
        default=5.0,
        help="merge straight chains at joints; 0 disables (degrees, after endpoint snap)",
    )
    p.add_argument(
        "--vector-rdp-epsilon",
        type=float,
        default=1.5,
        help="Douglas–Peucker epsilon along degree-2 chains (px); 0 disables",
    )
    p.add_argument("--mm-per-pixel", type=float, default=1.0)
    p.add_argument(
        "--viewer-layout-gap-mm",
        type=float,
        default=25.0,
        help="Gap in mm between panels in viewer/*_autodesk_layout.dxf",
    )
    p.add_argument(
        "--no-viewer-bundle",
        action="store_true",
        help="Do not write viewer/ (single combined layout DXF + models/ + upload hints)",
    )
    p.add_argument(
        "--emit-root-panel-dxfs",
        action="store_true",
        help="With viewer bundle, also copy stem_panel_*.dxf into output root (legacy layout)",
    )
    p.add_argument("--skip-ocr", action="store_true", help="Name outputs panel_NN DXFs; no EasyOCR required")
    p.add_argument("--ocr-gpu", action="store_true")
    p.add_argument(
        "--part-regex",
        type=str,
        default=None,
        help="Override default part-number regex (requires EasyOCR unless --skip-ocr)",
    )
    p.add_argument("--ocr-min-confidence", type=float, default=0.15)

    args = p.parse_args(argv)
    inp = args.input
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2
    is_pdf = inp.suffix.lower() == ".pdf"
    cfg = PanelDxfRunConfig(
        input_path=inp,
        output_dir=args.output_dir,
        is_pdf=is_pdf,
        pdf_page=args.pdf_page,
        pdf_dpi=args.pdf_dpi,
        max_side=None if args.max_side == 0 else args.max_side,
        denoise=not args.no_denoise,
        deskew=not args.no_deskew,
        min_line_length=args.min_line_length,
        panel_min_area=args.panel_min_area,
        panel_min_gap=args.panel_min_gap,
        panel_min_short_side_px=args.panel_min_short_side,
        panel_max_aspect_ratio=args.panel_max_aspect,
        segment_merge_distance_px=args.segment_merge_distance,
        vector_collinear_merge_angle_deg=args.vector_collinear_angle_deg,
        vector_polyline_rdp_epsilon_px=args.vector_rdp_epsilon,
        skip_ocr=args.skip_ocr,
        ocr_gpu=args.ocr_gpu,
        part_regex=args.part_regex,
        ocr_min_confidence=args.ocr_min_confidence,
        mm_per_pixel=args.mm_per_pixel,
        write_viewer_bundle=not args.no_viewer_bundle,
        viewer_layout_gap_mm=args.viewer_layout_gap_mm,
        emit_root_panel_dxfs=args.emit_root_panel_dxfs,
    )
    try:
        res = run_panel_dxfs(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1

    print(f"Manifest: {res.manifest_path}")
    print(f"Panels: {res.panel_count} -> DXF files: {len(res.dxf_paths)}")
    if res.viewer_primary_dxf:
        print(f"Viewer (upload this DXF alone): {res.viewer_primary_dxf}")
    for w in res.warnings:
        print(f"Warning: {w}", file=sys.stderr)
    return 0


def convert_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert raster/PDF sheets to DXF. Subcommands: `sheet …`, "
        "`panels …`, `ui-panels` (Streamlit preview).",
    )
    p.add_argument("input", type=Path, help="Path to PNG/JPG/TIFF/BMP or PDF")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("out"),
        help="Directory for DXF + manifest (default: ./out)",
    )
    p.add_argument("--pdf-page", type=int, default=0, help="Zero-based PDF page index")
    p.add_argument("--pdf-dpi", type=float, default=150.0, help="Rasterization DPI for PDF input")
    p.add_argument(
        "--max-side",
        type=int,
        default=4096,
        help="Downscale longest edge to this many pixels (set 0 to disable)",
    )
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--min-line-length", type=int, default=20, help="Hough min line length (pixels, processed)")
    p.add_argument(
        "--segment-merge-distance",
        type=float,
        default=5.0,
        help="merge nearby Hough segment endpoints (pixels); 0 skips snap before refinement",
    )
    p.add_argument(
        "--vector-collinear-angle-deg",
        type=float,
        default=5.0,
        help="merge collinear segments at shared joints (degrees); 0 disables",
    )
    p.add_argument(
        "--vector-rdp-epsilon",
        type=float,
        default=1.5,
        help="Douglas–Peucker simplify polylines (image px); 0 disables",
    )
    p.add_argument(
        "--link-mode",
        choices=("hybrid", "bbox", "nearest"),
        default="hybrid",
        help="How segments attach to OCR part labels",
    )
    p.add_argument("--padding-px", type=float, default=120.0, help="Expand OCR box for bbox linking")
    p.add_argument(
        "--max-nearest-px",
        type=float,
        default=180.0,
        help="Max distance (processed px) for nearest-neighbor linking",
    )
    p.add_argument(
        "--mm-per-pixel",
        type=float,
        default=1.0,
        help="DXF drawing units: mm per processed image pixel (calibrate from a known dimension)",
    )
    p.add_argument("--skip-ocr", action="store_true", help="Vectorize only; single assembly DXF")
    p.add_argument("--ocr-gpu", action="store_true", help="Use GPU for EasyOCR if available")
    p.add_argument(
        "--part-regex",
        type=str,
        default=None,
        help=r'Override part-number regex (default: \b\d{3,5}\b style)',
    )
    p.add_argument("--ocr-min-confidence", type=float, default=0.15)

    args = p.parse_args(argv)

    inp: Path = args.input
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    suffix = inp.suffix.lower()
    is_pdf = suffix == ".pdf"

    cfg = RunConfig(
        input_path=inp,
        output_dir=args.output_dir,
        is_pdf=is_pdf,
        pdf_page=args.pdf_page,
        pdf_dpi=args.pdf_dpi,
        max_side=None if args.max_side == 0 else args.max_side,
        denoise=not args.no_denoise,
        deskew=not args.no_deskew,
        min_line_length=args.min_line_length,
        segment_merge_distance_px=args.segment_merge_distance,
        vector_collinear_merge_angle_deg=args.vector_collinear_angle_deg,
        vector_polyline_rdp_epsilon_px=args.vector_rdp_epsilon,
        link_mode=args.link_mode,
        padding_px=args.padding_px,
        max_nearest_px=args.max_nearest_px,
        mm_per_pixel=args.mm_per_pixel,
        skip_ocr=args.skip_ocr,
        ocr_gpu=args.ocr_gpu,
        part_regex=args.part_regex,
        ocr_min_confidence=args.ocr_min_confidence,
    )

    try:
        res = run(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1

    print(f"Manifest: {res.manifest_path}")
    if res.part_ids:
        print(f"Parts: {', '.join(res.part_ids)}")
    else:
        print("Parts: (none — see manifest warnings)")
    print(f"Segments: {res.segment_count}, OCR boxes: {res.ocr_box_count}")
    for w in res.warnings:
        print(f"Warning: {w}", file=sys.stderr)
    return 0


def ui_panels_main(argv: list[str] | None = None) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print('Install Streamlit UI: pip install -e ".[ui]"', file=sys.stderr)
        return 1
    app = Path(drawing_to_dxf.__file__).resolve().parent / "ui_panels_app.py"
    if not app.is_file():
        print(f"Missing app: {app}", file=sys.stderr)
        return 1
    prefix = argv if argv is not None else []
    try:
        r = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app)] + prefix,
            check=False,
        )
    except KeyboardInterrupt:
        return 130
    return int(r.returncode)


def main(argv: list[str] | None = None) -> int:
    av = list(sys.argv[1:] if argv is None else argv)
    if av and av[0] == "sheet":
        return sheet_main(av[1:])
    if av and av[0] == "panels":
        return panels_main(av[1:])
    if av and av[0] == "ui-panels":
        return ui_panels_main(av[1:])
    return convert_main(av)


if __name__ == "__main__":
    raise SystemExit(main())
