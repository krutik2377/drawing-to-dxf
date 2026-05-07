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
    p.add_argument("--panel-min-area", type=int, default=15_000, help="Min panel area (px^2) after preprocess")
    p.add_argument("--panel-min-gap", type=int, default=48, help="Min white gutter width for gap splitter")
    p.add_argument("--ocr-gpu", action="store_true")
    p.add_argument(
        "--ai",
        choices=("none", "openai", "ollama", "gemini"),
        default="none",
        help="Structured extraction: gemini (free-tier Google AI Studio), "
        "local ollama+vision, or paid OpenAI-compatible API",
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
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model id for --ai gemini (overridable if a region exposes a different slug)",
    )
    p.add_argument(
        "--layout-cols",
        type=int,
        default=None,
        help="Columns in composite PNG (default: ~sqrt(N))",
    )


def sheet_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Split multi-part shop sheet to panels + optional AI JSON + composite PNG.",
        fromfile_prefix_chars="@",
    )
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
        gemini_model=args.gemini_model,
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
        fromfile_prefix_chars="@",
        epilog="Save repeated flags in a UTF-8 text file (one argument per line), then:  "
        "drawing-to-dxf panels @my_panels.args path/to/sheet.png",
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
        help="pixels: snap nearby segment endpoints after exploding skeleton polylines (0 disables)",
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
        help="Douglas-Peucker epsilon along degree-2 chains (px); 0 disables",
    )
    p.add_argument("--mm-per-pixel", type=float, default=1.0)
    p.add_argument(
        "--reconstruction",
        choices=("none", "shop", "full"),
        default="none",
        help="shop: topology/constraint/intel bundle; full: shop + unified engineering reconstruction suite + semantic DXF layers",
    )
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
    p.add_argument(
        "--no-panel-gap-fallback",
        action="store_true",
        help="Disable XY gutter/grid splitting fallback when ink appears as one large blob",
    )
    p.add_argument(
        "--no-mask-ocr-crops",
        action="store_true",
        help="Do not inpaint OCR text inside each cropped panel prior to skeleton tracing",
    )
    p.add_argument(
        "--panel-split",
        choices=("auto", "blob", "geometry_cc"),
        default="auto",
        help="Panel detection: contour/blob (legacy), geometry connected-components, or auto pick",
    )
    p.add_argument(
        "--trace-debug-dir",
        type=Path,
        default=None,
        help="Write per-panel PNG stages under this folder (original, masked, binary, skeleton, overlay)",
    )
    p.add_argument(
        "--full-rect-ocr-mask",
        action="store_true",
        help="Erase full OCR rectangles in each crop; default wipes text interiors only",
    )
    p.add_argument(
        "--no-soft-ink",
        action="store_true",
        help="Aggressive ruling-line suppression on adaptive ink mask",
    )
    p.add_argument(
        "--python-topology-clean",
        action="store_true",
        help="In-process lwpolyline tidy (topology_clean); default off—prefer AutoCAD MCP after export",
    )
    p.add_argument(
        "--no-arc-fit",
        action="store_true",
        help="Disable arc + circular-loop to CIRCLE fitting",
    )
    p.add_argument(
        "--skeleton-circles",
        action="store_true",
        help="Enable Hough circle pre-pass per panel (off by default on line art)",
    )
    p.add_argument(
        "--no-skeleton-circles",
        action="store_true",
        help="Force-disable Hough circles on panel crops",
    )
    p.add_argument(
        "--keep-title-corner-block",
        action="store_true",
        help="Keep bottom-right title-like blobs when using geometry_cc mode",
    )
    p.add_argument(
        "--vectorize-lsd-supplement",
        action="store_true",
        help="After skeleton tracing, add OpenCV Line Segment Detector strokes not already covered",
    )
    p.add_argument(
        "--vectorize-lsd-min-length",
        type=float,
        default=None,
        help="Min LSD segment length in pixels (default: follow --min-line-length)",
    )
    p.add_argument(
        "--geometry-bridge-gap",
        type=float,
        default=0.0,
        help="Bridge open polyline endpoints facing each other within this gap (pixels); 0 disables",
    )
    p.add_argument(
        "--python-topology-segment-repair",
        action="store_true",
        help="In-process junction snap + gap bridges on exploded segments; default off",
    )
    p.add_argument(
        "--topology-max-bridge-gap",
        type=float,
        default=None,
        help="Max gap (px) for segment endpoint bridges; default derives from merge distance",
    )
    p.add_argument(
        "--topology-junction-snap",
        type=float,
        default=3.0,
        help="Snap endpoints onto segment interiors within this distance (px); 0 disables",
    )
    p.add_argument(
        "--topology-bridge-dot-min",
        type=float,
        default=0.42,
        help="Minimum cosine alignment for directed gap bridges (0–1)",
    )
    p.add_argument(
        "--annotation-box-shrink-pad",
        type=float,
        default=0.0,
        help="Shrink effective OCR mask pad by this many px per box side",
    )
    p.add_argument(
        "--engineering-layout",
        action="store_true",
        help="Milder horizontal ruling suppression on ink mask",
    )
    p.add_argument(
        "--multi-scale-ink",
        action="store_true",
        help="Fuse second adaptive-threshold pass for thin strokes",
    )
    p.add_argument(
        "--no-hole-ring-protect",
        action="store_true",
        help="Disable hole-ring preservation before skeletonization",
    )
    p.add_argument(
        "--python-constraint-heal",
        action="store_true",
        help="In-process orthogonal 4-gon closure; default off—prefer constraints in CAD/MCP",
    )
    p.add_argument(
        "--no-dedupe-geometry-residuals",
        action="store_true",
        help="Keep redundant overlapping LINE residuals (LSD/double edges)",
    )
    p.add_argument(
        "--raster-dpi",
        type=float,
        default=None,
        help="Nominal raster DPI for scaling length thresholds vs 150dpi baseline (default: --pdf-dpi on PDF, else 150)",
    )
    p.add_argument(
        "--python-engineering-reconstruction-suite",
        action="store_true",
        help="Unified reconstruction suite on each panel (topology, CAD cleanup, dimensions, QA)",
    )
    p.add_argument(
        "--python-engineering-intel",
        action="store_true",
        help="In-process semantic/constraint/topology metrics on vectors; default off",
    )
    p.add_argument(
        "--python-cad-reconstruction",
        action="store_true",
        help="Stronger topology: ray intersection extend, axis/parallel snap, healed LINE export",
    )

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
        panel_gap_split_fallback=not args.no_panel_gap_fallback,
        panel_split_strategy=args.panel_split,
        exclude_corner_title_block=not args.keep_title_corner_block,
        segment_merge_distance_px=args.segment_merge_distance,
        vector_collinear_merge_angle_deg=args.vector_collinear_angle_deg,
        vector_polyline_rdp_epsilon_px=args.vector_rdp_epsilon,
        mask_annotation_via_ocr_crop=not args.no_mask_ocr_crops,
        mask_text_interior_only=not args.full_rect_ocr_mask,
        soft_ink_mask=not args.no_soft_ink,
        enable_topology_clean=args.python_topology_clean,
        enable_arc_fitting=not args.no_arc_fit,
        enable_loop_circle_fit=not args.no_arc_fit,
        trace_debug_dir=args.trace_debug_dir,
        skip_ocr=args.skip_ocr,
        ocr_gpu=args.ocr_gpu,
        part_regex=args.part_regex,
        ocr_min_confidence=args.ocr_min_confidence,
        mm_per_pixel=args.mm_per_pixel,
        write_viewer_bundle=not args.no_viewer_bundle,
        viewer_layout_gap_mm=args.viewer_layout_gap_mm,
        emit_root_panel_dxfs=args.emit_root_panel_dxfs,
        enable_skeleton_circles=args.skeleton_circles and not args.no_skeleton_circles,
        vectorize_lsd_supplement=args.vectorize_lsd_supplement,
        vectorize_lsd_min_length_px=args.vectorize_lsd_min_length,
        geometry_bridge_gap_px=args.geometry_bridge_gap,
        dedupe_geometry_residuals=not args.no_dedupe_geometry_residuals,
        raster_dpi_nominal=args.raster_dpi,
        enable_topology_segment_repair=args.python_topology_segment_repair,
        topology_max_bridge_gap_px=args.topology_max_bridge_gap,
        topology_junction_snap_px=args.topology_junction_snap,
        topology_bridge_direction_dot_min=args.topology_bridge_dot_min,
        annotation_box_shrink_from_pad_px=args.annotation_box_shrink_pad,
        engineering_layout=args.engineering_layout,
        multi_scale_ink=args.multi_scale_ink,
        protect_hole_rings=not args.no_hole_ring_protect,
        enable_constraint_heal=args.python_constraint_heal,
        enable_engineering_intel_passes=args.python_engineering_intel,
        reconstruction_preset=args.reconstruction if args.reconstruction != "none" else None,
        enable_engineering_reconstruction_suite=args.python_engineering_reconstruction_suite,
    )
    if args.reconstruction == "full":
        from dataclasses import replace

        from drawing_to_dxf.reconstruction_preset import apply_full_engineering_reconstruction_to_panel_config

        cfg = apply_full_engineering_reconstruction_to_panel_config(cfg)
        if args.no_skeleton_circles:
            cfg = replace(cfg, enable_skeleton_circles=False)
    elif args.reconstruction == "shop":
        from dataclasses import replace

        from drawing_to_dxf.reconstruction_preset import apply_shop_reconstruction_to_panel_config

        cfg = apply_shop_reconstruction_to_panel_config(cfg)
        if args.no_skeleton_circles:
            cfg = replace(cfg, enable_skeleton_circles=False)
    if args.python_cad_reconstruction:
        from drawing_to_dxf.reconstruction_preset import apply_python_cad_reconstruction_bundle_to_panel_config

        cfg = apply_python_cad_reconstruction_bundle_to_panel_config(cfg)
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
        description="Convert raster/PDF sheets to DXF. Subcommands: `sheet ...`, "
        "`panels ...`, `ui-panels` (Streamlit preview).",
        fromfile_prefix_chars="@",
    )
    p.add_argument("input", type=Path, help="Path to PNG/JPG/TIFF/BMP or PDF")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("out"),
        help="Directory for DXF + manifest (default: ./out)",
    )
    p.add_argument(
        "--reconstruction",
        choices=("none", "shop", "full"),
        default="none",
        help="shop/full Python reconstruction bundles; see manifest reconstruction_roadmap",
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
    p.add_argument(
        "--min-line-length",
        type=int,
        default=20,
        help="Skeleton branch / legacy Hough min span (pixels, processed image)",
    )
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
        help="Douglas-Peucker simplify polylines (image px); 0 disables",
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
    p.add_argument(
        "--full-rect-ocr-mask",
        action="store_true",
        help="Erase full OCR rectangles before tracing (legacy); default clears text interiors only",
    )
    p.add_argument(
        "--no-soft-ink",
        action="store_true",
        help="Apply aggressive horizontal ruling suppression on adaptive ink mask",
    )
    p.add_argument(
        "--python-topology-clean",
        action="store_true",
        help="In-process lwpolyline tidy (topology_clean); default off—use AutoCAD MCP after export",
    )
    p.add_argument(
        "--no-arc-fit",
        action="store_true",
        help="Disable arc approximation and circular loop to CIRCLE substitution",
    )
    p.add_argument(
        "--no-mask-annotations",
        action="store_true",
        help="Skip inpainting OCR text boxes before skeleton tracing",
    )
    p.add_argument(
        "--skeleton-circles",
        action="store_true",
        help="Enable Hough circles + hole validation (off by default; noisy on line-only drawings)",
    )
    p.add_argument(
        "--no-skeleton-circles",
        action="store_true",
        help="Force-disable Hough circles if enabled elsewhere",
    )
    p.add_argument(
        "--skeleton-text-pad-pixels",
        type=float,
        default=10.0,
        help="Expand OCR rectangles before wiping text strokes ahead of skeletonization",
    )
    p.add_argument(
        "--legacy-hough-vectorize",
        action="store_true",
        help="Use Canny + HoughLinesP instead of OCR-masked skeleton graphs",
    )
    p.add_argument(
        "--vectorize-lsd-supplement",
        action="store_true",
        help="Add OpenCV LSD segments that the skeleton path may miss",
    )
    p.add_argument(
        "--vectorize-lsd-min-length",
        type=float,
        default=None,
        help="Min LSD stroke in pixels (default: similar to --min-line-length)",
    )
    p.add_argument(
        "--geometry-bridge-gap",
        type=float,
        default=0.0,
        help="Bridge open polyline endpoints within this gap (px); 0 disables",
    )
    p.add_argument(
        "--python-topology-segment-repair",
        action="store_true",
        help="In-process junction snap + gap bridges on exploded segments; default off",
    )
    p.add_argument(
        "--topology-max-bridge-gap",
        type=float,
        default=None,
        help="Max gap (px) for segment endpoint bridges; default derives from merge distance",
    )
    p.add_argument(
        "--topology-junction-snap",
        type=float,
        default=3.0,
        help="Snap free endpoints onto nearby segment interiors within this distance (px); 0 disables",
    )
    p.add_argument(
        "--topology-bridge-dot-min",
        type=float,
        default=0.42,
        help="Minimum cosine alignment for directed gap bridges (0–1)",
    )
    p.add_argument(
        "--annotation-box-shrink-pad",
        type=float,
        default=0.0,
        help="Reduce effective OCR mask padding by this many px (preserves linework near annotations)",
    )
    p.add_argument(
        "--engineering-layout",
        action="store_true",
        help="Milder horizontal ruling suppression (preserves more construction/dimension ticks)",
    )
    p.add_argument(
        "--multi-scale-ink",
        action="store_true",
        help="Fuse a second adaptive-threshold pass to recover thin strokes",
    )
    p.add_argument(
        "--no-hole-ring-protect",
        action="store_true",
        help="Do not restore hole circle ink after small-object removal",
    )
    p.add_argument(
        "--python-constraint-heal",
        action="store_true",
        help="In-process orthogonal closure for nearly-rectangular 4-gons; default off",
    )
    p.add_argument(
        "--debug-export-dir",
        type=Path,
        default=None,
        help="Write raster debug PNGs and segments GeoJSON under this directory",
    )
    p.add_argument(
        "--no-export-ocr-text-dxf",
        action="store_true",
        help="Omit OCR_TEXT layer from DXF exports",
    )
    p.add_argument(
        "--no-dimension-hints-dxf",
        action="store_true",
        help="Omit DIMENSION_HINT heuristic overlay layer",
    )
    p.add_argument(
        "--no-dedupe-geometry-residuals",
        action="store_true",
        help="Keep overlapping duplicate residual segments",
    )
    p.add_argument(
        "--raster-dpi",
        type=float,
        default=None,
        help="Scale thresholds vs 150dpi (default: --pdf-dpi for PDF else 150)",
    )
    p.add_argument(
        "--profile-pipeline",
        action="store_true",
        help="Write preprocess/OCR/vectorize/export timings to manifest",
    )
    p.add_argument(
        "--python-engineering-intel",
        action="store_true",
        help="In-process semantic/constraint/topology-intel passes + manifest metrics; default off",
    )
    p.add_argument(
        "--python-cad-reconstruction",
        action="store_true",
        help="Stronger topology: ray intersection extension, axis/parallel regularization, "
        "and DXF export from healed LINE graph (circles/arcs kept)",
    )
    p.add_argument(
        "--semantic-seg-onnx",
        type=Path,
        default=None,
        help="Optional ONNX segmentation model: inpaint non-geometry classes before skeleton tracing "
        "(install: pip install -e \".[ml]\")",
    )
    p.add_argument(
        "--semantic-seg-config",
        type=Path,
        default=None,
        help="JSON: classes, suppress_for_skeleton, input_width/height, normalize, layout (nchw|nhwc)",
    )
    p.add_argument(
        "--semantic-seg-suppress-dimension",
        action="store_true",
        help="Also mask \"dimension\" class from segmentation (aggressive)",
    )
    p.add_argument(
        "--ollama-ocr-correct",
        action="store_true",
        help="After EasyOCR, batch-correct strings via local Ollama text model",
    )
    p.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    p.add_argument(
        "--ollama-text-model",
        type=str,
        default="llama3.2",
        help="Text-only model for --ollama-ocr-correct (e.g. llama3.2, mistral)",
    )
    p.add_argument(
        "--gemini-ocr-correct",
        action="store_true",
        help="After EasyOCR, batch-correct strings via Gemini (text-only; GEMINI_API_KEY or GOOGLE_API_KEY)",
    )
    p.add_argument(
        "--gemini-text-model",
        type=str,
        default="gemini-2.0-flash",
        help="Model id for --gemini-ocr-correct (Google AI Studio generateContent)",
    )
    p.add_argument(
        "--layered-dxf",
        action="store_true",
        help="Emit semantic LINE layers (GEOMETRY/DIMENSION/BORDER/ANNOTATION) from pixel labels + exploded segments",
    )
    p.add_argument(
        "--rule-based-semantics",
        action="store_true",
        help="Use rule-based class raster (OCR + ruling + frame band), merged with ONNX when both set",
    )
    p.add_argument(
        "--emit-linear-dimension-entities",
        action="store_true",
        help="Add ezdxf LINEAR DIMENSION entities where numeric OCR associates to an axis segment",
    )
    p.add_argument(
        "--python-engineering-reconstruction-suite",
        action="store_true",
        help="Unified reconstruction: topology refine, CAD graph cleanup, dimension objects, multilayer semantics, QA",
    )
    p.add_argument(
        "--topology-loop-close",
        type=float,
        default=0.0,
        help="Bridge nearly closed chains: connect deg-1 endpoints within this gap (pixels); 0=off",
    )

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
        mask_annotation_via_ocr=not args.no_mask_annotations,
        enable_skeleton_circles=args.skeleton_circles and not args.no_skeleton_circles,
        skeleton_annotation_pad_px=args.skeleton_text_pad_pixels,
        legacy_vectorize_hough=args.legacy_hough_vectorize,
        mask_text_interior_only=not args.full_rect_ocr_mask,
        soft_ink_mask=not args.no_soft_ink,
        enable_topology_clean=args.python_topology_clean,
        enable_arc_fitting=not args.no_arc_fit,
        enable_loop_circle_fit=not args.no_arc_fit,
        ocr_gpu=args.ocr_gpu,
        part_regex=args.part_regex,
        ocr_min_confidence=args.ocr_min_confidence,
        vectorize_lsd_supplement=args.vectorize_lsd_supplement,
        vectorize_lsd_min_length_px=args.vectorize_lsd_min_length,
        geometry_bridge_gap_px=args.geometry_bridge_gap,
        dedupe_geometry_residuals=not args.no_dedupe_geometry_residuals,
        raster_dpi_nominal=args.raster_dpi,
        profile_pipeline=args.profile_pipeline,
        enable_topology_segment_repair=args.python_topology_segment_repair,
        topology_max_bridge_gap_px=args.topology_max_bridge_gap,
        topology_junction_snap_px=args.topology_junction_snap,
        topology_bridge_direction_dot_min=args.topology_bridge_dot_min,
        annotation_box_shrink_from_pad_px=args.annotation_box_shrink_pad,
        engineering_layout=args.engineering_layout,
        multi_scale_ink=args.multi_scale_ink,
        protect_hole_rings=not args.no_hole_ring_protect,
        enable_constraint_heal=args.python_constraint_heal,
        debug_export_dir=args.debug_export_dir,
        export_ocr_text_to_dxf=not args.no_export_ocr_text_dxf,
        export_dimension_hints_to_dxf=not args.no_dimension_hints_dxf,
        enable_engineering_intel_passes=args.python_engineering_intel,
        reconstruction_preset=args.reconstruction if args.reconstruction != "none" else None,
        semantic_seg_onnx=args.semantic_seg_onnx,
        semantic_seg_config=args.semantic_seg_config,
        semantic_seg_suppress_dimension=args.semantic_seg_suppress_dimension,
        ollama_ocr_correct=args.ollama_ocr_correct,
        ollama_host=args.ollama_host,
        ollama_text_model=args.ollama_text_model,
        gemini_ocr_correct=args.gemini_ocr_correct,
        gemini_text_model=args.gemini_text_model,
        layered_dxf=args.layered_dxf,
        rule_based_semantics=args.rule_based_semantics,
        emit_linear_dimension_entities=args.emit_linear_dimension_entities,
        topology_loop_close_px=args.topology_loop_close,
        enable_engineering_reconstruction_suite=args.python_engineering_reconstruction_suite,
    )
    if args.reconstruction == "full":
        from drawing_to_dxf.reconstruction_preset import apply_full_engineering_reconstruction_to_run_config

        cfg = apply_full_engineering_reconstruction_to_run_config(cfg)
    elif args.reconstruction == "shop":
        from dataclasses import replace

        from drawing_to_dxf.reconstruction_preset import apply_shop_reconstruction_to_run_config

        cfg = apply_shop_reconstruction_to_run_config(cfg)
        if args.no_skeleton_circles:
            cfg = replace(cfg, enable_skeleton_circles=False)
    if args.python_cad_reconstruction:
        from drawing_to_dxf.reconstruction_preset import apply_python_cad_reconstruction_bundle_to_run_config

        cfg = apply_python_cad_reconstruction_bundle_to_run_config(cfg)

    try:
        res = run(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1

    print(f"Manifest: {res.manifest_path}")
    if res.part_ids:
        print(f"Parts: {', '.join(res.part_ids)}")
    else:
        print("Parts: (none - see manifest warnings)")
    print(
        f"Segments: {res.segment_count}, primitives: {res.primitive_count}, "
        f"OCR boxes: {res.ocr_box_count}"
    )
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
