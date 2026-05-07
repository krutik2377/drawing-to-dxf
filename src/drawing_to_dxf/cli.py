"""Command-line interface: Gemini vision → per-part Python → DXF assembly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drawing_to_dxf.component_sheet_report import (
    ComponentSheetExtractConfig,
    run_component_sheet_extract,
)
from drawing_to_dxf.gemini_cad_codegen import GeminiCadCodegenConfig, run_gemini_cad_codegen


def component_sheet_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Gemini vision → structured component sheet only (JSON + Markdown, 5–7 parts). "
            "Requires GEMINI_API_KEY or GOOGLE_API_KEY — no codegen, no DXF."
        ),
        fromfile_prefix_chars="@",
    )
    p.add_argument("input", type=Path, nargs="?", help="PNG/JPG/TIFF/BMP or PDF drawing")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("out_component_sheet"))
    p.add_argument("--pdf-page", type=int, default=0)
    p.add_argument("--pdf-dpi", type=float, default=150.0)
    p.add_argument(
        "--max-side",
        type=int,
        default=2048,
        help="Longest image edge sent to Gemini (0 = full resolution)",
    )
    p.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    p.add_argument(
        "--gemini-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for Gemini HTTP response",
    )
    p.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=8192,
        help="Gemini maxOutputTokens for sheet JSON only",
    )
    args = p.parse_args(argv)

    if args.input is None:
        p.print_help()
        return 2
    inp = args.input
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    cfg = ComponentSheetExtractConfig(
        input_path=inp,
        output_dir=args.output_dir,
        gemini_model=args.gemini_model,
        pdf_page=args.pdf_page,
        pdf_dpi=args.pdf_dpi,
        max_side=args.max_side,
        gemini_timeout_s=args.gemini_timeout,
        gemini_max_output_tokens=args.gemini_max_output_tokens,
    )
    try:
        res = run_component_sheet_extract(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    print(f"Component sheet JSON: {res.json_path}")
    print(f"Component sheet Markdown: {res.markdown_path}")
    print(f"Raw model text: {res.raw_text_path}")
    for w in res.warnings:
        print(f"Warning: {w}", file=sys.stderr)
    return 0


def gemini_codegen_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Gemini analyzes the drawing; returns JSON of Python scripts; each script writes a "
            "component DXF; outputs merge into one assembly. Requires GEMINI_API_KEY or "
            "GOOGLE_API_KEY. Executes model-generated Python locally — trusted models/inputs only."
        ),
        fromfile_prefix_chars="@",
    )
    p.add_argument("input", type=Path, nargs="?", help="PNG/JPG/TIFF/BMP or PDF drawing")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("out_gemini_codegen"))
    p.add_argument("--pdf-page", type=int, default=0)
    p.add_argument("--pdf-dpi", type=float, default=150.0)
    p.add_argument(
        "--max-side",
        type=int,
        default=2048,
        help="Longest image edge sent to Gemini (0 = full resolution)",
    )
    p.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    p.add_argument("--layout-gap-mm", type=float, default=40.0)
    p.add_argument(
        "--assembly-layout",
        choices=("horizontal", "grid"),
        default="horizontal",
        help="How to arrange part DXFs in the merged assembly DXF",
    )
    p.add_argument(
        "--grid-columns",
        type=int,
        default=3,
        help="Number of columns when --assembly-layout grid",
    )
    p.add_argument(
        "--sheet-report",
        action="store_true",
        help=(
            "Run an extra Gemini call first: *_component_sheet.json + .md "
            "(5–7 components, dimensions, BOM-style tables); adds one API round-trip"
        ),
    )
    p.add_argument(
        "--sheet-max-output-tokens",
        type=int,
        default=8192,
        help="maxOutputTokens for the sheet-report call only (--sheet-report)",
    )
    p.add_argument("--script-timeout", type=float, default=90.0)
    p.add_argument(
        "--gemini-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for Gemini HTTP response (large maxOutputTokens often needs 5–15+ min)",
    )
    p.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=65536,
        help="Gemini maxOutputTokens (65536≈Gemini Flash family cap for long multi-part JSON)",
    )
    args = p.parse_args(argv)

    if args.input is None:
        p.print_help()
        return 2
    inp = args.input
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2
    cfg = GeminiCadCodegenConfig(
        input_path=inp,
        output_dir=args.output_dir,
        gemini_model=args.gemini_model,
        pdf_page=args.pdf_page,
        pdf_dpi=args.pdf_dpi,
        max_side=args.max_side,
        layout_gap_mm=args.layout_gap_mm,
        assembly_layout=args.assembly_layout,
        grid_columns=args.grid_columns,
        script_timeout_s=args.script_timeout,
        gemini_timeout_s=args.gemini_timeout,
        gemini_max_output_tokens=args.gemini_max_output_tokens,
        sheet_report=args.sheet_report,
        sheet_gemini_max_output_tokens=args.sheet_max_output_tokens,
    )
    try:
        res = run_gemini_cad_codegen(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    print(f"Manifest: {res.manifest_path}")
    print(f"Raw model text: {res.raw_model_text_path}")
    if res.component_sheet_json:
        print(f"Component sheet JSON: {res.component_sheet_json}")
    if res.component_sheet_markdown:
        print(f"Component sheet Markdown: {res.component_sheet_markdown}")
    if res.assembly_dxf:
        print(f"Assembly DXF ({args.assembly_layout}): {res.assembly_dxf}")
    else:
        print("Assembly DXF: (not created — see manifest warnings)", file=sys.stderr)
    for w in res.warnings:
        print(f"Warning: {w}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    av = list(sys.argv[1:] if argv is None else argv)
    if not av:
        print(
            "Use: drawing-to-dxf gemini-codegen <image_or_pdf> [options]\n"
            "  or: drawing-to-dxf component-sheet <image_or_pdf> [options]\n"
            "  or: drawing-to-dxf <image_or_pdf> [options] (same as gemini-codegen)",
            file=sys.stderr,
        )
        return gemini_codegen_main(["-h"])
    if av[0] == "gemini-codegen":
        return gemini_codegen_main(av[1:])
    if av[0] == "component-sheet":
        return component_sheet_main(av[1:])
    if av[0] in ("-h", "--help"):
        # Default help is full codegen parser
        return gemini_codegen_main(av)
    if av[0].startswith("-"):
        return gemini_codegen_main(av)
    if Path(av[0]).exists():
        return gemini_codegen_main(av)
    print(
        f"Not a file: {av[0]!r}\n"
        "Use: drawing-to-dxf <image_or_pdf> [options]\n"
        "  or: drawing-to-dxf gemini-codegen <image_or_pdf> [options]\n"
        "  or: drawing-to-dxf component-sheet <image_or_pdf> [options]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
