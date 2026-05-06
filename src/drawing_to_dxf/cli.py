"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drawing_to_dxf.pipeline import RunConfig, run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert raster/PDF engineering sheets to DXF (per-part heuristic split + merged assembly)."
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
        print(f"Error: {e}", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
