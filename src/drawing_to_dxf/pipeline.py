"""End-to-end run: preprocess → vectorize → OCR → link → export DXF + manifest."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drawing_to_dxf.export_dxf import (
    export_merged_dxf,
    export_part_dxf,
    export_segments_only,
)
from drawing_to_dxf.link_parts import PartGroup, link_segments_to_parts
from drawing_to_dxf.ocr_extract import (
    default_part_pattern,
    extract_text_boxes,
    filter_part_candidates,
)
from drawing_to_dxf.preprocess import PreprocessResult, load_image_bgr, load_pdf_page_as_bgr, preprocess
from drawing_to_dxf.vectorize import Segment, extract_segments


@dataclass
class RunConfig:
    input_path: Path
    output_dir: Path
    is_pdf: bool
    pdf_page: int = 0
    pdf_dpi: float = 150.0
    max_side: int | None = 4096
    denoise: bool = True
    deskew: bool = True
    min_line_length: int = 20
    link_mode: str = "hybrid"
    padding_px: float = 120.0
    max_nearest_px: float = 180.0
    mm_per_pixel: float = 1.0
    skip_ocr: bool = False
    ocr_gpu: bool = False
    part_regex: str | None = None
    ocr_min_confidence: float = 0.15


@dataclass
class RunResult:
    manifest_path: Path
    warnings: list[str] = field(default_factory=list)
    part_ids: list[str] = field(default_factory=list)
    segment_count: int = 0
    ocr_box_count: int = 0


def _safe_stem(path: Path) -> str:
    s = path.stem
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "drawing"


def load_input_bgr(cfg: RunConfig) -> Any:
    p = cfg.input_path
    if cfg.is_pdf:
        img = load_pdf_page_as_bgr(str(p), page_index=cfg.pdf_page, dpi=cfg.pdf_dpi)
        if img is None:
            raise FileNotFoundError(f"Cannot read PDF page {cfg.pdf_page}: {p}")
        return img
    img = load_image_bgr(str(p))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img


def run(cfg: RunConfig) -> RunResult:
    warnings: list[str] = []
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(cfg.input_path)

    bgr = load_input_bgr(cfg)
    pre: PreprocessResult = preprocess(
        bgr,
        max_side=cfg.max_side,
        denoise=cfg.denoise,
        deskew=cfg.deskew,
    )
    if pre.scale < 1.0:
        warnings.append(
            f"Image downscaled for processing (scale={pre.scale:.4f}). "
            "Coordinates in DXF match the processed raster size, not the original file pixels."
        )

    min_len = max(5, int(round(cfg.min_line_length * pre.scale)))
    segs: list[Segment] = extract_segments(
        pre.gray,
        min_line_length=min_len,
    )
    if not segs:
        warnings.append("No line segments detected — try lowering min_line_length or tuning scan quality.")

    h, w = pre.gray.shape[:2]
    parts: list[PartGroup] = []
    unassigned: list[Segment] = list(segs)

    ocr_count = 0
    if not cfg.skip_ocr:
        try:
            boxes = extract_text_boxes(pre.gray, gpu=cfg.ocr_gpu)
            ocr_count = len(boxes)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"OCR failed ({type(e).__name__}: {e}); exporting vectorization only.")
            boxes = []
            cfg.skip_ocr = True  # force fallback behavior below

        if not cfg.skip_ocr and boxes:
            pat = re.compile(cfg.part_regex) if cfg.part_regex else default_part_pattern()
            labeled = filter_part_candidates(boxes, pat, min_confidence=cfg.ocr_min_confidence)
            if not labeled:
                warnings.append(
                    "OCR ran but no part numbers matched the regex — "
                    "try --part-regex or export for vector-only assembly.dxf."
                )
                parts, unassigned = [], list(segs)
            else:
                parts, unassigned = link_segments_to_parts(
                    segs,
                    labeled,
                    padding_px=cfg.padding_px,
                    max_nearest_px=cfg.max_nearest_px,
                    mode=cfg.link_mode,
                )
            if parts:
                empty = [p.part_id for p in parts if not p.segments]
                if empty:
                    warnings.append(
                        "Some matched part IDs have no linked geometry: "
                        + ", ".join(empty[:20])
                        + (" …" if len(empty) > 20 else "")
                    )
    else:
        parts = []

    merged_path = out_dir / f"{stem}_assembly_layers.dxf"
    per_part_paths: dict[str, Path] = {}

    if cfg.skip_ocr or not parts:
        if segs:
            export_segments_only(
                merged_path,
                segs,
                img_height_px=float(h),
                mm_per_pixel=cfg.mm_per_pixel,
            )
            if not cfg.skip_ocr:
                warnings.append("Wrote single vectorized DXF (no per-part split). See assembly_layers path.")
        else:
            merged_path = out_dir / f"{stem}_EMPTY.dxf"
            export_segments_only(merged_path, [], img_height_px=float(h), mm_per_pixel=cfg.mm_per_pixel)
    else:
        export_merged_dxf(
            merged_path,
            parts,
            unassigned,
            img_height_px=float(h),
            mm_per_pixel=cfg.mm_per_pixel,
        )
        for pg in parts:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in pg.part_id)
            outp = out_dir / f"{stem}_part_{safe}.dxf"
            export_part_dxf(
                outp,
                pg,
                img_height_px=float(h),
                mm_per_pixel=cfg.mm_per_pixel,
            )
            per_part_paths[pg.part_id] = outp

    manifest: dict[str, Any] = {
        "version": 1,
        "input": str(cfg.input_path.resolve()),
        "processed_size_px": {"width": int(w), "height": int(h)},
        "original_raster_shape": {"height": int(pre.original_shape[0]), "width": int(pre.original_shape[1])},
        "preprocess_scale_from_original": pre.scale,
        "mm_per_pixel": cfg.mm_per_pixel,
        "segment_count": len(segs),
        "ocr_text_boxes": ocr_count,
        "parts": [
            {
                "part_id": p.part_id,
                "segment_count": len(p.segments),
                "label_xy_px": {"x": p.label_center[0], "y": p.label_center[1]},
                "dxf": str(per_part_paths[p.part_id].resolve()) if p.part_id in per_part_paths else None,
            }
            for p in parts
        ],
        "outputs": {
            "assembly_layers_dxf": str(merged_path.resolve()),
            "per_part": {k: str(v.resolve()) for k, v in per_part_paths.items()},
        },
        "warnings": warnings,
        "parameters": {
            "link_mode": cfg.link_mode,
            "padding_px": cfg.padding_px,
            "max_nearest_px": cfg.max_nearest_px,
            "skip_ocr": cfg.skip_ocr,
            "min_line_length": cfg.min_line_length,
        },
    }

    manifest_path = out_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return RunResult(
        manifest_path=manifest_path,
        warnings=warnings,
        part_ids=[p.part_id for p in parts],
        segment_count=len(segs),
        ocr_box_count=ocr_count,
    )
