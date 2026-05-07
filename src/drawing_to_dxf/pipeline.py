"""End-to-end run: preprocess → vectorize → OCR → link → export DXF + manifest.

The roadmap checklist lives in ``raster_pipeline_flow.CANONICAL_PIPELINE_STEPS``; each manifest
includes ``pipeline_flow`` mapping steps to implementing symbols.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextlib import nullcontext

from drawing_to_dxf.export_dxf import (
    export_merged_dxf,
    export_part_dxf,
    export_segments_only,
)
from drawing_to_dxf.geometry_intel import geometry_quality_report, profile_stage
from drawing_to_dxf.geometry_model import VectorDrawing, exploded_segments_for_sampling
from drawing_to_dxf.link_geometry import link_vector_geometry_to_parts
from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.ocr_extract import (
    TextBox,
    default_part_pattern,
    extract_text_boxes,
    filter_part_candidates,
)
from drawing_to_dxf.preprocess import PreprocessResult, load_image_bgr, load_pdf_page_as_bgr, preprocess
from drawing_to_dxf.raster_pipeline_flow import pipeline_flow_manifest_rows
from drawing_to_dxf.vectorize import extract_skeleton_vector_bundle, _legacy_hough_extract_segments


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
    segment_merge_distance_px: float = 5.0
    vector_collinear_merge_angle_deg: float = 5.0
    vector_polyline_rdp_epsilon_px: float = 1.5
    link_mode: str = "hybrid"
    padding_px: float = 120.0
    max_nearest_px: float = 180.0
    mm_per_pixel: float = 1.0
    skip_ocr: bool = False
    mask_annotation_via_ocr: bool = True
    ocr_gpu: bool = False
    part_regex: str | None = None
    ocr_min_confidence: float = 0.15
    legacy_vectorize_hough: bool = False
    skeleton_annotation_pad_px: float = 10.0
    enable_skeleton_circles: bool = False
    mask_text_interior_only: bool = True
    soft_ink_mask: bool = True
    enable_topology_clean: bool = True
    enable_arc_fitting: bool = True
    enable_loop_circle_fit: bool = True
    vectorize_lsd_supplement: bool = False
    vectorize_lsd_min_length_px: float | None = None
    geometry_bridge_gap_px: float = 0.0
    dedupe_geometry_residuals: bool = True
    raster_dpi_nominal: float | None = None
    profile_pipeline: bool = False


@dataclass
class RunResult:
    manifest_path: Path
    warnings: list[str] = field(default_factory=list)
    part_ids: list[str] = field(default_factory=list)
    segment_count: int = 0
    primitive_count: int = 0
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
    """
    Execute the 20-step raster→DXF flow (see ``raster_pipeline_flow``).

    Call order: load (1) → preprocess grayscale/denoise/deskew/scale (2,4,5,16) → OCR (7) →
    vector bundle via adaptive ink mask + morphology + skeleton + circles (3,6,8–10,13) →
    topology/merge/snap (11,12,20) → link parts (15) → export CAD (17–19).
    """
    warnings: list[str] = []
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(cfg.input_path)

    bgr = load_input_bgr(cfg)
    timings: dict[str, float] = {}

    with profile_stage(timings, "preprocess") if cfg.profile_pipeline else nullcontext():
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

    nominal_dpi = cfg.raster_dpi_nominal if cfg.raster_dpi_nominal is not None else (cfg.pdf_dpi if cfg.is_pdf else 150.0)
    dpi_scale = float(nominal_dpi) / 150.0
    min_len = max(5, int(round(cfg.min_line_length * pre.scale * dpi_scale)))

    boxes: list[TextBox] = []
    ocr_count = 0
    with profile_stage(timings, "ocr") if cfg.profile_pipeline else nullcontext():
        if not cfg.skip_ocr:
            try:
                boxes = extract_text_boxes(pre.gray, gpu=cfg.ocr_gpu)
                ocr_count = len(boxes)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"OCR failed ({type(e).__name__}: {e}); exporting vectorization only.")
                boxes = []

    vd: VectorDrawing
    if cfg.legacy_vectorize_hough:
        with profile_stage(timings, "vectorize") if cfg.profile_pipeline else nullcontext():
            warnings.append("Legacy Hough vectorization is active (thin LINE-only output expected).")
            segs = _legacy_hough_extract_segments(
                pre.gray,
                min_line_length=min_len,
                merge_distance=cfg.segment_merge_distance_px,
                collinear_merge_angle_deg=cfg.vector_collinear_merge_angle_deg,
                polyline_rdp_epsilon_px=cfg.vector_polyline_rdp_epsilon_px,
            )
        vd = VectorDrawing(residual_segments=list(segs))
        primitive_totals = len(segs)
        if not segs:
            warnings.append("Legacy Hough found no segments — try lowering min_line_length.")
    else:
        with profile_stage(timings, "vectorize") if cfg.profile_pipeline else nullcontext():
            vd, segs = extract_skeleton_vector_bundle(
                pre.gray,
                annotation_boxes=boxes if (cfg.mask_annotation_via_ocr and not cfg.skip_ocr and boxes) else [],
                mask_annotation_regions=bool(cfg.mask_annotation_via_ocr and not cfg.skip_ocr and boxes),
                min_line_length=min_len,
                merge_distance=cfg.segment_merge_distance_px,
                collinear_merge_angle_deg=cfg.vector_collinear_merge_angle_deg,
                polyline_rdp_epsilon_px=cfg.vector_polyline_rdp_epsilon_px,
                annotation_pad_px=cfg.skeleton_annotation_pad_px,
                enable_circles=cfg.enable_skeleton_circles,
                mask_text_interior_only=cfg.mask_text_interior_only,
                soft_ink_mask=cfg.soft_ink_mask,
                enable_topology_clean=cfg.enable_topology_clean,
                enable_arc_fitting=cfg.enable_arc_fitting,
                enable_loop_circle_fit=cfg.enable_loop_circle_fit,
                lsd_supplement=cfg.vectorize_lsd_supplement,
                lsd_min_length_px=cfg.vectorize_lsd_min_length_px,
                geometry_bridge_gap_px=cfg.geometry_bridge_gap_px,
                dedupe_residual_segments=cfg.dedupe_geometry_residuals,
            )
        primitive_totals = len(vd.polylines) + len(vd.circles) + len(vd.arcs) + len(vd.residual_segments)
        if not segs:
            warnings.append(
                "No exploded line sampling segments after skeleton tracing — "
                "try lowering --min-line-length or improving raster contrast "
                "(geometry may still include circles/lwpolylines)."
            )

    g_quality = geometry_quality_report(vd, timings_ms=timings if cfg.profile_pipeline else {})

    h, w = pre.gray.shape[:2]

    labeled: list[tuple[TextBox, str]] = []
    if boxes:
        pat = re.compile(cfg.part_regex) if cfg.part_regex else default_part_pattern()
        labeled = filter_part_candidates(boxes, pat, min_confidence=cfg.ocr_min_confidence)

    parts: list[PartGroup] = []
    unassigned_geom: VectorDrawing = vd

    if labeled:
        parts, unassigned_geom = link_vector_geometry_to_parts(
            vd,
            labeled,
            padding_px=cfg.padding_px,
            max_nearest_px=cfg.max_nearest_px,
            mode=cfg.link_mode,
        )
        for pg in parts:
            if pg.vector_drawing is None:
                continue
            pg.segments[:] = exploded_segments_for_sampling(pg.vector_drawing)
        empty = [
            p.part_id
            for p in parts
            if (
                p.vector_drawing is None
                or (
                    len(p.vector_drawing.polylines) == 0
                    and len(p.vector_drawing.circles) == 0
                    and len(getattr(p.vector_drawing, "arcs", ())) == 0
                    and len(p.vector_drawing.residual_segments) == 0
                )
            )
        ]
        if empty:
            warnings.append(
                "Some matched part IDs have no linked geometry: "
                + ", ".join(empty[:20])
                + (" …" if len(empty) > 20 else "")
            )
    else:
        if not cfg.skip_ocr and boxes and not labeled:
            warnings.append(
                "OCR ran but no part numbers matched the regex — "
                "try --part-regex or expect a vector-only assembly DXF."
            )

    merged_path = out_dir / f"{stem}_assembly_layers.dxf"
    per_part_paths: dict[str, Path] = {}

    with profile_stage(timings, "export_dxf") if cfg.profile_pipeline else nullcontext():
        if not parts:
            if not vd.is_empty():
                export_segments_only(
                    merged_path,
                    segs,
                    img_height_px=float(h),
                    mm_per_pixel=cfg.mm_per_pixel,
                    vector_drawing=vd,
                )
                if labeled or not cfg.skip_ocr:
                    warnings.append("Wrote a single CAD-style skeleton DXF (no per-part split).")
            else:
                merged_path = out_dir / f"{stem}_EMPTY.dxf"
                export_segments_only(
                    merged_path,
                    [],
                    img_height_px=float(h),
                    mm_per_pixel=cfg.mm_per_pixel,
                    vector_drawing=None,
                )
        else:
            export_merged_dxf(
                merged_path,
                parts,
                unassigned_geom,
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
        "version": 2,
        "pipeline_flow": pipeline_flow_manifest_rows(),
        "input": str(cfg.input_path.resolve()),
        "processed_size_px": {"width": int(w), "height": int(h)},
        "original_raster_shape": {"height": int(pre.original_shape[0]), "width": int(pre.original_shape[1])},
        "preprocess_scale_from_original": pre.scale,
        "mm_per_pixel": cfg.mm_per_pixel,
        "segment_count": len(segs),
        "primitive_count": primitive_totals,
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
        "geometry_quality": {
            "confidence_score": g_quality.confidence_score,
            "mean_segment_length_px": g_quality.mean_segment_length_px,
            "polylines": g_quality.polyline_count,
            "closed_polylines": g_quality.closed_polyline_count,
            "circles": g_quality.circle_count,
            "arcs": g_quality.arc_count,
            "residual_segments": g_quality.residual_count,
            "rectangle_like_polylines": g_quality.rectangle_like_count,
        },
        "parameters": {
            "link_mode": cfg.link_mode,
            "padding_px": cfg.padding_px,
            "max_nearest_px": cfg.max_nearest_px,
            "skip_ocr": cfg.skip_ocr,
            "mask_annotation_via_ocr": cfg.mask_annotation_via_ocr,
            "min_line_length": cfg.min_line_length,
            "segment_merge_distance_px": cfg.segment_merge_distance_px,
            "vector_collinear_merge_angle_deg": cfg.vector_collinear_merge_angle_deg,
            "vector_polyline_rdp_epsilon_px": cfg.vector_polyline_rdp_epsilon_px,
            "legacy_vectorize_hough": cfg.legacy_vectorize_hough,
            "skeleton_annotation_pad_px": cfg.skeleton_annotation_pad_px,
            "enable_skeleton_circles": cfg.enable_skeleton_circles,
            "mask_text_interior_only": cfg.mask_text_interior_only,
            "soft_ink_mask": cfg.soft_ink_mask,
            "enable_topology_clean": cfg.enable_topology_clean,
            "enable_arc_fitting": cfg.enable_arc_fitting,
            "enable_loop_circle_fit": cfg.enable_loop_circle_fit,
            "vectorize_lsd_supplement": cfg.vectorize_lsd_supplement,
            "vectorize_lsd_min_length_px": cfg.vectorize_lsd_min_length_px,
            "geometry_bridge_gap_px": cfg.geometry_bridge_gap_px,
            "dedupe_geometry_residuals": cfg.dedupe_geometry_residuals,
            "raster_dpi_nominal": cfg.raster_dpi_nominal,
            "raster_dpi_effective": nominal_dpi,
            "profile_pipeline": cfg.profile_pipeline,
        },
    }

    if cfg.profile_pipeline and timings:
        manifest["pipeline_timings_ms"] = {k: round(v, 3) for k, v in sorted(timings.items())}

    manifest_path = out_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return RunResult(
        manifest_path=manifest_path,
        warnings=warnings,
        part_ids=[p.part_id for p in parts],
        segment_count=len(segs),
        primitive_count=primitive_totals,
        ocr_box_count=ocr_count,
    )
