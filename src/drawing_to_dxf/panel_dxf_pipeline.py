"""Preprocess → panel split → vectorize each crop → one DXF per panel.

Stages align with ``raster_pipeline_flow.CANONICAL_PIPELINE_STEPS``; manifests include
``pipeline_flow`` for the same 20-step checklist (panel split augments step 15).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drawing_to_dxf.export_dxf import (
    export_part_dxf,
    export_segments_only,
    export_viewer_layout_dxf,
)
from drawing_to_dxf.geometry_intel import count_corner_rectangles
from drawing_to_dxf.geometry_model import VectorDrawing, exploded_segments_for_sampling
from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.ocr_extract import (
    default_part_pattern,
    extract_text_boxes,
    filter_part_candidates,
)
from drawing_to_dxf.panel_debug import write_panel_trace_debug
from drawing_to_dxf.panel_split import split_panels
from drawing_to_dxf.preprocess import load_image_bgr, load_pdf_page_as_bgr, preprocess
from drawing_to_dxf.raster_pipeline_flow import pipeline_flow_manifest_rows
from drawing_to_dxf.vectorize import extract_skeleton_vector_bundle


@dataclass
class PanelDxfRunConfig:
    input_path: Path
    output_dir: Path
    is_pdf: bool = False
    pdf_page: int = 0
    pdf_dpi: float = 150.0
    max_side: int | None = 4096
    denoise: bool = True
    deskew: bool = True
    min_line_length: int = 20
    panel_min_area: int = 15_000
    panel_min_gap: int = 48
    panel_min_short_side_px: int = 80
    panel_max_aspect_ratio: float = 10.0
    panel_gap_split_fallback: bool = True
    panel_split_strategy: str = "auto"
    exclude_corner_title_block: bool = True
    segment_merge_distance_px: float = 5.0
    vector_collinear_merge_angle_deg: float = 5.0
    vector_polyline_rdp_epsilon_px: float = 1.5
    mask_annotation_via_ocr_crop: bool = True
    skeleton_annotation_pad_px: float = 5.0
    mask_text_interior_only: bool = True
    soft_ink_mask: bool = True
    enable_topology_clean: bool = True
    enable_arc_fitting: bool = True
    enable_loop_circle_fit: bool = True
    enable_skeleton_circles: bool = False
    skip_ocr: bool = False
    ocr_gpu: bool = False
    part_regex: str | None = None
    ocr_min_confidence: float = 0.15
    mm_per_pixel: float = 1.0
    write_viewer_bundle: bool = True
    viewer_layout_gap_mm: float = 25.0
    emit_root_panel_dxfs: bool = False
    trace_debug_dir: Path | None = None
    vectorize_lsd_supplement: bool = False
    vectorize_lsd_min_length_px: float | None = None
    geometry_bridge_gap_px: float = 0.0
    dedupe_geometry_residuals: bool = True
    raster_dpi_nominal: float | None = None


@dataclass
class PanelDxfRunResult:
    manifest_path: Path
    dxf_paths: list[Path]
    warnings: list[str] = field(default_factory=list)
    panel_count: int = 0
    viewer_bundle_dir: Path | None = None
    viewer_primary_dxf: Path | None = None


def _safe_stem(path: Path) -> str:
    s = path.stem
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "drawing"


def _load_bgr(cfg: PanelDxfRunConfig) -> np.ndarray:
    if cfg.is_pdf:
        img = load_pdf_page_as_bgr(str(cfg.input_path), page_index=cfg.pdf_page, dpi=cfg.pdf_dpi)
        if img is None:
            raise FileNotFoundError(f"Cannot read PDF page {cfg.pdf_page}: {cfg.input_path}")
        return img
    img = load_image_bgr(str(cfg.input_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {cfg.input_path}")
    return img


def _analyze_crop_easyocr(
    gray_crop: np.ndarray,
    cfg: PanelDxfRunConfig,
) -> tuple[list, str | None, tuple[float, float] | None]:
    """Return (boxes, best_part_id_match, center_of_that_label_or_None)."""
    pat = re.compile(cfg.part_regex) if cfg.part_regex else default_part_pattern()
    try:
        boxes_local = extract_text_boxes(gray_crop, gpu=cfg.ocr_gpu)
    except ImportError:
        return [], None, None
    labeled_local = filter_part_candidates(boxes_local, pat, min_confidence=cfg.ocr_min_confidence)
    if not labeled_local:
        return boxes_local, None, None
    labeled_local.sort(key=lambda t: (-t[0].confidence, t[0].y0))
    tb_hit, pid = labeled_local[0]
    return boxes_local, pid, tb_hit.center()


def run_panel_dxfs(cfg: PanelDxfRunConfig) -> PanelDxfRunResult:
    """
    Split the processed sheet into panels; write one DXF per panel (vectors + label).

    When ``write_viewer_bundle`` is True (default), per-panel DXFs live only under
    ``viewer/models/panel_XX.dxf`` (no duplicate ``Sample_panel_XX.dxf`` in the output
    root). Use ``emit_root_panel_dxfs`` to also write legacy root copies.

    OCR is optional: when enabled, filenames and PART label prefer the strongest
    regex-matched part id on that crop; otherwise panels are named ``panel_XX``.
    """
    warnings: list[str] = []
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(cfg.input_path)

    bgr = _load_bgr(cfg)
    pre = preprocess(
        bgr,
        max_side=cfg.max_side,
        denoise=cfg.denoise,
        deskew=cfg.deskew,
    )
    if pre.scale < 1.0:
        warnings.append(
            f"Image downscaled for processing (scale={pre.scale:.4f}). "
            "DXF coordinates match processed pixels; calibrate --mm-per-pixel if needed."
        )

    nominal_dpi = cfg.raster_dpi_nominal if cfg.raster_dpi_nominal is not None else cfg.pdf_dpi
    dpi_scale = float(nominal_dpi) / 150.0
    min_len = max(5, int(round(cfg.min_line_length * pre.scale * dpi_scale)))
    h_full, w_full = pre.gray.shape[:2]
    boxes = split_panels(
        pre.gray,
        min_area=cfg.panel_min_area,
        min_gap_px=cfg.panel_min_gap,
        min_short_side_px=cfg.panel_min_short_side_px,
        max_aspect_ratio=cfg.panel_max_aspect_ratio,
        gap_split_fallback=cfg.panel_gap_split_fallback,
        strategy=cfg.panel_split_strategy,
        exclude_corner_title_block=cfg.exclude_corner_title_block,
    )

    if len(boxes) == 1:
        warnings.append(
            "Only one panel detected — for multi-part grids, try lowering "
            "--panel-min-area or --panel-min-gap; dense assembly sheets may "
            "need Phase 2 view/title-based cropping (see docs/PER_PART_DXF_PLAN.md)."
        )

    dxf_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    filename_counts: dict[str, int] = {}
    layout_stack: list[tuple[str, PartGroup, float, float]] = []

    models_dir: Path | None = None
    if cfg.write_viewer_bundle:
        models_dir = out_dir / "viewer" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    skip_ocr_effective = cfg.skip_ocr
    if not skip_ocr_effective:
        try:
            import easyocr  # noqa: F401
        except ImportError:
            warnings.append("OCR requested but easyocr is not installed; use --skip-ocr or pip install -e '.[ocr]'.")
            skip_ocr_effective = True

    for i, (px, py, pw, ph) in enumerate(boxes):
        crop = pre.gray[py : py + ph, px : px + pw]
        crop_h, crop_w = crop.shape[:2]

        ocr_boxes_crop: list = []
        pid: str | None = None
        label_center: tuple[float, float] | None = None
        if not skip_ocr_effective:
            ocr_boxes_crop, pid, label_center = _analyze_crop_easyocr(crop, cfg)

        trace_dbg = cfg.trace_debug_dir is not None
        stage_images: dict[str, Any] = {}
        vd, segs = extract_skeleton_vector_bundle(
            crop,
            annotation_boxes=ocr_boxes_crop if (cfg.mask_annotation_via_ocr_crop and ocr_boxes_crop) else [],
            mask_annotation_regions=bool(cfg.mask_annotation_via_ocr_crop and len(ocr_boxes_crop) > 0),
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
            debug_stages=stage_images if trace_dbg else None,
        )
        if trace_dbg and cfg.trace_debug_dir is not None:
            write_panel_trace_debug(Path(cfg.trace_debug_dir), i, crop, vd, stage_images)
        exploded = exploded_segments_for_sampling(vd)

        if pid is None:
            part_id = f"panel_{i:02d}"
            primary = f"{stem}_panel_{i:02d}"
            lc = (0.5 * crop_w, max(12.0, min(48.0, 0.06 * crop_h)))
        else:
            part_id = pid
            safe_pid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in part_id)
            primary = f"{stem}_part_{safe_pid}"
            lc = label_center if label_center is not None else (0.5 * crop_w, max(12.0, min(48.0, 0.06 * crop_h)))

        n = filename_counts.get(primary, 0) + 1
        filename_counts[primary] = n
        root_name = f"{primary}.dxf" if n == 1 else f"{primary}_{i:02d}.dxf"

        if cfg.write_viewer_bundle:
            assert models_dir is not None
            outp = models_dir / f"panel_{i:02d}.dxf"
        else:
            outp = out_dir / root_name

        lc_x, lc_y = lc
        group = PartGroup(
            part_id=part_id,
            label_center=lc,
            label_box_pad=(lc_x - 1.0, lc_y - 1.0, lc_x + 1.0, lc_y + 1.0),
            segments=list(exploded if exploded else segs),
            vector_drawing=vd,
        )

        if not vd.is_empty():
            export_part_dxf(outp, group, img_height_px=float(crop_h), mm_per_pixel=cfg.mm_per_pixel)
        elif segs:
            export_part_dxf(outp, group, img_height_px=float(crop_h), mm_per_pixel=cfg.mm_per_pixel)
        else:
            export_segments_only(
                outp,
                [],
                img_height_px=float(crop_h),
                mm_per_pixel=cfg.mm_per_pixel,
                vector_drawing=None,
            )
            warnings.append(f"Panel {i}: skeleton produced no primitives — wrote empty DXF.")

        if cfg.emit_root_panel_dxfs and cfg.write_viewer_bundle:
            shutil.copy2(outp, out_dir / root_name)

        layout_stack.append((f"PANEL_{i:02d}", group, float(crop_h), float(crop_w)))

        dxf_paths.append(outp.resolve())
        records.append(
            {
                "index": i,
                "bbox_px": {"x": int(px), "y": int(py), "w": int(pw), "h": int(ph)},
                "crop_size_px": {"width": int(crop_w), "height": int(crop_h)},
                "part_id_guess": pid,
                "exported_part_id": part_id,
                "segment_count": len(exploded if exploded else segs),
                "primitive_polylines": len(vd.polylines),
                "primitive_circles": len(vd.circles),
                "primitive_arcs": len(vd.arcs),
                "primitive_residuals": len(vd.residual_segments),
                "rectangle_like_polylines": count_corner_rectangles(vd),
                "dxf": str(outp.resolve()),
            }
        )
    viewer_bundle_dir: Path | None = None
    viewer_primary_dxf: Path | None = None

    if cfg.write_viewer_bundle and layout_stack:
        viewer_root = out_dir / "viewer"
        layout_name = f"{stem}_autodesk_layout.dxf"
        viewer_primary_dxf = viewer_root / layout_name
        export_viewer_layout_dxf(
            viewer_primary_dxf,
            layout_stack,
            mm_per_pixel=cfg.mm_per_pixel,
            gap_mm=cfg.viewer_layout_gap_mm,
        )
        viewer_bundle_dir = viewer_root
        viewer_hint = {
            "purpose": "Autodesk Viewer and similar online CAD viewers",
            "upload_one_file_only": layout_name,
            "relative_path": f"viewer/{layout_name}",
            "secondary_option": "viewer/models/panel_XX.dxf (one upload at a time)",
            "omit_from_upload": ["*_panels_manifest.json", "*.json bundled as parent"],
        }
        (viewer_root / "autodesk_viewer_upload.json").write_text(
            json.dumps(viewer_hint, indent=2),
            encoding="utf-8",
        )
        (viewer_root / "README.txt").write_text(
            "Autodesk Viewer (https://viewer.autodesk.com/)\n"
            "-----------------------------------------------\n"
            f"- Upload ONLY:  {layout_name}\n"
            "- Do not multi-select DXF + JSON; manifests are metadata, not geometry.\n"
            "- Alternate: upload a single panel from models/panel_XX.dxf\n"
            "- Layer names PANEL_XX lines, PANEL_XX_LBL for PART labels.\n",
            encoding="utf-8",
        )

    manifest_path = out_dir / f"{stem}_panels_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "pipeline": "panels",
                "pipeline_flow": pipeline_flow_manifest_rows(),
                "input": str(cfg.input_path.resolve()),
                "processed_size_px": {"width": int(w_full), "height": int(h_full)},
                "preprocess_scale_from_original": pre.scale,
                "mm_per_pixel": cfg.mm_per_pixel,
                "panels": records,
                "warnings": warnings,
                "parameters": {
                    "skip_ocr_requested": cfg.skip_ocr,
                    "skip_ocr_effective": skip_ocr_effective,
                    "panel_min_area": cfg.panel_min_area,
                    "panel_min_gap": cfg.panel_min_gap,
                    "panel_min_short_side_px": cfg.panel_min_short_side_px,
                    "panel_max_aspect_ratio": cfg.panel_max_aspect_ratio,
                    "panel_gap_split_fallback": cfg.panel_gap_split_fallback,
                    "panel_split_strategy": cfg.panel_split_strategy,
                    "exclude_corner_title_block": cfg.exclude_corner_title_block,
                    "mask_annotation_via_ocr_crop": cfg.mask_annotation_via_ocr_crop,
                    "skeleton_annotation_pad_px": cfg.skeleton_annotation_pad_px,
                    "mask_text_interior_only": cfg.mask_text_interior_only,
                    "soft_ink_mask": cfg.soft_ink_mask,
                    "enable_topology_clean": cfg.enable_topology_clean,
                    "enable_arc_fitting": cfg.enable_arc_fitting,
                    "enable_loop_circle_fit": cfg.enable_loop_circle_fit,
                    "enable_skeleton_circles": cfg.enable_skeleton_circles,
                    "trace_debug_dir": str(Path(cfg.trace_debug_dir).resolve()) if cfg.trace_debug_dir else None,
                    "segment_merge_distance_px": cfg.segment_merge_distance_px,
                    "vector_collinear_merge_angle_deg": cfg.vector_collinear_merge_angle_deg,
                    "vector_polyline_rdp_epsilon_px": cfg.vector_polyline_rdp_epsilon_px,
                    "min_line_length": cfg.min_line_length,
                    "write_viewer_bundle": cfg.write_viewer_bundle,
                    "viewer_layout_gap_mm": cfg.viewer_layout_gap_mm,
                    "emit_root_panel_dxfs": cfg.emit_root_panel_dxfs,
                    "vectorize_lsd_supplement": cfg.vectorize_lsd_supplement,
                    "vectorize_lsd_min_length_px": cfg.vectorize_lsd_min_length_px,
                    "geometry_bridge_gap_px": cfg.geometry_bridge_gap_px,
                    "dedupe_geometry_residuals": cfg.dedupe_geometry_residuals,
                    "raster_dpi_nominal": cfg.raster_dpi_nominal,
                    "raster_dpi_effective": nominal_dpi,
                },
                "outputs": {
                    "manifest": str(manifest_path.resolve()),
                    "viewer_bundle_dir": str(viewer_bundle_dir.resolve()) if viewer_bundle_dir else None,
                    "viewer_primary_dxf": str(viewer_primary_dxf.resolve()) if viewer_primary_dxf else None,
                    "models_glob": "viewer/models/panel_*.dxf" if viewer_bundle_dir else None,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PanelDxfRunResult(
        manifest_path=manifest_path,
        dxf_paths=dxf_paths,
        warnings=warnings,
        panel_count=len(boxes),
        viewer_bundle_dir=viewer_bundle_dir,
        viewer_primary_dxf=viewer_primary_dxf,
    )
