"""Preprocess → panel split → vectorize each crop → one DXF per panel."""

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
from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.ocr_extract import (
    default_part_pattern,
    extract_text_boxes,
    filter_part_candidates,
)
from drawing_to_dxf.panel_split import split_panels
from drawing_to_dxf.preprocess import load_image_bgr, load_pdf_page_as_bgr, preprocess
from drawing_to_dxf.vectorize import Segment, extract_segments


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
    segment_merge_distance_px: float = 5.0
    vector_collinear_merge_angle_deg: float = 5.0
    vector_polyline_rdp_epsilon_px: float = 1.5
    skip_ocr: bool = False
    ocr_gpu: bool = False
    part_regex: str | None = None
    ocr_min_confidence: float = 0.15
    mm_per_pixel: float = 1.0
    write_viewer_bundle: bool = True
    viewer_layout_gap_mm: float = 25.0
    emit_root_panel_dxfs: bool = False


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


def _pick_part_id_from_ocr(
    gray_crop: np.ndarray,
    cfg: PanelDxfRunConfig,
) -> tuple[str | None, tuple[float, float] | None]:
    """Return best (part_id, label_center) from EasyOCR + regex, or (None, None)."""
    pat = re.compile(cfg.part_regex) if cfg.part_regex else default_part_pattern()
    try:
        boxes = extract_text_boxes(gray_crop, gpu=cfg.ocr_gpu)
    except ImportError:
        return None, None
    labeled = filter_part_candidates(boxes, pat, min_confidence=cfg.ocr_min_confidence)
    if not labeled:
        return None, None
    labeled.sort(key=lambda t: (-t[0].confidence, t[0].y0))
    tb, pid = labeled[0]
    return pid, tb.center()


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

    min_len = max(5, int(round(cfg.min_line_length * pre.scale)))
    h_full, w_full = pre.gray.shape[:2]
    boxes = split_panels(
        pre.gray,
        min_area=cfg.panel_min_area,
        min_gap_px=cfg.panel_min_gap,
        min_short_side_px=cfg.panel_min_short_side_px,
        max_aspect_ratio=cfg.panel_max_aspect_ratio,
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
        segs: list[Segment] = extract_segments(
            crop,
            min_line_length=min_len,
            merge_distance=cfg.segment_merge_distance_px,
            collinear_merge_angle_deg=cfg.vector_collinear_merge_angle_deg,
            polyline_rdp_epsilon_px=cfg.vector_polyline_rdp_epsilon_px,
        )

        pid: str | None = None
        label_center: tuple[float, float] | None = None
        if not skip_ocr_effective:
            pid, label_center = _pick_part_id_from_ocr(crop, cfg)

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
            segments=list(segs),
        )

        if segs:
            export_part_dxf(outp, group, img_height_px=float(crop_h), mm_per_pixel=cfg.mm_per_pixel)
        else:
            export_segments_only(outp, [], img_height_px=float(crop_h), mm_per_pixel=cfg.mm_per_pixel)
            warnings.append(f"Panel {i}: no line segments in crop — wrote empty DXF.")

        if cfg.emit_root_panel_dxfs and cfg.write_viewer_bundle:
            shutil.copy2(outp, out_dir / root_name)

        layout_stack.append((f"PANEL_{i:02d}", group, float(crop_h), float(crop_w)))

        dxf_paths.append(outp.resolve())
        records.append(
            {
                "index": i,
                "bbox_px": {"x": px, "y": py, "w": pw, "h": ph},
                "crop_size_px": {"width": int(crop_w), "height": int(crop_h)},
                "part_id_guess": pid,
                "exported_part_id": part_id,
                "segment_count": len(segs),
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
                    "segment_merge_distance_px": cfg.segment_merge_distance_px,
                    "vector_collinear_merge_angle_deg": cfg.vector_collinear_merge_angle_deg,
                    "vector_polyline_rdp_epsilon_px": cfg.vector_polyline_rdp_epsilon_px,
                    "min_line_length": cfg.min_line_length,
                    "write_viewer_bundle": cfg.write_viewer_bundle,
                    "viewer_layout_gap_mm": cfg.viewer_layout_gap_mm,
                    "emit_root_panel_dxfs": cfg.emit_root_panel_dxfs,
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
