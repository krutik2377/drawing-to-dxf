"""End-to-end run: preprocess → vectorize → OCR → link → export rough DXF + manifest.

Extraction stays in OpenCV/skeleton/OCR; **CAD-quality cleanup** (join, overkill, orthogonal
regularization, constraints) is expected **after export** via AutoCAD or an AutoCAD MCP backend—see
``cad_mcp_recipe`` and the ``cad_healing`` block in each manifest. Optional Python-side healing
flags exist for environments without MCP (``--python-*`` on the CLI).

**Semantics / layers:** ``--layered-dxf`` + rule-based and/or ONNX labels classify exploded
segments into GEOMETRY / DIMENSION / BORDER / ANNOTATION overlays; ``--rule-based-semantics``
merges heuristics with ONNX when both are used. ``--emit-linear-dimension-entities`` writes
ezdxf LINEAR dimensions where numeric OCR associates to an axis segment.

Roadmap rows: ``pipeline_flow`` (20 geometric steps) and ``engineering_intelligence_layers``
from ``raster_pipeline_flow`` (intelligence layers are mostly **off by default**; enable with
``--python-engineering-intel`` when needed). The **engineering reconstruction suite** (topology +
CAD graph cleanup + dimension objects + QA) is enabled with ``--reconstruction full`` or
``--python-engineering-reconstruction-suite`` — see ``engineering_reconstruction_suite`` and manifest
keys ``engineering_reconstruction_suite`` / ``engineering_reconstruction_capabilities``.

**Shop-sheet–style QA PNG:** with ``--debug-export-dir`` and semantic splits (``--reconstruction full``
or ``--layered-dxf`` / ``--rule-based-semantics``), the bundle includes ``*_semantic_preview.png``
(dark geometry, red dimensions, green part markers / text strokes) as a raster analogue to layered DXF.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextlib import nullcontext

from drawing_to_dxf.annotations_export import dimension_association_bundle
from drawing_to_dxf.cad_mcp_recipe import build_autocad_mcp_recipe, write_autocad_mcp_recipe_file
from drawing_to_dxf.cad_geometry_rebuild import vector_drawing_from_healed_segments
from drawing_to_dxf.debug_export import write_vectorization_debug_bundle
from drawing_to_dxf.drawing_metrics import extraction_metrics_report
from drawing_to_dxf.engineering_intel_passes import (
    apply_engineering_intelligence_vector_stage,
    topology_intel_on_exploded_segments,
)
from drawing_to_dxf.engineering_reconstruction_suite import apply_engineering_reconstruction_suite
from drawing_to_dxf.export_dxf import (
    export_merged_dxf,
    export_part_dxf,
    export_segments_only,
)
from drawing_to_dxf.hole_pattern import summarize_hole_patterns
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
from drawing_to_dxf.ocr_llm_correct import correct_text_boxes_gemini, correct_text_boxes_ollama
from drawing_to_dxf.preprocess import PreprocessResult, load_image_bgr, load_pdf_page_as_bgr, preprocess
from drawing_to_dxf.raster_pipeline_flow import (
    engineering_intelligence_manifest_rows,
    pipeline_flow_manifest_rows,
)
from drawing_to_dxf.reconstruction_roadmap import (
    engineering_reconstruction_capabilities_rows,
    reconstruction_roadmap_manifest_rows,
)
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
    enable_topology_clean: bool = False
    enable_arc_fitting: bool = True
    enable_loop_circle_fit: bool = True
    vectorize_lsd_supplement: bool = False
    vectorize_lsd_min_length_px: float | None = None
    geometry_bridge_gap_px: float = 0.0
    dedupe_geometry_residuals: bool = True
    raster_dpi_nominal: float | None = None
    profile_pipeline: bool = False
    enable_topology_segment_repair: bool = False
    topology_max_bridge_gap_px: float | None = None
    topology_junction_snap_px: float = 3.0
    topology_bridge_direction_dot_min: float = 0.42
    topology_intersection_extend_px: float = 0.0
    enable_cad_axis_regularization: bool = False
    enable_healed_vector_export: bool = False
    annotation_box_shrink_from_pad_px: float = 0.0
    engineering_layout: bool = False
    ruling_suppress_strength: float | None = None
    multi_scale_ink: bool = False
    protect_hole_rings: bool = True
    enable_constraint_heal: bool = False
    constraint_orthogonal_quad_corner_tol_deg: float = 14.0
    complete_full_arcs_to_circles_min_span_deg: float | None = 315.0
    debug_export_dir: Path | None = None
    export_ocr_text_to_dxf: bool = True
    export_dimension_hints_to_dxf: bool = True
    enable_engineering_intel_passes: bool = False
    reconstruction_preset: str | None = None
    semantic_seg_onnx: Path | None = None
    semantic_seg_config: Path | None = None
    semantic_seg_suppress_dimension: bool = False
    ollama_ocr_correct: bool = False
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_text_model: str = "llama3.2"
    gemini_ocr_correct: bool = False
    gemini_text_model: str = "gemini-2.0-flash"
    layered_dxf: bool = False
    rule_based_semantics: bool = False
    emit_linear_dimension_entities: bool = False
    topology_loop_close_px: float = 0.0
    enable_engineering_reconstruction_suite: bool = False


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
    Execute the raster→DXF flow (see ``raster_pipeline_flow``: 20 geometric steps plus five
    engineering-intelligence layers recorded in the manifest).

    Call order: load (1) → preprocess grayscale/denoise/deskew/scale (2,4,5,16) → OCR (7) →
    vector bundle via adaptive ink mask + morphology + skeleton + circles (3,6,8–10,13) →
    light segment merge/snap on samples (11) → link parts (15) → export rough CAD (17–19);
    CAD healing (20) is documented for AutoCAD MCP; Python healing is opt-in via ``RunConfig``.
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

    if (
        not cfg.skip_ocr
        and boxes
        and cfg.ollama_ocr_correct
    ):
        with profile_stage(timings, "ocr_llm") if cfg.profile_pipeline else nullcontext():
            boxes, ocr_llm_err = correct_text_boxes_ollama(
                boxes,
                host=cfg.ollama_host,
                model=cfg.ollama_text_model,
            )
            if ocr_llm_err:
                warnings.append(ocr_llm_err)

    if (
        not cfg.skip_ocr
        and boxes
        and cfg.gemini_ocr_correct
    ):
        with profile_stage(timings, "ocr_llm_gemini") if cfg.profile_pipeline else nullcontext():
            boxes, gem_err = correct_text_boxes_gemini(
                boxes,
                model=cfg.gemini_text_model,
            )
            if gem_err:
                warnings.append(gem_err)

    semantic_seg_meta: dict[str, Any] | None = None
    pixel_labels = None
    pixel_semantics_meta: dict[str, Any] = {}
    gray_for_vectorize = pre.gray

    if cfg.semantic_seg_onnx is not None:
        with profile_stage(timings, "semantic_seg") if cfg.profile_pipeline else nullcontext():
            try:
                from drawing_to_dxf.semantic_segment import semantic_prepare_gray_for_vectorize

                want_labels = bool(cfg.layered_dxf or cfg.rule_based_semantics)
                if want_labels:
                    gray_for_vectorize, pixel_labels, semantic_seg_meta = semantic_prepare_gray_for_vectorize(
                        pre.gray,
                        onnx_path=cfg.semantic_seg_onnx,
                        config_path=cfg.semantic_seg_config,
                        suppress_dimension=cfg.semantic_seg_suppress_dimension,
                        return_pixel_labels=True,
                    )
                else:
                    gray_for_vectorize, semantic_seg_meta = semantic_prepare_gray_for_vectorize(
                        pre.gray,
                        onnx_path=cfg.semantic_seg_onnx,
                        config_path=cfg.semantic_seg_config,
                        suppress_dimension=cfg.semantic_seg_suppress_dimension,
                    )
                if cfg.rule_based_semantics and pixel_labels is not None:
                    from drawing_to_dxf.raster_semantics import (
                        build_rule_based_pixel_labels,
                        merge_onnx_with_rule_bias,
                    )

                    rl, _rmeta = build_rule_based_pixel_labels(pre.gray, boxes if boxes else None)
                    pixel_labels = merge_onnx_with_rule_bias(pixel_labels, rl)
                    pixel_semantics_meta["hybrid_merge"] = True
            except ImportError as e:
                warnings.append(str(e))
            except Exception as e:  # noqa: BLE001
                warnings.append(
                    f"Semantic segmentation failed ({type(e).__name__}: {e}); using raw gray."
                )
                gray_for_vectorize = pre.gray
                pixel_labels = None
                if cfg.layered_dxf or cfg.rule_based_semantics:
                    try:
                        from drawing_to_dxf.raster_semantics import combined_pixel_labels

                        pixel_labels, pixel_semantics_meta = combined_pixel_labels(
                            pre.gray,
                            boxes if boxes else None,
                            onnx_path=None,
                            rule_based_semantics=True,
                            rule_only=True,
                        )
                        pixel_semantics_meta["onnx_load_error"] = f"{type(e).__name__}: {e}"
                        warnings.append(
                            "Falling back to rule-based pixel labels (--layered-dxf / --rule-based-semantics)."
                        )
                    except Exception as e2:  # noqa: BLE001
                        warnings.append(
                            f"Rule semantics fallback failed ({type(e2).__name__}: {e2}); no pixel labels."
                        )
    elif cfg.layered_dxf or cfg.rule_based_semantics:
        try:
            from drawing_to_dxf.raster_semantics import combined_pixel_labels

            rule_only = bool(cfg.layered_dxf and cfg.semantic_seg_onnx is None)
            pixel_labels, pixel_semantics_meta = combined_pixel_labels(
                pre.gray,
                boxes if boxes else None,
                onnx_path=None,
                rule_based_semantics=cfg.rule_based_semantics,
                rule_only=rule_only,
            )
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Rule semantics failed ({type(e).__name__}: {e}).")
            pixel_labels = None

    vd: VectorDrawing
    topology_metrics: dict[str, int] = {}
    intel_runtime: dict[str, Any] = {}
    stage_dbg: dict[str, Any] | None
    if cfg.debug_export_dir is not None:
        stage_dbg = {}
    else:
        stage_dbg = None
    if cfg.legacy_vectorize_hough:
        with profile_stage(timings, "vectorize") if cfg.profile_pipeline else nullcontext():
            warnings.append("Legacy Hough vectorization is active (thin LINE-only output expected).")
            segs = _legacy_hough_extract_segments(
                gray_for_vectorize,
                min_line_length=min_len,
                merge_distance=cfg.segment_merge_distance_px,
                collinear_merge_angle_deg=cfg.vector_collinear_merge_angle_deg,
                polyline_rdp_epsilon_px=cfg.vector_polyline_rdp_epsilon_px,
            )
        vd = VectorDrawing(residual_segments=list(segs))
        if not segs:
            warnings.append("Legacy Hough found no segments — try lowering min_line_length.")
        if cfg.enable_engineering_intel_passes:
            vd = apply_engineering_intelligence_vector_stage(
                vd,
                ocr_boxes=boxes if (not cfg.skip_ocr and boxes) else None,
                metrics=intel_runtime,
            )
            segs = exploded_segments_for_sampling(vd)
            topology_intel_on_exploded_segments(segs, intel_runtime)
    else:
        with profile_stage(timings, "vectorize") if cfg.profile_pipeline else nullcontext():
            vd, segs = extract_skeleton_vector_bundle(
                gray_for_vectorize,
                annotation_boxes=boxes if (cfg.mask_annotation_via_ocr and not cfg.skip_ocr and boxes) else [],
                mask_annotation_regions=bool(cfg.mask_annotation_via_ocr and not cfg.skip_ocr and boxes),
                min_line_length=min_len,
                merge_distance=cfg.segment_merge_distance_px,
                collinear_merge_angle_deg=cfg.vector_collinear_merge_angle_deg,
                polyline_rdp_epsilon_px=cfg.vector_polyline_rdp_epsilon_px,
                annotation_pad_px=cfg.skeleton_annotation_pad_px,
                annotation_box_shrink_from_pad_px=cfg.annotation_box_shrink_from_pad_px,
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
                enable_topology_segment_repair=cfg.enable_topology_segment_repair,
                topology_max_bridge_gap_px=cfg.topology_max_bridge_gap_px,
                topology_junction_snap_px=cfg.topology_junction_snap_px,
                topology_bridge_direction_dot_min=cfg.topology_bridge_direction_dot_min,
                topology_intersection_extend_px=cfg.topology_intersection_extend_px,
                enable_cad_axis_regularization=cfg.enable_cad_axis_regularization,
                enable_healed_vector_export=cfg.enable_healed_vector_export,
                topology_repair_metrics=topology_metrics,
                engineering_layout=cfg.engineering_layout,
                ruling_suppress_strength=cfg.ruling_suppress_strength,
                multi_scale_ink=cfg.multi_scale_ink,
                protect_hole_rings=cfg.protect_hole_rings,
                enable_constraint_heal=cfg.enable_constraint_heal,
                constraint_orthogonal_quad_corner_tol_deg=cfg.constraint_orthogonal_quad_corner_tol_deg,
                complete_full_arcs_to_circles_min_span_deg=cfg.complete_full_arcs_to_circles_min_span_deg,
                debug_stages=stage_dbg,
                enable_engineering_intel_passes=cfg.enable_engineering_intel_passes,
                engineering_intel_metrics=intel_runtime if cfg.enable_engineering_intel_passes else None,
            )
    if not segs and not cfg.legacy_vectorize_hough:
        warnings.append(
            "No exploded line sampling segments after skeleton tracing — "
            "try lowering --min-line-length or improving raster contrast "
            "(geometry may still include circles/lwpolylines)."
        )

    if cfg.topology_loop_close_px and segs:
        from drawing_to_dxf.topology_repair import bridge_almost_closed_loops

        segs2, n_lc = bridge_almost_closed_loops(list(segs), max_gap_px=float(cfg.topology_loop_close_px))
        segs = segs2
        if n_lc:
            topology_metrics["topology_loop_closures"] = int(n_lc)

    layered_segments: dict[str, list] | None = None
    layer_classify_counts: dict[str, int] = {}
    if pixel_labels is not None and segs:
        from drawing_to_dxf.segment_semantics import split_segments_by_semantic_layer

        layered_segments, layer_classify_counts = split_segments_by_semantic_layer(segs, pixel_labels)

    engineering_suite_manifest: dict[str, Any] | None = None
    if cfg.enable_engineering_reconstruction_suite and segs:
        suite_metrics: dict[str, Any] = {}
        vd, segs, suite_blob = apply_engineering_reconstruction_suite(
            vd,
            list(segs),
            ocr_boxes=boxes if (not cfg.skip_ocr and boxes) else None,
            layered_segments=layered_segments,
            dimension_associations=None,
            metrics_out=suite_metrics,
        )
        engineering_suite_manifest = suite_blob
        if pixel_labels is not None:
            layered_segments, layer_classify_counts = split_segments_by_semantic_layer(segs, pixel_labels)
        vd = vector_drawing_from_healed_segments(vd, list(segs))

    primitive_totals = len(vd.polylines) + len(vd.circles) + len(vd.arcs) + len(vd.residual_segments)

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

    dim_hint_segs: list = []
    dim_assoc_records: list = []
    dim_pool = layered_segments.get("DIMENSION") if layered_segments else None
    if boxes and segs and not cfg.skip_ocr:
        hints, dim_assoc_records, stubs = dimension_association_bundle(
            segs,
            boxes,
            dimension_candidate_segments=(dim_pool if dim_pool else None),
        )
        if cfg.export_dimension_hints_to_dxf:
            dim_hint_segs = list(hints) + list(stubs)

    ocr_for_dxf: list[TextBox] | None = (
        list(boxes) if (cfg.export_ocr_text_to_dxf and boxes and not cfg.skip_ocr) else None
    )

    debug_report: dict[str, Any] | None = None
    if cfg.debug_export_dir is not None:
        part_centers = (
            [(p.part_id, float(p.label_center[0]), float(p.label_center[1])) for p in parts]
            if parts
            else None
        )
        debug_report = write_vectorization_debug_bundle(
            cfg.debug_export_dir.resolve(),
            stem=stem,
            stages=stage_dbg if stage_dbg is not None else {},
            vector_drawing=vd,
            segments=segs,
            image_height_for_geojson=float(h),
            layered_segments=layered_segments,
            gray_for_semantic_preview=pre.gray if layered_segments else None,
            part_label_centers=part_centers,
        )

    xtra: dict[str, Any] = {}
    if topology_metrics:
        xtra["topology_segment_repair"] = topology_metrics
    if engineering_suite_manifest:
        xtra["engineering_reconstruction_passes"] = engineering_suite_manifest.get("pass_metrics", {})
    extraction_metrics = extraction_metrics_report(vd, segs, boxes, extra=xtra)

    sem_export = layered_segments if cfg.layered_dxf else None

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
                    ocr_text_boxes=ocr_for_dxf,
                    dimension_hint_segments=dim_hint_segs or None,
                    semantic_layer_segments=sem_export,
                    linear_dimension_associations=dim_assoc_records if cfg.emit_linear_dimension_entities else None,
                    emit_linear_dimension_entities=cfg.emit_linear_dimension_entities,
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
                ocr_text_boxes=ocr_for_dxf,
                dimension_hint_segments=dim_hint_segs or None,
                semantic_layer_segments=sem_export,
                linear_dimension_associations=dim_assoc_records if cfg.emit_linear_dimension_entities else None,
                emit_linear_dimension_entities=cfg.emit_linear_dimension_entities,
            )
            for pg in parts:
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in pg.part_id)
                outp = out_dir / f"{stem}_part_{safe}.dxf"
                export_part_dxf(
                    outp,
                    pg,
                    img_height_px=float(h),
                    mm_per_pixel=cfg.mm_per_pixel,
                    ocr_text_boxes=ocr_for_dxf,
                )
                per_part_paths[pg.part_id] = outp

    outputs_block: dict[str, Any] = {
        "assembly_layers_dxf": str(merged_path.resolve()),
        "per_part": {k: str(v.resolve()) for k, v in per_part_paths.items()},
    }
    if debug_report and debug_report.get("semantic_preview_png"):
        outputs_block["semantic_preview_png"] = debug_report["semantic_preview_png"]

    manifest: dict[str, Any] = {
        "version": 2,
        "pipeline_flow": pipeline_flow_manifest_rows(),
        "engineering_intelligence_layers": engineering_intelligence_manifest_rows(),
        "reconstruction_roadmap": reconstruction_roadmap_manifest_rows(),
        "engineering_reconstruction_capabilities": engineering_reconstruction_capabilities_rows(),
        "reconstruction_preset": cfg.reconstruction_preset,
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
        "outputs": outputs_block,
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
        "topology_segment_repair": topology_metrics if topology_metrics else {},
        "extraction_metrics": extraction_metrics,
        "dimension_hint_segment_count": len(dim_hint_segs),
        "dimension_association_count": len(dim_assoc_records),
        "dimension_associations": dim_assoc_records if dim_assoc_records else [],
        "hole_patterns": summarize_hole_patterns(vd.circles),
        "pixel_semantics": {
            **(pixel_semantics_meta or {}),
            "layer_classify_counts": layer_classify_counts,
            "label_map_available": pixel_labels is not None,
        },
        "engineering_intelligence_runtime": intel_runtime if cfg.enable_engineering_intel_passes else {},
        "engineering_reconstruction_suite": engineering_suite_manifest or {},
        "semantic_segmentation": semantic_seg_meta,
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
            "enable_topology_segment_repair": cfg.enable_topology_segment_repair,
            "topology_max_bridge_gap_px": cfg.topology_max_bridge_gap_px,
            "topology_junction_snap_px": cfg.topology_junction_snap_px,
            "topology_bridge_direction_dot_min": cfg.topology_bridge_direction_dot_min,
            "topology_intersection_extend_px": cfg.topology_intersection_extend_px,
            "enable_cad_axis_regularization": cfg.enable_cad_axis_regularization,
            "enable_healed_vector_export": cfg.enable_healed_vector_export,
            "annotation_box_shrink_from_pad_px": cfg.annotation_box_shrink_from_pad_px,
            "engineering_layout": cfg.engineering_layout,
            "ruling_suppress_strength": cfg.ruling_suppress_strength,
            "multi_scale_ink": cfg.multi_scale_ink,
            "protect_hole_rings": cfg.protect_hole_rings,
            "enable_constraint_heal": cfg.enable_constraint_heal,
            "constraint_orthogonal_quad_corner_tol_deg": cfg.constraint_orthogonal_quad_corner_tol_deg,
            "complete_full_arcs_to_circles_min_span_deg": cfg.complete_full_arcs_to_circles_min_span_deg,
            "debug_export_dir": str(cfg.debug_export_dir.resolve()) if cfg.debug_export_dir else None,
            "export_ocr_text_to_dxf": cfg.export_ocr_text_to_dxf,
            "export_dimension_hints_to_dxf": cfg.export_dimension_hints_to_dxf,
            "enable_engineering_intel_passes": cfg.enable_engineering_intel_passes,
            "reconstruction_preset": cfg.reconstruction_preset,
            "semantic_seg_onnx": str(cfg.semantic_seg_onnx) if cfg.semantic_seg_onnx else None,
            "semantic_seg_config": str(cfg.semantic_seg_config) if cfg.semantic_seg_config else None,
            "semantic_seg_suppress_dimension": cfg.semantic_seg_suppress_dimension,
            "ollama_ocr_correct": cfg.ollama_ocr_correct,
            "ollama_host": cfg.ollama_host,
            "ollama_text_model": cfg.ollama_text_model,
            "gemini_ocr_correct": cfg.gemini_ocr_correct,
            "gemini_text_model": cfg.gemini_text_model,
            "layered_dxf": cfg.layered_dxf,
            "rule_based_semantics": cfg.rule_based_semantics,
            "emit_linear_dimension_entities": cfg.emit_linear_dimension_entities,
            "topology_loop_close_px": cfg.topology_loop_close_px,
            "enable_engineering_reconstruction_suite": cfg.enable_engineering_reconstruction_suite,
        },
    }

    recipe_dxfs = [str(merged_path.resolve())]
    recipe_dxfs.extend(str(p.resolve()) for p in per_part_paths.values())
    recipe_path = out_dir / f"{stem}_autocad_mcp_recipe.json"
    write_autocad_mcp_recipe_file(
        recipe_path,
        build_autocad_mcp_recipe(dxfs=recipe_dxfs, mm_per_pixel=cfg.mm_per_pixel),
    )
    manifest["cad_healing"] = {
        "strategy": "hybrid_raster_extract_then_autocad_mcp",
        "recipe_file": str(recipe_path.resolve()),
        "python_inprocess_fallback": {
            "topology_clean": cfg.enable_topology_clean,
            "topology_segment_repair": cfg.enable_topology_segment_repair,
            "constraint_heal": cfg.enable_constraint_heal,
            "engineering_intel_passes": cfg.enable_engineering_intel_passes,
            "engineering_reconstruction_suite": cfg.enable_engineering_reconstruction_suite,
        },
    }

    if debug_report is not None:
        manifest["debug_export"] = debug_report

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
