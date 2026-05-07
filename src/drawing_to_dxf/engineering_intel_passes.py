"""Orchestrate engineering-intelligence passes after skeleton extraction (semantics, constraints, topology stats)."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

from drawing_to_dxf.constraint_heal import align_parallel_residual_clusters, snap_axis_aligned_residual_segments
from drawing_to_dxf.geometry_intel import count_corner_rectangles
from drawing_to_dxf.geometry_model import VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment


def _drawing_bbox(dwg: VectorDrawing) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for poly in dwg.polylines:
        for x, y in poly.points:
            xs.append(x)
            ys.append(y)
    for c in dwg.circles:
        xs.extend([c.cx - c.r, c.cx + c.r])
        ys.extend([c.cy - c.r, c.cy + c.r])
    for s in dwg.residual_segments:
        xs.extend([s.x1, s.x2])
        ys.extend([s.y1, s.y2])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def semantic_reasoning_snapshot(
    vd: VectorDrawing,
    ocr_boxes: Sequence[TextBox] | None,
    *,
    bbox_pad_px: float = 24.0,
) -> dict[str, Any]:
    """Coarse semantic metrics (part-ready geometry vs annotations)."""
    rects = int(count_corner_rectangles(vd))
    bbox = _drawing_bbox(vd)
    n_ocr = len(ocr_boxes or [])
    inside = 0
    if bbox and ocr_boxes:
        x0, y0, x1, y1 = bbox
        x0 -= bbox_pad_px
        y0 -= bbox_pad_px
        x1 += bbox_pad_px
        y1 += bbox_pad_px
        for tb in ocr_boxes:
            cx, cy = tb.center()
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                inside += 1
    density = "high" if n_ocr > 35 else ("medium" if n_ocr > 12 else "low")
    return {
        "semantic_rectangle_like_polylines": rects,
        "semantic_circle_primitives": len(vd.circles),
        "semantic_arc_primitives": len(vd.arcs),
        "semantic_polyline_chains": len(vd.polylines),
        "semantic_residual_segments": len(vd.residual_segments),
        "semantic_ocr_box_count": n_ocr,
        "semantic_ocr_boxes_in_geometry_extent": inside,
        "semantic_annotation_density": density,
    }


def cad_regularize_residual_segments(vd: VectorDrawing, *, dedupe_fn: Any | None = None) -> VectorDrawing:
    """Light dedupe after constraint snaps if caller supplies dedupe_segment_list."""
    if dedupe_fn is None or not vd.residual_segments:
        return vd
    rd = dedupe_fn(list(vd.residual_segments))
    if len(rd) == len(vd.residual_segments):
        return vd
    return VectorDrawing(
        polylines=list(vd.polylines),
        circles=list(vd.circles),
        arcs=list(vd.arcs),
        residual_segments=rd,
    )


def apply_engineering_intelligence_vector_stage(
    vd: VectorDrawing,
    *,
    ocr_boxes: Sequence[TextBox] | None,
    metrics: MutableMapping[str, Any],
    enable_semantic_snapshot: bool = True,
    enable_residual_axis_snap: bool = True,
    enable_parallel_cluster_snap: bool = True,
    residual_axis_snap_tol_deg: float = 3.0,
    residual_axis_snap_min_px: float = 10.0,
    parallel_cluster_angle_tol_deg: float = 2.5,
    parallel_cluster_offset_px: float = 2.0,
    parallel_cluster_min_size: int = 3,
    parallel_cluster_min_length_px: float = 28.0,
    dedupe_residuals_after_snaps: bool = True,
) -> VectorDrawing:
    """
    Mutate-stage intelligence on :class:`VectorDrawing` before exploding to sampling segments.

    Fills ``metrics`` with semantic + constraint counters.
    """
    if enable_semantic_snapshot:
        metrics.update(semantic_reasoning_snapshot(vd, ocr_boxes))

    rs = list(vd.residual_segments)
    if not rs:
        metrics.setdefault("constraint_residual_axis_snaps", 0)
        metrics.setdefault("constraint_parallel_cluster_moves", 0)
        return vd

    if enable_residual_axis_snap:
        rs, n_axis = snap_axis_aligned_residual_segments(
            rs,
            angle_tol_deg=residual_axis_snap_tol_deg,
            min_length_px=residual_axis_snap_min_px,
        )
        metrics["constraint_residual_axis_snaps"] = n_axis
    else:
        metrics.setdefault("constraint_residual_axis_snaps", 0)

    if enable_parallel_cluster_snap:
        rs, n_par = align_parallel_residual_clusters(
            rs,
            angle_tol_deg=parallel_cluster_angle_tol_deg,
            offset_snap_px=parallel_cluster_offset_px,
            min_cluster_size=parallel_cluster_min_size,
            min_length_px=parallel_cluster_min_length_px,
        )
        metrics["constraint_parallel_cluster_moves"] = n_par
    else:
        metrics.setdefault("constraint_parallel_cluster_moves", 0)

    vd2 = VectorDrawing(
        polylines=list(vd.polylines),
        circles=list(vd.circles),
        arcs=list(vd.arcs),
        residual_segments=rs,
    )

    if dedupe_residuals_after_snaps:
        from drawing_to_dxf.geometry_intel import dedupe_segment_list

        vd2 = cad_regularize_residual_segments(vd2, dedupe_fn=dedupe_segment_list)

    return vd2


def topology_intel_on_exploded_segments(
    segments: Sequence[Segment],
    metrics: MutableMapping[str, Any],
    *,
    quant_px: float = 2.0,
) -> None:
    from drawing_to_dxf.topology_intel import endpoint_degree_counts

    counts = endpoint_degree_counts(segments, quant_px=quant_px)
    for k, v in counts.items():
        metrics[f"topology_{k}"] = int(v)
