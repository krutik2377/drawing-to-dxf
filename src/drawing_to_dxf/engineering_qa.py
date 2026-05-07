"""Engineering QA: topology, dimensions, circles, rectangles — manifest-friendly reports."""

from __future__ import annotations

import math
from typing import Any, Sequence

from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing
from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.topology_intel import endpoint_degree_counts


def _corner_deviation_from_90(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    l1 = math.hypot(v1x, v1y)
    l2 = math.hypot(v2x, v2y)
    if l1 < 1e-9 or l2 < 1e-9:
        return 90.0
    ca = abs(v1x * v2x + v1y * v2y) / (l1 * l2)
    return abs(90.0 - math.degrees(math.acos(max(-1.0, min(1.0, ca)))))


def run_engineering_qa(
    vd: VectorDrawing,
    segments: Sequence[Segment],
    *,
    dimension_associations: Sequence[dict[str, Any]] | None = None,
    reconstructed_dimensions: Sequence[dict[str, Any]] | None = None,
    corner_tol_deg: float = 8.0,
    degree1_ratio_warn: float = 0.42,
) -> dict[str, Any]:
    """
    Lightweightvalidator for extraction QA (Python side). Returns counts and severity hints.
    """
    issues: list[dict[str, str]] = []

    open_or_skew_rects = 0
    for poly in vd.polylines:
        pts = poly.points
        if not poly.closed or len(pts) != 4:
            continue
        chained = pts + [pts[0], pts[1]]
        bad = False
        for i in range(1, 5):
            dev = _corner_deviation_from_90(chained[i - 1], chained[i], chained[i + 1])
            if dev > corner_tol_deg:
                bad = True
                break
        if bad:
            open_or_skew_rects += 1
    if open_or_skew_rects:
        issues.append(
            {
                "code": "open_or_skew_rectangle",
                "severity": "info",
                "detail": f"{open_or_skew_rects} closed 4-gons deviate > {corner_tol_deg}° from orthogonal",
            }
        )

    bad_circles = 0
    for c in vd.circles:
        if not math.isfinite(c.r) or c.r <= 0.15 or not math.isfinite(c.cx) or not math.isfinite(c.cy):
            bad_circles += 1
    if bad_circles:
        issues.append({"code": "invalid_circle", "severity": "warn", "detail": f"{bad_circles} circles have bad radius/center"})

    deg = endpoint_degree_counts(segments, quant_px=2.0) if segments else {}
    n_vert = max(int(deg.get("vertices_total", 0)), 1)
    d1 = int(deg.get("degree_1", 0))
    ratio_d1 = d1 / n_vert
    broken_topology = ratio_d1 > degree1_ratio_warn
    if broken_topology:
        issues.append(
            {
                "code": "broken_topology",
                "severity": "info",
                "detail": f"high endpoint openness (degree-1 / vertices = {ratio_d1:.3f})",
            }
        )

    disconnected_dims = 0
    if dimension_associations:
        for rec in dimension_associations:
            dpx = rec.get("nearest_axis_segment_dist_px")
            if isinstance(dpx, (int, float)) and float(dpx) > 55.0:
                disconnected_dims += 1
    weak_dim_objects = 0
    if reconstructed_dimensions:
        for obj in reconstructed_dimensions:
            conf = float(obj.get("object_confidence", 0.0))
            if conf < 0.22:
                weak_dim_objects += 1
            dist = obj.get("assoc_dist_text_to_dim_px")
            if isinstance(dist, (int, float)) and float(dist) > 70.0:
                disconnected_dims += 1
    if disconnected_dims:
        issues.append(
            {
                "code": "disconnected_dimension",
                "severity": "info",
                "detail": f"{disconnected_dims} dimension associations look far from axis strokes",
            }
        )
    if weak_dim_objects:
        issues.append(
            {
                "code": "low_confidence_dimension_object",
                "severity": "info",
                "detail": f"{weak_dim_objects} reconstructed dimensions have low confidence",
            }
        )

    warn_count = sum(1 for i in issues if i.get("severity") == "warn")
    return {
        "open_or_skew_rectangles": open_or_skew_rects,
        "invalid_circles": bad_circles,
        "broken_topology_estimate": broken_topology,
        "endpoint_degree_snapshot": deg,
        "disconnected_dimension_hints": disconnected_dims,
        "weak_dimension_objects": weak_dim_objects,
        "issue_list": issues,
        "summary_ok": warn_count == 0 and bad_circles == 0,
    }
