"""Heuristic dimension *object* reconstruction: dim line, extensions, arrows, measurement text."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment

_NUM = re.compile(r"\d")


def _axis_score(s: Segment) -> float:
    dx, dy = abs(s.x2 - s.x1), abs(s.y2 - s.y1)
    if dx < 1e-6 and dy < 1e-6:
        return 0.0
    return max(dx, dy) / (dx + dy)


def _len(s: Segment) -> float:
    return math.hypot(s.x2 - s.x1, s.y2 - s.y1)


def _dist_point_to_segment(px: float, py: float, s: Segment) -> tuple[float, float, float]:
    ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay), ax, ay
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy), qx, qy


def _nearest_axis_segment(
    px: float,
    py: float,
    pool: Sequence[Segment],
    *,
    min_len: float,
    min_axis: float,
) -> tuple[Segment | None, float]:
    best: Segment | None = None
    best_d = 1e18
    for s in pool:
        if _len(s) < min_len or _axis_score(s) < min_axis:
            continue
        d, _, _ = _dist_point_to_segment(px, py, s)
        if d < best_d:
            best_d = d
            best = s
    return best, best_d


def _extension_candidates(
    dim: Segment,
    pool: Sequence[Segment],
    *,
    max_dist: float = 22.0,
    min_len: float = 5.0,
    max_len: float = 120.0,
) -> list[dict[str, float]]:
    """Short axis-aligned strokes near dim line endpoints, perpendicular-ish."""
    ax, ay, bx, by = dim.x1, dim.y1, dim.x2, dim.y2
    mids = [(ax, ay), (bx, by)]
    out: list[dict[str, float]] = []
    daxis = _axis_score(dim)
    for s in pool:
        lg = _len(s)
        if lg < min_len or lg > max_len:
            continue
        if _axis_score(s) < 0.88:
            continue
        # must be roughly perpendicular to dim line if dim is axis-aligned
        if daxis > 0.92:
            horiz_dim = abs(by - ay) < 2.5
            horiz_s = abs(s.y2 - s.y1) < 2.5
            if horiz_dim and horiz_s:
                continue
            if not horiz_dim and not horiz_s:
                continue
        for mx, my in mids:
            d1 = math.hypot(s.x1 - mx, s.y1 - my)
            d2 = math.hypot(s.x2 - mx, s.y2 - my)
            if min(d1, d2) <= max_dist:
                out.append(
                    {
                        "x1": s.x1,
                        "y1": s.y1,
                        "x2": s.x2,
                        "y2": s.y2,
                        "endpoint_dist": round(min(d1, d2), 3),
                    }
                )
                break
    return out[:6]


def _arrow_candidates(
    dim: Segment,
    pool: Sequence[Segment],
    *,
    max_len: float = 14.0,
    min_len: float = 2.0,
    near_end_px: float = 18.0,
) -> list[dict[str, float]]:
    tips: list[dict[str, float]] = []
    ends = [(dim.x1, dim.y1), (dim.x2, dim.y2)]
    for s in pool:
        lg = _len(s)
        if lg < min_len or lg > max_len:
            continue
        mx, my = 0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2)
        for ex, ey in ends:
            if math.hypot(mx - ex, my - ey) <= near_end_px:
                tips.append(
                    {
                        "x1": s.x1,
                        "y1": s.y1,
                        "x2": s.x2,
                        "y2": s.y2,
                        "length_px": round(lg, 3),
                    }
                )
                break
    return tips[:8]


def reconstruct_dimension_objects(
    segments: Sequence[Segment],
    text_boxes: Sequence[TextBox] | None,
    *,
    dimension_layer_segments: Sequence[Segment] | None = None,
    min_text_confidence: float = 0.12,
    dim_line_min_px: float = 28.0,
    dim_line_axis_min: float = 0.9,
    max_text_to_dim_px: float = 85.0,
) -> list[dict[str, Any]]:
    """
    Build structured dimension dicts (JSON-serializable) from strokes + numeric OCR.

    Not a full ANSI dimension model — heuristic association for CAD export / QA.
    """
    if not segments or not text_boxes:
        return []
    pool = list(dimension_layer_segments) if dimension_layer_segments else list(segments)
    numeric_boxes = [
        tb
        for tb in text_boxes
        if tb.confidence >= min_text_confidence
        and (tb.text or "").strip()
        and _NUM.search(tb.text or "")
    ]
    if not numeric_boxes:
        return []

    used_seg_ids: set[int] = set()
    results: list[dict[str, Any]] = []
    for bi, tb in enumerate(numeric_boxes):
        cx, cy = tb.center()
        best, d = _nearest_axis_segment(
            cx,
            cy,
            pool,
            min_len=dim_line_min_px,
            min_axis=dim_line_axis_min,
        )
        if best is None or d > max_text_to_dim_px:
            continue
        sid = id(best)
        if sid in used_seg_ids:
            continue
        used_seg_ids.add(sid)
        exts = _extension_candidates(best, pool)
        arrows = _arrow_candidates(best, pool)
        txt = (tb.text or "").strip().replace("\n", " ")
        results.append(
            {
                "id": bi,
                "measurement_text": txt[:160],
                "text_confidence": round(float(tb.confidence), 4),
                "text_center_px": {"x": round(cx, 3), "y": round(cy, 3)},
                "dimension_line": {
                    "x1": best.x1,
                    "y1": best.y1,
                    "x2": best.x2,
                    "y2": best.y2,
                    "length_px": round(_len(best), 3),
                },
                "extension_lines": exts,
                "arrow_candidates": arrows,
                "assoc_dist_text_to_dim_px": round(float(d), 3),
                "object_confidence": round(max(0.08, min(1.0, float(tb.confidence) * (1.0 - d / 120.0))), 4),
            }
        )
    return results
