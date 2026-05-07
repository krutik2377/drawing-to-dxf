"""Heuristic dimension hints and OCR-adjacent geometry for CAD export (rule-based v1)."""

from __future__ import annotations

import math
import re
from typing import Sequence

from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment

_NUM = re.compile(r"\d")


def text_boxes_with_digits(boxes: Sequence[TextBox], *, min_confidence: float = 0.12) -> list[TextBox]:
    return [
        tb
        for tb in boxes
        if tb.confidence >= min_confidence
        and (tb.text or "").strip()
        and _NUM.search(tb.text or "")
    ]


def _dist_point_to_segment(
    px: float, py: float, s: Segment
) -> tuple[float, float, float, float]:
    ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 < 1e-12:
        d = math.hypot(px - ax, py - ay)
        return d, ax, ay, 0.0
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy), qx, qy, float(t)


def _seg_axis_score(s: Segment) -> float:
    dx, dy = abs(s.x2 - s.x1), abs(s.y2 - s.y1)
    if dx < 1e-6 and dy < 1e-6:
        return 0.0
    return max(dx, dy) / (dx + dy)


def _midpoint(s: Segment) -> tuple[float, float]:
    return (0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2))


def _segment_direction(s: Segment) -> tuple[float, float]:
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    lg = math.hypot(dx, dy)
    if lg < 1e-9:
        return (1.0, 0.0)
    return (dx / lg, dy / lg)


def _extension_stub_toward_point(s: Segment, tx: float, ty: float, *, max_stub_px: float = 28.0) -> Segment | None:
    """Short perpendicular tick from segment midpoint toward a target point."""
    mx, my = _midpoint(s)
    ux, uy = _segment_direction(s)
    px, py = -uy, ux  # perpendicular to stroke direction
    vx, vy = tx - mx, ty - my
    vl = math.hypot(vx, vy)
    if vl < 1e-6:
        return None
    # Choose perpendicular sign pointing toward text
    if vx * px + vy * py < 0:
        px, py = -px, -py
    span = min(max_stub_px, max(6.0, 0.35 * vl))
    return Segment(mx, my, mx + px * span, my + py * span)


def dimension_association_bundle(
    segments: Sequence[Segment],
    text_boxes: Sequence[TextBox],
    *,
    dimension_candidate_segments: Sequence[Segment] | None = None,
    hint_min_length_px: float = 36.0,
    hint_max_dist_to_text_px: float = 22.0,
    hint_min_axis_score: float = 0.92,
    assoc_max_dist_px: float = 48.0,
    assoc_min_axis_score: float = 0.85,
    assoc_min_length_px: float = 18.0,
    min_confidence: float = 0.12,
) -> tuple[list[Segment], list[dict[str, float | str]], list[Segment]]:
    """
    Dimension understanding pass: axis hints + OCR↔geometry association records + extension stubs.

    Returns ``(hint_segments, association_records, extension_stub_segments)``.
    Association records are JSON-serializable (strings/floats).
    """
    hints = dimension_hint_segments(
        segments,
        text_boxes,
        min_length_px=hint_min_length_px,
        max_dist_to_text_px=hint_max_dist_to_text_px,
        min_axis_score=hint_min_axis_score,
        min_confidence=min_confidence,
    )
    boxes = text_boxes_with_digits(text_boxes, min_confidence=min_confidence)
    assoc_pool: list[Segment] = list(dimension_candidate_segments) if dimension_candidate_segments else list(segments)
    if not boxes or not assoc_pool:
        return hints, [], []

    records: list[dict[str, float | str]] = []
    stubs: list[Segment] = []
    seen_tb: set[int] = set()

    for bi, tb in enumerate(boxes):
        cx, cy = tb.center()
        best_s: Segment | None = None
        best_d = assoc_max_dist_px + 1.0
        for s in assoc_pool:
            lg = math.hypot(s.x2 - s.x1, s.y2 - s.y1)
            if lg < assoc_min_length_px:
                continue
            if _seg_axis_score(s) < assoc_min_axis_score:
                continue
            dm, *_ = _dist_point_to_segment(cx, cy, s)
            if dm < best_d:
                best_d = dm
                best_s = s
        if best_s is None or best_d > assoc_max_dist_px:
            continue
        if bi in seen_tb:
            continue
        seen_tb.add(bi)
        txt = (tb.text or "").strip().replace("\n", " ")
        records.append(
            {
                "ocr_text": txt[:120],
                "nearest_axis_segment_dist_px": round(float(best_d), 3),
                "assoc_confidence": round(float(tb.confidence), 4),
                "segment_x1": round(float(best_s.x1), 3),
                "segment_y1": round(float(best_s.y1), 3),
                "segment_x2": round(float(best_s.x2), 3),
                "segment_y2": round(float(best_s.y2), 3),
            }
        )
        stub = _extension_stub_toward_point(best_s, cx, cy)
        if stub is not None:
            stubs.append(stub)

    return hints, records, stubs


def dimension_hint_segments(
    segments: Sequence[Segment],
    text_boxes: Sequence[TextBox],
    *,
    min_length_px: float = 36.0,
    max_dist_to_text_px: float = 22.0,
    min_axis_score: float = 0.92,
    min_confidence: float = 0.12,
) -> list[Segment]:
    """
    Long, near-axis-aligned segments that pass close to a numeric OCR string.

    These are exported on a separate layer as hints — not full dimension entities.
    """
    boxes = text_boxes_with_digits(text_boxes, min_confidence=min_confidence)
    if not boxes or not segments:
        return []

    def near_some_digit_box(s: Segment) -> bool:
        mid = (0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2))
        for tb in boxes:
            d, *_ = _dist_point_to_segment(mid[0], mid[1], s)
            if d <= max_dist_to_text_px:
                return True
            cx = 0.5 * (tb.x0 + tb.x1)
            cy = 0.5 * (tb.y0 + tb.y1)
            d2, _, _, _ = _dist_point_to_segment(cx, cy, s)
            if d2 <= max_dist_to_text_px:
                return True
        return False

    out: list[Segment] = []
    seen: set[tuple[float, float, float, float]] = set()
    for s in segments:
        lg = math.hypot(s.x2 - s.x1, s.y2 - s.y1)
        if lg < min_length_px:
            continue
        if _seg_axis_score(s) < min_axis_score:
            continue
        if not near_some_digit_box(s):
            continue
        key = (round(s.x1, 2), round(s.y1, 2), round(s.x2, 2), round(s.y2, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
