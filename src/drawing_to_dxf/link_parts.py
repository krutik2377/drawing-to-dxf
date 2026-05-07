"""Associate line segments with part labels (heuristic)."""

from __future__ import annotations

from dataclasses import dataclass, field

from drawing_to_dxf.geometry_model import VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox, _expand_box, point_in_box
from drawing_to_dxf.segment_types import Segment


@dataclass
class PartGroup:
    part_id: str
    label_center: tuple[float, float]
    label_box_pad: tuple[float, float, float, float]
    segments: list[Segment] = field(default_factory=list)
    vector_drawing: VectorDrawing | None = None


def _seg_id(seg: Segment) -> int:
    return id(seg)


def _dedupe_segments(segs: list[Segment]) -> list[Segment]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[Segment] = []
    for s in segs:
        t = (
            int(round(s.x1)),
            int(round(s.y1)),
            int(round(s.x2)),
            int(round(s.y2)),
        )
        if t in seen:
            continue
        seen.add(t)
        out.append(s)
    return out


def _distance_sq(p: tuple[float, float], q: tuple[float, float]) -> float:
    dx, dy = p[0] - q[0], p[1] - q[1]
    return dx * dx + dy * dy


def link_segments_to_parts(
    segments: list[Segment],
    labeled_boxes: list[tuple[TextBox, str]],
    *,
    padding_px: float = 120.0,
    max_nearest_px: float = 180.0,
    mode: str = "hybrid",
) -> tuple[list[PartGroup], list[Segment]]:
    """
    Assign segments to parts.

    Modes:
    - bbox: segment midpoint must fall inside expanded OCR bounding box.
    - nearest: assign each segment to nearest part centroid within max_nearest_px.
    - hybrid: bbox first, then nearest for leftovers (recommended).
    """
    if mode not in ("bbox", "nearest", "hybrid"):
        raise ValueError("mode must be bbox|nearest|hybrid")

    seen: dict[str, TextBox] = {}
    for tb, pid in labeled_boxes:
        if pid not in seen:
            seen[pid] = tb

    groups: dict[str, PartGroup] = {}
    for pid, tb in seen.items():
        groups[pid] = PartGroup(
            part_id=pid,
            label_center=tb.center(),
            label_box_pad=_expand_box(tb, padding_px),
            segments=[],
        )

    assigned_ids: set[int] = set()

    def assign_bbox() -> None:
        nonlocal assigned_ids
        for seg in segments:
            mx, my = seg.midpoint()
            for pg in groups.values():
                if point_in_box(mx, my, pg.label_box_pad):
                    pg.segments.append(seg)
                    assigned_ids.add(_seg_id(seg))
                    break

    def assign_nearest(pool: list[Segment]) -> None:
        nonlocal assigned_ids
        if not groups:
            return
        centers = [(pid, groups[pid].label_center) for pid in groups]
        max_d2 = max_nearest_px * max_nearest_px
        for seg in pool:
            if _seg_id(seg) in assigned_ids:
                continue
            mx, my = seg.midpoint()
            best_pid: str | None = None
            best_d2 = max_d2
            for pid, c in centers:
                d2 = _distance_sq((mx, my), c)
                if d2 < best_d2:
                    best_d2 = d2
                    best_pid = pid
            if best_pid is not None:
                groups[best_pid].segments.append(seg)
                assigned_ids.add(_seg_id(seg))

    if mode == "nearest":
        assign_nearest(segments)
    elif mode == "bbox":
        assign_bbox()
    else:
        assign_bbox()
        assign_nearest(segments)

    unassigned: list[Segment] = []
    for seg in segments:
        if _seg_id(seg) not in assigned_ids:
            unassigned.append(seg)

    for pg in groups.values():
        pg.segments = _dedupe_segments(pg.segments)

    return list(groups.values()), _dedupe_segments(unassigned)
