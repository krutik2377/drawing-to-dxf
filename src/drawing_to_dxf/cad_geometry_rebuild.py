"""Align exported VectorDrawing with healed segment topology (DXF ↔ graph repair)."""

from __future__ import annotations

import math

from drawing_to_dxf.geometry_model import CircleDef, VectorDrawing
from drawing_to_dxf.segment_types import Segment


def filter_segments_coincident_with_circles(
    segs: list[Segment],
    circles: list[CircleDef],
    *,
    rim_tol_px: float = 2.25,
    min_radius_px: float = 5.5,
) -> list[Segment]:
    """
    Drop LINE chords that only tessellate a known CIRCLE primitive so we do not duplicate
    rim geometry at export time.
    """
    if not circles or not segs:
        return list(segs)
    big = [c for c in circles if c.r >= min_radius_px]
    if not big:
        return list(segs)
    keep: list[Segment] = []
    tol = float(rim_tol_px)
    for s in segs:
        skip = False
        for c in big:
            ra = math.hypot(s.x1 - c.cx, s.y1 - c.cy)
            rb = math.hypot(s.x2 - c.cx, s.y2 - c.cy)
            da = abs(ra - c.r)
            db = abs(rb - c.r)
            mx, my = 0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2)
            dm = abs(math.hypot(mx - c.cx, my - c.cy) - c.r)
            if da < tol and db < tol and dm < tol * 1.15:
                skip = True
                break
        if not skip:
            keep.append(s)
    return keep


def vector_drawing_from_healed_segments(
    vd: VectorDrawing,
    healed_segs: list[Segment],
    *,
    rim_tol_px: float = 2.25,
) -> VectorDrawing:
    """
    Replace traced LWPolyline geometry with the repaired LINE network so DXF matches T/X
    junction healing. Preserves CIRCLE / ARC entities and strips redundant rim chords.
    """
    filtered = filter_segments_coincident_with_circles(
        healed_segs, list(vd.circles), rim_tol_px=rim_tol_px
    )
    return VectorDrawing(
        polylines=[],
        circles=list(vd.circles),
        arcs=list(vd.arcs),
        residual_segments=filtered,
    )
