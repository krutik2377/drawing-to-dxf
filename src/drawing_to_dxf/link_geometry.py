"""Assign traced vector primitives (polylines, circles, arcs) to OCR part groups."""

from __future__ import annotations

from drawing_to_dxf.geometry_model import VectorDrawing, poly_mass_centroid
from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.ocr_extract import TextBox, _expand_box, point_in_box


def _distance_sq(px: tuple[float, float], qx: tuple[float, float]) -> float:
    dx = px[0] - qx[0]
    dy = px[1] - qx[1]
    return dx * dx + dy * dy


def _nearest_pid(
    point: tuple[float, float],
    groups: dict[str, PartGroup],
    *,
    max_d2: float,
) -> str | None:
    best_pid: str | None = None
    bd = max_d2
    for pid, pg in groups.items():
        d2 = _distance_sq(point, pg.label_center)
        if d2 < bd:
            bd, best_pid = d2, pid
    return best_pid


def link_vector_geometry_to_parts(
    drawing: VectorDrawing,
    labeled_boxes: list[tuple[TextBox, str]],
    *,
    padding_px: float = 120.0,
    max_nearest_px: float = 180.0,
    mode: str = "hybrid",
) -> tuple[list[PartGroup], VectorDrawing]:
    """Partition primitives by expanded OCR bounding boxes then nearest centroid (hybrid)."""
    if mode not in ("bbox", "nearest", "hybrid"):
        raise ValueError("mode must be bbox|nearest|hybrid")

    seen_tb: dict[str, TextBox] = {}
    for tb, pid in labeled_boxes:
        if pid not in seen_tb:
            seen_tb[pid] = tb

    groups: dict[str, PartGroup] = {}
    for pid, tb in seen_tb.items():
        groups[pid] = PartGroup(
            part_id=pid,
            label_center=tb.center(),
            label_box_pad=_expand_box(tb, padding_px),
            segments=[],
            vector_drawing=VectorDrawing(),
        )

    max_d2 = max_nearest_px * max_nearest_px

    def bbox_pid(point: tuple[float, float]) -> str | None:
        hits = [pid for pid, pg in groups.items() if point_in_box(point[0], point[1], pg.label_box_pad)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            hits.sort(key=lambda pid: float(_distance_sq(groups[pid].label_center, point)))
            return hits[0]
        return None

    def assign(point: tuple[float, float]) -> str | None:
        if mode in ("bbox", "hybrid"):
            bp = bbox_pid(point)
            if bp is not None:
                return bp
            if mode == "bbox":
                return None
        return _nearest_pid(point, groups, max_d2=max_d2)

    rebuilt: dict[str, VectorDrawing] = {pid: VectorDrawing() for pid in groups}
    unassigned = VectorDrawing()

    for poly in drawing.polylines:
        cen = poly.points
        if not cen:
            continue
        pid = assign(poly_mass_centroid(cen))
        tgt = rebuilt.get(pid) if pid else None
        if tgt is None:
            unassigned.polylines.append(poly)
        else:
            tgt.polylines.append(poly)

    for circ in drawing.circles:
        pid = assign((circ.cx, circ.cy))
        tgt = rebuilt.get(pid) if pid else None
        if tgt is None:
            unassigned.circles.append(circ)
        else:
            tgt.circles.append(circ)

    for arc in drawing.arcs:
        pid = assign((arc.cx, arc.cy))
        tgt = rebuilt.get(pid) if pid else None
        if tgt is None:
            unassigned.arcs.append(arc)
        else:
            tgt.arcs.append(arc)

    for seg in drawing.residual_segments:
        mx, my = seg.midpoint()
        pid = assign((mx, my))
        tgt = rebuilt.get(pid) if pid else None
        if tgt is None:
            unassigned.residual_segments.append(seg)
        else:
            tgt.residual_segments.append(seg)

    out_parts: list[PartGroup] = []
    for pid, grp in groups.items():
        grp.vector_drawing = rebuilt.get(pid, VectorDrawing())
        out_parts.append(grp)

    return out_parts, unassigned
