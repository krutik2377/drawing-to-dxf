"""Write DXF files with ezdxf (AutoCAD-compatible; no AutoCAD install required)."""

from __future__ import annotations

from math import atan2, cos, degrees, radians, sin
from pathlib import Path

import ezdxf
from ezdxf import units

from drawing_to_dxf.geometry_model import VectorDrawing
from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.segment_types import Segment


def image_xy_to_dxf(x: float, y: float, img_height_px: float, scale: float) -> tuple[float, float]:
    """Map top-left image coordinates to DXF XY with Y up."""
    return (x * scale, (img_height_px - y) * scale)


def img_arc_point(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    th = radians(angle_deg % 360.0)
    return (cx + r * cos(th), cy + r * sin(th))


def _dxf_polar_angle(px: float, py: float, cx: float, cy: float, img_height_px: float, scale: float) -> float:
    dx, dy = image_xy_to_dxf(px, py, img_height_px, scale)
    dxc, dyc = image_xy_to_dxf(cx, cy, img_height_px, scale)
    return degrees(atan2(dy - dyc, dx - dxc))


def _normalize360(d: float) -> float:
    x = d % 360.0
    return x + 360.0 if x < 0 else x


def _emit_vector_entities(
    msp,
    drawing: VectorDrawing,
    *,
    img_height_px: float,
    scale: float,
    layer_base: str,
    offset_x: float = 0.0,
) -> None:
    """Emit LWPOLYLINE, ARC, CIRCLE; residual strokes as LINE."""
    lyr = layer_base[:255]

    for poly in drawing.polylines:
        pts_xy: list[tuple[float, float]] = []
        for px, py in poly.points:
            dx, dy = image_xy_to_dxf(px, py, img_height_px, scale)
            pts_xy.append((dx + offset_x, dy))

        clo = poly.closed and len(pts_xy) >= 3
        if len(pts_xy) >= 2:
            msp.add_lwpolyline(
                pts_xy,
                close=clo,
                dxfattribs={"layer": lyr},
            )

    for circ in drawing.circles:
        cx, cy = image_xy_to_dxf(circ.cx, circ.cy, img_height_px, scale)
        msp.add_circle(
            (cx + offset_x, cy),
            radius=float(circ.r) * scale,
            dxfattribs={"layer": lyr},
        )

    for arc in drawing.arcs:
        xa, ya = img_arc_point(arc.cx, arc.cy, arc.r, arc.start_angle_deg)
        xb, yb = img_arc_point(arc.cx, arc.cy, arc.r, arc.end_angle_deg)
        sa = _normalize360(_dxf_polar_angle(xa, ya, arc.cx, arc.cy, img_height_px, scale))
        ea = _normalize360(_dxf_polar_angle(xb, yb, arc.cx, arc.cy, img_height_px, scale))
        xd_c, yd_c = image_xy_to_dxf(arc.cx, arc.cy, img_height_px, scale)
        msp.add_arc(
            center=(xd_c + offset_x, yd_c),
            radius=float(arc.r) * scale,
            start_angle=float(sa),
            end_angle=float(ea),
            is_counter_clockwise=bool(arc.ccw),
            dxfattribs={"layer": lyr},
        )

    for s in drawing.residual_segments:
        x1, y1 = image_xy_to_dxf(s.x1, s.y1, img_height_px, scale)
        x2, y2 = image_xy_to_dxf(s.x2, s.y2, img_height_px, scale)
        msp.add_line((x1 + offset_x, y1), (x2 + offset_x, y2), dxfattribs={"layer": lyr})
def _emit_segment_lines(msp, segs: list[Segment], *, img_height_px: float, scale: float, lyr: str, ox: float) -> None:
    for s in segs:
        x1, y1 = image_xy_to_dxf(s.x1, s.y1, img_height_px, scale)
        x2, y2 = image_xy_to_dxf(s.x2, s.y2, img_height_px, scale)
        msp.add_line((x1 + ox, y1), (x2 + ox, y2), dxfattribs={"layer": lyr})


def export_viewer_layout_dxf(
    path: Path,
    panels: list[tuple[str, PartGroup, float, float]],
    *,
    mm_per_pixel: float = 1.0,
    gap_mm: float = 25.0,
) -> None:
    """
    Single DXF for web viewers: each panel on its own layer, laid out along +X with gaps.

    Autodesk Viewer (online) expects one primary CAD file per session; upload this file alone.
    Do not upload *_manifest.json with it.
    """
    doc = ezdxf.new(setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    offset_x = 0.0
    for layer_base, group, crop_h, crop_w in panels:
        layer_base_z = layer_base[:255]
        lbl_layer = f"{layer_base_z}_LBL"[:255]
        doc.layers.add(layer_base_z, color=7)
        doc.layers.add(lbl_layer, color=3)

        if group.vector_drawing is not None:
            _emit_vector_entities(
                msp,
                group.vector_drawing,
                img_height_px=float(crop_h),
                scale=mm_per_pixel,
                layer_base=layer_base_z,
                offset_x=offset_x,
            )
        else:
            _emit_segment_lines(
                msp,
                group.segments,
                img_height_px=float(crop_h),
                scale=mm_per_pixel,
                lyr=layer_base_z,
                ox=offset_x,
            )

        cx, cy = image_xy_to_dxf(
            group.label_center[0], group.label_center[1], float(crop_h), mm_per_pixel
        )
        hgt = max(2.5, 12.0 * mm_per_pixel)
        msp.add_text(
            f"PART {group.part_id}",
            dxfattribs={"height": hgt, "layer": lbl_layer, "insert": (cx + offset_x, cy)},
        )

        width_mm = float(crop_w) * mm_per_pixel
        offset_x += width_mm + gap_mm

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))


def export_part_dxf(
    path: Path,
    part: PartGroup,
    *,
    img_height_px: float,
    mm_per_pixel: float = 1.0,
    layer_geom: str = "GEOMETRY",
    layer_anno: str = "ANNOTATIONS",
) -> None:
    """Single-part drawing (vector primitives preferred) plus small PART label marker."""
    doc = ezdxf.new(setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    doc.layers.add(layer_geom, color=7)
    doc.layers.add(layer_anno, color=3)

    if part.vector_drawing is not None:
        _emit_vector_entities(
            msp,
            part.vector_drawing,
            img_height_px=float(img_height_px),
            scale=mm_per_pixel,
            layer_base=layer_geom,
            offset_x=0.0,
        )
    else:
        _emit_segment_lines(
            msp,
            part.segments,
            img_height_px=float(img_height_px),
            scale=mm_per_pixel,
            lyr=layer_geom,
            ox=0.0,
        )

    cx, cy = image_xy_to_dxf(part.label_center[0], part.label_center[1], float(img_height_px), mm_per_pixel)
    hgt = max(2.5, 12.0 * mm_per_pixel)
    msp.add_text(
        f"PART {part.part_id}",
        dxfattribs={"height": hgt, "layer": layer_anno, "insert": (cx, cy)},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))


def export_merged_dxf(
    path: Path,
    parts: list[PartGroup],
    unassigned: VectorDrawing,
    *,
    img_height_px: float,
    mm_per_pixel: float = 1.0,
) -> None:
    """One DXF with layered geometry per part id + UNASSIGNED layer."""
    doc = ezdxf.new(setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    def layer_for(pid: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in pid)
        return f"P_{safe}"[:255]

    for pg in parts:
        lyr = layer_for(pg.part_id)
        doc.layers.add(lyr, color=7)
        if pg.vector_drawing is not None:
            _emit_vector_entities(msp, pg.vector_drawing, img_height_px=float(img_height_px), scale=mm_per_pixel, layer_base=lyr)
        else:
            _emit_segment_lines(msp, pg.segments, img_height_px=float(img_height_px), scale=mm_per_pixel, lyr=lyr, ox=0.0)

    doc.layers.add("UNASSIGNED", color=1)
    _emit_vector_entities(
        msp,
        unassigned,
        img_height_px=float(img_height_px),
        scale=mm_per_pixel,
        layer_base="UNASSIGNED",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))


def export_segments_only(
    path: Path,
    segments: list[Segment],
    *,
    img_height_px: float,
    mm_per_pixel: float = 1.0,
    layer: str = "ALL_LINES",
    vector_drawing: VectorDrawing | None = None,
) -> None:
    """Vector-only assembly when no OCR parts matched (preferred: ``vector_drawing``)."""
    doc = ezdxf.new(setup=True)
    doc.units = units.MM
    msp = doc.modelspace()
    doc.layers.add(layer[:255], color=7)

    if vector_drawing is not None:
        _emit_vector_entities(
            msp,
            vector_drawing,
            img_height_px=float(img_height_px),
            scale=mm_per_pixel,
            layer_base=layer,
        )
    else:
        for s in segments:
            x1, y1 = image_xy_to_dxf(s.x1, s.y1, float(img_height_px), mm_per_pixel)
            x2, y2 = image_xy_to_dxf(s.x2, s.y2, float(img_height_px), mm_per_pixel)
            msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer[:255]})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
