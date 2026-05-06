"""Write DXF files with ezdxf (AutoCAD-compatible; no AutoCAD install required)."""

from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf import units

from drawing_to_dxf.link_parts import PartGroup
from drawing_to_dxf.vectorize import Segment


def image_xy_to_dxf(x: float, y: float, img_height_px: float, scale: float) -> tuple[float, float]:
    """Map top-left image coordinates to DXF XY with Y up."""
    return (x * scale, (img_height_px - y) * scale)


def export_part_dxf(
    path: Path,
    part: PartGroup,
    *,
    img_height_px: float,
    mm_per_pixel: float = 1.0,
    layer_geom: str = "GEOMETRY",
    layer_anno: str = "ANNOTATIONS",
) -> None:
    """Single-part drawing: segments + small label marker."""
    doc = ezdxf.new(setup=True)
    doc.units = units.MM
    msp = doc.modelspace()

    doc.layers.add(layer_geom, color=7)
    doc.layers.add(layer_anno, color=3)

    for s in part.segments:
        x1, y1 = image_xy_to_dxf(s.x1, s.y1, img_height_px, mm_per_pixel)
        x2, y2 = image_xy_to_dxf(s.x2, s.y2, img_height_px, mm_per_pixel)
        msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer_geom})

    cx, cy = image_xy_to_dxf(part.label_center[0], part.label_center[1], img_height_px, mm_per_pixel)
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
    unassigned: list[Segment],
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
        for s in pg.segments:
            x1, y1 = image_xy_to_dxf(s.x1, s.y1, img_height_px, mm_per_pixel)
            x2, y2 = image_xy_to_dxf(s.x2, s.y2, img_height_px, mm_per_pixel)
            msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": lyr})

    doc.layers.add("UNASSIGNED", color=1)
    for s in unassigned:
        x1, y1 = image_xy_to_dxf(s.x1, s.y1, img_height_px, mm_per_pixel)
        x2, y2 = image_xy_to_dxf(s.x2, s.y2, img_height_px, mm_per_pixel)
        msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": "UNASSIGNED"})

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))


def export_segments_only(
    path: Path,
    segments: list[Segment],
    *,
    img_height_px: float,
    mm_per_pixel: float = 1.0,
    layer: str = "ALL_LINES",
) -> None:
    """Fallback when no OCR parts detected: full vectorization."""
    doc = ezdxf.new(setup=True)
    doc.units = units.MM
    msp = doc.modelspace()
    doc.layers.add(layer, color=7)
    for s in segments:
        x1, y1 = image_xy_to_dxf(s.x1, s.y1, img_height_px, mm_per_pixel)
        x2, y2 = image_xy_to_dxf(s.x2, s.y2, img_height_px, mm_per_pixel)
        msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
