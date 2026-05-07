"""Structured vector geometry for skeleton-based extraction (prior to DXF export)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, radians, sin

from drawing_to_dxf.segment_types import Segment


def _bbox_center(pts: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


@dataclass
class CircleDef:
    cx: float
    cy: float
    r: float


@dataclass
class ArcDef:
    """Circular arc in original image coordinates before DXF Y-flip (ezdxf export adjusts angles)."""

    cx: float
    cy: float
    r: float
    start_angle_deg: float
    end_angle_deg: float
    ccw: bool = True


def poly_mass_centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    xs = sum(p[0] for p in pts) / max(len(pts), 1)
    ys = sum(p[1] for p in pts) / max(len(pts), 1)
    return (xs, ys)


@dataclass
class PolylineDef:
    points: list[tuple[float, float]]
    closed: bool = False


@dataclass
class VectorDrawing:
    """LWPolylines, circles, arcs; optional residual strokes as segments."""

    polylines: list[PolylineDef] = field(default_factory=list)
    circles: list[CircleDef] = field(default_factory=list)
    arcs: list[ArcDef] = field(default_factory=list)
    residual_segments: list[Segment] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.polylines and not self.circles and not self.arcs and not self.residual_segments

    def bbox_for_poly(self, poly: PolylineDef) -> tuple[float, float, float, float]:
        xs = [p[0] for p in poly.points]
        ys = [p[1] for p in poly.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def centroid_for_assignment(self, poly: PolylineDef) -> tuple[float, float]:
        return _bbox_center(poly.points)

    def merge_inplace(self, other: VectorDrawing) -> None:
        self.polylines.extend(other.polylines)
        self.circles.extend(other.circles)
        self.arcs.extend(other.arcs)
        self.residual_segments.extend(other.residual_segments)


def _angle_lerp_deg(a0: float, a1: float, frac: float, ccw: bool) -> float:
    """Degrees on [0,360), stepping the shorter/longer way according to orientation."""
    a0 = a0 % 360.0
    a1 = a1 % 360.0
    delta_ccw = (a1 - a0) % 360.0
    delta_cw = (360.0 - delta_ccw) % 360.0
    step = delta_ccw if ccw else -delta_cw
    ang = (a0 + frac * step) % 360.0
    return ang


def exploded_segments_for_sampling(dwg: VectorDrawing) -> list[Segment]:
    """
    Approximate primitives as LINE segments so legacy heuristics (midpoints) behave.
    """
    out: list[Segment] = []
    for poly in dwg.polylines:
        pts = poly.points
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            out.append(Segment(a[0], a[1], b[0], b[1]))
        if poly.closed and len(pts) >= 2:
            a, b = pts[-1], pts[0]
            out.append(Segment(a[0], a[1], b[0], b[1]))
    for c in dwg.arcs:
        n_seg = max(16, min(96, int(2 * pi * max(c.r, 1.0) / max(6.0, c.r / 14))))
        n_seg = max(16, min(160, n_seg))
        prev: tuple[float, float] | None = None
        for k in range(n_seg + 1):
            ta = _angle_lerp_deg(c.start_angle_deg, c.end_angle_deg, k / float(n_seg), c.ccw)
            th = radians(ta)
            px = c.cx + c.r * cos(th)
            py = c.cy + c.r * sin(th)
            if prev is None:
                prev = (px, py)
                continue
            a, bb = prev, (px, py)
            if hypot(bb[0] - a[0], bb[1] - a[1]) >= 0.85:
                out.append(Segment(a[0], a[1], bb[0], bb[1]))
            prev = bb
    for seg in dwg.residual_segments:
        out.append(seg)
    for c in dwg.circles:
        n = max(16, min(96, int(2 * 3.14159 * max(c.r, 1.0) / max(8.0, c.r / 12))))
        n = max(16, min(128, n))
        pts: list[tuple[float, float]] = []
        for k in range(n):
            theta = k * (2 * pi / n)
            pts.append((c.cx + c.r * cos(theta), c.cy + c.r * sin(theta)))
        pts.append(pts[0])
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if hypot(b[0] - a[0], b[1] - a[1]) >= 1.0:
                out.append(Segment(a[0], a[1], b[0], b[1]))
    return out
