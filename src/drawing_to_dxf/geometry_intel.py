"""Geometry intelligence pass: LSD supplement, dedupe, gap bridge, rectangle stats, quality."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing, exploded_segments_for_sampling
from drawing_to_dxf.segment_types import Segment


def lsd_extract_segments(
    gray_uint8: np.ndarray,
    *,
    min_length_px: float = 12.0,
    refine: bool = True,
) -> list[Segment]:
    """Line Segment Detector on grayscale (OpenCV createLineSegmentDetector)."""
    if gray_uint8.ndim != 2:
        raise ValueError("grayscale expected")
    if not hasattr(cv2, "createLineSegmentDetector"):
        return []

    refine_flag = cv2.LSD_REFINE_STD if refine else cv2.LSD_REFINE_NONE
    lsd = cv2.createLineSegmentDetector(refine_flag)
    lines, _, _, _ = lsd.detect(gray_uint8)
    if lines is None or len(lines) == 0:
        return []

    min_l2 = float(min_length_px) ** 2
    out: list[Segment] = []
    for row in lines.reshape(-1, 4):
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        dx, dy = x2 - x1, y2 - y1
        if dx * dx + dy * dy < min_l2:
            continue
        out.append(Segment(x1, y1, x2, y2))
    return out


def _seg_dir_unit(s: Segment) -> tuple[float, float] | None:
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    l = math.hypot(dx, dy)
    if l < 1e-9:
        return None
    return (dx / l, dy / l)


def _parallel_overlap_score(
    s1: Segment,
    s2: Segment,
    *,
    sin_thresh: float,
    max_perp_dist: float,
) -> float:
    """Return overlap length if nearly parallel and coincident, else 0."""
    u1 = _seg_dir_unit(s1)
    u2 = _seg_dir_unit(s2)
    if u1 is None or u2 is None:
        return 0.0
    cross = abs(u1[0] * u2[1] - u1[1] * u2[0])
    if cross > sin_thresh:
        return 0.0
    if u1[0] * u2[0] + u1[1] * u2[1] < 0:
        return 0.0

    m1 = ((s1.x1 + s1.x2) * 0.5, (s1.y1 + s1.y2) * 0.5)
    ax, ay = s1.x1, s1.y1
    vx, vy = s1.x2 - s1.x1, s1.y2 - s1.y1
    perp = abs(vx * (m1[1] - ay) - vy * (m1[0] - ax)) / max(math.hypot(vx, vy), 1e-9)
    m2 = ((s2.x1 + s2.x2) * 0.5, (s2.y1 + s2.y2) * 0.5)
    perp2 = abs(vx * (m2[1] - ay) - vy * (m2[0] - ax)) / max(math.hypot(vx, vy), 1e-9)
    if max(perp, perp2) > max_perp_dist:
        return 0.0

    def proj_t(px: float, py: float) -> float:
        return (px - ax) * u1[0] + (py - ay) * u1[1]

    ts = [
        proj_t(s1.x1, s1.y1),
        proj_t(s1.x2, s1.y2),
        proj_t(s2.x1, s2.y1),
        proj_t(s2.x2, s2.y2),
    ]
    lo, hi = min(ts), max(ts)
    return max(0.0, hi - lo)


def remove_redundant_segments_vs_reference(
    candidates: list[Segment],
    reference: list[Segment],
    *,
    angle_tol_deg: float = 4.0,
    max_perp_dist_px: float = 3.5,
    min_overlap_ratio: float = 0.45,
) -> list[Segment]:
    """Drop candidate segments that duplicate references (thick/double edges, skeleton+LSD)."""
    if not candidates:
        return []
    if not reference:
        return list(candidates)

    st = math.sin(math.radians(angle_tol_deg))
    kept: list[Segment] = []
    for c in candidates:
        cl = math.hypot(c.x2 - c.x1, c.y2 - c.y1)
        if cl < 1e-6:
            continue
        drop = False
        for r in reference:
            ov = _parallel_overlap_score(c, r, sin_thresh=st, max_perp_dist=max_perp_dist_px)
            rl = math.hypot(r.x2 - r.x1, r.y2 - r.y1)
            if ov > 0 and ov >= min_overlap_ratio * min(cl, rl):
                drop = True
                break
        if not drop:
            kept.append(c)
    return kept


def dedupe_segment_list(
    segments: list[Segment],
    *,
    angle_tol_deg: float = 4.0,
    max_perp_dist_px: float = 2.8,
    min_overlap_ratio: float = 0.5,
) -> list[Segment]:
    """Greedy dedupe within one list (O(n^2), fine for residual list sizes)."""
    if len(segments) < 2:
        return list(segments)
    st = math.sin(math.radians(angle_tol_deg))
    out: list[Segment] = []
    for s in segments:
        cl = math.hypot(s.x2 - s.x1, s.y2 - s.y1)
        dup = False
        for o in out:
            ov = _parallel_overlap_score(s, o, sin_thresh=st, max_perp_dist=max_perp_dist_px)
            ol = math.hypot(o.x2 - o.x1, o.y2 - o.y1)
            if ov > 0 and ov >= min_overlap_ratio * min(cl, ol):
                dup = True
                break
        if not dup:
            out.append(s)
    return out


def count_corner_rectangles(dwg: VectorDrawing, *, angle_tol_deg: float = 12.0) -> int:
    """Orthogonal quadrilaterals from closed polylines (4 vertices, ~90° corners)."""
    cos_slack = math.sin(math.radians(angle_tol_deg))
    n = 0
    for poly in dwg.polylines:
        if not poly.closed or len(poly.points) != 4:
            continue
        pts = list(poly.points) + [poly.points[0], poly.points[1]]
        ok = True
        for i in range(1, 5):
            ax, ay = pts[i - 1]
            bx, by = pts[i]
            cx, cy = pts[i + 1]
            v1x, v1y = bx - ax, by - ay
            v2x, v2y = cx - bx, cy - by
            l1 = math.hypot(v1x, v1y)
            l2 = math.hypot(v2x, v2y)
            if l1 < 1e-6 or l2 < 1e-6:
                ok = False
                break
            cost = abs(v1x * v2x + v1y * v2y) / (l1 * l2)
            if cost > cos_slack:
                ok = False
                break
        if ok:
            n += 1
    return n


def _end_tangent(poly: PolylineDef, *, at_start: bool) -> tuple[float, float] | None:
    pts = poly.points
    if len(pts) < 2:
        return None
    if at_start:
        ax, ay = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    else:
        ax, ay = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
    l = math.hypot(ax, ay)
    if l < 1e-9:
        return None
    return (ax / l, ay / l)


def bridge_open_polyline_gaps(
    dwg: VectorDrawing,
    *,
    max_gap_px: float,
    direction_dot_min: float = 0.42,
) -> VectorDrawing:
    """Add short 2-point polylines between open polyline endpoints facing each other."""
    if max_gap_px <= 0:
        return dwg

    open_ix = [i for i, p in enumerate(dwg.polylines) if not p.closed and len(p.points) >= 2]
    if len(open_ix) < 2:
        return dwg

    eps = max_gap_px * max_gap_px
    bridges: list[PolylineDef] = []
    used: set[tuple[int, bool]] = set()

    for ii, i in enumerate(open_ix):
        pi = dwg.polylines[i]
        for j in open_ix[ii + 1 :]:
            if j == i:
                continue
            pj = dwg.polylines[j]
            for i_head in (True, False):
                if (i, i_head) in used:
                    continue
                a = pi.points[0] if i_head else pi.points[-1]
                ta = _end_tangent(pi, at_start=i_head)
                if ta is None:
                    continue
                for j_head in (True, False):
                    if (j, j_head) in used:
                        continue
                    b = pj.points[0] if j_head else pj.points[-1]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    d2 = dx * dx + dy * dy
                    if d2 < 1.0 or d2 > eps:
                        continue
                    tb = _end_tangent(pj, at_start=j_head)
                    if tb is None:
                        continue
                    lab = math.hypot(dx, dy)
                    if lab < 1e-9:
                        continue
                    ua = (dx / lab, dy / lab)
                    if ta[0] * ua[0] + ta[1] * ua[1] < direction_dot_min:
                        continue
                    if tb[0] * (-ua[0]) + tb[1] * (-ua[1]) < direction_dot_min:
                        continue
                    bridges.append(PolylineDef(points=[a, b], closed=False))
                    used.add((i, i_head))
                    used.add((j, j_head))

    if not bridges:
        return dwg

    return VectorDrawing(
        polylines=list(dwg.polylines) + bridges,
        circles=list(dwg.circles),
        arcs=list(dwg.arcs),
        residual_segments=list(dwg.residual_segments),
    )


@dataclass
class GeometryQualityReport:
    polyline_count: int
    circle_count: int
    arc_count: int
    residual_count: int
    closed_polyline_count: int
    rectangle_like_count: int
    mean_segment_length_px: float
    confidence_score: float
    timings_ms: dict[str, float]


def geometry_quality_report(dwg: VectorDrawing, *, timings_ms: dict[str, float] | None = None) -> GeometryQualityReport:
    """Heuristic 0–1 score from coverage and primitive mix."""
    ref = exploded_segments_for_sampling(dwg)
    lens = [math.hypot(s.x2 - s.x1, s.y2 - s.y1) for s in ref]
    mean_len = sum(lens) / max(len(lens), 1)
    n_poly = len(dwg.polylines)
    n_closed = sum(1 for p in dwg.polylines if p.closed)
    rects = count_corner_rectangles(dwg)
    prim = n_poly + len(dwg.circles) + len(dwg.arcs) + len(dwg.residual_segments)

    if prim == 0:
        score = 0.0
    else:
        closed_ratio = n_closed / max(n_poly, 1) if n_poly else 0.0
        density = min(1.0, math.log1p(len(ref)) / math.log1p(400))
        circ_bonus = min(0.15, 0.05 * len(dwg.circles))
        score = 0.25 + 0.35 * density + 0.25 * closed_ratio + circ_bonus
        score = float(max(0.0, min(1.0, score)))

    return GeometryQualityReport(
        polyline_count=n_poly,
        circle_count=len(dwg.circles),
        arc_count=len(dwg.arcs),
        residual_count=len(dwg.residual_segments),
        closed_polyline_count=n_closed,
        rectangle_like_count=rects,
        mean_segment_length_px=float(mean_len),
        confidence_score=score,
        timings_ms=dict(timings_ms or {}),
    )


@contextmanager
def profile_stage(timings_ms: dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings_ms[name] = timings_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
