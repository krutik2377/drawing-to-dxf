"""Light constraint-based healing: orthogonal rectangle closure on near-rect polylines."""

from __future__ import annotations

import math

from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing
from drawing_to_dxf.segment_types import Segment


def _corner_angle_deg(
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


def orthogonal_close_quads(
    dwg: VectorDrawing,
    *,
    corner_tol_deg: float = 14.0,
    min_edge_px: float = 6.0,
) -> VectorDrawing:
    """Snap closed 4-vertex polylines with nearly 90° corners to axis-aligned rectangles."""
    polys_out: list[PolylineDef] = []
    for poly in dwg.polylines:
        pts = poly.points
        if not poly.closed or len(pts) != 4:
            polys_out.append(poly)
            continue
        chained = pts + [pts[0], pts[1]]
        ok = True
        for i in range(1, 5):
            ang = _corner_angle_deg(chained[i - 1], chained[i], chained[i + 1])
            if ang > corner_tol_deg:
                ok = False
                break
        if not ok:
            polys_out.append(poly)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if (x1 - x0) < min_edge_px or (y1 - y0) < min_edge_px:
            polys_out.append(poly)
            continue
        rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        polys_out.append(PolylineDef(points=rect, closed=True))

    return VectorDrawing(
        polylines=polys_out,
        circles=list(dwg.circles),
        arcs=list(dwg.arcs),
        residual_segments=list(dwg.residual_segments),
    )


def apply_constraint_heal(
    dwg: VectorDrawing,
    *,
    orthogonal_quads: bool = True,
    corner_tol_deg: float = 14.0,
) -> VectorDrawing:
    if not orthogonal_quads:
        return dwg
    return orthogonal_close_quads(dwg, corner_tol_deg=corner_tol_deg)


def snap_axis_aligned_residual_segments(
    segments: list[Segment],
    *,
    angle_tol_deg: float = 3.0,
    min_length_px: float = 8.0,
) -> tuple[list[Segment], int]:
    """
    Snap nearly horizontal / vertical residual LINE strokes to exact axis alignment.

    Preserves span along the long axis; short strokes are left unchanged.
    """
    if not segments or angle_tol_deg <= 0:
        return list(segments), 0
    out: list[Segment] = []
    changed = 0
    for s in segments:
        dx, dy = s.x2 - s.x1, s.y2 - s.y1
        lg = math.hypot(dx, dy)
        if lg < min_length_px:
            out.append(s)
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        na = ang if ang <= 90.0 else 180.0 - ang
        ns = s
        if na < angle_tol_deg:
            ym = 0.5 * (s.y1 + s.y2)
            ns = Segment(s.x1, ym, s.x2, ym)
        elif abs(na - 90.0) < angle_tol_deg:
            xm = 0.5 * (s.x1 + s.x2)
            ns = Segment(xm, s.y1, xm, s.y2)
        if abs(ns.x1 - s.x1) > 1e-6 or abs(ns.y1 - s.y1) > 1e-6 or abs(ns.x2 - s.x2) > 1e-6 or abs(ns.y2 - s.y2) > 1e-6:
            changed += 1
        out.append(ns)
    return out, changed


def align_parallel_residual_clusters(
    segments: list[Segment],
    *,
    angle_tol_deg: float = 2.5,
    offset_snap_px: float = 2.0,
    min_cluster_size: int = 3,
    min_length_px: float = 28.0,
) -> tuple[list[Segment], int]:
    """
    Within bands of nearly parallel long residuals, snap offsets perpendicular to the
    bundle toward a shared median (weak parallel-line constraint).

    Conservative: only clusters with ≥ ``min_cluster_size`` segments participate.
    """
    if not segments or angle_tol_deg <= 0 or offset_snap_px <= 0:
        return list(segments), 0
    rad = math.radians(angle_tol_deg)
    bins: dict[int, list[Segment]] = {}
    for s in segments:
        dx, dy = s.x2 - s.x1, s.y2 - s.y1
        lg = math.hypot(dx, dy)
        if lg < min_length_px:
            continue
        ang = math.atan2(dy, dx)
        key = int((ang % math.pi) / max(rad, 1e-6))
        bins.setdefault(key, []).append(s)

    moves = 0
    out_map: dict[tuple[float, float, float, float], Segment] = {}
    for s in segments:
        out_map[(s.x1, s.y1, s.x2, s.y2)] = s

    for _bk, group in bins.items():
        if len(group) < min_cluster_size:
            continue
        g0 = group[0]
        vx, vy = g0.x2 - g0.x1, g0.y2 - g0.y1
        ln = math.hypot(vx, vy)
        if ln < 1e-9:
            continue
        ux, uy = vx / ln, vy / ln
        px, py = -uy, ux
        offs: list[float] = []
        for s in group:
            mx, my = 0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2)
            offs.append(mx * px + my * py)
        offs.sort()
        median_o = offs[len(offs) // 2]
        for s in group:
            mx, my = 0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2)
            cur = mx * px + my * py
            delta = median_o - cur
            if abs(delta) < 0.5:
                continue
            step = max(-offset_snap_px, min(offset_snap_px, delta))
            if abs(step) < 0.15:
                continue
            dx, dy = step * px, step * py
            ns = Segment(s.x1 + dx, s.y1 + dy, s.x2 + dx, s.y2 + dy)
            out_map[(s.x1, s.y1, s.x2, s.y2)] = ns
            moves += 1

    rebuilt: list[Segment] = []
    for s in segments:
        rebuilt.append(out_map.get((s.x1, s.y1, s.x2, s.y2), s))
    return rebuilt, moves
