"""Post-vectorization tidy-up: orthogonal snap, collinear merges, prune tiny open ends."""

from __future__ import annotations

import math

from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing


def _angle_dxdy(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def snap_polyline_near_axis(
    pts: list[tuple[float, float]],
    *,
    angle_tol_deg: float = 3.25,
    min_edge_px: float = 2.0,
) -> list[tuple[float, float]]:
    if len(pts) < 2:
        return list(pts)
    out: list[tuple[float, float]] = [pts[0]]
    for i in range(1, len(pts)):
        px, py = out[-1]
        x, y = pts[i]
        dx, dy = x - px, y - py
        l = math.hypot(dx, dy)
        if l < min_edge_px:
            continue
        ang = _angle_dxdy(dx, dy) % 180.0
        if ang < angle_tol_deg or abs(ang - 90.0) < angle_tol_deg or abs(ang - 180.0) < angle_tol_deg:
            if ang < angle_tol_deg or abs(ang - 180.0) < angle_tol_deg:
                out.append((x, py))
            else:
                out.append((px, y))
            continue
        out.append((x, y))

    uniq: list[tuple[float, float]] = []
    for p in out:
        if uniq and math.hypot(p[0] - uniq[-1][0], p[1] - uniq[-1][1]) < 0.25:
            continue
        uniq.append(p)
    return uniq if len(uniq) >= 2 else list(pts)


def _triple_collinear(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    *,
    sin_thresh: float,
) -> bool:
    vx1, vy1 = b[0] - a[0], b[1] - a[1]
    vx2, vy2 = c[0] - b[0], c[1] - b[1]
    l1 = math.hypot(vx1, vy1)
    l2 = math.hypot(vx2, vy2)
    if l1 < 1e-6 or l2 < 1e-6:
        return True
    if vx1 * vx2 + vy1 * vy2 < -1e-6:
        return False
    cross = abs(vx1 * vy2 - vy1 * vx2)
    return cross <= sin_thresh * l1 * l2 + 1e-9


def merge_collinear_run(
    pts: list[tuple[float, float]],
    *,
    angle_tol_deg: float = 4.0,
    closed: bool = False,
) -> list[tuple[float, float]]:
    if len(pts) < 3:
        return list(pts)
    st = math.sin(math.radians(angle_tol_deg))
    if closed and len(pts) >= 4:
        chained = pts + [pts[0]]
        merged_o = merge_collinear_run(chained, angle_tol_deg=angle_tol_deg, closed=False)
        if merged_o and len(merged_o) >= 2 and merged_o[-1] == merged_o[0]:
            merged_o = merged_o[:-1]
        if merged_o == pts:
            return pts
        if len(merged_o) >= 3 and merged_o[-1] != merged_o[0]:
            return merge_collinear_run(merged_o + [merged_o[0]], angle_tol_deg=angle_tol_deg, closed=False)[
                :-1
            ]
        return merged_o

    work = list(pts)
    rounds = 0
    changed = True
    while changed and rounds < len(work) + 80:
        rounds += 1
        changed = False
        i = 0
        while i < len(work) - 2:
            a, b, c = work[i], work[i + 1], work[i + 2]
            if _triple_collinear(a, b, c, sin_thresh=st):
                work.pop(i + 1)
                changed = True
                if i > 0:
                    i -= 1
            else:
                i += 1
    return work


def prune_open_polyline_caps(
    pts: list[tuple[float, float]],
    *,
    min_edge_px: float,
) -> list[tuple[float, float]]:
    if len(pts) < 2 or min_edge_px <= 0:
        return list(pts)
    pts = list(pts)

    def head_len() -> float:
        return math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])

    def tail_len() -> float:
        return math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])

    while len(pts) >= 3 and head_len() < min_edge_px:
        pts.pop(1)

    while len(pts) >= 3 and tail_len() < min_edge_px:
        pts.pop(-2)

    return pts


def refine_vector_drawing(
    dwg: VectorDrawing,
    *,
    min_branch_px: float,
    orthogonal_snap_deg: float = 3.25,
    collinear_tol_deg: float = 4.0,
    prune_tip_px_mult: float = 0.6,
) -> VectorDrawing:
    """Return a shallow-refined drawing (does not mutate input)."""
    pr = prune_tip_px_mult * float(min_branch_px)
    polys_out: list[PolylineDef] = []
    for poly in dwg.polylines:
        if len(poly.points) < 2:
            continue
        s1 = snap_polyline_near_axis(poly.points, angle_tol_deg=orthogonal_snap_deg)
        if len(s1) < 2:
            continue
        s2 = merge_collinear_run(s1, angle_tol_deg=collinear_tol_deg, closed=poly.closed)
        if len(s2) < 2:
            continue
        if not poly.closed:
            s2 = prune_open_polyline_caps(s2, min_edge_px=pr)

        peri = sum(
            math.hypot(s2[i][0] - s2[i - 1][0], s2[i][1] - s2[i - 1][1])
            for i in range(1, len(s2))
        )
        if poly.closed and len(s2) >= 3:
            peri += math.hypot(s2[0][0] - s2[-1][0], s2[0][1] - s2[-1][1])

        if peri + 1e-6 < min_branch_px:
            continue
        polys_out.append(PolylineDef(points=s2, closed=poly.closed))

    return VectorDrawing(
        polylines=polys_out,
        circles=list(dwg.circles),
        arcs=list(dwg.arcs),
        residual_segments=list(dwg.residual_segments),
    )
