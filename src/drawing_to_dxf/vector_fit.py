"""Circle/arc hypotheses from traced polylines (complements Hough circle hints)."""

from __future__ import annotations

import math

from drawing_to_dxf.geometry_model import ArcDef, CircleDef, PolylineDef, VectorDrawing


def _fit_circle_three_point(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> tuple[float, float, float] | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = (
        ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by))
        / d
    )
    uy = (
        ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax))
        / d
    )
    r = math.hypot(ax - ux, ay - uy)
    if r < 8.0:
        return None
    return float(ux), float(uy), float(r)


def _circ_theta_deg(px: float, py: float, ux: float, uy: float) -> float:
    return math.degrees(math.atan2(py - uy, px - ux)) % 360.0


def _spans_deg_ccw(a0: float, a1: float) -> float:
    """CCW geometric angle from ``a0`` to ``a1`` in degrees."""
    return (a1 - a0) % 360.0


def _inside_ccw_sector(start: float, probe: float, end: float) -> bool:
    """True iff ``probe`` lies on CCW traversal from ``start`` to ``end`` (exclusive full tour)."""
    span = _spans_deg_ccw(start, end)
    if span <= 1e-6:
        return False
    off = (_spans_deg_ccw(start, probe))
    return off <= span + 0.05


def fit_arcs_replace_polylines(
    vd: VectorDrawing,
    *,
    max_dev_px: float = 2.85,
    min_arc_deg: float = 26.0,
    min_pts: int = 6,
) -> VectorDrawing:
    """
    Replace **open** polylines that fit a circular arc closely with DXF arcs.

    Preserves circles from upstream Hough extraction; skips closed loops (polygon / circle territory).
    """
    kept_polys: list[PolylineDef] = []
    new_arcs: list[ArcDef] = []

    for poly in vd.polylines:
        pts = poly.points
        if poly.closed or len(pts) < min_pts:
            kept_polys.append(poly)
            continue

        ia = 0
        ib = max(3, min(len(pts) - 3, len(pts) // 2))
        ic = len(pts) - 1
        fit = _fit_circle_three_point(pts[ia], pts[ib], pts[ic])
        if fit is None:
            kept_polys.append(poly)
            continue
        ux, uy, r = fit
        radial = [abs(math.hypot(p[0] - ux, p[1] - uy) - r) for p in pts]
        if max(radial) > max_dev_px:
            kept_polys.append(poly)
            continue

        ts = _circ_theta_deg(pts[ia][0], pts[ia][1], ux, uy)
        tm = _circ_theta_deg(pts[ib][0], pts[ib][1], ux, uy)
        te = _circ_theta_deg(pts[ic][0], pts[ic][1], ux, uy)

        if _inside_ccw_sector(ts, tm, te):
            arc_ccw = True
            theta_s, theta_e = ts, te
        elif _inside_ccw_sector(te, tm, ts):
            arc_ccw = True
            theta_s, theta_e = te, ts
        else:
            kept_polys.append(poly)
            continue

        span_geo = _spans_deg_ccw(theta_s, theta_e)
        if span_geo < min_arc_deg or span_geo > 330.0:
            kept_polys.append(poly)
            continue

        new_arcs.append(
            ArcDef(
                cx=ux,
                cy=uy,
                r=r,
                start_angle_deg=theta_s,
                end_angle_deg=theta_e,
                ccw=arc_ccw,
            )
        )

    return VectorDrawing(
        polylines=kept_polys,
        circles=list(vd.circles),
        arcs=list(vd.arcs) + new_arcs,
        residual_segments=list(vd.residual_segments),
    )


def fit_circles_from_closed_polylines(
    vd: VectorDrawing,
    *,
    circ_tol_px: float = 2.75,
    min_pts: int = 10,
) -> VectorDrawing:
    """Approximate circular **closed** polylines as CIRCLE primitives when radial error is low."""
    circles = list(vd.circles)
    kept_polys: list[PolylineDef] = []

    for poly in vd.polylines:
        ps = poly.points
        if len(ps) < min_pts or not poly.closed:
            kept_polys.append(poly)
            continue
        mid_idx = len(ps) // 2
        fu = _fit_circle_three_point(ps[0], ps[mid_idx], ps[len(ps) // 3])
        if fu is None:
            kept_polys.append(poly)
            continue
        ux, uy, r = fu
        radial = [abs(math.hypot(p[0] - ux, p[1] - uy) - r) for p in ps]
        if max(radial) > circ_tol_px:
            kept_polys.append(poly)
            continue

        redundant = False
        for oc in circles:
            if math.hypot(oc.cx - ux, oc.cy - uy) < max(22.0, 0.12 * max(r, oc.r)):
                if abs(oc.r - r) < circ_tol_px * 3:
                    redundant = True
                    break
        if redundant:
            continue
        circles.append(CircleDef(cx=ux, cy=uy, r=r))

    return VectorDrawing(
        polylines=kept_polys,
        circles=circles,
        arcs=list(vd.arcs),
        residual_segments=list(vd.residual_segments),
    )


def complete_near_full_arcs_to_circles(
    vd: VectorDrawing,
    *,
    min_span_deg: float = 315.0,
) -> VectorDrawing:
    """Promote almost-full arcs to circles to stabilize hole-like geometry in DXF."""
    if min_span_deg >= 360.0:
        return vd
    arcs_kept: list = []
    circles = list(vd.circles)
    for a in vd.arcs:
        span = _spans_deg_ccw(a.start_angle_deg, a.end_angle_deg) if a.ccw else _spans_deg_ccw(a.end_angle_deg, a.start_angle_deg)
        if span >= float(min_span_deg):
            circles.append(CircleDef(cx=a.cx, cy=a.cy, r=a.r))
        else:
            arcs_kept.append(a)
    return VectorDrawing(
        polylines=list(vd.polylines),
        circles=circles,
        arcs=arcs_kept,
        residual_segments=list(vd.residual_segments),
    )


def apply_polyline_fittings(
    vd: VectorDrawing,
    *,
    fit_arcs: bool = True,
    fit_circles_from_loops: bool = True,
    complete_full_arcs_to_circles_min_span_deg: float | None = 315.0,
) -> VectorDrawing:
    out = vd
    if fit_circles_from_loops:
        out = fit_circles_from_closed_polylines(out)
    if fit_arcs:
        out = fit_arcs_replace_polylines(out)
    if complete_full_arcs_to_circles_min_span_deg is not None:
        out = complete_near_full_arcs_to_circles(out, min_span_deg=float(complete_full_arcs_to_circles_min_span_deg))
    return out
