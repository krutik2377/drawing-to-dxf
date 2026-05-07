"""CAD-style graph cleanup: collinear JOIN chains, OVERKILL duplicates, near-aligned merges."""

from __future__ import annotations

import math
from collections import defaultdict

from drawing_to_dxf.segment_types import Segment


def _quant_key(x: float, y: float, q: float) -> tuple[int, int]:
    return (int(round(x / q)), int(round(y / q)))


def _angle_norm_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _seg_angle(s: Segment) -> float:
    return _angle_norm_deg(s.x2 - s.x1, s.y2 - s.y1)


def _seg_len(s: Segment) -> float:
    return math.hypot(s.x2 - s.x1, s.y2 - s.y1)


def merge_collinear_chains(
    segments: list[Segment],
    *,
    endpoint_quant_px: float = 1.8,
    angle_tol_deg: float = 2.8,
    min_edge_px: float = 0.85,
) -> tuple[list[Segment], int]:
    """
    JOIN-like pass: at degree-2 quantized junctions, merge two collinear segments into one.

    Preserves long structural strokes as single entities where the sketch chain is almost straight.
    """
    if len(segments) < 2:
        return list(segments), 0
    q = max(0.6, float(endpoint_quant_px))
    ang_tol = max(0.05, float(angle_tol_deg))

    segs: list[Segment | None] = [Segment(s.x1, s.y1, s.x2, s.y2) for s in segments]
    merges = 0

    def node_map() -> dict[tuple[int, int], list[tuple[int, bool]]]:
        m: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
        for i, s in enumerate(segs):
            if s is None:
                continue
            m[_quant_key(s.x1, s.y1, q)].append((i, True))
            m[_quant_key(s.x2, s.y2, q)].append((i, False))
        return m

    while True:
        nm = node_map()
        did = False
        def other_endpoint(s: Segment, node: tuple[int, int]) -> tuple[float, float] | None:
            p1, p2 = (s.x1, s.y1), (s.x2, s.y2)
            k1 = _quant_key(p1[0], p1[1], q)
            k2 = _quant_key(p2[0], p2[1], q)
            if k1 == node and k2 != node:
                return p2
            if k2 == node and k1 != node:
                return p1
            return None

        for node, inc in nm.items():
            if len(inc) != 2:
                continue
            i1, _e1f = inc[0]
            i2, _e2f = inc[1]
            if i1 == i2:
                continue
            s1 = segs[i1]
            s2 = segs[i2]
            if s1 is None or s2 is None:
                continue
            a1 = _seg_angle(s1)
            a2 = _seg_angle(s2)
            if abs(a1 - a2) > ang_tol and abs(abs(a1 - a2) - 180.0) > ang_tol:
                continue
            o1 = other_endpoint(s1, node)
            o2 = other_endpoint(s2, node)
            if o1 is None or o2 is None:
                continue
            ax, ay = o1
            bx, by = o2
            if math.hypot(ax - bx, ay - by) < min_edge_px:
                segs[i1] = None
                segs[i2] = None
                did = True
                merges += 1
                continue
            segs[i1] = Segment(ax, ay, bx, by)
            segs[i2] = None
            did = True
            merges += 1
        if not did:
            break
    out = [s for s in segs if s is not None and _seg_len(s) >= min_edge_px]
    return out, merges


def overkill_near_duplicate_segments(
    segments: list[Segment],
    *,
    parallel_angle_tol_deg: float = 1.25,
    lateral_tol_px: float = 1.35,
    overlap_frac_min: float = 0.72,
) -> tuple[list[Segment], int]:
    """
    OVERKILL-lite: drop shorter segment when it nearly coincides with a longer parallel one.
    """
    if len(segments) < 2:
        return list(segments), 0
    segs = [Segment(s.x1, s.y1, s.x2, s.y2) for s in segments]
    n = len(segs)
    drop = [False] * n
    ang_tol = max(0.05, float(parallel_angle_tol_deg))
    lat = max(0.2, float(lateral_tol_px))

    def midpoint(s: Segment) -> tuple[float, float]:
        return (0.5 * (s.x1 + s.x2), 0.5 * (s.y1 + s.y2))

    def dist_point_line(px: float, py: float, s: Segment) -> float:
        ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
        vx, vy = bx - ax, by - ay
        l2 = vx * vx + vy * vy
        if l2 < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
        qx, qy = ax + t * vx, ay + t * vy
        return math.hypot(px - qx, py - qy)

    def on_projection_interval(s_long: Segment, s_short: Segment) -> tuple[float, float]:
        ax, ay = s_long.x1, s_long.y1
        dx, dy = s_long.x2 - ax, s_long.y2 - ay
        lg = math.hypot(dx, dy)
        if lg < 1e-9:
            return 0.0, 0.0
        ux, uy = dx / lg, dy / lg
        def tp(px: float, py: float) -> float:
            return (px - ax) * ux + (py - ay) * uy
        ts = [tp(s_short.x1, s_short.y1), tp(s_short.x2, s_short.y2)]
        return min(ts), max(ts)

    removed = 0
    for i in range(n):
        if drop[i]:
            continue
        for j in range(n):
            if i == j or drop[j]:
                continue
            a, b = segs[i], segs[j]
            la, lb = _seg_len(a), _seg_len(b)
            if la < 4 or lb < 4:
                continue
            long_i, short_i = (i, j) if la >= lb else (j, i)
            if long_i == short_i:
                continue
            L, S = segs[long_i], segs[short_i]
            if abs(_seg_angle(L) - _seg_angle(S)) > ang_tol:
                continue
            mx, my = midpoint(S)
            if dist_point_line(mx, my, L) > lat:
                continue
            t0, t1 = on_projection_interval(L, S)
            lu = _seg_len(L)
            cov = max(0.0, min(t1, lu) - max(t0, 0.0))
            if lu <= 1e-9:
                continue
            if cov / min(_seg_len(S), lu) < overlap_frac_min:
                continue
            # drop shorter
            drop[short_i] = True
            removed += 1
    out = [segs[k] for k in range(n) if not drop[k]]
    return out, removed


def straighten_near_axis_segments(
    segments: list[Segment],
    *,
    angle_tol_deg: float = 6.0,
    min_length_px: float = 16.0,
) -> tuple[list[Segment], int]:
    """Snap almost-H/V strokes to exact horizontal or vertical (CAD regularization helper)."""
    if not segments:
        return segments, 0
    out: list[Segment] = []
    n = 0
    for s in segments:
        dx, dy = s.x2 - s.x1, s.y2 - s.y1
        lg = math.hypot(dx, dy)
        if lg < min_length_px:
            out.append(s)
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        na = ang if ang <= 90.0 else 180.0 - ang
        if na <= angle_tol_deg:
            out.append(Segment(s.x1, s.y1, s.x2, s.y1))
            n += 1
        elif 90.0 - na <= angle_tol_deg:
            out.append(Segment(s.x1, s.y1, s.x1, s.y2))
            n += 1
        else:
            out.append(s)
    return out, n
