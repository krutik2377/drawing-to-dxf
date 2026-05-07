"""Extract line segments from preprocessed grayscale drawings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from typing import DefaultDict

import cv2
import numpy as np


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    def midpoint(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))


def extract_segments(
    gray: np.ndarray,
    *,
    canny1: int = 40,
    canny2: int = 120,
    hough_threshold: int = 30,
    min_line_length: int = 20,
    max_line_gap: int = 12,
    merge_distance: float = 3.0,
    collinear_merge_angle_deg: float = 5.0,
    polyline_rdp_epsilon_px: float = 1.5,
) -> list[Segment]:
    """
    Edge detect + probabilistic Hough. Filters very short segments; optional endpoint merge.

    After Hough, optional refinements reduce zig-zag and stitch straight chains:
    - ``collinear_merge_angle_deg > 0``: merge consecutive collinear segments at shared joints.
    - ``polyline_rdp_epsilon_px > 0``: Douglas–Peucker simplify along degree-2 chains (curves / jitter).

    If ``merge_distance == 0``, refinement uses a 2 px snap for graph building only (Hough remains unsnapped).

    Limitations (expected on real drawings):
    - Dashed/hidden lines may break into fragments.
    - Thick linework can produce double edges.
    """
    if gray.ndim != 2:
        raise ValueError("grayscale expected")

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return []

    segs: list[Segment] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx * dx + dy * dy < min_line_length * min_line_length:
            continue
        segs.append(Segment(float(x1), float(y1), float(x2), float(y2)))

    if merge_distance > 0 and segs:
        segs = _merge_close_endpoints(segs, merge_distance)

    snap_for_refinement = merge_distance if merge_distance > 0 else 2.0

    if collinear_merge_angle_deg > 0 and segs:
        segs = _collapse_collinear_segments(segs, snap_for_refinement, collinear_merge_angle_deg)

    if polyline_rdp_epsilon_px > 0 and segs:
        segs = _simplify_degree2_chains_rdp(segs, snap_for_refinement, polyline_rdp_epsilon_px)

    return segs


def _dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _merge_close_endpoints(segs: list[Segment], d: float) -> list[Segment]:
    """Greedy merge of segments sharing endpoints within distance d (image pixels)."""
    pts: list[tuple[float, float]] = []
    for s in segs:
        pts.append((s.x1, s.y1))
        pts.append((s.x2, s.y2))

    def cluster_key(p: tuple[float, float]) -> tuple[int, int]:
        return (int(round(p[0] / d)), int(round(p[1] / d)))

    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for p in pts:
        k = cluster_key(p)
        buckets.setdefault(k, []).append(p)

    def canon(p: tuple[float, float]) -> tuple[float, float]:
        k = cluster_key(p)
        near = []
        for dk in (-1, 0, 1):
            for dl in (-1, 0, 1):
                near.extend(buckets.get((k[0] + dk, k[1] + dl), []))
        if not near:
            return p
        mx = sum(q[0] for q in near) / len(near)
        my = sum(q[1] for q in near) / len(near)
        return (mx, my)

    merged: list[Segment] = []
    for s in segs:
        p1 = canon((s.x1, s.y1))
        p2 = canon((s.x2, s.y2))
        if _dist(p1, p2) < 1.0:
            continue
        merged.append(Segment(p1[0], p1[1], p2[0], p2[1]))
    return merged


def _collapse_collinear_segments(segs: list[Segment], tol: float, angle_deg: float) -> list[Segment]:
    """Merge segment pairs that share a joint and lie on one straight ray."""
    if not segs:
        return []
    segs = _merge_close_endpoints(segs, tol)
    sin_thresh = float(np.sin(np.radians(angle_deg)))

    repeat_guard = 0
    while repeat_guard < 200:
        repeat_guard += 1
        merged_one = False
        n = len(segs)
        used = [False] * n
        out: list[Segment] = []
        for i in range(n):
            if used[i]:
                continue
            for j in range(n):
                if i == j or used[j]:
                    continue
                comb = _merge_two_if_collinear(segs[i], segs[j], tol, sin_thresh)
                if comb is not None:
                    out.append(comb)
                    used[i] = used[j] = True
                    merged_one = True
                    break
            if not used[i]:
                out.append(segs[i])
                used[i] = True
        segs = out
        if not merged_one:
            break
    return segs


def _merge_two_if_collinear(
    s1: Segment,
    s2: Segment,
    tol: float,
    sin_thresh: float,
) -> Segment | None:
    """If segments share an endpoint (within tol) and directions align, return the combined span."""
    a1 = (s1.x1, s1.y1)
    b1 = (s1.x2, s1.y2)
    a2 = (s2.x1, s2.y1)
    b2 = (s2.x2, s2.y2)

    trials: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]] = [
        (a1, b1, a2, b2),
        (a1, b1, b2, a2),
        (b1, a1, a2, b2),
        (b1, a1, b2, a2),
    ]
    for p, q, r, t in trials:
        if _dist(q, r) > tol:
            continue
        vx1, vy1 = q[0] - p[0], q[1] - p[1]
        vx2, vy2 = t[0] - r[0], t[1] - r[1]
        len1 = hypot(vx1, vy1)
        len2 = hypot(vx2, vy2)
        if len1 < 1e-9 or len2 < 1e-9:
            continue
        cross = vx1 * vy2 - vy1 * vx2
        if abs(cross) > sin_thresh * len1 * len2 + 1e-9:
            continue
        if vx1 * vx2 + vy1 * vy2 <= 0:
            continue
        return Segment(p[0], p[1], t[0], t[1])
    return None


def _simplify_degree2_chains_rdp(segs: list[Segment], tol: float, epsilon: float) -> list[Segment]:
    """Replace runs of degree-2 vertices with Ramer–Douglas–Peucker simplifications."""
    if not segs or epsilon <= 0:
        return segs

    segs = _merge_close_endpoints(segs, tol)
    keys_to_acc: dict[tuple[int, int], list[float]] = {}

    def pk(x: float, y: float) -> tuple[int, int]:
        ix = int(x / tol + 0.5)
        iy = int(y / tol + 0.5)
        k = (ix, iy)
        if k not in keys_to_acc:
            keys_to_acc[k] = [x, y, 1.0]
        else:
            keys_to_acc[k][0] += x
            keys_to_acc[k][1] += y
            keys_to_acc[k][2] += 1.0
        return k

    def pos(k: tuple[int, int]) -> tuple[float, float]:
        z = keys_to_acc[k]
        return (z[0] / z[2], z[1] / z[2])

    adj: DefaultDict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    seen_seg: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for s in segs:
        k1 = pk(s.x1, s.y1)
        k2 = pk(s.x2, s.y2)
        if k1 == k2:
            continue
        ek = (k1, k2) if k1 < k2 else (k2, k1)
        if ek in seen_seg:
            continue
        seen_seg.add(ek)
        adj[k1].add(k2)
        adj[k2].add(k1)

    if not adj:
        return segs

    junctions = {k for k in adj if len(adj[k]) != 2}
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    poly_pts: list[list[tuple[float, float]]] = []
    poly_closed: list[bool] = []

    def record_chain(chain_keys: list[tuple[int, int]], *, closed: bool = False) -> None:
        raw = [pos(c) for c in chain_keys]
        simp = _ramer_douglas_peucker(raw, epsilon)
        if len(simp) >= 2:
            poly_pts.append(simp)
            poly_closed.append(closed)

    def walk_from(prev: tuple[int, int], curr: tuple[int, int], anchor: tuple[int, int]) -> list[tuple[int, int]]:
        chain = [anchor]
        if prev != anchor:
            chain.append(prev)
        chain.append(curr)
        seen: set[tuple[int, int]] = set(chain)
        _p, c = prev, curr
        max_steps = max(len(adj) + 12, 64)
        steps = 0
        while c not in junctions and len(adj[c]) == 2 and steps < max_steps:
            steps += 1
            nbs = [x for x in adj[c] if x != _p]
            if len(nbs) != 1:
                break
            nxt = nbs[0]
            if nxt in seen:
                if nxt == anchor and len(chain) >= 3:
                    chain.append(nxt)
                break
            chain.append(nxt)
            seen.add(nxt)
            _p, c = c, nxt
        return chain

    def record_walk_chain(ch: list[tuple[int, int]]) -> None:
        if len(ch) >= 4 and ch[0] == ch[-1]:
            record_chain(ch[:-1], closed=True)
        else:
            record_chain(ch, closed=False)

    def mark_chain_edges(chain: list[tuple[int, int]]) -> None:
        for a, b in zip(chain[:-1], chain[1:], strict=False):
            visited_edges.add(frozenset((a, b)))

    for j in junctions:
        for nb in list(adj[j]):
            e0 = frozenset((j, nb))
            if e0 in visited_edges:
                continue
            ch = walk_from(j, nb, j)
            mark_chain_edges(ch)
            record_walk_chain(ch)

    if not junctions:
        start = next(iter(adj))
        nb = next(iter(adj[start]))
        if frozenset((start, nb)) not in visited_edges:
            chain_keys = [start, nb]
            prev, curr = start, nb
            guard = 0
            while curr != start or len(chain_keys) <= 2:
                guard += 1
                if guard > len(adj) + 5:
                    break
                nbs = [x for x in adj[curr] if x != prev]
                if len(nbs) != 1:
                    break
                nxt = nbs[0]
                chain_keys.append(nxt)
                prev, curr = curr, nxt
                if curr == start and len(chain_keys) > 2:
                    break
            if len(chain_keys) >= 3 and chain_keys[0] == chain_keys[-1]:
                mark_chain_edges(chain_keys)
                record_chain(chain_keys[:-1], closed=True)
            elif len(chain_keys) >= 2:
                mark_chain_edges(chain_keys)
                record_chain(chain_keys, closed=False)

    for k in adj:
        for nb in adj[k]:
            if frozenset((k, nb)) in visited_edges:
                continue
            ch = walk_from(k, nb, k)
            mark_chain_edges(ch)
            record_walk_chain(ch)

    out: list[Segment] = []
    for pts, closed in zip(poly_pts, poly_closed, strict=True):
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if _dist(a, b) >= 1.0:
                out.append(Segment(a[0], a[1], b[0], b[1]))
        if closed and len(pts) >= 2:
            a, b = pts[-1], pts[0]
            if _dist(a, b) >= 1.0:
                out.append(Segment(a[0], a[1], b[0], b[1]))
    return out if out else segs


def _ramer_douglas_peucker(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer–Douglas–Peucker polyline simplification (image pixel units)."""
    if len(points) < 3:
        return list(points)

    def perp_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        vx, vy = bx - ax, by - ay
        L = hypot(vx, vy)
        if L < 1e-12:
            return hypot(px - ax, py - ay)
        return abs(vx * (py - ay) - vy * (px - ax)) / L

    stack: list[tuple[int, int]] = [(0, len(points) - 1)]
    keep = {0, len(points) - 1}
    while stack:
        i0, i1 = stack.pop()
        dmax = 0.0
        imax = i0
        for i in range(i0 + 1, i1):
            d = perp_dist(points[i], points[i0], points[i1])
            if d > dmax:
                dmax = d
                imax = i
        if dmax > epsilon:
            keep.add(imax)
            stack.append((i0, imax))
            stack.append((imax, i1))

    order = sorted(keep)
    return [points[i] for i in order]
