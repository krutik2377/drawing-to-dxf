"""Heal exploded segment graphs: T-junction snaps, directed gap bridges, endpoint clustering."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import MutableMapping

import numpy as np

from drawing_to_dxf.segment_types import Segment


@dataclass
class TopologyRepairStats:
    gap_bridges_added: int = 0
    junction_snaps: int = 0
    intersection_extends: int = 0
    passes_run: int = 0

    def to_manifest_dict(self) -> dict[str, int]:
        return {
            "topology_gap_bridges": self.gap_bridges_added,
            "topology_junction_snaps": self.junction_snaps,
            "topology_intersection_extends": self.intersection_extends,
            "topology_repair_passes": self.passes_run,
        }


def _dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _seg_tangent_into_endpoint(s: Segment, *, at_first: bool) -> tuple[float, float] | None:
    ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
    if at_first:
        vx, vy = bx - ax, by - ay
    else:
        vx, vy = ax - bx, ay - by
    l = math.hypot(vx, vy)
    if l < 1e-9:
        return None
    return (vx / l, vy / l)


def _endpoint_coords(s: Segment, *, at_first: bool) -> tuple[float, float]:
    if at_first:
        return (s.x1, s.y1)
    return (s.x2, s.y2)


def _set_endpoint(segs: list[Segment], idx: int, *, at_first: bool, p: tuple[float, float]) -> None:
    s = segs[idx]
    if at_first:
        segs[idx] = Segment(p[0], p[1], s.x2, s.y2)
    else:
        segs[idx] = Segment(s.x1, s.y1, p[0], p[1])


def _project_point_to_segment(
    px: float,
    py: float,
    s: Segment,
) -> tuple[tuple[float, float], float, float]:
    ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 < 1e-12:
        q = (ax, ay)
        return q, 0.0, _dist((px, py), q)
    t = ((px - ax) * vx + (py - ay) * vy) / l2
    t_clamped = max(0.0, min(1.0, t))
    qx, qy = ax + t_clamped * vx, ay + t_clamped * vy
    return (qx, qy), float(t_clamped), _dist((px, py), (qx, qy))


def merge_close_segment_endpoints(segs: list[Segment], d: float) -> list[Segment]:
    """Same bucket policy as vectorize._merge_close_endpoints (mean per cluster)."""
    if d <= 0 or not segs:
        return segs
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
        near: list[tuple[float, float]] = []
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


def _round_pt(p: tuple[float, float], step: float = 0.25) -> tuple[int, int]:
    return (int(round(p[0] / step)), int(round(p[1] / step)))


def _try_junction_snaps(
    segs: list[Segment],
    *,
    junction_snap_px: float,
    endpoint_margin: float = 0.035,
) -> int:
    """Snap free endpoints onto nearby segment interiors (T / near-T). Returns move count."""
    if junction_snap_px <= 0 or len(segs) < 2:
        return 0
    moves = 0
    n = len(segs)
    for i in range(n):
        for at_first in (True, False):
            p = _endpoint_coords(segs[i], at_first=at_first)
            best_j = -1
            best_q: tuple[float, float] | None = None
            best_d = junction_snap_px + 1.0
            for j in range(n):
                if j == i:
                    continue
                q, t, d = _project_point_to_segment(p[0], p[1], segs[j])
                if t <= endpoint_margin or t >= 1.0 - endpoint_margin:
                    continue
                if d >= best_d:
                    continue
                best_d = d
                best_j = j
                best_q = q
            if best_j < 0 or best_q is None:
                continue
            v = (best_q[0] - p[0], best_q[1] - p[1])
            vl = math.hypot(v[0], v[1])
            if vl < 1e-9:
                continue
            _set_endpoint(segs, i, at_first=at_first, p=best_q)
            moves += 1
    return moves


def _ray_segment_hit(
    px: float,
    py: float,
    ux: float,
    uy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    max_t: float,
) -> tuple[float, tuple[float, float]] | None:
    """Ray P + t u (u unit, t >= 0) vs segment AB, t <= max_t. Returns (t, hit_point)."""
    vx, vy = bx - ax, by - ay
    denom = ux * vy - uy * vx
    if abs(denom) < 1e-10:
        return None
    wx, wy = ax - px, ay - py
    t = (wx * vy - wy * vx) / denom
    s = (wx * uy - wy * ux) / denom
    if t < 0.02 or t > max_t:
        return None
    if s < -0.03 or s > 1.03:
        return None
    qx, qy = ax + s * vx, ay + s * vy
    return (t, (qx, qy))


def extend_free_endpoints_to_intersections(
    segs: list[Segment],
    *,
    max_extend_px: float,
    endpoint_quant_px: float = 2.0,
) -> int:
    """
    For graph-degree-1 endpoints (quantized), extend along the stroke direction until the
    ray hits another segment within ``max_extend_px`` (near misses / broken X junctions).
    """
    if max_extend_px <= 0 or len(segs) < 2:
        return 0

    def qk(x: float, y: float) -> tuple[int, int]:
        return (int(round(x / endpoint_quant_px)), int(round(y / endpoint_quant_px)))

    ctr: Counter[tuple[int, int]] = Counter()
    for s in segs:
        ctr[qk(s.x1, s.y1)] += 1
        ctr[qk(s.x2, s.y2)] += 1

    moves = 0
    for i in range(len(segs)):
        moved_this_seg = False
        for at_first in (True, False):
            if moved_this_seg:
                break
            px, py = _endpoint_coords(segs[i], at_first=at_first)
            if ctr[qk(px, py)] != 1:
                continue
            tu = _seg_tangent_into_endpoint(segs[i], at_first=at_first)
            if tu is None:
                continue
            ux, uy = tu
            best_t: float | None = None
            best_q: tuple[float, float] | None = None
            for j in range(len(segs)):
                if j == i:
                    continue
                s2 = segs[j]
                hit = _ray_segment_hit(px, py, ux, uy, s2.x1, s2.y1, s2.x2, s2.y2, max_extend_px)
                if hit is None:
                    continue
                t, q = hit
                if best_t is None or t < best_t:
                    best_t, best_q = t, q
            if best_q is None or best_t is None:
                continue
            _set_endpoint(segs, i, at_first=at_first, p=best_q)
            moves += 1
            moved_this_seg = True
    return moves


def _try_gap_bridges(
    segs: list[Segment],
    *,
    max_bridge_gap_px: float,
    bridge_direction_dot_min: float,
) -> int:
    if max_bridge_gap_px <= 0:
        return 0
    added = 0
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    n = len(segs)
    endpoints: list[tuple[int, bool, tuple[float, float]]] = []
    for i in range(n):
        for at_first in (True, False):
            p = _endpoint_coords(segs[i], at_first=at_first)
            endpoints.append((i, at_first, p))

    for ii in range(len(endpoints)):
        i, i_first, p = endpoints[ii]
        tu_p = _seg_tangent_into_endpoint(segs[i], at_first=i_first)
        if tu_p is None:
            continue
        for jj in range(ii + 1, len(endpoints)):
            j, j_first, q = endpoints[jj]
            if i == j:
                continue
            d = _dist(p, q)
            if d <= 1.0 or d > max_bridge_gap_px:
                continue
            tu_q = _seg_tangent_into_endpoint(segs[j], at_first=j_first)
            if tu_q is None:
                continue
            lab = d
            ux = (q[0] - p[0]) / lab
            uy = (q[1] - p[1]) / lab
            if tu_p[0] * ux + tu_p[1] * uy < bridge_direction_dot_min:
                continue
            if tu_q[0] * (-ux) + tu_q[1] * (-uy) < bridge_direction_dot_min:
                continue
            k1 = _round_pt(p)
            k2 = _round_pt(q)
            ek = (k1, k2) if k1 <= k2 else (k2, k1)
            if ek in seen:
                continue
            seen.add(ek)
            segs.append(Segment(p[0], p[1], q[0], q[1]))
            added += 1
    return added


def repair_exploded_segments(
    segments: list[Segment],
    *,
    merge_distance: float,
    max_bridge_gap_px: float,
    junction_snap_px: float,
    bridge_direction_dot_min: float = 0.42,
    intersection_extend_px: float = 0.0,
    max_passes: int = 10,
    stats: TopologyRepairStats | None = None,
    manifest_metrics: MutableMapping[str, int] | None = None,
) -> list[Segment]:
    """
    Multi-pass heal: T snaps → bridges → optional ray intersection extension → endpoint merge.

    Runs until a pass adds no snaps, bridges, or extensions.
    """
    if not segments:
        return []
    segs = [Segment(s.x1, s.y1, s.x2, s.y2) for s in segments]
    total_snaps = 0
    total_bridges = 0
    total_ext = 0
    passes_run = 0
    eq = max(2.0, float(merge_distance) * 0.65) if merge_distance > 0 else 2.0
    for _ in range(max_passes):
        passes_run += 1
        ms = (
            _try_junction_snaps(segs, junction_snap_px=junction_snap_px)
            if junction_snap_px > 0
            else 0
        )
        mb = (
            _try_gap_bridges(
                segs,
                max_bridge_gap_px=max_bridge_gap_px,
                bridge_direction_dot_min=bridge_direction_dot_min,
            )
            if max_bridge_gap_px > 0
            else 0
        )
        me = (
            extend_free_endpoints_to_intersections(
                segs,
                max_extend_px=float(intersection_extend_px),
                endpoint_quant_px=eq,
            )
            if intersection_extend_px > 0
            else 0
        )
        total_snaps += ms
        total_bridges += mb
        total_ext += me
        if merge_distance > 0 and segs:
            segs = merge_close_segment_endpoints(segs, merge_distance)
        segs = [s for s in segs if _dist((s.x1, s.y1), (s.x2, s.y2)) >= 0.5]
        if ms == 0 and mb == 0 and me == 0:
            break

    if stats is not None:
        stats.gap_bridges_added = total_bridges
        stats.junction_snaps = total_snaps
        stats.intersection_extends = total_ext
        stats.passes_run = passes_run

    if manifest_metrics is not None:
        manifest_metrics["topology_gap_bridges"] = total_bridges
        manifest_metrics["topology_junction_snaps"] = total_snaps
        manifest_metrics["topology_intersection_extends"] = total_ext
        manifest_metrics["topology_repair_passes"] = passes_run

    return segs


def bridge_almost_closed_loops(
    segments: list[Segment],
    *,
    max_gap_px: float,
    endpoint_quant_px: float = 2.0,
) -> tuple[list[Segment], int]:
    """
    Connect degree-1 sketch endpoints that face each other within ``max_gap_px`` (loop closure).

    Complements gap bridges by targeting *short* remaining cracks in nearly closed chains.
    """
    if max_gap_px <= 0 or len(segments) < 2:
        return segments, 0

    segs = [Segment(s.x1, s.y1, s.x2, s.y2) for s in segments]

    def qk(x: float, y: float) -> tuple[int, int]:
        return (int(round(x / endpoint_quant_px)), int(round(y / endpoint_quant_px)))

    ctr: Counter[tuple[int, int]] = Counter()
    for s in segs:
        ctr[qk(s.x1, s.y1)] += 1
        ctr[qk(s.x2, s.y2)] += 1

    endpoints: list[tuple[int, bool, tuple[float, float]]] = []
    for i, s in enumerate(segs):
        for at_first in (True, False):
            px, py = (s.x1, s.y1) if at_first else (s.x2, s.y2)
            if ctr[qk(px, py)] == 1:
                endpoints.append((i, at_first, (px, py)))

    added = 0
    seen_e: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for a in range(len(endpoints)):
        ia, _fa, pa = endpoints[a]
        for b in range(a + 1, len(endpoints)):
            ib, _fb, pb = endpoints[b]
            if ia == ib:
                continue
            d = _dist(pa, pb)
            if d <= 0.5 or d > max_gap_px:
                continue
            ek1 = _round_pt(pa)
            ek2 = _round_pt(pb)
            ek = (ek1, ek2) if ek1 <= ek2 else (ek2, ek1)
            if ek in seen_e:
                continue
            seen_e.add(ek)
            segs.append(Segment(pa[0], pa[1], pb[0], pb[1]))
            added += 1

    return segs, added