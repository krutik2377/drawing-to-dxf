"""Extract line segments from preprocessed grayscale drawings."""

from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Any, DefaultDict, MutableMapping, Sequence

import cv2
import numpy as np

from drawing_to_dxf.geometry_model import VectorDrawing, exploded_segments_for_sampling
from drawing_to_dxf.geometry_intel import (
    bridge_open_polyline_gaps,
    dedupe_segment_list,
    lsd_extract_segments,
    remove_redundant_segments_vs_reference,
)
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.skeleton_vectorize import prepare_masked_gray
from drawing_to_dxf.constraint_heal import apply_constraint_heal
from drawing_to_dxf.engineering_intel_passes import (
    apply_engineering_intelligence_vector_stage,
    topology_intel_on_exploded_segments,
)
from drawing_to_dxf.topology_clean import refine_vector_drawing
from drawing_to_dxf.topology_repair import repair_exploded_segments
from drawing_to_dxf.vector_fit import apply_polyline_fittings


def extract_skeleton_vector_bundle(
    gray: np.ndarray,
    *,
    annotation_boxes: Sequence[TextBox] | None = None,
    mask_annotation_regions: bool = True,
    min_line_length: int = 20,
    merge_distance: float = 3.0,
    collinear_merge_angle_deg: float = 5.0,
    polyline_rdp_epsilon_px: float = 1.5,
    annotation_pad_px: float = 5.0,
    annotation_box_shrink_from_pad_px: float = 0.0,
    enable_circles: bool = True,
    mask_text_interior_only: bool = False,
    soft_ink_mask: bool = False,
    enable_topology_clean: bool = False,
    enable_arc_fitting: bool = False,
    enable_loop_circle_fit: bool = False,
    lsd_supplement: bool = False,
    lsd_min_length_px: float | None = None,
    geometry_bridge_gap_px: float = 0.0,
    dedupe_residual_segments: bool = True,
    enable_topology_segment_repair: bool = False,
    topology_max_bridge_gap_px: float | None = None,
    topology_junction_snap_px: float = 3.0,
    topology_bridge_direction_dot_min: float = 0.42,
    topology_repair_metrics: MutableMapping[str, int] | None = None,
    debug_stages: MutableMapping[str, np.ndarray] | None = None,
    engineering_layout: bool = False,
    ruling_suppress_strength: float | None = None,
    multi_scale_ink: bool = False,
    protect_hole_rings: bool = True,
    enable_constraint_heal: bool = False,
    constraint_orthogonal_quad_corner_tol_deg: float = 14.0,
    complete_full_arcs_to_circles_min_span_deg: float | None = None,
    enable_engineering_intel_passes: bool = False,
    engineering_intel_metrics: MutableMapping[str, Any] | None = None,
    topology_intersection_extend_px: float = 0.0,
    enable_cad_axis_regularization: bool = False,
    enable_healed_vector_export: bool = False,
) -> tuple[VectorDrawing, list[Segment]]:
    """Skeleton-graph vectorization (+ optional OCR masks) packaged with exploded segments for OCR linking."""
    if gray.ndim != 2:
        raise ValueError("grayscale expected")

    intel_sink: MutableMapping[str, Any] = (
        engineering_intel_metrics if engineering_intel_metrics is not None else {}
    )

    from drawing_to_dxf.skeleton_vectorize import extract_vector_drawing as _extract_vd

    boxes_arg = annotation_boxes if mask_annotation_regions else None

    vd = _extract_vd(
        gray,
        annotation_boxes=boxes_arg if boxes_arg else None,
        annotation_pad_px=float(annotation_pad_px),
        annotation_box_shrink_from_pad_px=float(annotation_box_shrink_from_pad_px),
        min_skeleton_branch_px=max(12, min(160, min_line_length)),
        rdp_tolerance_px=max(0.4, polyline_rdp_epsilon_px),
        enable_circles=enable_circles,
        mask_text_interior_only=mask_text_interior_only,
        soft_ink_mask=soft_ink_mask,
        engineering_layout=engineering_layout,
        ruling_suppress_strength=ruling_suppress_strength,
        multi_scale_ink=multi_scale_ink,
        protect_hole_rings=protect_hole_rings,
        debug_stages=debug_stages,
    )
    mb_px = float(max(12, min(160, min_line_length)))

    if enable_topology_clean:
        vd = refine_vector_drawing(vd, min_branch_px=mb_px)
    if enable_arc_fitting or enable_loop_circle_fit or complete_full_arcs_to_circles_min_span_deg is not None:
        vd = apply_polyline_fittings(
            vd,
            fit_arcs=enable_arc_fitting,
            fit_circles_from_loops=enable_loop_circle_fit,
            complete_full_arcs_to_circles_min_span_deg=complete_full_arcs_to_circles_min_span_deg,
        )

    if enable_constraint_heal:
        vd = apply_constraint_heal(
            vd,
            orthogonal_quads=True,
            corner_tol_deg=float(constraint_orthogonal_quad_corner_tol_deg),
        )

    if enable_engineering_intel_passes:
        vd = apply_engineering_intelligence_vector_stage(
            vd,
            ocr_boxes=list(annotation_boxes) if annotation_boxes else None,
            metrics=intel_sink,
        )

    if geometry_bridge_gap_px > 0:
        vd = bridge_open_polyline_gaps(vd, max_gap_px=geometry_bridge_gap_px)

    if lsd_supplement:
        mg = prepare_masked_gray(
            gray,
            annotation_boxes=boxes_arg if boxes_arg else None,
            annotation_pad_px=float(annotation_pad_px),
            annotation_box_shrink_from_pad_px=float(annotation_box_shrink_from_pad_px),
            mask_text_interior_only=mask_text_interior_only,
        )
        lsd_floor = float(lsd_min_length_px) if lsd_min_length_px is not None else float(max(min_line_length, 10))
        raw_lsd = lsd_extract_segments(mg, min_length_px=max(6.0, lsd_floor * 0.85))
        ref = exploded_segments_for_sampling(vd)
        kept = remove_redundant_segments_vs_reference(raw_lsd, ref)
        kept = dedupe_segment_list(kept)
        if kept:
            vd = VectorDrawing(
                polylines=list(vd.polylines),
                circles=list(vd.circles),
                arcs=list(vd.arcs),
                residual_segments=list(vd.residual_segments) + kept,
            )

    if dedupe_residual_segments and vd.residual_segments:
        rd = dedupe_segment_list(list(vd.residual_segments))
        if len(rd) != len(vd.residual_segments):
            vd = VectorDrawing(
                polylines=list(vd.polylines),
                circles=list(vd.circles),
                arcs=list(vd.arcs),
                residual_segments=rd,
            )

    segs_raw = exploded_segments_for_sampling(vd)
    sq = float(min_line_length * min_line_length)
    segs = [s for s in segs_raw if hypot(s.x2 - s.x1, s.y2 - s.y1) >= min_line_length * 0.85]

    if enable_topology_segment_repair and segs:
        auto_bridge = (
            topology_max_bridge_gap_px
            if topology_max_bridge_gap_px is not None
            else max(4.0, float(merge_distance) * 1.15)
        )
        repair_merge_d = float(merge_distance) if merge_distance > 0 else 2.0
        segs = repair_exploded_segments(
            segs,
            merge_distance=repair_merge_d,
            max_bridge_gap_px=float(auto_bridge),
            junction_snap_px=float(topology_junction_snap_px),
            bridge_direction_dot_min=float(topology_bridge_direction_dot_min),
            intersection_extend_px=float(topology_intersection_extend_px),
            stats=None,
            manifest_metrics=topology_repair_metrics,
        )

    if enable_cad_axis_regularization and segs:
        from drawing_to_dxf.constraint_heal import (
            align_parallel_residual_clusters,
            snap_axis_aligned_residual_segments,
        )

        segs, n_ax = snap_axis_aligned_residual_segments(
            segs,
            angle_tol_deg=2.85,
            min_length_px=max(8.0, float(min_line_length) * 0.55),
        )
        segs, n_pr = align_parallel_residual_clusters(
            segs,
            angle_tol_deg=2.35,
            offset_snap_px=max(1.5, float(merge_distance) * 0.4),
            min_cluster_size=3,
            min_length_px=max(22.0, float(min_line_length) * 1.05),
        )
        if topology_repair_metrics is not None:
            topology_repair_metrics["axis_snap_residuals"] = int(n_ax)
            topology_repair_metrics["parallel_cluster_snaps"] = int(n_pr)

    if merge_distance > 0 and segs:
        segs = _merge_close_endpoints(segs, merge_distance)
    snap_for_refinement = merge_distance if merge_distance > 0 else 2.0
    if collinear_merge_angle_deg > 0 and segs:
        segs = _collapse_collinear_segments(segs, snap_for_refinement, collinear_merge_angle_deg)
    if polyline_rdp_epsilon_px > 0 and segs:
        segs = _simplify_degree2_chains_rdp(segs, snap_for_refinement, polyline_rdp_epsilon_px)
    trimmed = []
    for s in segs:
        if hypot(s.x2 - s.x1, s.y2 - s.y1) * hypot(s.x2 - s.x1, s.y2 - s.y1) >= sq * 0.25:
            trimmed.append(s)
    out_segs = trimmed if trimmed else segs_raw
    if enable_engineering_intel_passes:
        topology_intel_on_exploded_segments(out_segs, intel_sink)
    if enable_healed_vector_export and out_segs:
        from drawing_to_dxf.cad_geometry_rebuild import vector_drawing_from_healed_segments

        vd = vector_drawing_from_healed_segments(vd, out_segs)
    return vd, out_segs


def _legacy_hough_extract_segments(
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
    legacy_vectorize_hough: bool = False,
    annotation_boxes: Sequence[TextBox] | None = None,
    mask_annotation_regions: bool = True,
    annotation_pad_px: float = 10.0,
    annotation_box_shrink_from_pad_px: float = 0.0,
    enable_circles: bool = True,
    mask_text_interior_only: bool = False,
    soft_ink_mask: bool = False,
    enable_topology_clean: bool = False,
    enable_arc_fitting: bool = False,
    enable_loop_circle_fit: bool = False,
    lsd_supplement: bool = False,
    lsd_min_length_px: float | None = None,
    geometry_bridge_gap_px: float = 0.0,
    dedupe_residual_segments: bool = True,
    enable_topology_segment_repair: bool = False,
    topology_max_bridge_gap_px: float | None = None,
    topology_junction_snap_px: float = 3.0,
    topology_bridge_direction_dot_min: float = 0.42,
    topology_repair_metrics: MutableMapping[str, int] | None = None,
    engineering_layout: bool = False,
    ruling_suppress_strength: float | None = None,
    multi_scale_ink: bool = False,
    protect_hole_rings: bool = True,
    enable_constraint_heal: bool = False,
    constraint_orthogonal_quad_corner_tol_deg: float = 14.0,
    complete_full_arcs_to_circles_min_span_deg: float | None = None,
    enable_engineering_intel_passes: bool = False,
    engineering_intel_metrics: MutableMapping[str, Any] | None = None,
    topology_intersection_extend_px: float = 0.0,
    enable_cad_axis_regularization: bool = False,
    enable_healed_vector_export: bool = False,
) -> list[Segment]:
    """Primary path: OCR-mask aware skeleton-graph tracing (:mod:`drawing_to_dxf.skeleton_vectorize`)."""
    _ = canny1, canny2, hough_threshold, max_line_gap  # legacy kwargs retained for callers/tests

    if legacy_vectorize_hough:
        return _legacy_hough_extract_segments(
            gray,
            canny1=canny1,
            canny2=canny2,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            merge_distance=merge_distance,
            collinear_merge_angle_deg=collinear_merge_angle_deg,
            polyline_rdp_epsilon_px=polyline_rdp_epsilon_px,
        )

    _vd, sg = extract_skeleton_vector_bundle(
        gray,
        annotation_boxes=annotation_boxes,
        mask_annotation_regions=mask_annotation_regions,
        min_line_length=min_line_length,
        merge_distance=merge_distance,
        collinear_merge_angle_deg=collinear_merge_angle_deg,
        polyline_rdp_epsilon_px=polyline_rdp_epsilon_px,
        annotation_pad_px=annotation_pad_px,
        annotation_box_shrink_from_pad_px=annotation_box_shrink_from_pad_px,
        enable_circles=enable_circles,
        mask_text_interior_only=mask_text_interior_only,
        soft_ink_mask=soft_ink_mask,
        enable_topology_clean=enable_topology_clean,
        enable_arc_fitting=enable_arc_fitting,
        enable_loop_circle_fit=enable_loop_circle_fit,
        lsd_supplement=lsd_supplement,
        lsd_min_length_px=lsd_min_length_px,
        geometry_bridge_gap_px=geometry_bridge_gap_px,
        dedupe_residual_segments=dedupe_residual_segments,
        enable_topology_segment_repair=enable_topology_segment_repair,
        topology_max_bridge_gap_px=topology_max_bridge_gap_px,
        topology_junction_snap_px=topology_junction_snap_px,
        topology_bridge_direction_dot_min=topology_bridge_direction_dot_min,
        topology_repair_metrics=topology_repair_metrics,
        engineering_layout=engineering_layout,
        ruling_suppress_strength=ruling_suppress_strength,
        multi_scale_ink=multi_scale_ink,
        protect_hole_rings=protect_hole_rings,
        enable_constraint_heal=enable_constraint_heal,
        constraint_orthogonal_quad_corner_tol_deg=constraint_orthogonal_quad_corner_tol_deg,
        complete_full_arcs_to_circles_min_span_deg=complete_full_arcs_to_circles_min_span_deg,
        enable_engineering_intel_passes=enable_engineering_intel_passes,
        engineering_intel_metrics=engineering_intel_metrics,
        topology_intersection_extend_px=topology_intersection_extend_px,
        enable_cad_axis_regularization=enable_cad_axis_regularization,
        enable_healed_vector_export=enable_healed_vector_export,
    )
    return sg


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
