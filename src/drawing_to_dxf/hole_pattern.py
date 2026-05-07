"""Bolt / hole pattern hints from extracted circle primitives (manifest metrics)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from drawing_to_dxf.geometry_model import CircleDef


def _cluster_indices(xs: Sequence[float], ys: Sequence[float], tol_px: float) -> list[list[int]]:
    """Greedy single-linkage by centroid distance."""
    n = len(xs)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d <= tol_px:
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())


def summarize_hole_patterns(
    circles: Sequence[CircleDef],
    *,
    cluster_tol_px: float | None = None,
) -> dict[str, Any]:
    """
    Cheap grid / repeat summary for hole detection QA (not full bolt-pattern solver).

    Returns JSON-friendly counts and cluster size histogram.
    """
    if not circles:
        return {"circle_count": 0, "clusters": 0, "largest_cluster": 0}

    xs = [float(c.cx) for c in circles]
    ys = [float(c.cy) for c in circles]
    rs = [float(c.r) for c in circles]
    med_r = sorted(rs)[len(rs) // 2]
    tol = float(cluster_tol_px) if cluster_tol_px is not None else max(12.0, 2.8 * med_r)

    clusters = _cluster_indices(xs, ys, tol)
    sizes = [len(c) for c in clusters]
    sizes.sort(reverse=True)
    hist = Counter(sizes)
    return {
        "circle_count": len(circles),
        "cluster_tol_px": round(tol, 3),
        "clusters": len(clusters),
        "largest_cluster": sizes[0] if sizes else 0,
        "cluster_size_histogram": {str(k): hist[k] for k in sorted(hist.keys())},
    }
