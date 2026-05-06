"""Extract line segments from preprocessed grayscale drawings."""

from __future__ import annotations

from dataclasses import dataclass

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
) -> list[Segment]:
    """
    Edge detect + probabilistic Hough. Filters very short segments; optional endpoint merge.

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
