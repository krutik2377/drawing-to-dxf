"""Map exploded segments to semantic layers using a pixel label raster."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.semantic_segment import DEFAULT_CLASS_NAMES


def _sample_points_along_segment(s: Segment, n: int = 7) -> list[tuple[float, float]]:
    ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
    if n < 2:
        n = 2
    return [
        (ax + (bx - ax) * i / (n - 1), ay + (by - ay) * i / (n - 1))
        for i in range(n)
    ]


def classify_segment_majority(
    s: Segment,
    labels: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> str:
    """Return winning class name via midpoint + interior samples (nearest-neighbor labels)."""
    names = tuple(class_names) if class_names is not None else DEFAULT_CLASS_NAMES
    h, w = labels.shape[:2]
    pts = _sample_points_along_segment(s, 9)
    votes: list[int] = []
    for x, y in pts:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        li = int(labels[yi, xi])
        if 0 <= li < len(names):
            votes.append(li)
    if not votes:
        return names[0] if names else "background"
    best = Counter(votes).most_common(1)[0][0]
    return names[best]


def split_segments_by_semantic_layer(
    segments: Sequence[Segment],
    labels: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[dict[str, list[Segment]], dict[str, int]]:
    """
    Split segments into GEOMETRY / DIMENSION / BORDER / TEXT buckets for DXF layers.

    BORDER uses the ``title_block`` class index when only ONNX-style names exist.
    """
    names = tuple(class_names) if class_names is not None else DEFAULT_CLASS_NAMES
    name_to_idx = {n: i for i, n in enumerate(names)}
    geom_idx = name_to_idx.get("geometry", 1)
    dim_idx = name_to_idx.get("dimension", 3)
    text_idx = name_to_idx.get("text", 2)
    border_idx = name_to_idx.get("title_block", name_to_idx.get("border", 4))

    out: dict[str, list[Segment]] = {
        "GEOMETRY": [],
        "DIMENSION": [],
        "BORDER": [],
        "TEXT": [],
        "OTHER": [],
    }
    counts = {k: 0 for k in out}

    for s in segments:
        # Fast path: majority class index from samples
        cls_name = classify_segment_majority(s, labels, names)
        idx = name_to_idx.get(cls_name, -1)
        if idx == dim_idx or cls_name == "dimension":
            bucket = "DIMENSION"
        elif idx == text_idx or cls_name in ("text", "table", "symbol"):
            bucket = "TEXT"
        elif idx == border_idx or cls_name in ("title_block", "border"):
            bucket = "BORDER"
        elif idx == geom_idx or cls_name == "geometry":
            bucket = "GEOMETRY"
        else:
            bucket = "OTHER"
        out[bucket].append(Segment(s.x1, s.y1, s.x2, s.y2))
        counts[bucket] += 1

    return out, counts
