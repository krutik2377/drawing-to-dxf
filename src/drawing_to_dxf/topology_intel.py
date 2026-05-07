"""Graph-side topology reasoning on exploded LINE segments (degrees, junction hints)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from drawing_to_dxf.segment_types import Segment


def endpoint_degree_counts(
    segments: Sequence[Segment],
    *,
    quant_px: float = 2.0,
) -> dict[str, int]:
    """
    Quantized vertex incidence counts.

    Returns counts of vertices by topological degree (1 / 2 / 3 / 4+).
    """
    if quant_px <= 0 or not segments:
        return {"vertices_total": 0, "degree_1": 0, "degree_2": 0, "degree_3": 0, "degree_4_plus": 0}

    def q(x: float, y: float) -> tuple[int, int]:
        return (int(round(x / quant_px)), int(round(y / quant_px)))

    deg: Counter[tuple[int, int]] = Counter()
    for s in segments:
        deg[q(s.x1, s.y1)] += 1
        deg[q(s.x2, s.y2)] += 1

    c1 = c2 = c3 = c4p = 0
    for _, d in deg.items():
        if d == 1:
            c1 += 1
        elif d == 2:
            c2 += 1
        elif d == 3:
            c3 += 1
        else:
            c4p += 1

    return {
        "vertices_total": len(deg),
        "degree_1": c1,
        "degree_2": c2,
        "degree_3": c3,
        "degree_4_plus": c4p,
        "t_like_vertices_estimate": c3 + c4p,
    }
