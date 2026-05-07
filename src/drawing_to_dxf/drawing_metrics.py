"""Lightweight extraction metrics for manifests and regression checks (Phase 0)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from drawing_to_dxf.geometry_model import VectorDrawing, exploded_segments_for_sampling
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment


def _quantize(p: tuple[float, float], q: float = 2.0) -> tuple[int, int]:
    return (int(round(p[0] / q)), int(round(p[1] / q)))


def estimate_endpoint_junctions(segments: Sequence[Segment], quant: float = 2.5) -> dict[str, int]:
    """Bucket quantized endpoints; junction count ≈ vertices with degree ≠ 2 in segment graph."""
    deg: dict[tuple[int, int], int] = defaultdict(int)
    for s in segments:
        for p in ((s.x1, s.y1), (s.x2, s.y2)):
            deg[_quantize(p, quant)] += 1
    n_j = sum(1 for _k, d in deg.items() if d != 2)
    n_end = sum(1 for _k, d in deg.items() if d == 1)
    return {
        "endpoint_buckets": len(deg),
        "junction_like_buckets": n_j,
        "degree_one_buckets": n_end,
    }


def mean_segment_length(segments: Sequence[Segment]) -> float:
    if not segments:
        return 0.0
    lens = [math.hypot(s.x2 - s.x1, s.y2 - s.y1) for s in segments]
    return float(sum(lens) / len(lens))


def extraction_metrics_report(
    vd: VectorDrawing,
    sample_segments: Sequence[Segment],
    ocr_boxes: Sequence[TextBox],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ref = exploded_segments_for_sampling(vd)
    ej = estimate_endpoint_junctions(sample_segments if sample_segments else ref)
    return {
        "circle_count": len(vd.circles),
        "arc_count": len(vd.arcs),
        "polyline_count": len(vd.polylines),
        "residual_segment_count": len(vd.residual_segments),
        "exploded_segment_count": len(ref),
        "sample_segment_count": len(sample_segments),
        "mean_sample_segment_length_px": round(mean_segment_length(sample_segments), 3),
        "ocr_text_box_count": len(ocr_boxes),
        "endpoint_analysis_quant_px": 2.5,
        **ej,
        **dict(extra or {}),
    }
