"""Extended engineering intelligence: arcs→circles, confidence, clusters, patterns, symbols, centerlines."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Sequence

from drawing_to_dxf.geometry_model import ArcDef, CircleDef, VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment


def arc_span_deg(a: ArcDef) -> float:
    """Sweep magnitude in degrees along arc orientation."""
    s = a.start_angle_deg % 360.0
    e = a.end_angle_deg % 360.0
    if a.ccw:
        return (e - s) % 360.0
    return (s - e) % 360.0


def promote_near_full_arcs_to_circles(
    vd: VectorDrawing,
    *,
    min_span_deg: float = 300.0,
) -> tuple[VectorDrawing, int]:
    """
    Complete broken circles represented as high-span arcs (circle/hole reconstruction helper).
    """
    thr = float(min_span_deg)
    if thr <= 0 or not vd.arcs:
        return vd, 0
    new_circles = list(vd.circles)
    kept: list[ArcDef] = []
    promoted = 0
    for a in vd.arcs:
        if arc_span_deg(a) >= thr - 1e-6:
            new_circles.append(CircleDef(a.cx, a.cy, a.r))
            promoted += 1
        else:
            kept.append(a)
    if promoted == 0:
        return vd, 0
    return (
        VectorDrawing(
            polylines=list(vd.polylines),
            circles=new_circles,
            arcs=kept,
            residual_segments=list(vd.residual_segments),
        ),
        promoted,
    )


def segment_primitive_confidence(s: Segment, *, min_len_boost: float = 24.0) -> float:
    """
    Score 0–1 for line-like primitives (reject jitter / noise). Axis-aligned strokes score higher.
    """
    lg = math.hypot(s.x2 - s.x1, s.y2 - s.y1)
    if lg < 1.5:
        return 0.0
    dx, dy = abs(s.x2 - s.x1), abs(s.y2 - s.y1)
    axis = max(dx, dy) / (dx + dy + 1e-9)
    len_score = min(1.0, lg / max(min_len_boost, 12.0))
    return float(max(0.0, min(1.0, 0.35 * axis + 0.65 * len_score)))


def filter_segments_by_confidence(
    segments: list[Segment],
    *,
    min_score: float = 0.12,
) -> tuple[list[Segment], int]:
    """Drop very low-confidence vectors (primitive confidence scoring)."""
    if min_score <= 0:
        return list(segments), 0
    out: list[Segment] = []
    dropped = 0
    for s in segments:
        sc = segment_primitive_confidence(s)
        if sc >= min_score:
            out.append(s)
        else:
            dropped += 1
    return out, dropped


def _angle_bucket(s: Segment, *, n_bins: int = 36) -> int:
    dx, dy = s.x2 - s.x1, s.y2 - s.y1
    ang = math.atan2(dy, dx)
    u = (ang + math.pi) / (2 * math.pi)
    return int(max(0, min(n_bins - 1, round(u * n_bins))))


def cluster_parallel_segments(
    segments: Sequence[Segment],
    *,
    n_angle_bins: int = 36,
    min_cluster: int = 3,
) -> tuple[list[list[int]], dict[str, int]]:
    """Group segment indices with similar orientation (repeated entity clustering)."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(segments):
        buckets[_angle_bucket(s, n_bins=n_angle_bins)].append(i)
    clusters = [idxs for idxs in buckets.values() if len(idxs) >= min_cluster]
    stats = {
        "parallel_clusters": len(clusters),
        "largest_cluster": max((len(c) for c in clusters), default=0),
    }
    return clusters, stats


def infer_regular_grid_pitch(
    centers: Sequence[tuple[float, float]],
    *,
    max_pairs: int = 8000,
) -> dict[str, Any]:
    """Infer dominant Δx / Δy spacing for repeated hole centers (structural pattern inference)."""
    pts = list(centers)
    n = len(pts)
    if n < 4:
        return {"ok": False, "reason": "not_enough_points"}
    dxs: list[float] = []
    dys: list[float] = []
    checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            if checked > max_pairs:
                break
            checked += 1
            dxs.append(abs(pts[i][0] - pts[j][0]))
            dys.append(abs(pts[i][1] - pts[j][1]))
        if checked > max_pairs:
            break
    if not dxs:
        return {"ok": False}

    def dominant_positive(vals: list[float], *, tol_ratio: float = 0.06) -> float | None:
        positives = [v for v in vals if v > 3.0]
        if len(positives) < 8:
            return None
        positives.sort()
        # histogram by quantization
        q = max(2.0, sum(positives) / len(positives) * tol_ratio * 4)
        buckets: dict[int, list[float]] = defaultdict(list)
        for v in positives:
            buckets[int(round(v / q))].append(v)
        best = None
        best_n = 0
        for _k, vs in buckets.items():
            if len(vs) > best_n:
                best_n = len(vs)
                best = sum(vs) / len(vs)
        return best

    px = dominant_positive(dxs)
    py = dominant_positive(dys)
    return {
        "ok": px is not None or py is not None,
        "pitch_x_px": round(float(px), 3) if px else None,
        "pitch_y_px": round(float(py), 3) if py else None,
        "pairs_sampled": min(checked, max_pairs),
    }


def symmetry_about_axis_hint(
    segments: Sequence[Segment],
    *,
    sample_cap: int = 400,
) -> dict[str, Any]:
    """Cheap symmetry probe: compare H vs V segment count balance (not full reflection test)."""
    if not segments:
        return {"symmetry_score": 0.0}
    h = v = o = 0
    for i, s in enumerate(segments):
        if i >= sample_cap:
            break
        dx, dy = abs(s.x2 - s.x1), abs(s.y2 - s.y1)
        if dx < 1.5 and dy >= 6:
            v += 1
        elif dy < 1.5 and dx >= 6:
            h += 1
        else:
            o += 1
    tot = max(h + v + o, 1)
    bal = 1.0 - abs(h - v) / tot
    return {
        "symmetry_score": round(float(max(0.0, min(1.0, bal))), 4),
        "axis_seg_counts": {"horizontal": h, "vertical": v, "other": o},
    }


_WELDISH = re.compile(r"\b(weld|fillet|bevel|groove)\b", re.I)
_SECTION = re.compile(r"^[A-Z]\d?$|SECTION|DETAIL", re.I)


def engineering_symbol_candidates(
    ocr_boxes: Sequence[TextBox] | None,
    *,
    segments: Sequence[Segment] | None = None,
) -> list[dict[str, Any]]:
    """Heuristic weld / section / GD&T *hints* from OCR tokens (not trained detector)."""
    out: list[dict[str, Any]] = []
    if not ocr_boxes:
        return out
    for tb in ocr_boxes:
        t = (tb.text or "").strip()
        if not t:
            continue
        cx, cy = tb.center()
        sym = None
        if _WELDISH.search(t):
            sym = "weld_annotation"
        elif _SECTION.search(t):
            sym = "section_mark"
        elif any(ch in t for ch in "⌭⌯⏤⊥∥◎"):
            sym = "gdt_like"
        if sym:
            item: dict[str, Any] = {
                "kind": sym,
                "text_excerpt": t[:80],
                "center_px": {"x": round(cx, 2), "y": round(cy, 2)},
            }
            out.append(item)
    # Center-mark ticks: very short segments near circle holes (optional)
    if segments and ocr_boxes:
        for tb in ocr_boxes:
            if len(tb.text or "") > 3:
                continue
            cx, cy = tb.center()
            near = 0
            for s in segments:
                if math.hypot(0.5 * (s.x1 + s.x2) - cx, 0.5 * (s.y1 + s.y2) - cy) < 20:
                    near += 1
            if near >= 2:
                out.append(
                    {
                        "kind": "center_mark_candidate",
                        "text_excerpt": (tb.text or "")[:20],
                        "center_px": {"x": round(cx, 2), "y": round(cy, 2)},
                    }
                )
    return out[:400]


def centerline_alignment_segments(
    vd: VectorDrawing,
    *,
    min_length_px: float = 40.0,
) -> list[dict[str, float]]:
    """Long axis-aligned segments as centerline / alignment references."""
    refs: list[dict[str, float]] = []
    for s in vd.residual_segments:
        dx, dy = abs(s.x2 - s.x1), abs(s.y2 - s.y1)
        lg = math.hypot(dx, dy)
        if lg < min_length_px:
            continue
        axis = max(dx, dy) / (dx + dy + 1e-9)
        if axis < 0.94:
            continue
        refs.append(
            {
                "x1": s.x1,
                "y1": s.y1,
                "x2": s.x2,
                "y2": s.y2,
                "length_px": round(lg, 3),
            }
        )
    return refs[:200]


def associate_ocr_semantic_links(
    ocr_boxes: Sequence[TextBox] | None,
    segments: Sequence[Segment],
    *,
    max_dist_px: float = 95.0,
) -> list[dict[str, Any]]:
    """Semantic OCR association: link each text box to nearest geometry stroke."""
    if not ocr_boxes or not segments:
        return []
    out: list[dict[str, Any]] = []
    for tb in ocr_boxes:
        cx, cy = tb.center()
        best_d = max_dist_px + 1.0
        best: Segment | None = None
        for s in segments:
            ax, ay, bx, by = s.x1, s.y1, s.x2, s.y2
            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy
            if l2 < 1e-12:
                d = math.hypot(cx - ax, cy - ay)
            else:
                t = max(0.0, min(1.0, ((cx - ax) * vx + (cy - ay) * vy) / l2))
                qx, qy = ax + t * vx, ay + t * vy
                d = math.hypot(cx - qx, cy - qy)
            if d < best_d:
                best_d = d
                best = s
        if best is not None and best_d <= max_dist_px:
            out.append(
                {
                    "ocr_text": ((tb.text or "").strip())[:120],
                    "nearest_segment_dist_px": round(float(best_d), 3),
                    "segment": {
                        "x1": best.x1,
                        "y1": best.y1,
                        "x2": best.x2,
                        "y2": best.y2,
                    },
                    "confidence": round(float(tb.confidence), 4),
                }
            )
    return out


def expand_multilayer_semantic_bucketing(
    layered_segments: dict[str, list[Segment]] | None,
) -> dict[str, Any]:
    """
    Multi-layer semantic processing summary (geometry / dimensions / OCR strokes / symbols / annotations).

    When pixel semantics are unavailable, SYMBOLS/ANNOTATIONS may be empty — bucket counts still publish.
    """
    if not layered_segments:
        return {
            "geometry": [],
            "dimensions": [],
            "ocr_text_strokes": [],
            "symbols": [],
            "annotations": [],
            "counts": {},
        }
    geom = list(layered_segments.get("GEOMETRY", []))
    dim = list(layered_segments.get("DIMENSION", []))
    border = list(layered_segments.get("BORDER", []))
    text = list(layered_segments.get("TEXT", []))
    other = list(layered_segments.get("OTHER", []))
    symbols: list[Segment] = []
    annotations: list[Segment] = []
    remainder: list[Segment] = []
    for s in other:
        lg = math.hypot(s.x2 - s.x1, s.y2 - s.y1)
        dx, dy = abs(s.x2 - s.x1), abs(s.y2 - s.y1)
        axis = max(dx, dy) / (dx + dy + 1e-9)
        if lg < 14.0 and axis < 0.86:
            symbols.append(s)
        elif lg < 30.0 and 0.88 <= axis <= 0.98:
            annotations.append(s)
        else:
            remainder.append(s)
    return {
        "geometry": geom + remainder,
        "dimensions": dim,
        "ocr_text_strokes": text,
        "symbols": symbols,
        "annotations": border + annotations,
        "counts": {
            "geometry": len(geom) + len(remainder),
            "dimensions": len(dim),
            "ocr_text_strokes": len(text),
            "symbols": len(symbols),
            "annotations": len(border) + len(annotations),
        },
    }
