"""Visual preview helpers: panel boxes + optional OCR part-id inventory (no hardcoded counts)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drawing_to_dxf.ocr_extract import (
    default_part_pattern,
    extract_text_boxes,
    filter_part_candidates,
)
from drawing_to_dxf.panel_split import split_panels


@dataclass
class PanelsPreviewData:
    boxes: list[tuple[int, int, int, int]]
    panel_count: int
    annotated_bgr: np.ndarray


def annotate_panel_boxes(
    gray: np.ndarray,
    *,
    boxes: list[tuple[int, int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), thickness)
        cv2.putText(
            vis,
            str(i),
            (x + 4, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return vis


def preview_panels(
    gray: np.ndarray,
    *,
    min_area: int = 15_000,
    min_gap_px: int = 48,
    min_short_side_px: int = 80,
    max_aspect_ratio: float = 10.0,
    morph_close: int = 11,
) -> PanelsPreviewData:
    """Run gutter/blob splitter on processed grayscale and draw bounding boxes."""
    boxes = split_panels(
        gray,
        min_area=min_area,
        min_gap_px=min_gap_px,
        min_short_side_px=min_short_side_px,
        max_aspect_ratio=max_aspect_ratio,
        morph_close=morph_close,
    )
    vis = annotate_panel_boxes(gray, boxes=boxes)
    return PanelsPreviewData(boxes=list(boxes), panel_count=len(boxes), annotated_bgr=vis)


def ocr_distinct_part_ids(
    gray: np.ndarray,
    *,
    gpu: bool = False,
    min_confidence: float = 0.15,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    OCR + default part-number regex → distinct IDs (best confidence kept per ID).

    This count can differ from ``panel_count`` (IDs can repeat, sit in gutters, etc.).
    """
    pat = default_part_pattern()
    tbs = extract_text_boxes(gray, gpu=gpu)
    labeled = filter_part_candidates(tbs, pat, min_confidence=min_confidence)
    best_conf: dict[str, float] = {}
    for tb, pid in labeled:
        c = tb.confidence
        if pid not in best_conf or c > best_conf[pid]:
            best_conf[pid] = float(c)
    pairs = sorted(best_conf.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pid for pid, _ in pairs], pairs
