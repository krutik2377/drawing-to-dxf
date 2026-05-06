"""Split multi-part shop sheets into rectangular panels (white gutters between figures)."""

from __future__ import annotations

import cv2
import numpy as np


def _smooth_1d(a: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    p = np.pad(a.astype(np.float64), (pad, pad), mode="edge")
    c = np.cumsum(np.insert(p, 0, 0))
    return (c[k:] - c[:-k]) / float(k)


def _gap_split_indices(proj: np.ndarray, min_run: int, rel_thresh: float) -> list[int]:
    """Return split indices in the middle of long low-ink runs (gutters)."""
    m = float(proj.max()) + 1e-6
    t = max(m * rel_thresh, 1.0)
    low = proj < t
    splits: list[int] = []
    i = 0
    n = len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        if j - i >= min_run:
            splits.append((i + j) // 2)
        i = j
    return splits


def _boxes_from_xy_splits(
    h: int, w: int, xs: list[int], ys: list[int]
) -> list[tuple[int, int, int, int]]:
    xs = [0] + [int(x) for x in xs if 0 < x < w] + [w]
    ys = [0] + [int(y) for y in ys if 0 < y < h] + [h]
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    boxes: list[tuple[int, int, int, int]] = []
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            x0, x1 = xs[xi], xs[xi + 1]
            y0, y1 = ys[yi], ys[yi + 1]
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def filter_strip_like_panels(
    boxes: list[tuple[int, int, int, int]],
    *,
    min_short_side_px: int,
    max_aspect_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """
    Drop gutter / margin rectangles that grid splitting often yields: very narrow
    or very tall ribbons (high aspect ratio), or tiles shorter than ``min_short_side_px``.

    These usually look like meaningless vertical captions or scan borders in CAD viewers,
    not real detail blocks (see sampled DXFs from dense assembly sheets).
    """
    if max_aspect_ratio < 1.0:
        raise ValueError("max_aspect_ratio must be >= 1")
    out: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes:
        if w < 1 or h < 1:
            continue
        short = float(min(w, h))
        long = float(max(w, h))
        if short < float(min_short_side_px):
            continue
        if long > max_aspect_ratio * short + 1e-6:
            continue
        out.append((x, y, w, h))
    return out


def _filter_boxes_by_ink(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    min_area: int,
    min_ink_frac: float,
) -> list[tuple[int, int, int, int]]:
    ink = (gray < 235).astype(np.float32)
    h, w = gray.shape
    out: list[tuple[int, int, int, int]] = []
    for (x, y, bw, bh) in boxes:
        if bw <= 1 or bh <= 1:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        patch = ink[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if patch.sum() < min_ink_frac * patch.size:
            continue
        area = (x1 - x0) * (y1 - y0)
        if area < min_area:
            continue
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def split_panels(
    gray: np.ndarray,
    *,
    min_area: int = 15_000,
    min_gap_px: int = 48,
    rel_thresh: float = 0.02,
    morph_close: int = 11,
    min_short_side_px: int = 80,
    max_aspect_ratio: float = 10.0,
) -> list[tuple[int, int, int, int]]:
    """
    Return axis-aligned panel bounding boxes (x, y, w, h) in pixel coords.

    Strategy:
    1. Try contour blobs after light morph closing (connects strokes inside a figure).
    2. If one dominant blob covers most of the sheet, fall back to X/Y gap splitting.

    Gap grids often introduce long **strip** tiles (narrow vertical gutters with ink).
    Those are dropped using ``min_short_side_px`` and ``max_aspect_ratio``. If stripping
    loses all useful tiles, the pre-trim ink-filtered set is used; if still invalid,
    the full sheet is returned as one panel.
    """
    if gray.ndim != 2:
        raise ValueError("grayscale expected")

    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return [(0, 0, w, h)]

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, morph_close | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes_cv: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw0, bh0 = cv2.boundingRect(c)
        if bw0 * bh0 < min_area:
            continue
        boxes_cv.append((x, y, bw0, bh0))

    # Single huge blob spanning the sheet → gap split
    if boxes_cv:
        boxes_cv.sort(key=lambda b: b[2] * b[3], reverse=True)
        lx, ly, lw, lh = boxes_cv[0]
        if lw * lh > 0.65 * w * h and len(boxes_cv) <= 2:
            proj_x = _smooth_1d(bw.sum(axis=0), k=81)
            proj_y = _smooth_1d(bw.sum(axis=1), k=81)
            xs = _gap_split_indices(proj_x, min_run=min_gap_px, rel_thresh=rel_thresh)
            ys = _gap_split_indices(proj_y, min_run=min_gap_px, rel_thresh=rel_thresh)
            merged_boxes = _boxes_from_xy_splits(h, w, xs, ys)
            filtered = _filter_boxes_by_ink(gray, merged_boxes, min_area=min_area, min_ink_frac=0.003)
            trimmed = filter_strip_like_panels(
                filtered,
                min_short_side_px=min_short_side_px,
                max_aspect_ratio=max_aspect_ratio,
            )
            if len(trimmed) >= 2:
                trimmed.sort(key=lambda b: (b[1], b[0]))
                return trimmed
            if len(filtered) >= 2:
                filtered.sort(key=lambda b: (b[1], b[0]))
                return filtered
            return [(0, 0, w, h)]

    if not boxes_cv:
        return [(0, 0, w, h)]

    # Non-overlapping sort + NMS-like: keep largest boxes that aren't mostly inside another
    boxes_cv.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes_cv:
        x, y, bw0, bh0 = b
        xa, ya = x + bw0, y + bh0
        inside = False
        for (kx, ky, kw, kh) in kept:
            xb, yb = kx + kw, ky + kh
            if x >= kx and y >= ky and xa <= xb and ya <= yb:
                inside = True
                break
        if inside:
            continue
        kept.append(b)

    kept = _filter_boxes_by_ink(gray, kept, min_area=min_area, min_ink_frac=0.002)
    kept = filter_strip_like_panels(
        kept,
        min_short_side_px=min_short_side_px,
        max_aspect_ratio=max_aspect_ratio,
    )
    if not kept:
        return [(0, 0, w, h)]
    kept.sort(key=lambda b: (b[1] // max(h // 50, 1), b[0]))
    return kept
