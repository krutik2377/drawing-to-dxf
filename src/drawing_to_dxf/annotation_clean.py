"""Mask annotations (text, coarse hatch, thin rulings) before skeleton vectorization."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from drawing_to_dxf.ocr_extract import TextBox


def _inflate_box(
    x0: float, y0: float, x1: float, y1: float,
    *,
    pad: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    xi0 = int(max(0, np.floor(x0 - pad)))
    yi0 = int(max(0, np.floor(y0 - pad)))
    xi1 = int(min(w - 1, np.ceil(x1 + pad)))
    yi1 = int(min(h - 1, np.ceil(y1 + pad)))
    return xi0, yi0, max(xi1, xi0 + 1), max(yi1, yi0 + 1)


def apply_text_masks(
    gray: np.ndarray,
    boxes: Sequence[TextBox],
    *,
    pad_px: float = 4.0,
) -> np.ndarray:
    """Set OCR bounding regions to white paper background (suppress text strokes)."""
    out = gray.copy()
    h, w = out.shape[:2]
    v255 = np.uint8(255)
    for tb in boxes:
        x0, y0, x1, y1 = _inflate_box(tb.x0, tb.y0, tb.x1, tb.y1, pad=pad_px, w=w, h=h)
        out[y0:y1, x0:x1] = v255
    return out


def apply_text_masks_interior_only(
    gray: np.ndarray,
    boxes: Sequence[TextBox],
    *,
    pad_px: float = 4.0,
    erode_iters: int = 1,
) -> np.ndarray:
    """
    Wipe OCR text **interiors** inside each inflated box instead of rectangular blocks.

    The crop is locally binarized; only connected ink pixels overlapping the OCR box centre
    (after light erosion) are cleared. Nearby linework touching the annotation frame is left intact.
    """
    out = gray.copy().astype(np.uint8)
    h, w = out.shape[:2]
    v255 = np.uint8(255)
    for tb in boxes:
        x0, y0, x1, y1 = _inflate_box(tb.x0, tb.y0, tb.x1, tb.y1, pad=pad_px, w=w, h=h)
        crop = out[y0:y1, x0:x1].copy()
        ch, cw = crop.shape[:2]
        if cw < 3 or ch < 3:
            continue
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8), connectivity=8)
        mx = max(1, cw // 2)
        my = max(1, ch // 2)
        lid = int(lab[my, mx])
        if lid <= 0:
            continue
        if n_cc < 2:
            continue
        mask = (lab == lid).astype(np.uint8)
        if erode_iters > 0:
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, ek, iterations=int(erode_iters))
        iy, ix = np.nonzero(mask)
        if iy.size == 0:
            mask = (lab == lid).astype(np.uint8)
            iy, ix = np.nonzero(mask)
        if iy.size > 10 * cw * ch:
            continue
        out[y0 + iy, x0 + ix] = v255

    return out


def suppress_hatches_and_dimensions(
    binary_ink_white: np.ndarray,
    *,
    max_horizontal_frac: float = 0.05,
    max_horizontal_kernel_px: int = 101,
    vertical_open_k: tuple[int, int] = (1, 5),
    diag_close_k: int = 3,
) -> np.ndarray:
    """
    Operates on BINARY INV image used by OCR-style workflows: strokes = 255, background = 0.

    - Opening with a horizontal SE removes predominantly horizontal thin rulings often used
      for dimensions and coarse hatch scaffolding.
    - Light diagonal-ish closing helps reconnect fragmented real geometry after opening.
    """
    if binary_ink_white.ndim != 2:
        raise ValueError("single-channel binary expected")

    w = int(binary_ink_white.shape[1])
    hh = max(5, min(int(round(w * max_horizontal_frac)) | 1, max_horizontal_kernel_px))
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (hh, vertical_open_k[1]))
    opened = cv2.morphologyEx(binary_ink_white, cv2.MORPH_OPEN, horiz, iterations=1)

    kd = max(3, diag_close_k | 1)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kd, kd))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cross, iterations=1)
    return cleaned


def inks_mask_from_gray(
    gray: np.ndarray,
    *,
    thresh_block: int = 31,
    thresh_c: int = 7,
    suppress_ruling_lines: bool = True,
) -> np.ndarray:
    """
    Robust ink mask (foreground = 255) for technical drawings — adaptive threshold on inverted sense.

    ``suppress_ruling_lines=False`` skips horizontal dimension/hatch suppression (safer for thin
    geometry when combined with skeleton tracing).
    """
    if gray.ndim != 2:
        raise ValueError("grayscale expected")
    gb = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        gb,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(7, thresh_block | 1),
        thresh_c,
    )
    if suppress_ruling_lines:
        bw = suppress_hatches_and_dimensions(bw)
    return bw


def geometry_split_ink_mask(gray: np.ndarray, *, thresh_block: int = 31, thresh_c: int = 7) -> np.ndarray:
    """Foreground ink for panel splitting: adaptive threshold only (no horizontal opening)."""
    return inks_mask_from_gray(gray, thresh_block=thresh_block, thresh_c=thresh_c, suppress_ruling_lines=False)

