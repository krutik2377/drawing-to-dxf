"""Rule-based and hybrid pixel semantics for layer policy (no ONNX required).

Builds a per-pixel class label image aligned with ``semantic_segment.DEFAULT_CLASS_NAMES``.
Used to classify exploded segments onto GEOMETRY / DIMENSION / BORDER layers and to
augment ONNX segmentation when ``rule_based_semantics`` is enabled.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.semantic_segment import DEFAULT_CLASS_NAMES, run_semantic_seg_labels

_NAME_TO_IDX = {n: i for i, n in enumerate(DEFAULT_CLASS_NAMES)}


def build_rule_based_pixel_labels(
    gray: np.ndarray,
    ocr_boxes: Sequence[TextBox] | None,
    *,
    border_band_px: int = 18,
    ocr_dilate_px: int = 12,
    ruling_kernel_max: int = 71,
    ruling_iter: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Heuristic labels: ink → geometry by default; OCR regions → text; frame band → title_block
    (reused as BORDER export); long H/V morph hits → dimension.

    Returns ``(labels HxW int32, meta)``.
    """
    if gray.ndim != 2:
        raise ValueError("grayscale expected")
    h, w = gray.shape[:2]
    labels = np.full((h, w), _NAME_TO_IDX["background"], dtype=np.int32)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    bin_inv = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    ink = bin_inv > 0
    labels[ink] = _NAME_TO_IDX["geometry"]

    # Border / frame ink (long strokes hugging the sheet edge)
    bb = max(3, min(border_band_px, h // 8, w // 8))
    edge = np.zeros_like(ink, dtype=bool)
    edge[:bb, :] = True
    edge[-bb:, :] = True
    edge[:, :bb] = True
    edge[:, -bb:] = True
    border_ink = ink & edge
    labels[border_ink] = _NAME_TO_IDX["title_block"]  # manifest as BORDER layer in export

    # OCR text regions
    text_idx = _NAME_TO_IDX["text"]
    k = max(1, int(ocr_dilate_px))
    mask_txt = np.zeros((h, w), dtype=np.uint8)
    if ocr_boxes:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        for tb in ocr_boxes:
            x0 = max(0, int(tb.x0))
            y0 = max(0, int(tb.y0))
            x1 = min(w, int(tb.x1))
            y1 = min(h, int(tb.y1))
            if x1 > x0 and y1 > y0:
                mask_txt[y0:y1, x0:x1] = 255
        mask_txt = cv2.dilate(mask_txt, ker, iterations=1)
        labels[mask_txt > 0] = text_idx

    # Dimension-like ruling: long horizontal and vertical morph openings on ink
    dim_idx = _NAME_TO_IDX["dimension"]
    kmax = max(15, min(ruling_kernel_max, max(h, w) // 12))
    khor = cv2.getStructuringElement(cv2.MORPH_RECT, (kmax, 1))
    kver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kmax))
    hor = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, khor, iterations=int(ruling_iter))
    ver = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kver, iterations=int(ruling_iter))
    ruling = (hor > 0) | (ver > 0)
    # Do not label text interiors as dimension strokes
    ruling_and_ink = ruling & ink
    if mask_txt.any():
        ruling_and_ink &= mask_txt == 0
    labels[ruling_and_ink] = dim_idx

    meta = {
        "rule_semantics": True,
        "border_band_px": int(bb),
        "ocr_dilate_px": int(k),
        "ruling_kernel_max": int(kmax),
        "pixel_frac_geometry": float(np.mean(labels == _NAME_TO_IDX["geometry"])),
        "pixel_frac_dimension": float(np.mean(labels == dim_idx)),
        "pixel_frac_text": float(np.mean(labels == text_idx)),
        "pixel_frac_border": float(np.mean(labels == _NAME_TO_IDX["title_block"])),
    }
    return labels, meta


def merge_onnx_with_rule_bias(
    onnx_labels: np.ndarray,
    rule_labels: np.ndarray,
    *,
    prefer_onnx_geometry: bool = True,
) -> np.ndarray:
    """
    When both exist, keep ONNX text/dimension where confident; fill gaps with rules.

    ``onnx_labels`` and ``rule_labels`` must match shape.
    """
    if onnx_labels.shape != rule_labels.shape:
        raise ValueError("label shape mismatch")
    out = np.array(onnx_labels, copy=True)
    bg = _NAME_TO_IDX["background"]
    # Where ONNX left background but rules see ink, adopt rule class
    rule_ink = rule_labels != bg
    onnx_bg = out == bg
    if prefer_onnx_geometry:
        adopt = rule_ink & onnx_bg
        out[adopt] = rule_labels[adopt]
    else:
        out = np.where(rule_ink & (out == bg), rule_labels, out)
    # Strengthen dimension where both agree on ink non-text
    dim_r = rule_labels == _NAME_TO_IDX["dimension"]
    dim_o = out == _NAME_TO_IDX["dimension"]
    out[dim_r & ~dim_o] = _NAME_TO_IDX["dimension"]
    return out


def onnx_pixel_labels(
    gray: np.ndarray,
    onnx_path: Any,
    *,
    config_path: Any = None,
    providers: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ONNX segmentation and return full-resolution int32 labels + meta."""
    from pathlib import Path

    from drawing_to_dxf.semantic_segment import build_model_config

    mc = build_model_config(Path(onnx_path), config_path=Path(config_path) if config_path else None, providers=providers)
    labels, meta = run_semantic_seg_labels(gray, mc)
    meta["class_names"] = list(mc.class_names)
    return labels, meta


def combined_pixel_labels(
    gray: np.ndarray,
    ocr_boxes: Sequence[TextBox] | None,
    *,
    onnx_path: Any | None = None,
    config_path: Any | None = None,
    rule_based_semantics: bool = False,
    rule_only: bool = False,
    providers: Sequence[str] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Prefer ONNX when ``onnx_path`` is set; optionally merge with rule-based labels.

    Use ``rule_only=True`` (or ``rule_based_semantics=True`` with no ONNX) to build
    rule-based labels for layered export without a model.
    """
    meta: dict[str, Any] = {}
    if onnx_path is not None:
        ol, ometa = onnx_pixel_labels(gray, onnx_path, config_path=config_path, providers=providers)
        meta["onnx"] = ometa
        if rule_based_semantics:
            rule_lbl, rmeta = build_rule_based_pixel_labels(gray, ocr_boxes)
            meta["rule_based"] = rmeta
            merged = merge_onnx_with_rule_bias(ol, rule_lbl)
            meta["hybrid"] = True
            return merged, meta
        return ol, meta

    if rule_based_semantics or rule_only:
        rule_lbl, rmeta = build_rule_based_pixel_labels(gray, ocr_boxes)
        meta["rule_based"] = rmeta
        return rule_lbl, meta

    return None, meta
