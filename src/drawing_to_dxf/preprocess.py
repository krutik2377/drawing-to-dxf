"""Image load, safe paths, resize, optional denoise and mild deskew."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Grayscale image ready for vectorization, with geometry metadata."""

    gray: np.ndarray
    scale: float  # multiply original pixel coord by this to get gray coord (if downscaled)
    original_shape: tuple[int, int]  # H, W of source before optional resize


def load_image_bgr(path: str) -> np.ndarray | None:
    """
    Load BGR image with Unicode path support on Windows.
    cv2.imread fails on non-ASCII paths; use imdecode + fromfile.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def load_pdf_page_as_bgr(path: str, page_index: int = 0, dpi: float = 150) -> np.ndarray | None:
    """Rasterize one PDF page to BGR using PyMuPDF."""
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise RuntimeError("PDF input requires pymupdf. pip install pymupdf") from e

    doc = fitz.open(path)
    try:
        if page_index < 0 or page_index >= len(doc):
            return None
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        doc.close()

    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif arr.shape[2] == 3 and pix.n == 3:
        # Pixmap is RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _deskew_gray(gray: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """
    Rotate to cancel small skew using the minimum-area rectangle of the text/ink mask.
    No-op if angle unreliable or image too small.
    """
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return gray

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.01 * h * w:
        return gray

    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > max_angle_deg or abs(angle) < 0.1:
        return gray

    center = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess(
    bgr: np.ndarray,
    *,
    max_side: int | None = 4096,
    denoise: bool = True,
    deskew: bool = True,
) -> PreprocessResult:
    """
    Convert to grayscale, optionally denoise, deskew, and downscale large images.

    Edge cases:
    - Uniform/empty images still pass through; downstream may find no lines.
    - max_side None disables downscaling (may exhaust RAM on huge scans).
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image")

    orig_h, orig_w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if denoise:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)

    if deskew:
        gray = _deskew_gray(gray)

    scale = 1.0
    h, w = gray.shape[:2]
    if max_side is not None:
        m = max(h, w)
        if m > max_side:
            scale = max_side / float(m)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return PreprocessResult(
        gray=gray,
        scale=scale,
        original_shape=(orig_h, orig_w),
    )
