"""OCR labels (part numbers) from drawing raster."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class TextBox:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))


def _expand_box(tb: TextBox, pad: float) -> tuple[float, float, float, float]:
    return (tb.x0 - pad, tb.y0 - pad, tb.x1 + pad, tb.y1 + pad)


def boxes_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def point_in_box(x: float, y: float, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def extract_text_boxes(
    gray: np.ndarray,
    *,
    languages: list[str] | None = None,
    gpu: bool = False,
    quantize: bool = True,
) -> list[TextBox]:
    """
    Run EasyOCR. First invocation downloads models.

    Edge cases:
    - Empty/blank image → usually empty list (not an error).
    - If easyocr not installed, raises ImportError with message.
    """
    try:
        import easyocr
    except ImportError as e:
        raise ImportError(
            "easyocr is required for OCR. Install dependencies: pip install -r requirements.txt"
        ) from e

    if languages is None:
        languages = ["en", "de"]

    if gpu:
        reader = easyocr.Reader(languages, gpu=True, quantize=quantize, verbose=False)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*pin_memory.*accelerator.*",
                category=UserWarning,
            )
            reader = easyocr.Reader(languages, gpu=False, quantize=quantize, verbose=False)
    # easyocr expects BGR or RGB; grayscale 3-channel
    if gray.ndim == 2:
        img = np.stack([gray, gray, gray], axis=-1)
    else:
        img = gray

    results = reader.readtext(img, detail=1, paragraph=False)
    out: list[TextBox] = []
    for (_bbox, text, conf) in results:
        if text is None:
            continue
        t = str(text).strip()
        if not t:
            continue
        xs = [float(p[0]) for p in _bbox]
        ys = [float(p[1]) for p in _bbox]
        out.append(
            TextBox(
                text=t,
                confidence=float(conf),
                x0=min(xs),
                y0=min(ys),
                x1=max(xs),
                y1=max(ys),
            )
        )
    return out


def filter_part_candidates(
    boxes: list[TextBox],
    pattern: re.Pattern[str],
    min_confidence: float = 0.15,
) -> list[tuple[TextBox, str]]:
    """Return (box, matched_part_id) for each OCR hit matching regex (e.g. 3–5 digit part numbers)."""
    hits: list[tuple[TextBox, str]] = []
    for tb in boxes:
        if tb.confidence < min_confidence:
            continue
        m = pattern.search(tb.text.replace(" ", ""))
        if not m:
            continue
        part_id = m.group(1) if m.lastindex else m.group(0)
        hits.append((tb, part_id))
    return hits


def default_part_pattern() -> re.Pattern[str]:
    # Typical stamped drawing callouts: 701, 1024, etc.
    return re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
