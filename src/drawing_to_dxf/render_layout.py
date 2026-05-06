"""Compose multiple panel crops into one shop-sheet-style PNG (vector-style layout, not GenAI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class PanelTile:
    """One cell in the composite sheet."""

    image_bgr: np.ndarray
    title: str
    meta: dict[str, Any]


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "C:\\Windows\\Fonts\\arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_composite_png(
    tiles: list[PanelTile],
    out_path: Path,
    *,
    cols: int | None = None,
    margin: int = 24,
    title_bar: int = 40,
    max_cell_width: int = 520,
    max_cell_height: int = 420,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Pack tiles in a rough grid (sorted input order: top-to-bottom, left-to-right per row).

    This reproduces the *layout class* of typical shop drawings (many parts on one sheet)
    by **compositing real crops**, not by generative image models (which are not dimensionally safe).
    """
    if not tiles:
        raise ValueError("no tiles")

    n = len(tiles)
    c = int(np.ceil(np.sqrt(n))) if cols is None else max(1, cols)
    r = int(np.ceil(n / c))

    scaled: list[tuple[np.ndarray, str]] = []
    for t in tiles:
        h, w = t.image_bgr.shape[:2]
        s = min(max_cell_width / max(w, 1), max_cell_height / max(h, 1), 1.0)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        if (nw, nh) != (w, h):
            img = cv2.resize(t.image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            img = t.image_bgr
        title = t.title.strip() or "(no title)"
        scaled.append((img, title))

    cell_w = max(margin + x[0].shape[1] for x in scaled)
    cell_h = max(title_bar + margin + x[0].shape[0] for x in scaled)

    sheet_w = margin + c * (cell_w + margin)
    sheet_h = margin + r * (cell_h + margin)

    canvas = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(canvas)
    font_title = _try_font(15)

    idx = 0
    for row in range(r):
        for col in range(c):
            if idx >= n:
                break
            img_bgr, title = scaled[idx]
            idx += 1
            x0 = margin + col * (cell_w + margin)
            y0 = margin + row * (cell_h + margin)
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tx = x0 + (cell_w - pil.width) // 2
            ty = y0 + title_bar + (cell_h - title_bar - pil.height) // 2
            canvas.paste(pil, (tx, ty))
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(200, 200, 200))
            draw.text((x0 + 6, y0 + 4), title[:120], fill=(0, 0, 0), font=font_title)
        if idx >= n:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), format="PNG")
