"""Raster debug dumps for per-panel vectorization (split + trace review)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from drawing_to_dxf.geometry_model import VectorDrawing, _angle_lerp_deg

def write_panel_trace_debug(
    out_dir: Path,
    panel_index: int,
    gray: np.ndarray,
    vector_drawing: VectorDrawing,
    stages: dict[str, np.ndarray],
) -> Path:
    """
    Write ``original.png``, ``masked.png``, ``binary.png``, ``skeleton.png``, ``final_overlay.png``.

    ``stages`` keys: ``masked_gray`` (uint8 HxW), ``binary_ink`` (uint8 HxW foreground 255),
    ``skeleton_bool`` (uint8 skeleton 255).
    """
    dst = out_dir / f"panel_{panel_index:02d}"
    dst.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dst / "original.png"), gray)
    mg = stages.get("masked_gray")
    if mg is not None:
        cv2.imwrite(str(dst / "masked.png"), mg)
    bi = stages.get("binary_ink")
    if bi is not None:
        cv2.imwrite(str(dst / "binary.png"), bi)
    sk = stages.get("skeleton_bool")
    if sk is not None:
        cv2.imwrite(str(dst / "skeleton.png"), sk)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    col = (42, 183, 231)  # BGR — bright cyan on gray
    tip = (32, 92, 252)
    for poly in vector_drawing.polylines:
        pts = np.array([[p[0], p[1]] for p in poly.points], dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], poly.closed, color=col, thickness=1, lineType=cv2.LINE_AA)
    for c in vector_drawing.circles:
        cv2.circle(vis, (int(round(c.cx)), int(round(c.cy))), int(round(c.r)), tip, 1, cv2.LINE_AA)
    for a in vector_drawing.arcs:
        n_arc = max(24, min(96, int(0.42 * float(a.r))))
        apx: list[tuple[int, int]] = []
        for k in range(n_arc + 1):
            ang = _angle_lerp_deg(a.start_angle_deg, a.end_angle_deg, k / float(n_arc), a.ccw)
            th = math.radians(ang)
            apx.append(
                (int(round(a.cx + a.r * math.cos(th))), int(round(a.cy + a.r * math.sin(th))))
            )
        for u in range(len(apx) - 1):
            cv2.line(vis, apx[u], apx[u + 1], col, 1, cv2.LINE_AA)
    res_col = (188, 89, 255)  # BGR magenta — residuals / LSD supplement
    for s in vector_drawing.residual_segments:
        p0 = (int(round(s.x1)), int(round(s.y1)))
        p1 = (int(round(s.x2)), int(round(s.y2)))
        cv2.line(vis, p0, p1, res_col, 1, cv2.LINE_AA)
    cv2.imwrite(str(dst / "final_overlay.png"), vis)
    return dst
