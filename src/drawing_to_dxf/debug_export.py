"""Debug artifact export: per-stage PNGs, GeoJSON line sampler (Phase 0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import cv2
import numpy as np

from drawing_to_dxf.geometry_model import VectorDrawing, exploded_segments_for_sampling
from drawing_to_dxf.segment_types import Segment

# BGR preview colors aligned with :mod:`drawing_to_dxf.export_dxf` ACI choices
# (geometry dark, dimensions red, borders magenta, text/labels green).
_SEMANTIC_PREVIEW_ORDER: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("GEOMETRY", (40, 40, 40)),
    ("OTHER", (40, 40, 40)),
    ("BORDER", (255, 0, 255)),
    ("DIMENSION", (0, 0, 255)),
    ("TEXT", (0, 200, 0)),
)


def render_semantic_layer_preview(
    gray: np.ndarray,
    layered_segments: Mapping[str, Sequence[Segment]],
    *,
    part_label_centers: Sequence[tuple[str, float, float]] | None = None,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Raster debug view: classify-colored strokes on the processed grayscale sheet.

    Mirrors *DXF overlay* layering (``GEOMETRY`` / ``DIMENSION`` / ``BORDER`` / ``TEXT``),
    comparable to shop-detail sheets where part callouts are green-toned and geometry is dark.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be HxW uint8")
    vis = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for bucket, bgr in _SEMANTIC_PREVIEW_ORDER:
        segs = layered_segments.get(bucket)
        if not segs:
            continue
        for s in segs:
            p0 = (int(round(s.x1)), int(round(s.y1)))
            p1 = (int(round(s.x2)), int(round(s.y2)))
            cv2.line(vis, p0, p1, bgr, line_thickness, lineType=cv2.LINE_AA)
    if part_label_centers:
        for pid, cx, cy in part_label_centers:
            t = str(pid).strip()
            if not t:
                continue
            ix, iy = int(round(cx)), int(round(cy))
            cv2.circle(vis, (ix, iy), 4, (0, 220, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(
                vis,
                f"P{t}",
                (ix + 6, iy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1,
                lineType=cv2.LINE_AA,
            )
    return vis


def write_stage_pngs(out_dir: Path, stages: Mapping[str, np.ndarray]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, arr in stages.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:120]
        p = out_dir / f"{safe}.png"
        to_write: np.ndarray
        if arr.ndim == 2:
            to_write = arr.astype(np.uint8)
        elif arr.dtype == bool:
            to_write = (arr.astype(np.uint8) * 255)
        else:
            to_write = arr
        cv2.imwrite(str(p), to_write)
        written.append(p)
    return written


def segments_to_geojson(
    segments: Sequence[Segment],
    path: Path,
    *,
    image_height_px: float | None = None,
    properties: Mapping[str, Any] | None = None,
) -> None:
    feats: list[dict[str, Any]] = []
    base_props = dict(properties or {})
    for i, s in enumerate(segments):
        p = {
            **base_props,
            "id": i,
            "length_px": float(((s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2) ** 0.5),
        }
        if image_height_px is not None:
            # Optional GIS-style coords: flip Y
            y0 = float(image_height_px - s.y1)
            y1 = float(image_height_px - s.y2)
        else:
            y0, y1 = float(s.y1), float(s.y2)
        feats.append(
            {
                "type": "Feature",
                "properties": p,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(s.x1), y0], [float(s.x2), y1]],
                },
            }
        )
    fc = {"type": "FeatureCollection", "features": feats}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, indent=2), encoding="utf-8")


def write_vectorization_debug_bundle(
    out_dir: Path,
    *,
    stem: str,
    stages: MutableMapping[str, np.ndarray] | None,
    vector_drawing: VectorDrawing | None = None,
    segments: Sequence[Segment] | None = None,
    image_height_for_geojson: float | None = None,
    layered_segments: Mapping[str, Sequence[Segment]] | None = None,
    gray_for_semantic_preview: np.ndarray | None = None,
    part_label_centers: Sequence[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    """Write ``stages`` PNGs plus optional GeoJSON of segments and semantic tint PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"dir": str(out_dir.resolve()), "png": [], "geojson": None, "semantic_preview_png": None}
    if stages:
        sub = out_dir / f"{stem}_stages"
        report["png"] = [str(p.resolve()) for p in write_stage_pngs(sub, stages)]
    segs = list(segments) if segments is not None else (
        exploded_segments_for_sampling(vector_drawing) if vector_drawing is not None else []
    )
    if segs:
        gj = out_dir / f"{stem}_segments.geojson"
        segments_to_geojson(segs, gj, image_height_px=image_height_for_geojson, properties={"stem": stem})
        report["geojson"] = str(gj.resolve())
    if (
        layered_segments is not None
        and gray_for_semantic_preview is not None
        and any(layered_segments.get(b) for b, _c in _SEMANTIC_PREVIEW_ORDER)
    ):
        prev = render_semantic_layer_preview(
            gray_for_semantic_preview,
            layered_segments,
            part_label_centers=part_label_centers,
        )
        pp = out_dir / f"{stem}_semantic_preview.png"
        cv2.imwrite(str(pp), prev)
        report["semantic_preview_png"] = str(pp.resolve())
    return report
