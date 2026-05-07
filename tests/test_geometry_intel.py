import numpy as np

from drawing_to_dxf.geometry_intel import (
    count_corner_rectangles,
    geometry_quality_report,
    lsd_extract_segments,
)
from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing


def test_lsd_extract_finds_long_edge() -> None:
    img = np.full((120, 160), 235, dtype=np.uint8)
    img[58:62, 20:142] = 18
    segs = lsd_extract_segments(img, min_length_px=30.0)
    assert len(segs) >= 1
    longest = max(segs, key=lambda s: (s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2)
    assert abs(longest.y2 - longest.y1) < 8.0


def test_count_corner_rectangles_square() -> None:
    sq = PolylineDef(points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], closed=True)
    dwg = VectorDrawing(polylines=[sq])
    assert count_corner_rectangles(dwg) == 1


def test_geometry_quality_report_nonempty() -> None:
    dwg = VectorDrawing(polylines=[PolylineDef(points=[(0, 0), (5, 0)], closed=False)])
    r = geometry_quality_report(dwg)
    assert r.polyline_count == 1
    assert 0.0 <= r.confidence_score <= 1.0


def test_dedupe_merged_residuals() -> None:
    from drawing_to_dxf.vectorize import extract_skeleton_vector_bundle

    gray = np.full((200, 300), 255, dtype=np.uint8)
    gray[49:53, 40:262] = 0
    gray[40:182, 149:154] = 0
    vd, _segs = extract_skeleton_vector_bundle(
        gray,
        annotation_boxes=None,
        mask_annotation_regions=False,
        min_line_length=8,
        lsd_supplement=True,
        dedupe_residual_segments=True,
    )
    assert isinstance(vd.residual_segments, list)
