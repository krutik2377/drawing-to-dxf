"""Tests for post-Hough collinear merge and Douglas–Peucker polyline simplification."""

from drawing_to_dxf.vectorize import (
    Segment,
    _collapse_collinear_segments,
    _ramer_douglas_peucker,
    _simplify_degree2_chains_rdp,
)


def test_collinear_chain_collapses_to_single_segment() -> None:
    segs = [
        Segment(0, 0, 10, 0),
        Segment(10, 0, 20, 0),
        Segment(20, 0, 30, 0),
    ]
    out = _collapse_collinear_segments(segs, tol=1.0, angle_deg=8.0)
    assert len(out) == 1
    o = out[0]
    assert abs(o.x1 - 0) < 0.01 and abs(o.y1 - 0) < 0.01
    assert abs(o.x2 - 30) < 0.5 and abs(o.y2 - 0) < 0.01


def test_rdp_simplifies_almost_straight_polyline() -> None:
    pts = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0), (10.0, 0.0)]
    out = _ramer_douglas_peucker(pts, epsilon=0.75)
    assert len(out) < len(pts)
    assert out[0] == pts[0]
    assert out[-1] == pts[-1]


def test_rdp_short_list_unchanged() -> None:
    pts = [(0.0, 0.0), (5.0, 1.0)]
    assert _ramer_douglas_peucker(pts, epsilon=2.0) == pts


def test_simplify_degree2_preserves_single_merged_line() -> None:
    long_seg = [Segment(0, 0, 40, 0)]
    out = _simplify_degree2_chains_rdp(long_seg, tol=1.0, epsilon=1.5)
    assert len(out) >= 1
    assert any(abs(s.y1) < 0.1 and abs(s.y2) < 0.1 for s in out)
