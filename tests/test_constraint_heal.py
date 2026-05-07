"""Tests for orthogonal rectangle constraint healing."""

from drawing_to_dxf.constraint_heal import orthogonal_close_quads
from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing


def test_near_square_snaps_to_axis_aligned_rect() -> None:
    pts = [(0.0, 1.1), (99.2, 0.0), (100.0, 80.4), (1.8, 81.0)]
    vd = VectorDrawing(polylines=[PolylineDef(points=pts, closed=True)])
    out = orthogonal_close_quads(vd, corner_tol_deg=20.0, min_edge_px=4.0)
    assert len(out.polylines) == 1
    r = out.polylines[0].points
    assert len(r) == 4
    xs = sorted({p[0] for p in r})
    ys = sorted({p[1] for p in r})
    assert xs[0] <= 1.9 and xs[-1] >= 99.0
    assert ys[0] <= 1.2 and ys[-1] >= 80.0
