"""Tests for exploded-segment graph repair (bridges, T-snaps)."""

from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.topology_repair import repair_exploded_segments


def test_directed_gap_bridge_connects_collinear_gapped_chain() -> None:
    segs = [
        Segment(0.0, 0.0, 9.0, 0.0),
        Segment(11.0, 0.0, 20.0, 0.0),
    ]
    out = repair_exploded_segments(
        list(segs),
        merge_distance=2.0,
        max_bridge_gap_px=4.0,
        junction_snap_px=0.0,
        bridge_direction_dot_min=0.35,
        max_passes=6,
    )
    xs = sorted({p[0] for s in out for p in ((s.x1, s.y1), (s.x2, s.y2))})
    assert min(xs) <= 0.1
    assert max(xs) >= 19.9


def test_junction_snap_moves_stub_onto_crossbar() -> None:
    segs = [
        Segment(0.0, 0.0, 20.0, 0.0),
        Segment(10.0, -8.0, 10.0, -0.8),
    ]
    out = repair_exploded_segments(
        list(segs),
        merge_distance=1.5,
        max_bridge_gap_px=0.0,
        junction_snap_px=2.5,
        max_passes=4,
    )
    ends = {(s.x1, s.y1) for s in out} | {(s.x2, s.y2) for s in out}
    assert any(abs(x - 10.0) < 0.55 and abs(y) < 0.35 for x, y in ends), (
        f"expected a vertex near the T on the crossbar, got {ends!r}"
    )


def test_intersection_ray_extend_reaches_crossing() -> None:
    segs = [
        Segment(0.0, 100.0, 20.0, 100.0),
        Segment(10.0, 92.0, 10.0, 200.0),
    ]
    out = repair_exploded_segments(
        list(segs),
        merge_distance=1.5,
        max_bridge_gap_px=0.0,
        junction_snap_px=0.0,
        intersection_extend_px=12.0,
        max_passes=6,
    )
    ends = {(round(s.x1, 2), round(s.y1, 2)) for s in out} | {(round(s.x2, 2), round(s.y2, 2)) for s in out}
    assert any(abs(x - 10.0) < 0.4 and abs(y - 100.0) < 0.4 for x, y in ends), f"got {ends!r}"
