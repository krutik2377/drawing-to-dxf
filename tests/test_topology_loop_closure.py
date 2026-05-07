"""Loop closure bridging."""

from __future__ import annotations

from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.topology_repair import bridge_almost_closed_loops


def test_bridge_almost_closed_loops_adds_segment() -> None:
    # Open square: three edges
    segs = [
        Segment(0, 0, 100, 0),
        Segment(100, 0, 100, 100),
        Segment(100, 100, 0, 100),
    ]
    out, n = bridge_almost_closed_loops(list(segs), max_gap_px=150.0)
    assert n >= 1
    assert len(out) == len(segs) + n
