"""Tests for panel strip/sliver filtering (gutter strips from grid splitter)."""

from drawing_to_dxf.panel_split import filter_strip_like_panels


def test_filter_strip_like_removes_vertical_gutter_column() -> None:
    boxes = [
        (0, 0, 900, 750),
        (910, 0, 54, 750),
        (1000, 0, 400, 700),
    ]
    out = filter_strip_like_panels(boxes, min_short_side_px=80, max_aspect_ratio=10.0)
    assert len(out) == 2
    xs = {x for x, y, w, h in out}
    assert 910 not in xs


def test_filter_strip_like_keeps_square_detail() -> None:
    boxes = [(0, 0, 133, 133)]
    out = filter_strip_like_panels(boxes, min_short_side_px=80, max_aspect_ratio=10.0)
    assert len(out) == 1
