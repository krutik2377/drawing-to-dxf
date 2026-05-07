"""Hough circle gating: hole-like rings vs line-art false positives."""

import cv2
import numpy as np

from drawing_to_dxf.geometry_model import CircleDef
from drawing_to_dxf.skeleton_vectorize import _filter_circles_by_ink_ring


def _disk_ring_ink(h: int, w: int, cx: int, cy: int, r: int, th: int) -> np.ndarray:
    """Binary 0/255 with ink on an annulus (drill hole)."""
    ink = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ink, (cx, cy), r, 255, th, lineType=cv2.LINE_AA)
    return ink


def test_filter_keeps_hole_like_circle() -> None:
    ink = _disk_ring_ink(200, 200, 100, 100, 40, 3)
    circles = [CircleDef(cx=100.0, cy=100.0, r=40.0)]
    kept = _filter_circles_by_ink_ring(circles, ink)
    assert len(kept) == 1


def test_filter_keeps_small_filled_drill_dot() -> None:
    ink = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(ink, (60, 60), 8, 255, thickness=-1, lineType=cv2.LINE_AA)
    c = CircleDef(cx=60.0, cy=60.0, r=8.0)
    kept = _filter_circles_by_ink_ring([c], ink)
    assert len(kept) == 1


def test_filter_drops_corner_blob_circle() -> None:
    """Solid ink blob (no hollow interior) should fail interior-ink test."""
    ink = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(ink, (100, 100), 50, 255, thickness=-1, lineType=cv2.LINE_AA)
    circles = [CircleDef(cx=100.0, cy=100.0, r=48.0)]
    kept = _filter_circles_by_ink_ring(circles, ink)
    assert len(kept) == 0
