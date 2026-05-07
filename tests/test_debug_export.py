"""Tests for semantic preview and debug bundle outputs."""

from __future__ import annotations

import numpy as np
import pytest

from drawing_to_dxf.debug_export import render_semantic_layer_preview, write_vectorization_debug_bundle
from drawing_to_dxf.segment_types import Segment


def test_render_semantic_preview_requires_2d() -> None:
    with pytest.raises(ValueError, match="gray must be HxW"):
        render_semantic_layer_preview(np.zeros((5, 5, 3), dtype=np.uint8), {})


def test_write_vectorization_debug_bundle_semantic_png(tmp_path) -> None:
    gray = np.full((40, 40), 250, dtype=np.uint8)
    layered = {
        "GEOMETRY": [Segment(2.0, 20.0, 38.0, 20.0)],
        "DIMENSION": [],
        "BORDER": [],
        "TEXT": [],
        "OTHER": [],
    }
    r = write_vectorization_debug_bundle(
        tmp_path,
        stem="t",
        stages=None,
        segments=[Segment(2, 20, 38, 20)],
        layered_segments=layered,
        gray_for_semantic_preview=gray,
    )
    expected = tmp_path / "t_semantic_preview.png"
    assert r.get("semantic_preview_png") == str(expected.resolve())
    assert expected.exists()
    assert expected.stat().st_size > 80
