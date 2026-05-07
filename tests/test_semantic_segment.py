"""Semantic segmentation helpers (no ONNX required in CI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drawing_to_dxf.semantic_segment import (
    SemanticSegModelConfig,
    _detect_layout,
    suppression_mask_from_labels,
)


def test_detect_layout_nchw() -> None:
    lay = _detect_layout((1, 7, 64, 64), None)
    assert lay.kind == "nchw"


def test_detect_layout_nhwc() -> None:
    lay = _detect_layout((1, 64, 64, 7), None)
    assert lay.kind == "nhwc"


def test_suppression_mask() -> None:
    cfg = SemanticSegModelConfig(
        onnx_path=Path("dummy.onnx"),
        class_names=("bg", "geometry", "text", "dimension", "title_block", "table", "symbol"),
        suppress_for_skeleton=("text", "title_block"),
        suppress_dimension=True,
    )
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[:, 0] = 2  # text
    labels[0, 1] = 3  # dimension
    m = suppression_mask_from_labels(labels, cfg)
    assert m.shape == labels.shape
    assert m[:, 0].all()
    assert m[0, 1]


def test_import_raises_without_onnxruntime() -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        onnxruntime = None  # type: ignore[assignment]
    if onnxruntime is not None:
        return

    from drawing_to_dxf.semantic_segment import run_semantic_seg_labels

    gray = np.zeros((10, 10), dtype=np.uint8)
    cfg = SemanticSegModelConfig(
        onnx_path=Path("missing.onnx"),
    )
    try:
        run_semantic_seg_labels(gray, cfg)
    except ImportError as e:
        assert "onnxruntime" in str(e).lower()
    else:
        raise AssertionError("expected ImportError")
