"""Tests for rule-based raster semantics and segment splitting."""

from __future__ import annotations

import numpy as np

from drawing_to_dxf.raster_semantics import build_rule_based_pixel_labels, combined_pixel_labels
from drawing_to_dxf.segment_semantics import split_segments_by_semantic_layer
from drawing_to_dxf.segment_types import Segment


def test_rule_labels_shape_and_dim_channel() -> None:
    gray = np.full((64, 64), 240, dtype=np.uint8)
    gray[30:34, 10:54] = 20  # horizontal stroke
    labels, meta = build_rule_based_pixel_labels(gray, [])
    assert labels.shape == (64, 64)
    assert "rule_semantics" in meta
    dim_frac = float(np.mean(labels == 3))  # dimension class index
    assert dim_frac > 0.0


def test_combined_rule_only() -> None:
    gray = np.ones((32, 32), dtype=np.uint8) * 250
    labels, meta = combined_pixel_labels(gray, None, rule_only=True)
    assert labels is not None
    assert "rule_based" in meta


def test_split_segments_by_layer() -> None:
    h, w = 100, 100
    labels = np.zeros((h, w), dtype=np.int32)
    labels[40:60, 40:60] = 1  # geometry
    labels[50, :] = 3  # dimension row
    segs = [Segment(50.0, 50.0, 50.0, 90.0), Segment(10.0, 80.0, 90.0, 80.0)]
    buckets, counts = split_segments_by_semantic_layer(segs, labels)
    assert "DIMENSION" in buckets
    assert sum(counts.values()) == 2
