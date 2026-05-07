"""Tests for engineering intelligence passes (constraints, topology stats, dimension bundle)."""

from __future__ import annotations

from drawing_to_dxf.annotations_export import dimension_association_bundle
from drawing_to_dxf.constraint_heal import align_parallel_residual_clusters, snap_axis_aligned_residual_segments
from drawing_to_dxf.engineering_intel_passes import (
    apply_engineering_intelligence_vector_stage,
    semantic_reasoning_snapshot,
    topology_intel_on_exploded_segments,
)
from drawing_to_dxf.geometry_model import PolylineDef, VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.topology_intel import endpoint_degree_counts


def test_snap_axis_aligned_residual_segments_horizontal() -> None:
    segs = [Segment(0.0, 10.0, 50.0, 11.2)]
    out, n = snap_axis_aligned_residual_segments(segs, angle_tol_deg=5.0, min_length_px=5.0)
    assert n == 1
    assert abs(out[0].y1 - out[0].y2) < 1e-5


def test_endpoint_degree_counts_corner() -> None:
    segs = [
        Segment(0.0, 0.0, 10.0, 0.0),
        Segment(10.0, 0.0, 10.0, 10.0),
    ]
    d = endpoint_degree_counts(segs, quant_px=1.0)
    assert d["vertices_total"] == 3
    assert d["degree_1"] == 2
    assert d["degree_2"] == 1


def test_dimension_association_bundle_finds_numeric_near_axis_segment() -> None:
    boxes = [TextBox("12.5 mm", 0.95, 40.0, 40.0, 120.0, 70.0)]
    segs = [Segment(0.0, 55.0, 220.0, 56.0)]
    hints, recs, stubs = dimension_association_bundle(segs, boxes, assoc_max_dist_px=90.0)
    assert hints
    assert recs
    assert stubs


def test_semantic_snapshot_counts_rect_like() -> None:
    vd = VectorDrawing(
        polylines=[
            PolylineDef(points=[(0, 0), (10, 0), (10, 10), (0, 10)], closed=True),
        ]
    )
    snap = semantic_reasoning_snapshot(vd, [])
    assert snap["semantic_rectangle_like_polylines"] >= 1


def test_engineering_intel_vector_stage_populates_metrics() -> None:
    vd = VectorDrawing(
        residual_segments=[
            Segment(0.0, 100.0, 80.0, 101.0),
            Segment(10.0, 50.0, 11.0, 130.0),
        ]
    )
    m: dict = {}
    vd2 = apply_engineering_intelligence_vector_stage(
        vd,
        ocr_boxes=None,
        metrics=m,
        enable_parallel_cluster_snap=False,
    )
    assert "constraint_residual_axis_snaps" in m
    assert len(vd2.residual_segments) == 2


def test_topology_intel_on_exploded_writes_keys() -> None:
    m: dict = {}
    topology_intel_on_exploded_segments(
        [Segment(0, 0, 5, 0), Segment(5, 0, 5, 5)],
        m,
    )
    assert "topology_degree_1" in m


def test_parallel_cluster_moves_three_horizontal_lines() -> None:
    segs = [
        Segment(0.0, 10.0, 40.0, 10.0),
        Segment(0.0, 14.0, 40.0, 14.0),
        Segment(0.0, 18.0, 40.0, 18.0),
    ]
    out, moves = align_parallel_residual_clusters(
        segs,
        offset_snap_px=4.0,
        min_cluster_size=3,
        min_length_px=10.0,
    )
    assert moves >= 1
    assert len(out) == 3
