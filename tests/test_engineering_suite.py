"""Tests for engineering reconstruction suite and helpers."""

from __future__ import annotations

from drawing_to_dxf.cad_graph_passes import merge_collinear_chains
from drawing_to_dxf.dimension_semantic_rebuild import reconstruct_dimension_objects
from drawing_to_dxf.engineering_intel_extended import (
    arc_span_deg,
    promote_near_full_arcs_to_circles,
    segment_primitive_confidence,
)
from drawing_to_dxf.engineering_qa import run_engineering_qa
from drawing_to_dxf.geometry_model import ArcDef, CircleDef, VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment


def test_promote_near_full_arc_to_circle() -> None:
    vd = VectorDrawing(
        arcs=[ArcDef(cx=50.0, cy=50.0, r=10.0, start_angle_deg=0.0, end_angle_deg=350.0, ccw=True)]
    )
    vd2, n = promote_near_full_arcs_to_circles(vd, min_span_deg=300.0)
    assert n == 1
    assert len(vd2.circles) == 1
    assert not vd2.arcs


def test_segment_primitive_confidence_axis() -> None:
    s = Segment(0.0, 10.0, 80.0, 10.0)
    assert segment_primitive_confidence(s) > 0.7


def test_merge_collinear_chain() -> None:
    segs = [
        Segment(0.0, 0.0, 5.0, 0.0),
        Segment(5.0, 0.0, 12.0, 0.0),
    ]
    out, n = merge_collinear_chains(segs, endpoint_quant_px=1.0, angle_tol_deg=3.0)
    assert n >= 1
    assert len(out) == 1


def test_reconstruct_dimension_objects_numeric() -> None:
    boxes = [TextBox("10 mm", 0.9, 40.0, 40.0, 70.0, 62.0)]
    segs = [Segment(0.0, 52.0, 200.0, 52.0)]
    objs = reconstruct_dimension_objects(segs, boxes)
    assert objs
    assert objs[0].get("dimension_line")


def test_engineering_qa_reports_invalid_circle() -> None:
    vd_bad = VectorDrawing(circles=[CircleDef(cx=1.0, cy=1.0, r=-2.0)])
    qa = run_engineering_qa(vd_bad, [Segment(0, 0, 1, 1)])
    assert qa["invalid_circles"] >= 1


def test_arc_span_deg_ccw() -> None:
    a = ArcDef(cx=0, cy=0, r=1, start_angle_deg=0, end_angle_deg=90, ccw=True)
    assert abs(arc_span_deg(a) - 90.0) < 1e-6
