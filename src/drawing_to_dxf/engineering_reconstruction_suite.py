"""Unified engineering reconstruction suite: topology, CAD cleanup, dimensions, QA, semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence

from drawing_to_dxf.cad_graph_passes import (
    merge_collinear_chains,
    overkill_near_duplicate_segments,
    straighten_near_axis_segments,
)
from drawing_to_dxf.dimension_semantic_rebuild import reconstruct_dimension_objects
from drawing_to_dxf.engineering_intel_extended import (
    associate_ocr_semantic_links,
    centerline_alignment_segments,
    cluster_parallel_segments,
    engineering_symbol_candidates,
    expand_multilayer_semantic_bucketing,
    filter_segments_by_confidence,
    infer_regular_grid_pitch,
    promote_near_full_arcs_to_circles,
    symmetry_about_axis_hint,
)
from drawing_to_dxf.engineering_qa import run_engineering_qa
from drawing_to_dxf.geometry_model import VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.segment_types import Segment
from drawing_to_dxf.topology_repair import bridge_almost_closed_loops, repair_exploded_segments


@dataclass
class EngineeringReconstructionSuiteConfig:
    """Tunable pass set for ``apply_engineering_reconstruction_suite``."""

    refine_topology: bool = True
    topology_merge_distance_px: float = 5.5
    topology_max_bridge_gap_px: float = 12.0
    topology_junction_snap_px: float = 4.0
    topology_bridge_direction_dot_min: float = 0.42
    topology_intersection_extend_px: float = 9.0
    loop_close_gap_px: float = 8.0
    confidence_drop_min_score: float = 0.1
    straighten_axis: bool = True
    merge_chains: bool = True
    overkill_dupes: bool = True
    promote_arcs_min_span_deg: float = 300.0


def apply_engineering_reconstruction_suite(
    vd: VectorDrawing,
    segments: list[Segment],
    *,
    ocr_boxes: Sequence[TextBox] | None,
    layered_segments: dict[str, list[Segment]] | None,
    dimension_associations: list[dict[str, Any]] | None,
    suite_cfg: EngineeringReconstructionSuiteConfig | None = None,
    metrics_out: MutableMapping[str, Any] | None = None,
) -> tuple[VectorDrawing, list[Segment], dict[str, Any]]:
    """
    Run ordered Python-side passes covering topology healing, CAD regularization, dimension
    objects, multilayer semantics, pattern/symbol hints, and QA.

    Returns ``(vector_drawing, healed_segments, manifest_blob)``.
    """
    cfg = suite_cfg or EngineeringReconstructionSuiteConfig()
    blob: dict[str, Any] = {"suite": "engineering_reconstruction_v1"}
    m = metrics_out if metrics_out is not None else {}
    blob["pass_metrics"] = m

    vd2, n_arc_prom = promote_near_full_arcs_to_circles(vd, min_span_deg=cfg.promote_arcs_min_span_deg)
    m["arcs_promoted_to_circles"] = n_arc_prom

    segs = [Segment(s.x1, s.y1, s.x2, s.y2) for s in segments]

    segs, n_drop = filter_segments_by_confidence(segs, min_score=cfg.confidence_drop_min_score)
    m["low_confidence_segments_dropped"] = n_drop

    if cfg.refine_topology:
        topo_manifest: dict[str, int] = {}
        segs = repair_exploded_segments(
            segs,
            merge_distance=cfg.topology_merge_distance_px,
            max_bridge_gap_px=cfg.topology_max_bridge_gap_px,
            junction_snap_px=cfg.topology_junction_snap_px,
            bridge_direction_dot_min=cfg.topology_bridge_direction_dot_min,
            intersection_extend_px=cfg.topology_intersection_extend_px,
            manifest_metrics=topo_manifest,
        )
        m.update(topo_manifest)

    if cfg.loop_close_gap_px > 0:
        segs, n_lc = bridge_almost_closed_loops(segs, max_gap_px=cfg.loop_close_gap_px)
        m["topology_loop_closures_suite"] = n_lc

    if cfg.straighten_axis:
        segs, n_st = straighten_near_axis_segments(segs)
        m["axis_straightened_segments"] = n_st

    if cfg.merge_chains:
        segs, n_mc = merge_collinear_chains(segs)
        m["collinear_chain_merges"] = n_mc

    if cfg.overkill_dupes:
        segs, n_ok = overkill_near_duplicate_segments(segs)
        m["overkill_duplicates_removed"] = n_ok

    multilayer = expand_multilayer_semantic_bucketing(layered_segments)
    blob["multilayer_semantics"] = {
        "counts": multilayer.get("counts", {}),
        "note": "geometry bucket merges OTHER remainder after SYMBOL/ANNOTATION split",
    }

    dim_pool = layered_segments.get("DIMENSION") if layered_segments else None
    dim_objs = reconstruct_dimension_objects(
        segs,
        list(ocr_boxes) if ocr_boxes else None,
        dimension_layer_segments=dim_pool,
    )
    blob["reconstructed_dimensions"] = dim_objs

    centers = [(float(c.cx), float(c.cy)) for c in vd2.circles]
    blob["hole_pitch_inference"] = infer_regular_grid_pitch(centers)
    _clusters, cstats = cluster_parallel_segments(segs, min_cluster=4)
    blob["repeated_parallel_clusters"] = cstats
    blob["symmetry_hint"] = symmetry_about_axis_hint(segs)
    blob["centerline_reference_segments"] = centerline_alignment_segments(vd2)
    blob["engineering_symbol_hints"] = engineering_symbol_candidates(
        ocr_boxes,
        segments=segs,
    )
    blob["semantic_ocr_geometry_links"] = associate_ocr_semantic_links(ocr_boxes, segs)

    qa = run_engineering_qa(
        vd2,
        segs,
        dimension_associations=dimension_associations,
        reconstructed_dimensions=dim_objs,
    )
    blob["engineering_qa"] = qa

    blob["thin_line_preservation"] = {
        "strategy": "segmentation-time protect_hole_rings + short strokes classified in multilayer OTHER split",
        "morphology_note": "ink mask morphology remains upstream; suite does not re-erode vectors",
    }

    return vd2, segs, blob
