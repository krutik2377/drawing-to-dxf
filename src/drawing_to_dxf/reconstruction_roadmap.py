"""Research-aligned engineering reconstruction stages (present vs planned).

Maps the gap between **raster vector traces** and **fully reconstructed shop drawings**:
semantic CAD reconstruction must run largely *before* AutoCAD/MCP polishing—see
``RECONSTRUCTION_STAGES`` and manifests ``reconstruction_roadmap``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReconstructionStageSpec:
    """One layer in the target architecture: raster → semantics → structured CAD → MCP."""

    stage_id: int
    title: str
    status: str  # implemented | partial | planned
    implementation: str


# Ordered stack: what exists in-repo vs what remains research / product work.
RECONSTRUCTION_STAGES: tuple[ReconstructionStageSpec, ...] = (
    ReconstructionStageSpec(
        1,
        "Raster acquisition and preprocess",
        "implemented",
        "preprocess.preprocess: denoise, deskew, scale; grayscale for ink mask + skeleton.",
    ),
    ReconstructionStageSpec(
        2,
        "Ink / geometry vs text separation",
        "partial",
        "OCR masks + annotation_clean text interior inpainting; dimension text not structured as DIM entities.",
    ),
    ReconstructionStageSpec(
        3,
        "Primitive extraction (lines, arcs, circles)",
        "partial",
        "skeleton_vectorize + vectorize; optional Hough circles (--skeleton-circles); vector_fit arcs; "
        "fragments remain without strong reconstruction.",
    ),
    ReconstructionStageSpec(
        4,
        "Topology repair and graph connectivity",
        "partial",
        "topology_repair.repair_exploded_segments; topology_clean.refine_vector_drawing; "
        "engineering_intel axis/cluster snaps when enabled (--python-* or --reconstruction shop).",
    ),
    ReconstructionStageSpec(
        5,
        "Constraint and orthogonal regularization",
        "partial",
        "constraint_heal orthogonal quads; residual snaps in engineering_intel_passes; "
        "full constraint solver not implemented.",
    ),
    ReconstructionStageSpec(
        6,
        "Dimension system reconstruction",
        "partial",
        "annotations_export.dimension_association_bundle + DIMENSION_HINT layer; "
        "no full extension-line / arrow DIM block model.",
    ),
    ReconstructionStageSpec(
        7,
        "Semantic entity classification",
        "partial",
        "segment_semantics + raster_semantics / ONNX; engineering_intel_extended.symbol_candidates; "
        "engineering_reconstruction_suite multilayer bucket expansion.",
    ),
    ReconstructionStageSpec(
        8,
        "Structured DXF / layer model",
        "implemented",
        "export_dxf: LWPOLYLINE, LINE, ARC, CIRCLE, TEXT, layers per part/panel.",
    ),
    ReconstructionStageSpec(
        9,
        "Native CAD polish (MCP / interactive)",
        "implemented",
        "cad_mcp_recipe: JOIN, OVERKILL, PEDIT, layers—assumes reasonable upstream geometry.",
    ),
)


def reconstruction_roadmap_manifest_rows() -> list[dict[str, str | int]]:
    """JSON-serializable rows for manifests and external planning tools."""
    return [
        {
            "stage": s.stage_id,
            "title": s.title,
            "status": s.status,
            "implementation": s.implementation,
        }
        for s in RECONSTRUCTION_STAGES
    ]


def engineering_reconstruction_capabilities_rows() -> list[dict[str, str]]:
    """
    Twenty capability areas (user roadmap) mapped to in-repo modules.

    ``status``: *implemented* = runnable Python pass with tests or stable export path;
    *partial* = heuristic / best-effort; *planned* = not started beyond hooks/metrics.
    """
    return [
        {
            "id": "1",
            "title": "Graph-based topology reconstruction",
            "status": "partial",
            "modules": "topology_repair, engineering_reconstruction_suite",
        },
        {
            "id": "2",
            "title": "Constraint-based CAD regularization",
            "status": "partial",
            "modules": "constraint_heal, cad_graph_passes.straighten_near_axis_segments, engineering_intel_passes",
        },
        {
            "id": "3",
            "title": "Dimension-semantic reconstruction",
            "status": "partial",
            "modules": "dimension_semantic_rebuild, annotations_export",
        },
        {
            "id": "4",
            "title": "Circle and hole reconstruction",
            "status": "partial",
            "modules": "engineering_intel_extended.promote_near_full_arcs_to_circles, hole_pattern",
        },
        {
            "id": "5",
            "title": "Multi-layer semantic processing",
            "status": "partial",
            "modules": "segment_semantics, engineering_intel_extended.expand_multilayer_semantic_bucketing",
        },
        {
            "id": "6",
            "title": "Engineering-aware snapping",
            "status": "partial",
            "modules": "topology_repair.merge_close_segment_endpoints, vectorize merges, constraint_heal snaps",
        },
        {
            "id": "7",
            "title": "Topology healing pass",
            "status": "partial",
            "modules": "engineering_reconstruction_suite, topology_repair.repair_exploded_segments",
        },
        {
            "id": "8",
            "title": "Long-line continuity preservation",
            "status": "partial",
            "modules": "cad_graph_passes.merge_collinear_chains",
        },
        {
            "id": "9",
            "title": "Thin-line preservation",
            "status": "partial",
            "modules": "vectorize protect_hole_rings + manifest note in suite (no re-morph in suite)",
        },
        {
            "id": "10",
            "title": "Structural pattern inference",
            "status": "partial",
            "modules": "engineering_intel_extended.infer_regular_grid_pitch, symmetry_about_axis_hint",
        },
        {
            "id": "11",
            "title": "CAD-quality geometric cleanup",
            "status": "partial",
            "modules": "cad_graph_passes.overkill_near_duplicate_segments, straighten_near_axis_segments",
        },
        {
            "id": "12",
            "title": "Rectangle and closed-shape reconstruction",
            "status": "partial",
            "modules": "constraint_heal.orthogonal_close_quads, topology_repair.bridge_almost_closed_loops",
        },
        {
            "id": "13",
            "title": "Intersection inference",
            "status": "partial",
            "modules": "topology_repair.extend_free_endpoints_to_intersections",
        },
        {
            "id": "14",
            "title": "Semantic OCR association",
            "status": "partial",
            "modules": "engineering_intel_extended.associate_ocr_semantic_links, link_geometry",
        },
        {
            "id": "15",
            "title": "Centerline detection",
            "status": "partial",
            "modules": "engineering_intel_extended.centerline_alignment_segments",
        },
        {
            "id": "16",
            "title": "Engineering symbol detector",
            "status": "partial",
            "modules": "engineering_intel_extended.engineering_symbol_candidates",
        },
        {
            "id": "17",
            "title": "Primitive confidence scoring",
            "status": "partial",
            "modules": "engineering_intel_extended.segment_primitive_confidence",
        },
        {
            "id": "18",
            "title": "Repeated entity clustering",
            "status": "partial",
            "modules": "engineering_intel_extended.cluster_parallel_segments",
        },
        {
            "id": "19",
            "title": "Final CAD healing stage",
            "status": "partial",
            "modules": "cad_graph_passes, cad_mcp_recipe (AutoCAD JOIN/OVERKILL)",
        },
        {
            "id": "20",
            "title": "Engineering QA validator",
            "status": "partial",
            "modules": "engineering_qa.run_engineering_qa",
        },
    ]
