"""Canonical 20-step raster → CAD pipeline (roadmap ↔ code).

The numbered list matches the product workflow. **Conceptual order** here may differ
slightly from call order in code where physics dictates better results (e.g. denoise and
deskew run on continuous-tone gray before binarization; adaptive threshold is applied
when building the ink mask inside vectorization).

Consumers: manifests from ``pipeline.run`` and ``panel_dxf_pipeline.run_panel_dxfs``,
and developer orientation when extending stages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStepSpec:
    """One roadmap step with pointers into this package."""

    step_id: int
    title: str
    implementation: str


# Canonical 20 steps — titles kept aligned with the roadmap wording.
CANONICAL_PIPELINE_STEPS: tuple[PipelineStepSpec, ...] = (
    PipelineStepSpec(
        1,
        "Load raster image",
        "preprocess.load_image_bgr, preprocess.load_pdf_page_as_bgr; "
        "pipeline.load_input_bgr / panel_dxf_pipeline._load_bgr",
    ),
    PipelineStepSpec(
        2,
        "Convert to grayscale",
        "preprocess.preprocess (cv2.cvtColor BGR→GRAY)",
    ),
    PipelineStepSpec(
        3,
        "Apply adaptive thresholding",
        "annotation_clean.inks_mask_from_gray "
        "(cv2.adaptiveThreshold ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV)",
    ),
    PipelineStepSpec(
        4,
        "Perform deskewing/alignment",
        "preprocess._deskew_gray (invoked from preprocess.preprocess when deskew=True)",
    ),
    PipelineStepSpec(
        5,
        "Remove noise",
        "preprocess.preprocess bilateral denoise; skeleton_vectorize.extract_vector_drawing "
        "remove_small_objects on foreground before skeletonize",
    ),
    PipelineStepSpec(
        6,
        "Apply morphology closing to reconnect broken lines",
        "annotation_clean.suppress_hatches_and_dimensions MORPH_CLOSE on binary ink "
        "(when suppress_ruling_lines=True); preprocess._deskew_gray uses a small close on Otsu mask",
    ),
    PipelineStepSpec(
        7,
        "Separate OCR/text layer from geometry layer",
        "ocr_extract.extract_text_boxes; annotation_clean.apply_text_masks / "
        "apply_text_masks_interior_only before inks_mask_from_gray",
    ),
    PipelineStepSpec(
        8,
        "Detect line segments using LSD/Hough Transform",
        "Primary path: skeleton_graph.trace_skeleton_polylines. Alternates: "
        "vectorize._legacy_hough_extract_segments (HoughLinesP). Optional supplement: "
        "geometry_intel.lsd_extract_segments (cv2.createLineSegmentDetector) via "
        "--vectorize-lsd-supplement in vectorize.extract_skeleton_vector_bundle.",
    ),
    PipelineStepSpec(
        9,
        "Detect circles/arcs using HoughCircles",
        "skeleton_vectorize.detect_circles (optional, off by default): tighter Hough params, "
        "rim + **hole-interior** validation on binary ink, hard cap on count. "
        "Enable with --skeleton-circles. vector_fit.apply_polyline_fittings for arc / "
        "loop-to-CIRCLE when arc fitting is on.",
    ),
    PipelineStepSpec(
        10,
        "Detect rectangles and closed polylines",
        "Closed loops: skeleton traces with closed=True → geometry_model.PolylineDef.closed. "
        "Orthogonal 4-gons tallied as rectangle_like_polylines (geometry_intel.count_corner_rectangles); "
        "no separate DXF RECTANGLE entity type.",
    ),
    PipelineStepSpec(
        11,
        "Snap nearby endpoints together",
        "vectorize.extract_skeleton_vector_bundle → _merge_close_endpoints on exploded segments",
    ),
    PipelineStepSpec(
        12,
        "Merge collinear and overlapping lines",
        "topology_clean.refine_vector_drawing; vectorize._collapse_collinear_segments; "
        "merge_collinear runs inside topology_clean",
    ),
    PipelineStepSpec(
        13,
        "Reconstruct topology graph/connectivity",
        "skeleton_graph.skeleton_adjacency_graph + trace_skeleton_polylines",
    ),
    PipelineStepSpec(
        14,
        "Detect dimensions, arrows, and extension lines",
        "Partial: annotation_clean.suppress_hatches_and_dimensions removes long horizontal "
        "ruling motifs; no dedicated arrow/dimension classifier.",
    ),
    PipelineStepSpec(
        15,
        "Group related entities into drawing parts/blocks",
        "pipeline: link_geometry.link_vector_geometry_to_parts + link_parts.PartGroup; "
        "panels: panel_split.split_panels then per-crop export",
    ),
    PipelineStepSpec(
        16,
        "Preserve original scaling and coordinates",
        "PreprocessResult.original_shape, scale; RunConfig.mm_per_pixel; "
        "export_dxf.image_xy_to_dxf consistent Y-flip + scale in manifests",
    ),
    PipelineStepSpec(
        17,
        "Convert primitives into CAD entities (LINE, ARC, CIRCLE, TEXT, etc.)",
        "export_dxf._emit_vector_entities: LWPOLYLINE, ARC, CIRCLE, LINE residuals; "
        "TEXT via PartGroup label points on part layers",
    ),
    PipelineStepSpec(
        18,
        "Generate structured DXF entities/layers",
        "export_dxf.export_merged_dxf / export_part_dxf / export_segments_only "
        "(layer-per-part or GEOMETRY/LABEL conventions)",
    ),
    PipelineStepSpec(
        19,
        "Export final DXF file",
        "ezdxf writes in export_dxf; paths recorded in *_manifest.json",
    ),
    PipelineStepSpec(
        20,
        "Run post-processing cleanup and validation",
        "topology_clean.refine_vector_drawing; vector_fit.apply_polyline_fittings; "
        "vectorize length filters; optional geometry_intel bridge_open_polyline_gaps + residual dedupe; "
        "geometry_intel.geometry_quality_report (confidence heuristic) + optional pipeline_timings_ms; "
        "pipeline warnings in manifest.",
    ),
)


def pipeline_flow_manifest_rows() -> list[dict[str, int | str]]:
    """JSON-serializable rows for manifests and tooling."""
    return [
        {"step": s.step_id, "title": s.title, "implementation": s.implementation}
        for s in CANONICAL_PIPELINE_STEPS
    ]
