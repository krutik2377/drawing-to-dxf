"""Presets that bundle Python-side reconstruction passes for stronger DXF before CAD/MCP polish."""

from __future__ import annotations

from dataclasses import replace

from drawing_to_dxf.panel_dxf_pipeline import PanelDxfRunConfig
from drawing_to_dxf.pipeline import RunConfig


def apply_full_engineering_reconstruction_to_run_config(cfg: RunConfig) -> RunConfig:
    """
    Maximum Python-side reconstruction before DXF: shop bundle + unified engineering suite +
    semantic layering for geometry/dimension/text buckets (rule-based raster semantics).
    """
    cfg2 = apply_shop_reconstruction_to_run_config(cfg)
    return replace(
        cfg2,
        reconstruction_preset="full",
        enable_engineering_reconstruction_suite=True,
        rule_based_semantics=True,
        layered_dxf=True,
        topology_loop_close_px=max(cfg2.topology_loop_close_px, 8.0),
        emit_linear_dimension_entities=True,
    )


def apply_full_engineering_reconstruction_to_panel_config(cfg: PanelDxfRunConfig) -> PanelDxfRunConfig:
    """Panel export variant of :func:`apply_full_engineering_reconstruction_to_run_config`."""
    cfg2 = apply_shop_reconstruction_to_panel_config(cfg)
    return replace(
        cfg2,
        reconstruction_preset="full",
        enable_engineering_reconstruction_suite=True,
        rule_based_semantics=True,
        layered_dxf=True,
    )


def apply_shop_reconstruction_to_run_config(cfg: RunConfig) -> RunConfig:
    """
    Steel / plate shop-sheet style: enable topology + constraint + intel passes and
    tune merges, bridges, and hole detection. Does not replace full semantic reconstruction.
    """
    return replace(
        cfg,
        reconstruction_preset="shop",
        enable_topology_clean=True,
        enable_topology_segment_repair=True,
        enable_constraint_heal=True,
        enable_engineering_intel_passes=True,
        segment_merge_distance_px=max(cfg.segment_merge_distance_px, 10.0),
        vector_polyline_rdp_epsilon_px=max(cfg.vector_polyline_rdp_epsilon_px, 2.0),
        vector_collinear_merge_angle_deg=min(cfg.vector_collinear_merge_angle_deg, 4.0)
        if cfg.vector_collinear_merge_angle_deg > 0
        else cfg.vector_collinear_merge_angle_deg,
        topology_max_bridge_gap_px=cfg.topology_max_bridge_gap_px
        if cfg.topology_max_bridge_gap_px is not None
        else 12.0,
        geometry_bridge_gap_px=max(cfg.geometry_bridge_gap_px, 3.0),
        topology_junction_snap_px=max(cfg.topology_junction_snap_px, 4.0),
        enable_skeleton_circles=True,
        engineering_layout=True,
        topology_intersection_extend_px=max(cfg.topology_intersection_extend_px, 7.0),
        enable_cad_axis_regularization=True,
        enable_healed_vector_export=True,
    )


def apply_shop_reconstruction_to_panel_config(cfg: PanelDxfRunConfig) -> PanelDxfRunConfig:
    """Same intent as :func:`apply_shop_reconstruction_to_run_config` for panel exports."""
    return replace(
        cfg,
        reconstruction_preset="shop",
        enable_topology_clean=True,
        enable_topology_segment_repair=True,
        enable_constraint_heal=True,
        enable_engineering_intel_passes=True,
        segment_merge_distance_px=max(cfg.segment_merge_distance_px, 10.0),
        vector_polyline_rdp_epsilon_px=max(cfg.vector_polyline_rdp_epsilon_px, 2.0),
        vector_collinear_merge_angle_deg=min(cfg.vector_collinear_merge_angle_deg, 4.0)
        if cfg.vector_collinear_merge_angle_deg > 0
        else cfg.vector_collinear_merge_angle_deg,
        topology_max_bridge_gap_px=cfg.topology_max_bridge_gap_px
        if cfg.topology_max_bridge_gap_px is not None
        else 12.0,
        geometry_bridge_gap_px=max(cfg.geometry_bridge_gap_px, 3.0),
        topology_junction_snap_px=max(cfg.topology_junction_snap_px, 4.0),
        enable_skeleton_circles=True,
        engineering_layout=True,
        topology_intersection_extend_px=max(cfg.topology_intersection_extend_px, 7.0),
        enable_cad_axis_regularization=True,
        enable_healed_vector_export=True,
    )


def apply_python_cad_reconstruction_bundle_to_run_config(cfg: RunConfig) -> RunConfig:
    """Healed topology export + ray extensions + axis alignment without full shop preset."""
    return replace(
        cfg,
        enable_topology_segment_repair=True,
        topology_max_bridge_gap_px=cfg.topology_max_bridge_gap_px
        if cfg.topology_max_bridge_gap_px is not None
        else 10.0,
        topology_junction_snap_px=max(cfg.topology_junction_snap_px, 4.0),
        topology_intersection_extend_px=max(cfg.topology_intersection_extend_px, 7.0),
        enable_cad_axis_regularization=True,
        enable_healed_vector_export=True,
        segment_merge_distance_px=max(cfg.segment_merge_distance_px, 8.0),
    )


def apply_python_cad_reconstruction_bundle_to_panel_config(cfg: PanelDxfRunConfig) -> PanelDxfRunConfig:
    """Same intent as :func:`apply_python_cad_reconstruction_bundle_to_run_config` for panel exports."""
    return replace(
        cfg,
        enable_topology_segment_repair=True,
        topology_max_bridge_gap_px=cfg.topology_max_bridge_gap_px
        if cfg.topology_max_bridge_gap_px is not None
        else 10.0,
        topology_junction_snap_px=max(cfg.topology_junction_snap_px, 4.0),
        topology_intersection_extend_px=max(cfg.topology_intersection_extend_px, 7.0),
        enable_cad_axis_regularization=True,
        enable_healed_vector_export=True,
        segment_merge_distance_px=max(cfg.segment_merge_distance_px, 8.0),
    )
