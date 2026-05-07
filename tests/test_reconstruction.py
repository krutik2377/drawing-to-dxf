"""Tests for reconstruction roadmap and shop preset."""

from __future__ import annotations

from pathlib import Path

from drawing_to_dxf.panel_dxf_pipeline import PanelDxfRunConfig
from drawing_to_dxf.pipeline import RunConfig
from drawing_to_dxf.reconstruction_preset import (
    apply_shop_reconstruction_to_panel_config,
    apply_shop_reconstruction_to_run_config,
)
from drawing_to_dxf.reconstruction_roadmap import RECONSTRUCTION_STAGES, reconstruction_roadmap_manifest_rows


def test_reconstruction_roadmap_non_empty() -> None:
    rows = reconstruction_roadmap_manifest_rows()
    assert len(rows) == len(RECONSTRUCTION_STAGES)
    assert rows[0]["stage"] == 1
    assert "status" in rows[0]


def test_shop_preset_enables_python_passes() -> None:
    base = RunConfig(
        input_path=Path("a.png"),
        output_dir=Path("out"),
        is_pdf=False,
    )
    cfg = apply_shop_reconstruction_to_run_config(base)
    assert cfg.enable_topology_clean is True
    assert cfg.enable_topology_segment_repair is True
    assert cfg.enable_constraint_heal is True
    assert cfg.enable_engineering_intel_passes is True
    assert cfg.reconstruction_preset == "shop"
    assert cfg.segment_merge_distance_px >= 10.0
    assert cfg.enable_skeleton_circles is True
    assert cfg.engineering_layout is True


def test_shop_panel_preset_matches() -> None:
    p = PanelDxfRunConfig(input_path=Path("a.png"), output_dir=Path("out"))
    q = apply_shop_reconstruction_to_panel_config(p)
    assert q.reconstruction_preset == "shop"
    assert q.enable_topology_segment_repair is True
