import numpy as np
import pytest

pytest.importorskip("cv2")

import cv2  # noqa: E402

from drawing_to_dxf.panel_dxf_pipeline import PanelDxfRunConfig, run_panel_dxfs


def test_run_panel_dxfs_writes_one_dxf_per_panel(tmp_path):
    gray = np.full((400, 700), 255, dtype=np.uint8)
    gray[40:220, 40:260] = 30
    gray[40:220, 420:640] = 30
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    inp = tmp_path / "twoblobs.png"
    cv2.imwrite(str(inp), bgr)

    out_dir = tmp_path / "panel_out"
    cfg = PanelDxfRunConfig(
        input_path=inp,
        output_dir=out_dir,
        skip_ocr=True,
        max_side=None,
        denoise=False,
        deskew=False,
        panel_min_area=8000,
        panel_min_gap=24,
        min_line_length=8,
        mm_per_pixel=1.0,
    )
    res = run_panel_dxfs(cfg)
    assert res.panel_count >= 2
    assert len(res.dxf_paths) == res.panel_count
    assert res.manifest_path.exists()
    for pth in res.dxf_paths:
        assert pth.is_file()
    assert res.viewer_primary_dxf is not None
    assert res.viewer_primary_dxf.exists()
    assert (out_dir / "viewer" / "models" / "panel_00.dxf").exists()
    assert (out_dir / "viewer" / "README.txt").exists()


def test_run_panel_dxfs_optional_no_viewer_bundle(tmp_path):
    gray = np.full((400, 700), 255, dtype=np.uint8)
    gray[40:220, 40:260] = 30
    gray[40:220, 420:640] = 30
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    inp = tmp_path / "twoblobs_nv.png"
    cv2.imwrite(str(inp), bgr)
    out_dir = tmp_path / "nv"
    cfg = PanelDxfRunConfig(
        input_path=inp,
        output_dir=out_dir,
        skip_ocr=True,
        max_side=None,
        denoise=False,
        deskew=False,
        panel_min_area=8000,
        panel_min_gap=24,
        min_line_length=8,
        write_viewer_bundle=False,
        mm_per_pixel=1.0,
    )
    res = run_panel_dxfs(cfg)
    assert res.viewer_primary_dxf is None
    assert not (out_dir / "viewer").exists()
