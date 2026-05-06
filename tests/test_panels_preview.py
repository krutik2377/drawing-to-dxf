import numpy as np
import pytest

pytest.importorskip("cv2")

from drawing_to_dxf.panels_preview import preview_panels


def test_preview_panels_counts_two_blobs() -> None:
    g = np.full((400, 700), 255, dtype=np.uint8)
    g[40:220, 40:260] = 30
    g[40:220, 420:640] = 30
    pv = preview_panels(g, min_area=8000, min_gap_px=24, morph_close=9)
    assert pv.panel_count >= 2
    assert pv.annotated_bgr.shape[:2] == g.shape[:2]
