import numpy as np

from drawing_to_dxf.panel_split import split_panels
from drawing_to_dxf.render_layout import PanelTile, render_composite_png


def test_split_panels_finds_two_blobs() -> None:
    g = np.full((400, 700), 255, dtype=np.uint8)
    g[40:220, 40:260] = 30
    g[40:220, 420:640] = 30
    boxes = split_panels(g, min_area=8000, morph_close=9)
    assert len(boxes) >= 2


def test_render_composite_creates_png(tmp_path) -> None:
    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    bgr[:] = (240, 240, 240)
    out = tmp_path / "c.png"
    render_composite_png(
        [PanelTile(image_bgr=bgr, title="2x TEST 123 MS", meta={})],
        out,
        cols=1,
        max_cell_width=200,
        max_cell_height=200,
    )
    assert out.exists()
    assert out.stat().st_size > 50
