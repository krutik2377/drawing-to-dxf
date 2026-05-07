import numpy as np

from drawing_to_dxf.export_dxf import export_segments_only
from drawing_to_dxf.preprocess import preprocess
from drawing_to_dxf.vectorize import extract_segments


def _synthetic_gray() -> np.ndarray:
    img = np.full((200, 300), 255, dtype=np.uint8)
    # Thick strokes survive adaptive threshold + skeletonizer (thin 1px lines do not).
    img[49:53, 40:262] = 0
    img[40:182, 149:154] = 0
    return img


def test_extract_segments_finds_lines() -> None:
    gray = _synthetic_gray()
    segs = extract_segments(gray, min_line_length=10, hough_threshold=15)
    assert len(segs) >= 2


def test_preprocess_and_scale() -> None:
    bgr = np.dstack([_synthetic_gray()] * 3)
    r = preprocess(bgr, max_side=100, denoise=False, deskew=False)
    assert r.gray.shape[0] <= 100 and r.gray.shape[1] <= 100
    assert 0 < r.scale <= 1.0


def test_export_dxf_writes(tmp_path) -> None:
    from drawing_to_dxf.vectorize import Segment

    segs = [Segment(0, 0, 100, 0), Segment(0, 0, 0, 100)]
    outp = tmp_path / "t.dxf"
    export_segments_only(outp, segs, img_height_px=100.0, mm_per_pixel=1.0)
    assert outp.exists()
    text = outp.read_text(encoding="utf-8", errors="ignore")
    assert "LINE" in text or "LWPOLYLINE" in text.upper()
