"""
Streamlit UI: tune split + OCR part-ID inventory, preview counts, export DXFs.

Run:
    pip install -e ".[ui,ocr]"
    drawing-to-dxf ui-panels

Or:

    python -m streamlit run src/drawing_to_dxf/ui_panels_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from drawing_to_dxf.panel_dxf_pipeline import PanelDxfRunConfig, run_panel_dxfs
from drawing_to_dxf.panels_preview import ocr_distinct_part_ids, preview_panels
from drawing_to_dxf.preprocess import load_image_bgr, load_pdf_page_as_bgr, preprocess


@st.cache_data(show_spinner="Loading drawing…")
def _decode_grayscale_bytes(file_bytes: bytes, orig_name: str, max_side_val: int) -> tuple[np.ndarray, bool]:
    """Return (processed grayscale np.ndarray, is_pdf flag). Caller owns array copy semantics."""
    suf = Path(orig_name).suffix.lower() or ".png"
    if suf == ".jpeg":
        suf = ".jpg"
    is_pdf = suf == ".pdf"
    fd, path = tempfile.mkstemp(suffix=suf)
    try:
        os.write(fd, file_bytes)
        os.close(fd)
        fd = -1
        pth = Path(path)
        if is_pdf:
            bgr = load_pdf_page_as_bgr(str(pth), page_index=0, dpi=150.0)
        else:
            bgr = load_image_bgr(str(pth))
        if bgr is None:
            return np.zeros((1, 1), dtype=np.uint8), is_pdf
        ms = None if max_side_val == 0 else int(max_side_val)
        pre = preprocess(bgr, max_side=ms, denoise=True, deskew=True)
        return pre.gray.copy(), is_pdf
    finally:
        try:
            if fd >= 0:
                os.close(fd)
        except OSError:
            pass
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _write_temp_upload(file_bytes: bytes, orig_name: str) -> tuple[Path, bool]:
    suf = Path(orig_name).suffix.lower() or ".png"
    if suf == ".jpeg":
        suf = ".jpg"
    is_pdf = suf == ".pdf"
    fd, path = tempfile.mkstemp(suffix=suf)
    try:
        os.write(fd, file_bytes)
        os.close(fd)
        return Path(path), is_pdf
    except Exception:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main() -> None:
    st.set_page_config(page_title="drawing-to-dxf — panel preview", layout="wide")
    st.title("Panel detection & DXF export")
    st.caption(
        "**Panels detected** counts orange boxes from gutter/blob split. "
        "**Distinct OCR part IDs** lists regex matches on stitched text — use both; neither is fixed."
    )

    up = st.file_uploader(
        "Drawing upload",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
    )

    ms_col, mm_col = st.columns(2)
    with ms_col:
        max_side = int(
            st.number_input("Preprocess max side (0 = no downscale)", min_value=0, value=4096, step=128)
        )
    with mm_col:
        mm_per_px = float(st.number_input("mm per pixel", min_value=0.0001, value=1.0, format="%f"))

    st.subheader("Panel splitter parameters")
    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        min_area = int(st.slider("min_area", 1000, 100000, 15_000, step=500))
    with a2:
        min_gap = int(st.slider("min_gap_px", 8, 220, 48))
    with a3:
        min_side = int(st.slider("min_short_side_px", 16, 220, 80))
    with a4:
        max_aspect = float(st.slider("max_aspect_ratio", 4.0, 24.0, 10.0, step=0.5))
    with a5:
        morph_close = int(st.slider("morph_close", 3, 21, 11, step=2))
        if morph_close % 2 == 0:
            morph_close |= 1

    st.subheader("Vectorization & Autodesk bundle")
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        min_ll = int(st.slider("min_line_length", 5, 80, 20))
    with v2:
        merge_d = float(st.slider("segment_merge_distance_px", 0.0, 12.0, 5.0, step=0.5))
    with v3:
        col_ang = float(st.slider("vector_collinear_angle_deg (0=off)", 0.0, 15.0, 5.0, step=0.5))
    with v4:
        rdp_eps = float(st.slider("vector RDP epsilon px (0=off)", 0.0, 6.0, 1.5, step=0.25))
    with v5:
        viewer_gap = float(st.slider("viewer_layout_gap_mm", 5.0, 100.0, 25.0, step=5.0))

    do_ocr = st.checkbox(
        "OCR part-ID inventory (preview)",
        value=True,
        help="Distinct 3–5 digit matches after EasyOCR. Requires pip install -e '.[ocr]'",
    )
    ocr_gpu = st.checkbox("OCR GPU", value=False)

    chk_export_ocr = st.checkbox(
        "Use OCR naming on export",
        value=False,
        help="Writes part-number filenames where detected; installs EasyOCR if missing.",
    )

    out_dir = st.text_input("Output directory", value="out_panels_ui")

    col_a, col_b = st.columns(2)
    with col_a:
        analyze = st.button("Analyze (preview)")
    with col_b:
        export = st.button("Export DXFs → output directory")

    if up is None:
        st.info("Upload a sheet to inspect automatic counts.")
        return

    blob = up.getvalue()
    name = up.name
    gray_arr, _ = _decode_grayscale_bytes(blob, name, max_side)
    split_kw = dict(
        min_area=min_area,
        min_gap_px=min_gap,
        min_short_side_px=min_side,
        max_aspect_ratio=max_aspect,
        morph_close=morph_close,
    )

    pv = preview_panels(gray_arr, **split_kw)

    if analyze:
        rgb = pv.annotated_bgr[:, :, ::-1]
        st.image(rgb, caption="Splitter panel indices [0 … N-1]", use_container_width=True)
        st.metric("Panels detected (splitter)", int(pv.panel_count))

        if do_ocr:
            try:
                ids, pairs = ocr_distinct_part_ids(gray_arr, gpu=ocr_gpu, min_confidence=0.15)
                st.metric("Distinct OCR part-number candidates", len(ids))
                if pairs:
                    st.dataframe(
                        [{"part_id": p, "best_confidence": round(c, 3)} for p, c in pairs],
                        hide_index=True,
                    )
            except ImportError:
                st.warning("EasyOCR not installed — run `pip install -e \".[ocr]\"` for ID inventory.")
            except Exception as e:  # noqa: BLE001
                st.warning(f"OCR failed: {e}")

    if export:
        tmp_in, is_pdf2 = _write_temp_upload(blob, name)
        try:
            pcfg = PanelDxfRunConfig(
                input_path=tmp_in,
                output_dir=Path(out_dir).resolve(),
                is_pdf=is_pdf2,
                pdf_page=0,
                pdf_dpi=150.0,
                max_side=None if max_side == 0 else max_side,
                min_line_length=min_ll,
                panel_min_area=min_area,
                panel_min_gap=min_gap,
                panel_min_short_side_px=min_side,
                panel_max_aspect_ratio=max_aspect,
                segment_merge_distance_px=merge_d,
                vector_collinear_merge_angle_deg=col_ang,
                vector_polyline_rdp_epsilon_px=rdp_eps,
                skip_ocr=not chk_export_ocr,
                ocr_gpu=ocr_gpu,
                mm_per_pixel=mm_per_px,
                write_viewer_bundle=True,
                viewer_layout_gap_mm=viewer_gap,
            )
            with st.spinner("Exporting…"):
                res = run_panel_dxfs(pcfg)
            st.success(f"Manifest: `{res.manifest_path}`")
            st.success(f"DXF count: {len(res.dxf_paths)}")
            if res.viewer_primary_dxf:
                st.success(f"Autodesk single upload: `{res.viewer_primary_dxf}`")
            for w in res.warnings:
                st.warning(w)
        finally:
            try:
                tmp_in.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
