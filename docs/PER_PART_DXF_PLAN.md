# Plan: master drawing → one DXF per component

## Goal

- **Input:** Full engineering sheet (PNG/JPG/PDF page), e.g. `Resource/Sample.png` or gridded shop-detail sheets like `Resource/SampleOutput.png`.
- **Output:** **Multiple DXF files**—ideally one per fab part / per detail block—not a single merged “scribble” of the whole page.

## Why the old path failed

- **`drawing-to-dxf` (default convert):** One global Hough pass + OCR + heuristic linking of segments to part labels. On dense sheets, segments from other views or dimensions attach to the wrong part → nonsense per-part DXFs.
- **`drawing-to-dxf sheet`:** Splits panels and builds a **composite PNG** + JSON; it did **not** emit **per-panel DXFs**.

## Strategy (phased)

### Phase 1 — **Panel-first DXF export** (implemented now)

1. Preprocess the full page (same as existing pipeline: denoise, deskew, optional downscale).
2. **`split_panels`** on the processed grayscale: rectangular regions separated by white gutters (good fit for **grid shop-detail** layouts).
3. For **each** panel crop:
   - Run **vectorization only inside the crop** (Canny + Hough).
   - Write **one DXF** per panel (geometry + optional label).
4. Optional **OCR on the crop** to name files / label with a detected part id (`3–5` digit default regex); if OCR is off or no match, use `panel_00`, `panel_01`, …

**CLI:** `drawing-to-dxf panels <input> -o <dir> [options]`

**Manifest:** `*_panels_manifest.json` lists each panel’s bbox, segment count, output path, and warnings.

**Limits:** Assembly sheets where **one ink blob covers the whole page** (e.g. some tower elevations) may still yield **one** panel until gutter-based splitting or view detection improves (Phase 2).

### Phase 2 — Smarter regioning (not in this change)

- Title-line / frame detection: expand OCR boxes for “… **721**” into stable **part windows**.
- Optional: split **views** (elevation / plan / details) before part extraction on large assemblies.

### Phase 3 — Richer geometry (later)

- Circles for holes (`CIRCLE` in DXF), polyline fitting, optional PDF **vector** import instead of pixel tracing.

## Success criteria for Phase 1

- On synthetic / gridded tests, **N panels → N DXF files** with segment counts per file in the manifest.
- With `--skip-ocr`, **no EasyOCR** dependency for the happy path.
- Documentation points interview users to `panels` for **per-block** exports vs full-page `convert`.

## References in code

- Panel detection: `src/drawing_to_dxf/panel_split.py`
- Vectorize: `src/drawing_to_dxf/vectorize.py`
- DXF write: `src/drawing_to_dxf/export_dxf.py`
- New: `src/drawing_to_dxf/panel_dxf_pipeline.py`, CLI `panels` in `src/drawing_to_dxf/cli.py`
