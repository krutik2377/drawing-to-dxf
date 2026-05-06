# Clean and run — `drawing-to-dxf`

Short checklist to reset outputs and run the project. Paths assume the repo root is `HyzenInterview`.

---

## 1. Prerequisites

- **Python 3.10+** installed and on `PATH`.
- Terminal open in the project folder:

  `cd c:\Users\admin\Desktop\HyzenInterview` (adjust if yours differs).

---

## 2. Clean previous outputs

Removes **`out/`**, **`out_sheet/`**, **`out_panels/`**, and **`.pytest_cache/`** if they exist (safe to run anytime):

```bash
python scripts/clean_outputs.py
```

To **also delete `.venv`** (full dependency reset):

```bash
python scripts/clean_outputs.py --include-venv
```

After that, do **section 3** again before running.

---

## 3. Environment (first time, or after `--include-venv`)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\activate
pip install -e ".[ocr]"
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -e ".[ocr]"
```

- **`[ocr]`** installs **EasyOCR** (for part numbers / text).  
- For DXF-only vectorization without splitting by part IDs: `pip install -e .` and use `--skip-ocr` when running (see below).

---

## 4. Run — main interview flow (one drawing → many component DXFs)

**Input:** PNG, JPG, TIFF, BMP, or **PDF** (first page by default).  
**Output:** AutoCAD‑compatible **DXF** files (open in AutoCAD; Save As DWG if you need DWG).

### Recommended: one DXF per panel (gridded shop-detail sheets)

Splits the page on **white gutters**, then **vectorizes each crop** separately (avoids full-page “wrong lines attached to part” behavior). See **`docs/PER_PART_DXF_PLAN.md`**.

```bash
drawing-to-dxf panels "C:\path\to\shop_details.png" -o out_panels --skip-ocr
```

With **EasyOCR** installed, drop `--skip-ocr` so filenames / labels can use detected part numbers (`pip install -e ".[ocr]"`).

**PDF example:**

```bash
drawing-to-dxf panels "C:\path\to\sheet.pdf" --pdf-page 0 --pdf-dpi 200 -o out_panels --skip-ocr
```

Tuning: `--panel-min-area`, `--panel-min-gap`, `--panel-min-short-side`, `--panel-max-aspect` (drop gutter ribbons), `--segment-merge-distance`, `--min-line-length`, `--mm-per-pixel`, `--max-side` (`0` = no downscale).

### Legacy: full-page convert (OCR + heuristic part linking)

```bash
drawing-to-dxf "C:\path\to\your_drawing.png" -o out
```

**Vectorize only** (no EasyOCR; single assembly DXF):

```bash
drawing-to-dxf "C:\path\to\your_drawing.png" --skip-ocr -o out
```

### What appears in `out_panels/` (panels command)

| File | Meaning |
|------|--------|
| `<stem>_panels_manifest.json` | Run metadata (keep local; not for Autodesk upload) |
| `viewer/<stem>_autodesk_layout.dxf` | **Primary** single file for [Autodesk Viewer](https://viewer.autodesk.com/) |
| `viewer/models/panel_XX.dxf` | One DXF per panel (no duplicate `Sample_panel_XX` in output root by default) |
| `viewer/README.txt`, `viewer/autodesk_viewer_upload.json` | Upload instructions |

Add `--emit-root-panel-dxfs` if you still want `Sample_panel_XX.dxf` copies in the output root.

### What appears in `out/` (default convert command)

| File | Meaning |
|------|--------|
| `<stem>_manifest.json` | Settings, counts, warnings, paths |
| `<stem>_assembly_layers.dxf` | Full sheet as lines |
| `<stem>_part_<id>.dxf` | One DXF per detected part (when OCR + linking work) |

Optional tuning (convert): `--part-regex`, `--padding-px`, `--max-nearest-px`, `--mm-per-pixel` (scale). See **`README.md`**.

### Interactive preview (recommended for unknown sheets)

Tune gutter split sliders and preview **automatic panel counts** plus **distinct OCR part-number hits** — no baked-in part totals.

```bash
pip install -e ".[ui,ocr]"
drawing-to-dxf ui-panels
```

Opens Streamlit in the browser: upload the sheet → **Analyze** → **Export** when satisfied.

---

## 5. Run — optional shop sheet workflow (panels + AI + composite PNG)

For **multi‑panel** fabrication layouts (split page → optional VLM JSON → grid PNG), not the main tower/single‑assembly DXF path:

```bash
drawing-to-dxf sheet "C:\path\to\shop_sheet.png" -o out_sheet --ai none
```

Use `--ai ollama` or `--ai openai` when configured; details in **`README.md`**.

---

## 6. Tests (optional)

```bash
pip install -e ".[ocr,dev]"
pytest -q
```

---

## 7. Diagrams

Workflow charts: **`docs/diagrams/workflows.md`**.

---

## Quick cycle for repeated demos

```text
python scripts/clean_outputs.py
.\.venv\Scripts\activate
drawing-to-dxf panels "your_file.png" -o out_panels --skip-ocr
```
