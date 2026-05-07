"""Experimental parallel path: Gemini analyzes a drawing image and returns Python source per
component; scripts are executed to emit DXF files, then merged into one horizontal layout.

**Security:** runs model-generated Python with the same privileges as the user process. Use only
on trusted inputs and in an isolated environment for untrusted models/images.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import ezdxf
import ezdxf.units
from ezdxf import bbox
from ezdxf.addons.importer import Importer
from ezdxf.math import Matrix44

from drawing_to_dxf.ai_structured import call_gemini_generate_content_raw
from drawing_to_dxf.preprocess import load_image_bgr, load_pdf_page_as_bgr


CODEGEN_SYSTEM_INSTRUCTION = """You are a senior mechanical/structural drafting assistant. The input is **always** treated as an **engineering production context** (shop drawings, part portfolios, lattice/component sheets, or assembly elevations). Your job is to **extract distinct engineering/manufacturing components** so each output DXF is usable **as a hand-off** to downstream CAD, nesting, or CAM—not a loose sketch.

Return ONLY valid JSON (no markdown fences, no commentary). Shape:
[
  {
    "part_740": "<full Python 3 source as one JSON string>",
    "part_721": "<full Python 3 source as one JSON string>"
  }
]

Split across several inner objects if needed; merge all key/value pairs logically.

**CRITICAL — stay within output token limits:** The whole answer is one JSON blob; long Python strings get truncated and break parsing. Each part's program must be **terse executable code only**:
- **No design narrative, no "re-interpreting" paragraphs, no duplicate explanations** inside the source strings. At most a few one-line `#` notes (datum, EST dims).
- Target **roughly 80–160 lines per part**; use lists/loops for repeated holes, not copy-paste blocks.
- Prefer simpler outlines (lines/circles/LWPolylines) over essay-length comments. **You must emit complete, valid JSON** with every string closed and final `]` present.

**Component count (required):** **Target 5–8 top-level JSON keys** (each key = one DXF script). **Never output more than 8.** Aim for **at least 5** whenever the drawing has enough distinct callouts/details/BOM cut items (see bullets below); count separate items as separate keys unless identical multiples are explicit (then one key + QTY in ANNOTATION).

---

## Goal: manufacturing-style extraction (match professional shop output)

1. **Find identifiable parts** and **stay within 5–8 components**:
   - **Numbered part blocks** (e.g. 3-digit callouts, index items): prefer **one JSON key per distinct numbered detail**, using `part_740`, `id_12`, `comp_05`, etc., to match what is printed.
   - **Tables / BOM rows**: read **QTY**, **material** (e.g. IS2062), **section** (ISA 90x90x8, PL 10), **thickness**, **length** where shown; echo the important fields as **MTEXT/TEXT** on layer ANNOTATION in that part's DXF.
   - **Assemblies / elevations** (tower, frame): do **not** output a single "whole tower" blob unless the image is only one assembly outline. **Decompose** into **fabrication-oriented pieces**: legs, cross-arm L/R, brace panels, gussets, base plates—each its own key when the drawing implies separate cut/purchased items, **until you reach at least 5** distinct components where the geometry supports it.
   - **If the sheet shows more than 8 parts:** keep the **8** most important (numbered details first, then structural plates/members others depend on); **do not** exceed 8 keys.
   - **If you have fewer than 5 clearly separable fabricated items:** **subdivide** only where the drawing gives separate views, outlines, or callouts for distinct plates/members so you reach **5** without inventing fictitious hardware. If the image **unambiguously** shows fewer than five real parts, output only those justified items rather than hallucinating extras.

2. **Part count discipline**: Prefer **5–8** concise, traceable scripts over one huge script or dozens of tiny ones. Merge duplicates only when the drawing explicitly says one detail applies to **identical** multiples—then **one script** with QTY called out in ANNOTATION. **Keep each script lean** (minimal boilerplate and comments, factor repeated geometry with small loops) so the total JSON stays short and fits typical token limits.

3. **Geometry fidelity**
   - Build the **2D plan/profile** of that part: outline as `LWPOLYLINE`/`LINE`, chamfers/cuts as segmented lines, holes as `msp.add_circle` in mm.
   - **Dimensions**: **Prefer numbers printed on the image.** Read extension lines and digit strings (including comma decimals if European style). Use **Ø / R** callouts for hole and fillet radii (circle radius = Ø/2).
   - **Hole grids**: When coordinates are chained from edges, reproduce those **X/Y from datum** in code variables; place circles accordingly.
   - If a number is **unreadable**, interpolate from proportions **once**, prefix annotation with `EST` or `~`, and keep relative layout consistent.

4. **Annotations (required in every script)**
   - Layers: create `GEOMETRY`, `DIMENSION`, `ANNOTATION` (e.g. `doc.layers.add("GEOMETRY")`).
   - Place **visible text**: part ID/title, **overall W×L** or length, **hole notes** (`Ø18`, `4× Ø24 PCD…`), **thickness/section** (`PL10`, `ISA 75×75×6`), **QTY** and **material** if present.
   - Use `msp.add_mtext` or `msp.add_text` with heights ~2–5 mm (readable at 1:1).

5. **Key naming**
   - **Primary**: printed detail/callout number.
   - **Secondary**: role + index (`cross_arm_top`, `gusset_10`) if no number is visible.

6. **Script contract** (each value string is a full program)
   - Imports: **only** `sys`, `ezdxf`, `math`, optional `pathlib`.
   - Read `out = sys.argv[1]`; `doc = ezdxf.new("R2010", setup=True)`; `doc.units = ezdxf.units.MM`.
   - Draw **only** that part's geometry + annotations; `doc.saveas(out)` once.
   - No subprocess, network, file reads beyond save, `eval`/`exec`.

7. **Order of work inside your head** (must reflect in code quality)
   - Locate part boundary → read overall dims → list holes (Ø + position) → outline → circles → dimensions text.
   - Keep code **structured** (variables for datum, plate_w, plate_h, hole list loops where patterns repeat).

Prioritize **traceability**: another engineer should see **which printed dimensions** drove the DXF. Stay compact in Python but **do not** skip holes or part metadata when they appear on the sheet."""




def _gemini_api_key() -> str:
    k = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not k:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini CAD codegen.")
    return k


def _load_input_bgr(path: Path, *, pdf_page: int, pdf_dpi: float) -> Any:
    p = str(path.resolve())
    suf = path.suffix.lower()
    if suf == ".pdf":
        img = load_pdf_page_as_bgr(p, page_index=pdf_page, dpi=pdf_dpi)
    else:
        img = load_image_bgr(p)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def _bgr_to_png_bytes(bgr: Any, max_side: int) -> bytes:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if max_side > 0 and m > max_side:
        s = max_side / float(m)
        bgr = cv2.resize(
            bgr,
            (max(1, int(w * s)), max(1, int(h * s))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG for Gemini request")
    return buf.tobytes()


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _first_top_level_json_start(s: str) -> int:
    """Index of the first `{` or `[` that starts the payload (not inside a JSON string)."""
    in_string = False
    escape = False
    for i, c in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c in "{[":
            return i
    return -1


def _decode_json_codegen_root(t: str) -> Any:
    t = t.strip()
    exc: json.JSONDecodeError | None = None
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        exc = e
    start = _first_top_level_json_start(t)
    if start < 0:
        raise ValueError(
            "Expected a JSON object or array in the model response"
        ) from exc
    fragment = t[start:]
    try:
        data, _end = json.JSONDecoder().raw_decode(fragment)
        return data
    except json.JSONDecodeError as e:
        suffix = ""
        msg = str(e)
        if "Unterminated string" in msg or "Expecting" in msg:
            suffix = (
                "; model output likely hit the token limit (use "
                "`--gemini-max-output-tokens 65536` if below the model cap, or "
                "fewer/lighter components in the prompt / smaller image)."
            )
        raise ValueError(f"{msg}{suffix}") from e


def parse_codegen_json_payload(text: str) -> dict[str, str]:
    """Parse Gemini response into component_id -> python source."""
    t = _strip_json_fence(text)
    data = _decode_json_codegen_root(t)
    out: dict[str, str] = {}
    chunks: list[dict[Any, Any]] = []
    if isinstance(data, dict):
        chunks = [data]
    elif isinstance(data, list):
        chunks = [x for x in data if isinstance(x, dict)]
    else:
        raise ValueError("Top-level JSON must be an object or array of objects")
    for item in chunks:
        for k, v in item.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            key = k.strip()
            if not key:
                continue
            code = str(v).replace("\r\n", "\n")
            if code:
                out[key] = code
    if not out:
        raise ValueError("No component scripts found in parsed JSON")
    return out


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-+]+", "_", name.strip())[:64].strip("_").strip("-")
    return s or "component"


def merge_dxfs_horizontal(
    labeled_paths: list[tuple[str, Path]],
    out_path: Path,
    *,
    gap_mm: float = 40.0,
) -> None:
    """Append each DXF's MODELSPACE into a new document, translating along +X."""
    dst = ezdxf.new("R2010", setup=True)
    dst.units = ezdxf.units.MM
    cursor_x = 0.0

    for _label, src_path in labeled_paths:
        if not src_path.is_file():
            continue
        try:
            src = ezdxf.readfile(str(src_path))
        except Exception:
            continue
        msp_src = src.modelspace()
        try:
            ext = bbox.extents(msp_src)
            min_x = float(ext.extmin.x) if ext.has_data else 0.0
            width = float(ext.size.x) if ext.has_data else 80.0
        except Exception:
            min_x, width = 0.0, 80.0

        n_before = len(list(dst.modelspace()))
        try:
            Importer(src, dst).import_modelspace()
        except Exception:
            continue
        ents = list(dst.modelspace())[n_before:]
        tx = cursor_x - min_x
        for e in ents:
            try:
                e.transform(Matrix44.translate(tx, 0.0, 0.0))
            except Exception:
                continue
        cursor_x += max(width, 1.0) + float(gap_mm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst.saveas(str(out_path))


def merge_dxfs_grid(
    labeled_paths: list[tuple[str, Path]],
    out_path: Path,
    *,
    gap_mm: float = 40.0,
    ncols: int = 3,
) -> None:
    """Append each DXF's MODELSPACE into a new document, tiling in rows of ``ncols`` along +X, next row −Y."""
    if ncols < 1:
        ncols = 1
    dst = ezdxf.new("R2010", setup=True)
    dst.units = ezdxf.units.MM
    cursor_x = 0.0
    current_row_y = 0.0
    row_max_h = 0.0

    for idx, (_label, src_path) in enumerate(labeled_paths):
        if not src_path.is_file():
            continue
        col = idx % ncols
        if col == 0 and idx > 0:
            current_row_y -= row_max_h + float(gap_mm)
            cursor_x = 0.0
            row_max_h = 0.0
        try:
            src = ezdxf.readfile(str(src_path))
        except Exception:
            continue
        msp_src = src.modelspace()
        try:
            ext = bbox.extents(msp_src)
            min_x = float(ext.extmin.x) if ext.has_data else 0.0
            min_y = float(ext.extmin.y) if ext.has_data else 0.0
            width = float(ext.size.x) if ext.has_data else 80.0
            height = float(ext.size.y) if ext.has_data else 80.0
        except Exception:
            min_x, min_y = 0.0, 0.0
            width, height = 80.0, 80.0

        n_before = len(list(dst.modelspace()))
        try:
            Importer(src, dst).import_modelspace()
        except Exception:
            continue
        ents = list(dst.modelspace())[n_before:]
        tx = cursor_x - min_x
        ty = current_row_y - min_y
        for e in ents:
            try:
                e.transform(Matrix44.translate(tx, ty, 0.0))
            except Exception:
                continue
        cursor_x += max(width, 1.0) + float(gap_mm)
        row_max_h = max(row_max_h, max(height, 1.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst.saveas(str(out_path))


@dataclass
class GeminiCadCodegenConfig:
    input_path: Path
    output_dir: Path
    gemini_model: str = "gemini-2.5-flash"
    pdf_page: int = 0
    pdf_dpi: float = 150.0
    max_side: int = 2048
    layout_gap_mm: float = 40.0
    assembly_layout: str = "horizontal"
    grid_columns: int = 3
    script_timeout_s: float = 90.0
    gemini_timeout_s: float = 600.0
    gemini_max_output_tokens: int = 65536
    sheet_report: bool = False
    sheet_gemini_max_output_tokens: int = 8192


@dataclass
class GeminiCadCodegenResult:
    manifest_path: Path
    assembly_dxf: Path | None
    raw_model_text_path: Path
    component_scripts: dict[str, Path] = field(default_factory=dict)
    component_dxfs: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    script_errors: dict[str, str] = field(default_factory=dict)
    component_sheet_json: Path | None = None
    component_sheet_markdown: Path | None = None
    component_sheet_warnings: list[str] = field(default_factory=list)


def run_gemini_cad_codegen(cfg: GeminiCadCodegenConfig) -> GeminiCadCodegenResult:
    api_key = _gemini_api_key()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.input_path.stem
    warnings: list[str] = []
    script_errors: dict[str, str] = {}
    sheet_json_path: Path | None = None
    sheet_md_path: Path | None = None
    sheet_warnings: list[str] = []

    bgr = _load_input_bgr(cfg.input_path, pdf_page=cfg.pdf_page, pdf_dpi=cfg.pdf_dpi)
    png_bytes = _bgr_to_png_bytes(bgr, cfg.max_side)

    if cfg.sheet_report:
        from drawing_to_dxf.component_sheet_report import (
            ComponentSheetExtractConfig,
            run_component_sheet_extract,
        )

        try:
            ser = run_component_sheet_extract(
                ComponentSheetExtractConfig(
                    input_path=cfg.input_path,
                    output_dir=cfg.output_dir,
                    gemini_model=cfg.gemini_model,
                    pdf_page=cfg.pdf_page,
                    pdf_dpi=cfg.pdf_dpi,
                    max_side=cfg.max_side,
                    gemini_timeout_s=cfg.gemini_timeout_s,
                    gemini_max_output_tokens=cfg.sheet_gemini_max_output_tokens,
                )
            )
            sheet_json_path = ser.json_path
            sheet_md_path = ser.markdown_path
            sheet_warnings = list(ser.warnings)
            for sw in sheet_warnings:
                warnings.append(f"Sheet report: {sw}")
        except Exception as e:
            warnings.append(f"Sheet report failed: {e}")

    raw_text = call_gemini_generate_content_raw(
        api_key=api_key,
        model=cfg.gemini_model,
        png_bytes=png_bytes,
        user_prompt=CODEGEN_SYSTEM_INSTRUCTION,
        timeout_s=cfg.gemini_timeout_s,
        max_output_tokens=cfg.gemini_max_output_tokens,
    )
    raw_path = cfg.output_dir / f"{stem}_gemini_codegen_raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    try:
        components = parse_codegen_json_payload(raw_text)
    except ValueError as e:
        warnings.append(f"JSON parse failed: {e}")
        manifest = {
            "stem": stem,
            "input": str(cfg.input_path.resolve()),
            "warnings": warnings,
            "raw_text_path": str(raw_path.resolve()),
            "components": {},
            "assembly_layout": cfg.assembly_layout,
            "grid_columns": cfg.grid_columns,
            "component_sheet": {
                "json": str(sheet_json_path.resolve()) if sheet_json_path else None,
                "markdown": str(sheet_md_path.resolve()) if sheet_md_path else None,
                "warnings": sheet_warnings,
            },
        }
        mp = cfg.output_dir / f"{stem}_gemini_codegen_manifest.json"
        mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return GeminiCadCodegenResult(
            manifest_path=mp,
            assembly_dxf=None,
            raw_model_text_path=raw_path,
            warnings=warnings,
            component_sheet_json=sheet_json_path,
            component_sheet_markdown=sheet_md_path,
            component_sheet_warnings=sheet_warnings,
        )

    n_comp = len(components)
    if n_comp < 5:
        warnings.append(
            f"Only {n_comp} component script(s) returned; prompt targets 5–8 distinct parts when the drawing supports them — "
            "try --gemini-max-output-tokens 65536, a sharper image, or a model with clearer part callouts."
        )
    elif n_comp > 8:
        warnings.append(
            f"{n_comp} component script(s) returned; prompt asks for at most 8 — "
            "verify the model respected the limit or tighten the prompt."
        )

    scripts_dir = cfg.output_dir / "gemini_codegen_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    dxfs_dir = cfg.output_dir / "gemini_codegen_dxfs"
    dxfs_dir.mkdir(parents=True, exist_ok=True)

    script_paths: dict[str, Path] = {}
    dxf_paths: dict[str, Path] = {}

    for comp_id, code in components.items():
        stem_c = _safe_stem(comp_id)
        py_path = scripts_dir / f"{stem_c}.py"
        py_path.write_text(code, encoding="utf-8")
        script_paths[comp_id] = py_path
        out_dxf = dxfs_dir / f"{stem_c}.dxf"
        py_abs = py_path.resolve()
        dxf_abs = out_dxf.resolve()
        try:
            proc = subprocess.run(
                [sys.executable, str(py_abs), str(dxf_abs)],
                capture_output=True,
                text=True,
                timeout=cfg.script_timeout_s,
                cwd=str(scripts_dir.resolve()),
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
                script_errors[comp_id] = err[:4000]
                warnings.append(f"Script failed for {comp_id}: {err[:500]}")
                continue
            if not out_dxf.is_file():
                script_errors[comp_id] = "Script exited 0 but DXF missing"
                warnings.append(f"No DXF written for {comp_id}")
                continue
            dxf_paths[comp_id] = out_dxf
        except subprocess.TimeoutExpired:
            script_errors[comp_id] = "timeout"
            warnings.append(f"Script timeout for {comp_id}")

    assembly: Path | None = None
    if dxf_paths:
        ordered = [(k, dxf_paths[k]) for k in components if k in dxf_paths]
        assembly = cfg.output_dir / f"{stem}_gemini_codegen_assembly.dxf"
        try:
            lay = (cfg.assembly_layout or "horizontal").strip().lower()
            if lay == "grid":
                merge_dxfs_grid(
                    ordered,
                    assembly,
                    gap_mm=cfg.layout_gap_mm,
                    ncols=int(cfg.grid_columns),
                )
            else:
                merge_dxfs_horizontal(ordered, assembly, gap_mm=cfg.layout_gap_mm)
        except Exception as e:
            warnings.append(f"Assembly merge failed: {e}")
            assembly = None

    manifest = {
        "stem": stem,
        "input": str(cfg.input_path.resolve()),
        "gemini_model": cfg.gemini_model,
        "gemini_max_output_tokens": cfg.gemini_max_output_tokens,
        "component_count_requested": len(components),
        "warnings": warnings,
        "raw_text_path": str(raw_path.resolve()),
        "components": {k: {"script": str(v.resolve())} for k, v in script_paths.items()},
        "dxfs": {k: str(v.resolve()) for k, v in dxf_paths.items()},
        "script_errors": script_errors,
        "assembly_dxf": str(assembly.resolve()) if assembly else None,
        "assembly_layout": cfg.assembly_layout,
        "grid_columns": cfg.grid_columns,
        "component_sheet": {
            "json": str(sheet_json_path.resolve()) if sheet_json_path else None,
            "markdown": str(sheet_md_path.resolve()) if sheet_md_path else None,
            "warnings": sheet_warnings,
        },
    }
    mp = cfg.output_dir / f"{stem}_gemini_codegen_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return GeminiCadCodegenResult(
        manifest_path=mp,
        assembly_dxf=assembly,
        raw_model_text_path=raw_path,
        component_scripts=script_paths,
        component_dxfs=dxf_paths,
        warnings=warnings,
        script_errors=script_errors,
        component_sheet_json=sheet_json_path,
        component_sheet_markdown=sheet_md_path,
        component_sheet_warnings=sheet_warnings,
    )
