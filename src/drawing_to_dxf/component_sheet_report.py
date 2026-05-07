"""Vision extraction → structured component sheet (JSON + Markdown), no codegen."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from drawing_to_dxf.ai_structured import call_gemini_generate_content_raw
from drawing_to_dxf.preprocess import load_image_bgr, load_pdf_page_as_bgr

COMPONENT_SHEET_PROMPT = """You are a senior structural / mechanical drafting analyst. The image is an engineering **component sheet** (shop drawing, detail portfolio, or tower members layout).

Return **ONLY** valid JSON (no markdown fences, no commentary). Shape:
{
  "sheet_title": "string — main title from title block or heading if visible",
  "drawing_number": "string or null",
  "scale_note": "string or null",
  "revision": "string or null",
  "date_drawn": "string or null",
  "components": [
    {
      "item_index": "string or number — printed detail/BOM index (e.g. 2, 10, 701)",
      "name": "string — full line like \\"LEG (ISA 90X90X8)\\"",
      "views": ["short labels for each orthographic/schematic view shown, e.g. elevation, plan, section"],
      "dimension_callouts": [
        { "description": "string", "value": "string — numeric text as printed", "unit": "mm or null" }
      ],
      "holes_and_features": "string or null — e.g. Ø18 HOLES (6 NOS.)",
      "summary_table": { "SECTION": "…", "LENGTH": "…", "QTY": "…", "MATERIAL": "…", "THICKNESS": "…", "DIAMETER": "…" }
    }
  ]
}

**Rules:**
- Extract exactly **5 to 7** components that are clearly separate **fabrication or supply** items (angles, plates, bolts, …). Prefer numbered details in order if the sheet lists many.
- **summary_table**: include only keys you can justify from the drawing; omit keys entirely if absent (empty object `{}` allowed).
- **dimension_callouts**: list printed dimensions, lengths, widths, heights, angles — as many discrete numbers as readable.
- Never invent BOM quantities or steel grades; use null or omit unknown fields.
- Keep JSON compact but complete."""

SHEET_EXTRACT_OUTPUT_TOKENS = 8192


def _gemini_api_key() -> str:
    k = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not k:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for component sheet extraction.")
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
        if c == "{":
            return i
    return -1


def decode_json_object_strict(text: str) -> dict[str, Any]:
    """Parse Gemini response → single JSON object (component sheet payload)."""
    t = _strip_json_fence(text)
    try:
        data = json.loads(t)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = _first_top_level_json_start(t)
    if start < 0:
        raise ValueError("Expected JSON object in model response")
    fragment = t[start:]
    data, _end = json.JSONDecoder().raw_decode(fragment)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    return data


def component_sheet_dict_to_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    title = data.get("sheet_title") or "Component sheet extraction"
    lines.append(f"## {title}")
    lines.append("")
    meta = []
    for key, lab in (
        ("drawing_number", "Drawing No."),
        ("scale_note", "Scale"),
        ("revision", "Rev"),
        ("date_drawn", "Date"),
    ):
        v = data.get(key)
        if v:
            meta.append(f"| {lab} | {v} |")
    if meta:
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        lines.extend(meta)
        lines.append("")

    comps = data.get("components")
    if not isinstance(comps, list):
        comps = []

    lines.append(f"### Components ({len(comps)})")
    lines.append("")

    for i, c in enumerate(comps, start=1):
        if not isinstance(c, dict):
            continue
        num = c.get("item_index", "")
        name = c.get("name") or f"Part {i}"
        heading = f"#### {i}. {num} — {name}" if num != "" else f"#### {i}. {name}"
        lines.append(heading)

        views = c.get("views")
        if isinstance(views, list) and views:
            lines.append("")
            lines.append("- **Views:** " + ", ".join(str(v) for v in views if v))

        dc = c.get("dimension_callouts")
        if isinstance(dc, list) and dc:
            lines.append("")
            lines.append("| Description | Value | Unit |")
            lines.append("| --- | --- | --- |")
            for row in dc:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("description", "")).replace("|", "\\|"),
                            str(row.get("value", "")).replace("|", "\\|"),
                            str(row.get("unit") or "").replace("|", "\\|"),
                        ]
                    )
                    + " |"
                )

        holes = c.get("holes_and_features")
        if holes:
            lines.append("")
            lines.append(f"- **Holes / features:** {holes}")

        st = c.get("summary_table")
        if isinstance(st, dict) and st:
            lines.append("")
            lines.append("| Key | Value |")
            lines.append("| --- | --- |")
            for k, v in st.items():
                if v is None or v == "":
                    continue
                lines.append(
                    f"| {str(k).replace('|', '\\|')} | {str(v).replace('|', '\\|')} |"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


@dataclass
class ComponentSheetExtractConfig:
    input_path: Path
    output_dir: Path
    gemini_model: str = "gemini-2.5-flash"
    pdf_page: int = 0
    pdf_dpi: float = 150.0
    max_side: int = 2048
    gemini_timeout_s: float = 180.0
    gemini_max_output_tokens: int = SHEET_EXTRACT_OUTPUT_TOKENS


@dataclass
class ComponentSheetExtractResult:
    json_path: Path
    markdown_path: Path
    raw_text_path: Path
    data: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def run_component_sheet_extract(cfg: ComponentSheetExtractConfig) -> ComponentSheetExtractResult:
    api_key = _gemini_api_key()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.input_path.stem
    warnings: list[str] = []

    bgr = _load_input_bgr(cfg.input_path, pdf_page=cfg.pdf_page, pdf_dpi=cfg.pdf_dpi)
    png_bytes = _bgr_to_png_bytes(bgr, cfg.max_side)

    raw = call_gemini_generate_content_raw(
        api_key=api_key,
        model=cfg.gemini_model,
        png_bytes=png_bytes,
        user_prompt=COMPONENT_SHEET_PROMPT,
        timeout_s=cfg.gemini_timeout_s,
        max_output_tokens=cfg.gemini_max_output_tokens,
    )
    raw_path = cfg.output_dir / f"{stem}_component_sheet_raw.txt"
    raw_path.write_text(raw, encoding="utf-8")

    try:
        data = decode_json_object_strict(raw)
    except ValueError as e:
        warnings.append(f"JSON parse failed: {e}")
        data = {
            "sheet_title": None,
            "parse_error": str(e),
            "components": [],
            "raw_text_path": str(raw_path.resolve()),
        }

    comps = data.get("components")
    n = len(comps) if isinstance(comps, list) else 0
    if n < 5:
        warnings.append(
            f"Only {n} component(s) extracted; target was 5–7 — image quality or title block may limit visibility."
        )
    elif n > 7:
        warnings.append(
            f"{n} components in JSON; prompt asked for 5–7 — verify or trim in post-processing."
        )

    jp = cfg.output_dir / f"{stem}_component_sheet.json"
    jp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    md = cfg.output_dir / f"{stem}_component_sheet.md"
    md.write_text(component_sheet_dict_to_markdown(data), encoding="utf-8")

    return ComponentSheetExtractResult(
        json_path=jp,
        markdown_path=md,
        raw_text_path=raw_path,
        data=data,
        warnings=warnings,
    )
