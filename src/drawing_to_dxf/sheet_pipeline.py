"""Multi-panel shop sheet: split → OCR context → optional VLM → composite PNG + JSON manifest."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drawing_to_dxf.ai_structured import (
    AIExtractResult,
    build_user_prompt,
    call_ollama_chat,
    call_openai_compatible,
)
from drawing_to_dxf.ocr_extract import extract_text_boxes
from drawing_to_dxf.panel_split import split_panels
from drawing_to_dxf.preprocess import load_image_bgr, load_pdf_page_as_bgr, preprocess
from drawing_to_dxf.render_layout import PanelTile, render_composite_png


def _ocr_context_for_panel(gray: np.ndarray, gpu: bool) -> str:
    try:
        boxes = extract_text_boxes(gray, gpu=gpu)
    except Exception:  # noqa: BLE001
        return ""
    lines = [f"{b.text} ({b.confidence:.2f})" for b in sorted(boxes, key=lambda t: (t.y0, t.x0))]
    return "\n".join(lines[:80])


def _panel_title(ai: dict[str, Any] | None, ocr_hint: str) -> str:
    if ai:
        hg = ai.get("header_guess")
        pid = ai.get("part_id")
        if hg and pid:
            return f"{hg}  [{pid}]"
        if hg:
            return str(hg)
        if pid:
            return f"Part {pid}"
    first = ocr_hint.split("\n", 1)[0] if ocr_hint else ""
    return first[:100] if first else "Panel"


@dataclass
class SheetRunConfig:
    input_path: Path
    output_dir: Path
    is_pdf: bool = False
    pdf_page: int = 0
    pdf_dpi: float = 150.0
    max_side: int | None = 4096
    panel_min_area: int = 15_000
    panel_min_gap: int = 48
    ocr_gpu: bool = False
    ai_provider: str = "none"  # none | ollama | openai
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llava"
    layout_cols: int | None = None


@dataclass
class SheetRunResult:
    manifest_path: Path
    composite_png: Path
    warnings: list[str] = field(default_factory=list)
    panel_count: int = 0


def _load_bgr(cfg: SheetRunConfig) -> np.ndarray:
    if cfg.is_pdf:
        img = load_pdf_page_as_bgr(str(cfg.input_path), page_index=cfg.pdf_page, dpi=cfg.pdf_dpi)
        if img is None:
            raise FileNotFoundError(f"Cannot read PDF page {cfg.pdf_page}: {cfg.input_path}")
        return img
    img = load_image_bgr(str(cfg.input_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {cfg.input_path}")
    return img


def run_sheet(cfg: SheetRunConfig) -> SheetRunResult:
    warnings: list[str] = []
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.input_path.stem) or "sheet"

    bgr = _load_bgr(cfg)
    pre = preprocess(bgr, max_side=cfg.max_side, denoise=True, deskew=True)
    boxes = split_panels(
        pre.gray,
        min_area=cfg.panel_min_area,
        min_gap_px=cfg.panel_min_gap,
    )

    if len(boxes) == 1:
        warnings.append(
            "Only one panel detected — for multi-part sheets, try lowering --panel-min-area "
            "or crop tighter; contour/gap splitter may need tuning."
        )

    tiles: list[PanelTile] = []
    records: list[dict[str, Any]] = []

    api_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", cfg.openai_base_url)
    ollama_host = os.environ.get("OLLAMA_HOST", cfg.ollama_host)

    for i, (x, y, w, h) in enumerate(boxes):
        crop_bgr = pre.gray[y : y + h, x : x + w]
        crop_bgr3 = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".png", crop_bgr3)
        png_bytes = buf.tobytes()

        ocr_ctx = _ocr_context_for_panel(crop_bgr, gpu=cfg.ocr_gpu)
        ai_json: dict[str, Any] | None = None
        ai_err: str | None = None

        if cfg.ai_provider == "openai":
            if not api_key:
                ai_err = "OPENAI_API_KEY missing"
            else:
                try:
                    res: AIExtractResult = call_openai_compatible(
                        base_url=base_url,
                        api_key=api_key,
                        model=cfg.openai_model,
                        png_bytes=png_bytes,
                        user_prompt=build_user_prompt(ocr_ctx),
                    )
                    ai_json = res.data
                except Exception as e:  # noqa: BLE001
                    ai_err = f"{type(e).__name__}: {e}"
        elif cfg.ai_provider == "ollama":
            try:
                res = call_ollama_chat(
                    host=ollama_host,
                    model=cfg.ollama_model,
                    png_bytes=png_bytes,
                    user_prompt=build_user_prompt(ocr_ctx),
                )
                ai_json = res.data
            except Exception as e:  # noqa: BLE001
                ai_err = f"{type(e).__name__}: {e}"
        elif cfg.ai_provider != "none":
            warnings.append(f"Unknown ai_provider {cfg.ai_provider!r}; use none|openai|ollama")

        if ai_err:
            warnings.append(f"Panel {i}: AI skipped/failed: {ai_err}")

        title = _panel_title(ai_json, ocr_ctx)
        tiles.append(PanelTile(image_bgr=crop_bgr3, title=title, meta=ai_json or {}))
        records.append(
            {
                "index": i,
                "bbox_px": {"x": x, "y": y, "w": w, "h": h},
                "ocr_excerpt": ocr_ctx[:2000],
                "ai": ai_json,
            }
        )
        panel_png = out / f"{stem}_panel_{i:02d}.png"
        cv2.imwrite(str(panel_png), crop_bgr3)

    composite = out / f"{stem}_composite_layout.png"
    render_composite_png(tiles, composite, cols=cfg.layout_cols)

    manifest_path = out / f"{stem}_sheet_manifest.json"
    manifest = {
        "version": 1,
        "input": str(cfg.input_path.resolve()),
        "processed_size_px": {
            "width": int(pre.gray.shape[1]),
            "height": int(pre.gray.shape[0]),
        },
        "ai_provider": cfg.ai_provider,
        "panels": records,
        "outputs": {
            "composite_png": str(composite.resolve()),
            "panels_glob": f"{stem}_panel_*.png",
        },
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SheetRunResult(
        manifest_path=manifest_path,
        composite_png=composite,
        warnings=warnings,
        panel_count=len(boxes),
    )
