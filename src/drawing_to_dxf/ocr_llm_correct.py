"""Optional OCR string cleanup via local Ollama or Gemini text (no vision required)."""

from __future__ import annotations

import json
import os
import re

from drawing_to_dxf.ai_structured import call_gemini_generate_text
from drawing_to_dxf.ocr_extract import TextBox


def _parse_json_string_list(text: str) -> list[str]:
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError("No JSON array in model output")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array")
    return [str(x).strip() for x in data]


def correct_text_boxes_ollama(
    boxes: list[TextBox],
    *,
    host: str,
    model: str,
    timeout_s: float = 120.0,
) -> tuple[list[TextBox], str | None]:
    """
    Batch-correct EasyOCR strings with a local text LLM. On parse/network failure,
    returns the original boxes and a short error string for manifest warnings.
    """
    if not boxes:
        return [], None
    from drawing_to_dxf.ai_structured import call_ollama_text

    texts = [b.text for b in boxes]
    payload = json.dumps(texts, ensure_ascii=False)
    prompt = (
        "You fix OCR errors on structural / mechanical drawings.\n"
        "Input: one JSON array of strings in reading order.\n"
        "Output: one JSON array of the SAME LENGTH with corrected strings only.\n"
        "Rules: fix obvious confusions in numeric tokens (e.g. O→0, l→1, Z→2, S→5 "
        "where context is clearly numeric); preserve units and letters in part codes; "
        "if unsure, keep the original token.\n"
        "Return nothing except the JSON array.\n\n"
        f"{payload}"
    )
    try:
        raw = call_ollama_text(host=host, model=model, user_prompt=prompt, timeout_s=timeout_s)
        fixed = _parse_json_string_list(raw)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as e:
        return list(boxes), f"Ollama OCR correction failed: {type(e).__name__}: {e}"
    if len(fixed) != len(boxes):
        return list(boxes), "Ollama OCR correction: output length mismatch; using raw OCR"
    out: list[TextBox] = []
    for tb, t in zip(boxes, fixed, strict=True):
        out.append(
            TextBox(
                text=t,
                confidence=tb.confidence,
                x0=tb.x0,
                y0=tb.y0,
                x1=tb.x1,
                y1=tb.y1,
            )
        )
    return out, None


def correct_text_boxes_gemini(
    boxes: list[TextBox],
    *,
    model: str,
    timeout_s: float = 120.0,
    api_key: str | None = None,
) -> tuple[list[TextBox], str | None]:
    """
    Batch-correct EasyOCR strings with Gemini (text-only). On parse/network failure,
    returns the original boxes and a short error string for manifest warnings.
    """
    if not boxes:
        return [], None
    key = (
        (api_key or "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not key:
        return list(boxes), "Gemini OCR correction skipped: GEMINI_API_KEY or GOOGLE_API_KEY missing"

    texts = [b.text for b in boxes]
    payload = json.dumps(texts, ensure_ascii=False)
    prompt = (
        "You fix OCR errors on structural / mechanical drawings.\n"
        "Input: one JSON array of strings in reading order.\n"
        "Output: one JSON array of the SAME LENGTH with corrected strings only.\n"
        "Rules: fix obvious confusions in numeric tokens (e.g. O→0, l→1, Z→2, S→5 "
        "where context is clearly numeric); preserve units and letters in part codes; "
        "if unsure, keep the original token.\n"
        "Return nothing except the JSON array.\n\n"
        f"{payload}"
    )
    try:
        raw = call_gemini_generate_text(
            api_key=key, model=model, user_prompt=prompt, timeout_s=timeout_s
        )
        fixed = _parse_json_string_list(raw)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as e:
        return list(boxes), f"Gemini OCR correction failed: {type(e).__name__}: {e}"
    if len(fixed) != len(boxes):
        return list(boxes), "Gemini OCR correction: output length mismatch; using raw OCR"
    out: list[TextBox] = []
    for tb, t in zip(boxes, fixed, strict=True):
        out.append(
            TextBox(
                text=t,
                confidence=tb.confidence,
                x0=tb.x0,
                y0=tb.y0,
                x1=tb.x1,
                y1=tb.y1,
            )
        )
    return out, None
