"""Optional vision LLM calls: OpenAI-compatible HTTPS or local Ollama (free locally)."""

from __future__ import annotations

import base64
import json
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class AIExtractResult:
    raw_text: str
    data: dict[str, Any]


def _b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def build_user_prompt(ocr_context: str) -> str:
    return (
        "You are assisting with a structural steel shop drawing panel. "
        "Use the image and the noisy OCR snippet below.\n"
        "Return ONE JSON object only (no markdown), keys:\n"
        '  "part_id": string or null,\n'
        '  "quantity": number or null,\n'
        '  "material_note": string or null,\n'
        '  "header_guess": string or null,\n'
        '  "dimensions_mm_guess": array of strings (free text),\n'
        '  "hole_notes": string or null,\n'
        '  "confidence_0_1": number\n'
        "Rules: guess conservatively; use null when unsure.\n"
        f"OCR context:\n{ocr_context}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        if chunks:
            return "\n".join(chunks)
    return str(content)


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    png_bytes: bytes,
    user_prompt: str,
    timeout_s: float = 120.0,
) -> AIExtractResult:
    """Chat Completions with image (OpenAI / Azure / any compatible gateway)."""
    b64 = _b64_png(png_bytes)
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI-compatible HTTP {e.code}: {err}") from e

    try:
        raw = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response: {payload!r}") from e

    return AIExtractResult(
        raw_text=_message_text(raw),
        data=_extract_json_object(_message_text(raw)),
    )


def call_ollama_chat(
    *,
    host: str,
    model: str,
    png_bytes: bytes,
    user_prompt: str,
    timeout_s: float = 180.0,
) -> AIExtractResult:
    """Local Ollama /api/chat with vision models (llava, llama3.2-vision, etc.). No API key."""
    b64 = _b64_png(png_bytes)
    url = host.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
                "images": [b64],
            }
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    raw = payload.get("message", {}).get("content", "")
    if not raw:
        raise RuntimeError(f"Unexpected Ollama response: {payload!r}")
    return AIExtractResult(raw_text=str(raw), data=_extract_json_object(_message_text(raw)))


