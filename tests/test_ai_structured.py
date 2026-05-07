"""Tests for optional LLM wrappers (Gemini mocked; no API key needed)."""

from __future__ import annotations

import json
from unittest.mock import patch

from drawing_to_dxf.ai_structured import call_gemini_generate_content


def test_call_gemini_generate_content_parses_json(monkeypatch) -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"part_id":"A1","quantity":2,"confidence_0_1":0.8}'},
                    ]
                }
            }
        ]
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    class FakeHttpResponse:
        """Mimic urlopen() return value: context manager whose __enter__ exposes .read()."""

        def __enter__(self) -> FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            # Full body in one read, like http.client HTTPResponse after a successful request.
            return body_bytes

    with patch(
        "drawing_to_dxf.ai_structured.urllib.request.urlopen",
        return_value=FakeHttpResponse(),
    ):
        out = call_gemini_generate_content(
            api_key="fake",
            model="gemini-2.5-flash",
            png_bytes=b"\x89PNG\r\n\x1a\n",
            user_prompt='Return {"hello": true}',
            timeout_s=5.0,
        )
    assert out.data["part_id"] == "A1"
    assert out.data["quantity"] == 2
