"""Ollama and Gemini OCR correction helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.ocr_llm_correct import (
    _parse_json_string_list,
    correct_text_boxes_gemini,
)


def test_parse_json_string_list() -> None:
    assert _parse_json_string_list('prefix ["a", "b"] tail') == ["a", "b"]


def test_parse_json_string_list_invalid() -> None:
    with pytest.raises(ValueError):
        _parse_json_string_list("no array here")


def test_correct_text_boxes_gemini_parses_response() -> None:
    boxes = [
        TextBox(text="l0mm", confidence=0.9, x0=0, y0=0, x1=10, y1=10),
        TextBox(text="PART-O01", confidence=0.8, x0=1, y0=1, x1=20, y1=20),
    ]
    with patch("drawing_to_dxf.ocr_llm_correct.call_gemini_generate_text") as gem:
        gem.return_value = 'Sure.\n["10mm", "PART-001"]'
        out, err = correct_text_boxes_gemini(
            boxes, model="gemini-2.0-flash", api_key="test-key"
        )
    assert err is None
    assert len(out) == 2
    assert out[0].text == "10mm"
    assert out[1].text == "PART-001"
    assert out[0].confidence == 0.9
    gem.assert_called_once()
    call_kw = gem.call_args.kwargs
    assert call_kw["api_key"] == "test-key"
    assert call_kw["model"] == "gemini-2.0-flash"
    assert "l0mm" in call_kw["user_prompt"]


def test_correct_text_boxes_gemini_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    boxes = [TextBox(text="a", confidence=1.0, x0=0, y0=0, x1=1, y1=1)]
    out, err = correct_text_boxes_gemini(boxes, model="gemini-2.0-flash")
    assert out == boxes
    assert err is not None
    assert "GEMINI_API_KEY" in err or "GOOGLE_API_KEY" in err
