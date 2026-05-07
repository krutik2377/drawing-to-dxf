"""Unit tests for component sheet extraction helpers (no live API)."""

from __future__ import annotations

import json

import pytest

from drawing_to_dxf.component_sheet_report import (
    component_sheet_dict_to_markdown,
    decode_json_object_strict,
)


def test_decode_json_object_strict_strips_fence() -> None:
    inner = {"sheet_title": "T", "components": [{"name": "A"}]}
    text = f"```json\n{json.dumps(inner)}\n```"
    assert decode_json_object_strict(text)["sheet_title"] == "T"


def test_decode_json_object_strict_raw_decode_fragment() -> None:
    txt = 'prefix noise\n{"a": 1}'
    assert decode_json_object_strict(txt) == {"a": 1}


def test_decode_json_object_strict_rejects_not_object() -> None:
    with pytest.raises(ValueError, match="object"):
        decode_json_object_strict("[1,2]")


def test_component_sheet_dict_to_markdown_tables() -> None:
    md = component_sheet_dict_to_markdown(
        {
            "sheet_title": "My sheet",
            "drawing_number": "D-001",
            "components": [
                {
                    "item_index": 2,
                    "name": "LEG (ISA 90)",
                    "views": ["elevation", "section"],
                    "dimension_callouts": [
                        {"description": "Length", "value": "22500", "unit": "mm"}
                    ],
                    "holes_and_features": "Ø18",
                    "summary_table": {"SECTION": "ISA90", "QTY": "4"},
                }
            ],
        }
    )
    assert "My sheet" in md
    assert "D-001" in md
    assert "LEG (ISA 90)" in md
    assert "22500" in md
    assert "ISA90" in md
