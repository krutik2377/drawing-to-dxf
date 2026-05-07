"""Unit tests for Gemini CAD codegen helpers (no live API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import pytest

from drawing_to_dxf.gemini_cad_codegen import (
    merge_dxfs_grid,
    merge_dxfs_horizontal,
    parse_codegen_json_payload,
)


def test_parse_codegen_json_payload_accepts_array_of_objects() -> None:
    text = r"""
    [
      {"bracket_A": "import ezdxf\n", "plate_B": "import sys\n"},
      {"clip_C": "print(1)\n"}
    ]
    """
    d = parse_codegen_json_payload(text)
    assert set(d.keys()) == {"bracket_A", "plate_B", "clip_C"}
    assert "ezdxf" in d["bracket_A"]


def test_parse_codegen_json_payload_strips_fence() -> None:
    inner = json.dumps([{"only": "x = 1\n"}])
    text = f"```json\n{inner}\n```"
    d = parse_codegen_json_payload(text)
    assert d == {"only": "x = 1\n"}


def test_merge_dxfs_grid_ncols_two(tmp_path: Path) -> None:
    tiles = []
    for i, tag in enumerate(["a", "b", "c"]):
        p = tmp_path / f"{tag}.dxf"
        doc = ezdxf.new()
        doc.modelspace().add_line((0, i * 10), (80, i * 10))
        doc.saveas(str(p))
        tiles.append((tag, p))
    out = tmp_path / "grid.dxf"
    merge_dxfs_grid(tiles, out, gap_mm=5.0, ncols=2)
    assert out.is_file()
    doc = ezdxf.readfile(str(out))
    lines = list(doc.modelspace().query("LINE"))
    assert len(lines) >= 3


def test_merge_dxfs_horizontal(tmp_path: Path) -> None:
    a = tmp_path / "a.dxf"
    b = tmp_path / "b.dxf"
    d1 = ezdxf.new()
    d1.modelspace().add_line((0, 0), (100, 0))
    d1.saveas(str(a))
    d2 = ezdxf.new()
    d2.modelspace().add_line((0, 0), (0, 50))
    d2.saveas(str(b))
    out = tmp_path / "m.dxf"
    merge_dxfs_horizontal([("a", a), ("b", b)], out, gap_mm=10.0)
    assert out.is_file()
    doc = ezdxf.readfile(str(out))
    lines = list(doc.modelspace().query("LINE"))
    assert len(lines) >= 2


def test_parse_codegen_json_payload_accepts_root_object_with_brackets_in_code() -> None:
    payload = '{"p1": "import sys\\nout = sys.argv[1]\\n"}'
    text = f"```json\n{payload}\n```"
    d = parse_codegen_json_payload(text)
    assert "p1" in d
    assert "argv[1]" in d["p1"]


def test_parse_codegen_json_payload_rejects_primitive_root() -> None:
    with pytest.raises(ValueError, match="object or array"):
        parse_codegen_json_payload("42")
