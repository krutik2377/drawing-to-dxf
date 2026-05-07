"""AutoCAD / MCP post-processing hints for rough DXF exports.

Raster extraction (this package) produces geometry candidates; professional CAD cleanup
(JOIN, OVERKILL, constraint-style regularization) is delegated to AutoCAD—or an MCP server
that drives it—rather than reimplemented in Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_autocad_mcp_recipe(
    *,
    dxfs: list[str],
    mm_per_pixel: float,
) -> dict[str, Any]:
    """Structured recipe for an MCP agent or human operator using AutoCAD."""
    return {
        "version": 2,
        "description": (
            "Post-process rough DXF from drawing-to-dxf in AutoCAD (or via an AutoCAD MCP server). "
            "Upstream steps handle raster interpretation, OCR separation, semantic layer stubs, "
            "and primitive extraction; use the layer stubs (GEOMETRY, DIMENSION*, BORDER) as guides."
        ),
        "input_dxfs": list(dxfs),
        "scale_hint_mm_per_pixel": float(mm_per_pixel),
        "suggested_autocad_steps": [
            "OPEN — load a DXF from input_dxfs.",
            "AUDIT — report/fix entity errors.",
            "ZOOM E — review extents.",
            "LAYER — freeze DIMENSION_HINT / DIMENSION_ASSOC while cleaning part geometry if needed.",
            "QSELECT — filter by layer (GEOMETRY, UNASSIGNED, P_* parts, BORDER, OCR_TEXT).",
            "OVERKILL — remove duplicate overlays; start with a small tolerance scaled in mm.",
            "JOIN — merge collinear contiguous LINE segments (especially on GEOMETRY / P_* layers).",
            "PEDIT — Join / Multiple — convert fragmented LINE chains to LWPOLYLINE as appropriate.",
            "FILLET — corner cleanup where design intent is orthogonal metal/plate outlines.",
            "ALIGN — align inserts or duplicated panels if multi-block shop layout.",
            "DIMENSION / DIMSTYLE — replace DIMENSION_ASSOC stubs with drafting-standard dims if required.",
            "BLOCK — repeated plate outlines / symbols to insertable blocks.",
            "PURGE — unused layers and styles.",
            "SAVEAS — native DWG when satisfied.",
        ],
        "recommended_command_macro_order": [
            "OVERKILL→JOIN→PEDIT Join→FILLET (selective)→PURGE",
        ],
        "mcp_agent_hint": (
            "Expose these as invocations against the AutoCAD command line or .NET API. "
            "Prefer the assembly or layout DXF when a single file represents the sheet; "
            "otherwise batch panel / per-part DXFs. Run topology healing (JOIN/OVERKILL) after "
            "reviewing semantic overlay layers from raster classification."
        ),
    }


def write_autocad_mcp_recipe_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
