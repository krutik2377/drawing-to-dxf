"""Lightweight line segment — shared by vectorize, geometry, linker, and DXF export."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    def midpoint(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
