"""OCR-mask + skeleton + graph traces + circle hints → structured vector primitives."""

from __future__ import annotations

import math
from typing import MutableMapping, Sequence

import cv2
import numpy as np
from shapely.geometry import LineString
from skimage.morphology import remove_small_objects, skeletonize

from drawing_to_dxf.annotation_clean import (
    apply_text_masks,
    apply_text_masks_interior_only,
    inks_mask_from_gray,
)
from drawing_to_dxf.geometry_model import CircleDef, PolylineDef, VectorDrawing
from drawing_to_dxf.ocr_extract import TextBox
from drawing_to_dxf.skeleton_graph import trace_skeleton_polylines


def prepare_masked_gray(
    gray: np.ndarray,
    *,
    annotation_boxes: Sequence[TextBox] | None,
    annotation_pad_px: float = 5.0,
    mask_text_interior_only: bool = False,
) -> np.ndarray:
    """Grayscale after OCR masking (same prep as :func:`extract_vector_drawing`)."""
    if gray.ndim != 2:
        raise ValueError("grayscale expected")
    masked_gray = gray
    if annotation_boxes:
        if mask_text_interior_only:
            masked_gray = apply_text_masks_interior_only(
                masked_gray, annotation_boxes, pad_px=annotation_pad_px
            )
        else:
            masked_gray = apply_text_masks(masked_gray, annotation_boxes, pad_px=annotation_pad_px)
    return masked_gray


def detect_circles(
    gray_clean: np.ndarray,
    *,
    dp: float = 1.2,
    min_dist_px: float = 48.0,
    param1: int = 120,
    param2: int = 44,
    min_radius_px: int = 6,
    max_radius_px: int = 0,
) -> list[CircleDef]:
    blur = cv2.medianBlur(gray_clean, 5)
    h, w = blur.shape[:2]
    diag = math.hypot(float(w), float(h))
    min_dist_eff = max(float(min_dist_px), 0.038 * diag)
    if max_radius_px > 0:
        mr = int(max_radius_px)
    else:
        mr = min(int(0.42 * max(h, w)), int(0.19 * diag))
        mr = max(mr, min_radius_px + 6)

    def _one_hough(p2: int, min_r: int, max_r: int, md: float) -> list[CircleDef]:
        arr = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=float(dp),
            minDist=float(md),
            param1=param1,
            param2=p2,
            minRadius=max(1, int(min_r)),
            maxRadius=max(1, int(max_r)),
        )
        if arr is None:
            return []
        return [CircleDef(cx=float(row[0]), cy=float(row[1]), r=float(row[2])) for row in arr[0]]

    primary = _one_hough(param2, min_radius_px, mr, min_dist_eff)
    mr_small = max(8, min(36, mr))
    small_md = max(14.0, min_dist_eff * 0.55, 0.018 * diag)
    secondary = _one_hough(36, max(4, min_radius_px - 2), mr_small, small_md)

    return _dedupe_circles(primary + secondary, merge_dist_px=min(min_dist_eff, 36.0))


def _dedupe_circles(circles: list[CircleDef], *, merge_dist_px: float) -> list[CircleDef]:
    merged: list[CircleDef] = []
    md2 = merge_dist_px * merge_dist_px
    for c in circles:
        redundant = False
        for m in merged:
            dx = c.cx - m.cx
            dy = c.cy - m.cy
            if dx * dx + dy * dy <= md2 and abs(c.r - m.r) < max(12.0, 0.2 * max(c.r, m.r, 1.0)):
                redundant = True
                break
        if not redundant:
            merged.append(c)
    return merged


def _circle_circumference_ink_fraction(
    c: CircleDef,
    ink_fg: np.ndarray,
    *,
    samples: int = 72,
) -> float:
    """Fraction of circumference samples that land on foreground ink."""
    h, w = ink_fg.shape[:2]
    if samples <= 0 or c.r < 1.5:
        return 0.0
    rr = float(c.r)
    hits = 0
    for k in range(samples):
        th = 2.0 * math.pi * k / float(samples)
        ix = int(round(c.cx + rr * math.cos(th)))
        iy = int(round(c.cy + rr * math.sin(th)))
        if 0 <= ix < w and 0 <= iy < h and ink_fg[iy, ix]:
            hits += 1
    return hits / float(samples)


def _circle_interior_ink_fraction(
    c: CircleDef,
    ink_raw_fg: np.ndarray,
    *,
    radial_fracs: tuple[float, ...] | None = None,
    spokes: int = 20,
) -> float:
    """
    Mean ink density inside the circle (not on the rim). True drill holes are mostly paper/background
    inside the ring; line-art interior is often ink.
    """
    h, w = ink_raw_fg.shape[:2]
    if c.r < 2.5:
        return 1.0
    rr = float(c.r)
    if radial_fracs is None:
        radial_fracs = (0.10, 0.18, 0.26, 0.34) if rr < 14.0 else (0.18, 0.28, 0.38, 0.48)
    hits = 0
    total = 0
    for k in range(spokes):
        th = 2.0 * math.pi * k / float(spokes)
        cos_t, sin_t = math.cos(th), math.sin(th)
        for rf in radial_fracs:
            r = rr * rf
            if r < 1.15:
                continue
            ix = int(round(c.cx + r * cos_t))
            iy = int(round(c.cy + r * sin_t))
            if 0 <= ix < w and 0 <= iy < h:
                total += 1
                if ink_raw_fg[iy, ix]:
                    hits += 1
    return hits / float(max(total, 1))


def _filter_circles_by_ink_ring(
    circles: list[CircleDef],
    ink_binary_255: np.ndarray,
    *,
    min_ring_fraction: float = 0.56,
    max_interior_ink_hollow: float = 0.36,
    max_circles: int = 120,
    ring_dilate: int = 2,
) -> list[CircleDef]:
    """
    Drop line-art Hough false positives while keeping:

    - **Hollow** holes (rim on ink, interior mostly paper),
    - **Filled** drill dots (small disks — interior mostly ink, rim on ink),
    - **Strong small** rims where interior sampling is ambiguous.
    """
    if not circles or ink_binary_255.ndim != 2:
        return circles
    raw_fg = ink_binary_255 > 0
    ring_fg = raw_fg
    if ring_dilate > 0:
        ksz = ring_dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        ring_fg = cv2.dilate(ring_fg.astype(np.uint8), kernel, iterations=1).astype(bool)

    def _accept(c: CircleDef, ring: float, inside: float) -> tuple[float, bool]:
        r = float(c.r)
        # Filled drill symbol (common in shop drawings)
        if r <= 24.0 and ring >= 0.48 and 0.38 <= inside <= 1.0:
            return (ring + 0.25 * min(inside, 0.95), True)
        # Anti-aliased / subpixel tiny dots: strong interior, slightly weaker rim
        if r <= 12.0 and inside >= 0.42 and ring >= 0.34 and inside <= 1.0:
            return (ring + 0.22 * min(inside, 0.95), True)
        # Through-hole ring
        if ring >= min_ring_fraction and inside <= max_interior_ink_hollow:
            return (ring - 0.52 * inside, True)
        # Small feature: lenient interior (sampling crosses thin ring stroke)
        if r <= 15.0 and ring >= 0.52 and inside <= 0.48:
            return (ring - 0.35 * inside, True)
        # High-confidence rim (Hough sharp on real curve)
        if r <= 26.0 and ring >= 0.76 and inside <= 0.50:
            return (ring - 0.40 * inside, True)
        return (0.0, False)

    scored: list[tuple[float, CircleDef]] = []
    for c in circles:
        ring = _circle_circumference_ink_fraction(c, ring_fg, samples=80)
        inside = _circle_interior_ink_fraction(c, raw_fg)
        score, ok = _accept(c, ring, inside)
        if ok:
            scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    if len(scored) > max_circles:
        scored = scored[:max_circles]
    return [c for _s, c in scored]


def _supplement_circles_from_round_ink_blobs(
    ink_binary_255: np.ndarray,
    existing: Sequence[CircleDef],
    *,
    min_r: float = 3.0,
    max_r: float = 32.0,
    min_circularity: float = 0.74,
    max_add: int = 72,
) -> list[CircleDef]:
    """
    Catch small round **ink** blobs Hough often misses (solid drill dots, heavy rings after threshold).
    """
    if ink_binary_255.ndim != 2:
        return []
    ink_u8 = np.where(np.asarray(ink_binary_255) > 0, np.uint8(255), np.uint8(0)).astype(np.uint8)
    contours, _ = cv2.findContours(ink_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand: list[CircleDef] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 18.0 or area > math.pi * max_r * max_r * 1.35:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1e-3:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        if circ < min_circularity:
            continue
        (_, _), r_enc = cv2.minEnclosingCircle(cnt)
        if r_enc < min_r or r_enc > max_r:
            continue
        fill_ratio = area / max(math.pi * r_enc * r_enc, 1e-3)
        if fill_ratio < 0.38:
            continue
        m = cv2.moments(cnt)
        if m["m00"] < 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        cand.append(CircleDef(cx=cx, cy=cy, r=float(r_enc)))

    if not cand:
        return []
    md = max(8.0, min_r * 2.4)
    merged = _dedupe_circles(cand, merge_dist_px=md)

    def _near_existing(c: CircleDef) -> bool:
        for e in existing:
            d2 = (c.cx - e.cx) ** 2 + (c.cy - e.cy) ** 2
            if d2 <= (max(10.0, 1.1 * max(c.r, e.r))) ** 2:
                return True
        return False

    extra = [c for c in merged if not _near_existing(c)]
    scored: list[tuple[float, CircleDef]] = []
    for c in extra:
        ring = _circle_circumference_ink_fraction(c, ink_u8 > 0, samples=48)
        inside = _circle_interior_ink_fraction(c, ink_u8 > 0)
        if ring < 0.45:
            continue
        if inside < 0.28 and ring < 0.58:
            continue
        scored.append((ring + 0.2 * min(inside, 0.9), c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _s, c in scored[:max_add]]


def _erase_circumference_rings(binary_ink_255: np.ndarray, circles: Sequence[CircleDef]) -> np.ndarray:
    """Zero-out circumference bands so skeletonization does not re-trace circles."""
    out = binary_ink_255.copy()
    for c in circles:
        rr = float(c.r)
        if rr <= 1.5:
            continue
        ix, iy = int(round(c.cx)), int(round(c.cy))
        thickness = (
            max(3, min(41, int(0.045 * rr)))
            if rr <= 460
            else max(8, min(71, int(0.03 * rr)))
        )
        cv2.circle(out, (ix, iy), int(round(rr)), color=0, thickness=int(thickness), lineType=cv2.LINE_AA)
    return out


def _simplify_polyline_xy(xy: list[tuple[float, float]], *, tol_px: float) -> list[tuple[float, float]]:
    if len(xy) < 3 or tol_px <= 0:
        return list(xy)
    geom = LineString(xy)
    sim = geom.simplify(tolerance=tol_px, preserve_topology=False)
    if sim.geom_type == "LineString":
        pts = [(float(px), float(py)) for px, py in sim.coords]
        return pts if len(pts) >= 2 else list(xy)
    if sim.geom_type == "GeometryCollection":
        parts: list[list[tuple[float, float]]] = []
        for g in getattr(sim, "geoms", ()):
            if g.geom_type == "LineString" and len(g.coords) >= 2:
                parts.append([(float(px), float(py)) for px, py in g.coords])
        if len(parts) == 1:
            return parts[0]
    coords = getattr(sim, "coords", ())
    pts = [(float(cx), float(cy)) for cx, cy in coords]
    return pts if len(pts) >= 2 else list(xy)


def extract_vector_drawing(
    gray: np.ndarray,
    *,
    annotation_boxes: Sequence[TextBox] | None,
    annotation_pad_px: float = 5.0,
    min_skeleton_branch_px: int = 28,
    rdp_tolerance_px: float = 2.25,
    enable_circles: bool = True,
    mask_text_interior_only: bool = False,
    soft_ink_mask: bool = False,
    debug_stages: MutableMapping[str, np.ndarray] | None = None,
) -> VectorDrawing:
    """
    OCR-mask → ink morphology → circumference removal from Hough circles → skeletonize → graph traces → Shapely simplify.

    ``annotation_boxes=None`` skips masking rectangles (dimensions/text may remain).

    With ``mask_text_interior_only=True``, OCR boxes wipe only inferred letter interiors (narrower aggression).

    ``soft_ink_mask=True`` skips horizontal dimension-line suppression ahead of adaptive threshold.

    ``debug_stages`` optional dict filled with debug image arrays (masked gray ``uint8``, binary/skeleton masks).
    """
    if gray.ndim != 2:
        raise ValueError("grayscale expected")

    masked_gray = prepare_masked_gray(
        gray,
        annotation_boxes=annotation_boxes,
        annotation_pad_px=annotation_pad_px,
        mask_text_interior_only=mask_text_interior_only,
    )

    ink = inks_mask_from_gray(masked_gray, suppress_ruling_lines=not soft_ink_mask)

    circles: list[CircleDef] = []
    if enable_circles:
        circles = detect_circles(masked_gray)
        circles = _filter_circles_by_ink_ring(circles, ink)
        blob_extra = _supplement_circles_from_round_ink_blobs(ink, circles)
        if blob_extra:
            circles = _dedupe_circles(circles + blob_extra, merge_dist_px=22.0)

    ink = _erase_circumference_rings(ink, circles)

    fg = ink > 0
    min_keep_px = max(48, int(min_skeleton_branch_px) * 8)
    fg = remove_small_objects(fg, min_size=min_keep_px, connectivity=2)
    sk_bool = skeletonize(fg.astype(bool))

    if debug_stages is not None:
        debug_stages["masked_gray"] = masked_gray
        debug_stages["binary_ink"] = ink
        debug_stages["skeleton_bool"] = (sk_bool.astype(np.uint8)) * np.uint8(255)

    raw_paths_xy, closes = trace_skeleton_polylines(sk_bool)

    polylines_out: list[PolylineDef] = []
    simp_tol = max(0.4, float(rdp_tolerance_px))
    for pts, clo in zip(raw_paths_xy, closes, strict=True):
        if len(pts) < 2:
            continue
        simp = list(pts)
        if len(simp) >= 3:
            simp = _simplify_polyline_xy(simp, tol_px=simp_tol)
        if clo and len(simp) >= 3:
            dup = simp + [simp[0]]
            dup = _simplify_polyline_xy(dup, tol_px=simp_tol)
            if len(dup) >= 4 and dup[0] == dup[-1]:
                dup = dup[:-1]
            simp = dup

        if len(simp) < 2:
            continue
        cnt = cv2.arcLength(np.array(simp, dtype=np.float32).reshape(-1, 1, 2), bool(clo))
        if cnt + 1e-6 < float(min_skeleton_branch_px):
            continue

        uniq = len({(round(x * 10), round(y * 10)) for x, y in simp})
        if uniq < 2:
            continue
        polylines_out.append(PolylineDef(points=simp, closed=clo and uniq >= 3))

    return VectorDrawing(polylines=polylines_out, circles=list(circles), arcs=[], residual_segments=[])
