"""Optional semantic segmentation (ONNX) to mask non-geometry before skeleton tracing.

Requires: ``pip install -e ".[ml]"`` (onnxruntime). Supply a user-trained or converted
U-Net / SegFormer-style model; this module only handles a common NCHW or NHWC logits layout
and builds pixel-wise suppression masks for configured class names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "geometry",
    "text",
    "dimension",
    "title_block",
    "table",
    "symbol",
)

DEFAULT_SUPPRESS_FOR_SKELETON: tuple[str, ...] = (
    "text",
    "title_block",
    "table",
    "symbol",
)


@dataclass(frozen=True)
class SemanticSegLayout:
    """How class logits are arranged in the ONNX output tensor."""

    kind: str  # "nchw" | "nhwc"


@dataclass
class SemanticSegModelConfig:
    onnx_path: Path
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    suppress_for_skeleton: tuple[str, ...] = DEFAULT_SUPPRESS_FOR_SKELETON
    suppress_dimension: bool = False
    input_width: int = 512
    input_height: int = 512
    normalize: str = "zero_one"  # zero_one | imagenet | none
    layout: SemanticSegLayout | None = None
    providers: Sequence[str] | None = None


def load_semantic_seg_config(path: Path | None) -> dict[str, Any]:
    """Load optional JSON next to ONNX: class names, suppress lists, input size."""
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_model_config(
    onnx_path: Path,
    *,
    config_path: Path | None = None,
    input_width: int | None = None,
    input_height: int | None = None,
    suppress_dimension: bool = False,
    providers: Sequence[str] | None = None,
) -> SemanticSegModelConfig:
    raw = load_semantic_seg_config(config_path)
    names = tuple(raw["classes"]) if isinstance(raw.get("classes"), list) else DEFAULT_CLASS_NAMES
    sup = (
        tuple(raw["suppress_for_skeleton"])
        if isinstance(raw.get("suppress_for_skeleton"), list)
        else DEFAULT_SUPPRESS_FOR_SKELETON
    )
    iw = int(raw["input_width"]) if raw.get("input_width") is not None else (input_width or 512)
    ih = int(raw["input_height"]) if raw.get("input_height") is not None else (input_height or 512)
    norm = str(raw.get("normalize", "zero_one"))
    layout_raw = raw.get("layout")
    layout: SemanticSegLayout | None = None
    if layout_raw in ("nchw", "nhwc"):
        layout = SemanticSegLayout(str(layout_raw))
    sd = bool(raw.get("suppress_dimension", suppress_dimension))
    return SemanticSegModelConfig(
        onnx_path=onnx_path,
        class_names=names,
        suppress_for_skeleton=sup,
        suppress_dimension=sd,
        input_width=iw,
        input_height=ih,
        normalize=norm,
        layout=layout,
        providers=providers,
    )


def _detect_layout(shape: tuple[int, int, int, int], explicit: SemanticSegLayout | None) -> SemanticSegLayout:
    if explicit is not None:
        return explicit
    _, d1, d2, d3 = shape
    if d1 <= 64 and d1 < d2 and d1 < d3:
        return SemanticSegLayout("nchw")
    if d3 <= 64 and d3 < d1 and d3 < d2:
        return SemanticSegLayout("nhwc")
    return SemanticSegLayout("nchw")


def _logits_to_labels(logits: np.ndarray, layout: SemanticSegLayout) -> np.ndarray:
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D segmentation output, got shape {logits.shape}")
    if layout.kind == "nchw":
        return np.argmax(logits[0], axis=0).astype(np.int32)
    return np.argmax(logits[0], axis=2).astype(np.int32)


def _gray_to_model_input(
    gray: np.ndarray,
    size_hw: tuple[int, int],
    normalize: str,
) -> np.ndarray:
    h, w = size_hw
    g = gray.astype(np.float32)
    if g.ndim != 2:
        raise ValueError("grayscale expected")
    resized = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    if normalize == "none":
        tensor = bgr
    elif normalize == "imagenet":
        tensor = bgr
        tensor -= np.array([123.675, 116.28, 103.53], dtype=np.float32)
        tensor /= np.array([58.395, 57.12, 57.375], dtype=np.float32)
    else:  # zero_one
        tensor = bgr * (1.0 / 255.0)
    return np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def run_semantic_seg_labels(
    gray: np.ndarray,
    cfg: SemanticSegModelConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Semantic segmentation requires onnxruntime. Install: pip install -e \".[ml]\""
        ) from e

    h0, w0 = gray.shape[:2]
    sess_opts = ort.SessionOptions()
    prov = list(cfg.providers) if cfg.providers else None
    session = ort.InferenceSession(
        str(cfg.onnx_path.resolve()),
        sess_options=sess_opts,
        providers=prov or None,
    )
    in_meta = session.get_inputs()[0]
    in_name = in_meta.name
    shape = in_meta.shape
    ih, iw = cfg.input_height, cfg.input_width
    if isinstance(shape, list) and len(shape) == 4:
        def _dim(x: Any) -> int | None:
            if isinstance(x, int):
                return int(x)
            return None

        dh, dw = _dim(shape[2]), _dim(shape[3])
        if dh and dw:
            ih, iw = dh, dw

    tensor = _gray_to_model_input(gray, (ih, iw), cfg.normalize)
    outs = session.run(None, {in_name: tensor})
    if not outs:
        raise RuntimeError("ONNX model returned no outputs")
    logits = np.asarray(outs[0], dtype=np.float32)
    layout = _detect_layout(tuple(int(x) for x in logits.shape), cfg.layout)
    labels_small = _logits_to_labels(logits, layout)
    labels = cv2.resize(
        labels_small.astype(np.float32),
        (w0, h0),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
    meta = {
        "onnx_input_name": in_name,
        "layout": layout.kind,
        "logits_shape": list(logits.shape),
        "num_classes": int(logits.shape[1 if layout.kind == "nchw" else 3]),
    }
    return labels, meta


def suppression_mask_from_labels(
    labels: np.ndarray,
    cfg: SemanticSegModelConfig,
) -> np.ndarray:
    """Boolean mask (H, W): True where pixels should be cleared (set to white) before tracing."""
    name_to_idx = {n: i for i, n in enumerate(cfg.class_names)}
    clear: np.ndarray = np.zeros(labels.shape, dtype=bool)
    for name in cfg.suppress_for_skeleton:
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        clear |= labels == idx
    if cfg.suppress_dimension:
        idx = name_to_idx.get("dimension")
        if idx is not None:
            clear |= labels == idx
    return clear


def apply_suppression_to_gray(gray: np.ndarray, suppress: np.ndarray) -> np.ndarray:
    out = gray.copy()
    out[suppress] = np.uint8(255)
    return out


def semantic_prepare_gray_for_vectorize(
    gray: np.ndarray,
    *,
    onnx_path: Path,
    config_path: Path | None = None,
    suppress_dimension: bool = False,
    providers: Sequence[str] | None = None,
    return_pixel_labels: bool = False,
) -> (
    tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, np.ndarray, dict[str, Any]]
):
    """
    Run ONNX segmentation and inpaint suppressed classes to paper white.
    Returns (modified_gray, debug_manifest dict), or with ``return_pixel_labels=True``
    also the full-resolution int32 label raster (same class indices as the model).
    """
    mc = build_model_config(
        onnx_path,
        config_path=config_path,
        suppress_dimension=suppress_dimension,
        providers=providers,
    )
    labels, meta = run_semantic_seg_labels(gray, mc)
    mask = suppression_mask_from_labels(labels, mc)
    cleared = apply_suppression_to_gray(gray, mask)
    meta.update(
        {
            "class_names": list(mc.class_names),
            "suppress_for_skeleton": list(mc.suppress_for_skeleton),
            "suppress_dimension": mc.suppress_dimension,
            "suppressed_pixel_frac": float(np.mean(mask)),
        }
    )
    if return_pixel_labels:
        return cleared, labels, meta
    return cleared, meta
